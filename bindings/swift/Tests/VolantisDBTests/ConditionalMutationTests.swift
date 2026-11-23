import Foundation
import XCTest

import VolantisDB

@available(macOS 14.0, *)
final class ConditionalMutationTests: XCTestCase {
    func testConditionalUpsertCommitsOrReportsFirstConflict() async throws {
        let root = testRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await openRecords(at: root)
        let first = DocumentID(high: 3, low: 1)
        let second = DocumentID(high: 4, low: 2)
        let new = DocumentID(high: 5, low: 3)
        let original = try await store.upsert([
            document(first, revision: 1), document(second, revision: 2)
        ])
        let replacement = document(first, revision: 2, kind: "updated")
        let next = document(second, revision: 3)
        let created = document(new, revision: 1)

        do {
            _ = try await store.upsert(conditional: [
                ConditionalUpsertDocument(document: replacement),
                ConditionalUpsertDocument(document: next, expectedRevision: .exactly(1)),
                ConditionalUpsertDocument(document: created, expectedRevision: .absent)
            ])
            XCTFail("conditional upsert accepted a mismatched revision")
        } catch let conflict as RevisionConflict {
            XCTAssertEqual(conflict.code, .revisionConflict)
            XCTAssertEqual(conflict.index, 1)
            XCTAssertEqual(conflict.id, second)
            XCTAssertEqual(conflict.expectedRevision, .exactly(1))
            XCTAssertEqual(conflict.currentRevision, 2)
        }
        let unchanged = try await store.get([first, second, new])
        XCTAssertEqual(unchanged.generation, original.generation)
        XCTAssertEqual(unchanged.documents.map { $0?.revision }, [1, 2, nil])

        let committed = try await store.upsert(conditional: [
            ConditionalUpsertDocument(document: replacement),
            ConditionalUpsertDocument(document: next, expectedRevision: .exactly(2)),
            ConditionalUpsertDocument(document: created, expectedRevision: .absent)
        ])
        let result = try await store.get([first, second, new])
        XCTAssertGreaterThan(committed.sequence, original.sequence)
        XCTAssertGreaterThan(committed.generation, original.generation)
        XCTAssertEqual(result.generation, committed.generation)
        XCTAssertEqual(result.documents.map { $0?.revision }, [2, 3, 1])
        XCTAssertEqual(result.documents.first??.text, "updated")
        XCTAssertEqual(result.documents.first??.attributes[1], .string("updated"))
        try await store.close()
    }

    func testConditionalDeleteCommitsOrReportsFirstConflict() async throws {
        let root = testRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await openRecords(at: root)
        let live = DocumentID(high: 3, low: 1)
        let missing = DocumentID(high: 4, low: 2)
        let survivor = DocumentID(high: 5, low: 3)
        let original = try await store.upsert([
            document(live, revision: 7), document(survivor, revision: 2)
        ])

        do {
            _ = try await store.delete(conditional: [
                DeleteTarget(id: live, expectedRevision: .exactly(7)),
                DeleteTarget(id: missing, expectedRevision: .exactly(1))
            ])
            XCTFail("conditional delete accepted an exact revision for a missing document")
        } catch let conflict as RevisionConflict {
            XCTAssertEqual(conflict.code, .revisionConflict)
            XCTAssertEqual(conflict.index, 1)
            XCTAssertEqual(conflict.id, missing)
            XCTAssertEqual(conflict.expectedRevision, .exactly(1))
            XCTAssertNil(conflict.currentRevision)
        }
        let unchanged = try await store.get([live, missing, survivor])
        XCTAssertEqual(unchanged.generation, original.generation)
        XCTAssertEqual(unchanged.documents.map { $0?.revision }, [7, nil, 2])

        let committed = try await store.delete(conditional: [
            DeleteTarget(id: live, expectedRevision: .exactly(7)),
            DeleteTarget(id: missing, expectedRevision: .absent)
        ])
        let result = try await store.get([live, missing, survivor])
        XCTAssertGreaterThan(committed.sequence, original.sequence)
        XCTAssertGreaterThan(committed.generation, original.generation)
        XCTAssertEqual(result.generation, committed.generation)
        XCTAssertEqual(result.documents.map { $0?.revision }, [nil, nil, 2])
        try await store.close()
    }

    func testDeleteWhereRemovesMatchesAndRejectsInvalidFilter() async throws {
        let root = testRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await openRecords(at: root)
        let sealed = DocumentID(high: 3, low: 1)
        let active = DocumentID(high: 4, low: 2)
        let survivor = DocumentID(high: 5, low: 3)
        let ids = [sealed, active, survivor]
        _ = try await store.upsert([document(sealed, revision: 1, kind: "remove")])
        _ = try await store.seal()
        let original = try await store.upsert([
            document(active, revision: 1, kind: "remove"), document(survivor, revision: 1)
        ])

        do {
            _ = try await store.deleteWhere(.eq(99, .string("remove")))
            XCTFail("deleteWhere accepted an undeclared attribute")
        } catch let error as VolantisError {
            XCTAssertEqual(error, .invalidArgument)
        }
        let unchanged = try await store.get(ids)
        XCTAssertEqual(unchanged.generation, original.generation)
        XCTAssertEqual(unchanged.documents.map { $0?.id }, ids.map(Optional.some))

        let filter = Filter.and([.exists(1), .eq(1, .string("remove"))])
        let removed = try await store.deleteWhere(filter)
        XCTAssertEqual(removed.deleted, 2)
        XCTAssertGreaterThan(removed.generation, original.generation)
        let remaining = try await store.get(ids)
        XCTAssertEqual(remaining.documents.map { $0?.id }, [nil, nil, survivor])
        XCTAssertEqual(remaining.generation, removed.generation)

        let noMatch = try await store.deleteWhere(filter)
        XCTAssertEqual(noMatch.deleted, 0)
        XCTAssertEqual(noMatch.generation, removed.generation)
        try await store.close()
    }

    func testDeleteWhereAcceptsABIFilterNodeLimit() async throws {
        let root = testRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await openRecords(at: root)
        // The C ABI allows 2^20 nodes, including the root.
        let filter = Filter.or(Array(repeating: .exists(1), count: (1 << 20) - 1))
        let report = try await store.deleteWhere(filter)
        XCTAssertEqual(report.deleted, 0)
        try await store.close()
    }

    private func testRoot() -> URL {
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/test-namespaces", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
    }

    private func openRecords(at root: URL) async throws -> VolantisStore {
        try await VolantisStore.openNamespace(
            root: root,
            name: "records",
            spec: NamespaceSpec(
                attributes: [
                    AttributeDefinition(id: 1, name: "kind", type: .dictionaryString, nullable: false)
                ],
                vectorSpace: nil
            )
        )
    }

    private func document(_ id: DocumentID, revision: UInt64, kind: String = "keep") -> IngestDocument {
        IngestDocument(
            id: id, revision: revision, timestamp: 0, vector: [], text: kind,
            attributes: [1: .string(kind)]
        )
    }
}
