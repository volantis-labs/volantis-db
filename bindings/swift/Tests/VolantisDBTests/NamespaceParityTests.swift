import Foundation
import XCTest
import VolantisDB

@available(macOS 14.0, *)
final class NamespaceParityTests: XCTestCase {
    private let recordSpec = NamespaceSpec(attributes: [], vectorSpace: nil)
    private let readOnly = OpenOptions(accessMode: .readOnly, excludeFromBackup: false)

    func testOpenNamespaceWithTokenizer() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let spec = NamespaceSpec(attributes: [], vectorSpace: nil, tokenizerProfile: .voice)
        let store = try await VolantisStore.openNamespace(root: root, name: "transcript", spec: spec)
        _ = try await store.upsert([document(1, text: "twenty five")])
        try await store.quiesce()
        try await store.resume()
        let result = try await store.query(text: "25", options: QueryOptions(k: 1))
        XCTAssertEqual(result.hits.map(\.documentID), [id(1)])
        try await store.close()

        var changed = spec
        changed.tokenizerProfile = .code
        do {
            _ = try await VolantisStore.openNamespace(root: root, name: "transcript", spec: changed)
            XCTFail("reopened a voice namespace as code")
        } catch let error as VolantisError {
            XCTAssertEqual(error, .epochMismatch)
        }
    }

    func testNamespaceBatch() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let tagged = NamespaceSpec(
            attributes: [AttributeDefinition(id: 1, name: "obsolete", type: .bool, nullable: true)],
            vectorSpace: nil,
            tokenizerProfile: .code
        )
        try await seed(root, "left", tagged, [
            document(1), document(2), document(3, attributes: [1: .bool(true)])
        ])
        try await seed(root, "right", recordSpec, [])
        let left = NamespaceMutation(
            name: "left",
            spec: tagged,
            upserts: [NamespaceUpsert(document(1, revision: 2, text: "replacement"), expectedRevision: .exactly(1))],
            deletes: [id(2)],
            deleteWhere: .eq(1, .bool(true))
        )
        let right = NamespaceMutation(
            name: "right",
            spec: recordSpec,
            upserts: [NamespaceUpsert(document(4), expectedRevision: .absent)]
        )
        var invalid = left
        invalid.deleteWhere = .eq(999, .bool(true))
        do {
            _ = try await VolantisStore.namespaceBatch(root: root, participants: [right, invalid])
            XCTFail("batch accepted a filter on an undeclared attribute")
        } catch let error as VolantisError {
            XCTAssertEqual(error, .invalidArgument)
        }
        let beforeLeft = try await read(root, "left", tagged, [id(1), id(2), id(3)])
        let beforeRight = try await read(root, "right", recordSpec, [id(4)])
        XCTAssertEqual(beforeLeft.documents.map { $0?.revision }, [1, 1, 1])
        XCTAssertEqual(beforeRight.missingCount, 1)

        let generations = try await VolantisStore.namespaceBatch(root: root, participants: [right, left])
        let afterLeft = try await read(root, "left", tagged, [id(1), id(2), id(3)])
        let afterRight = try await read(root, "right", recordSpec, [id(4)])
        XCTAssertEqual(afterLeft.documents.map { $0?.revision }, [2, nil, nil])
        XCTAssertEqual(afterLeft.documents.first??.text, "replacement")
        XCTAssertEqual(afterRight.documents.map { $0?.id }, [id(4)])
        XCTAssertEqual(generations.count, 2)
        let rightGeneration = try XCTUnwrap(generations.first)
        let leftGeneration = try XCTUnwrap(generations.last)
        // Reopen may checkpoint past the generation acknowledged by the batch.
        XCTAssertGreaterThan(rightGeneration, beforeRight.generation)
        XCTAssertLessThanOrEqual(rightGeneration, afterRight.generation)
        XCTAssertGreaterThan(leftGeneration, beforeLeft.generation)
        XCTAssertLessThanOrEqual(leftGeneration, afterLeft.generation)
    }

    func testNamespaceBatchLive() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let left = try await VolantisStore.openNamespace(root: root, name: "left", spec: recordSpec)
        let right = try await VolantisStore.openNamespace(root: root, name: "right", spec: recordSpec)
        _ = try await left.upsert([document(1)])
        let participants = [
            LiveNamespaceMutation(
                store: right,
                mutation: NamespaceMutation(name: "right", spec: recordSpec, upserts: [NamespaceUpsert(document(2))])
            ),
            LiveNamespaceMutation(
                store: left,
                mutation: NamespaceMutation(name: "left", spec: recordSpec, deletes: [id(1)])
            )
        ]

        let generations = try await VolantisStore.namespaceBatchLive(root: root, participants: participants)
        let afterLeft = try await left.get([id(1)])
        let afterRight = try await right.get([id(2)])
        XCTAssertEqual(afterLeft.missingCount, 1)
        XCTAssertEqual(afterRight.documents.map { $0?.id }, [id(2)])
        XCTAssertNotEqual(afterRight.generation, afterLeft.generation)
        XCTAssertEqual(generations, [afterRight.generation, afterLeft.generation])

        try await left.close()
        do {
            _ = try await VolantisStore.namespaceBatchLive(root: root, participants: participants)
            XCTFail("live batch accepted a closed writer")
        } catch let error as VolantisError {
            XCTAssertEqual(error, .closed)
        }
        try await right.close()
    }

    func testNamespaceBatchAcceptsABIFilterNodeLimit() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let spec = cascadeSpec()
        try await seed(root, "left", spec, [document(1, attributes: [1: .id128(id(2))])])
        try await seed(root, "right", recordSpec, [])
        let filter = Filter.or(Array(repeating: .exists(1), count: (1 << 20) - 1))
        _ = try await VolantisStore.namespaceBatch(root: root, participants: [
            NamespaceMutation(name: "left", spec: spec, deleteWhere: filter),
            NamespaceMutation(name: "right", spec: recordSpec)
        ])
        let after = try await read(root, "left", spec, [id(1)])
        XCTAssertEqual(after.missingCount, 1)
    }

    func testDeclareCascade() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let spec = cascadeSpec()
        try await seed(root, "parent", spec, [])
        try await seed(root, "child", spec, [])
        let participants = ["parent", "child"].map { CascadeParticipant(name: $0, spec: spec) }
        let edge = CascadeDeclaration(parentIndex: 0, childIndex: 1, attributeID: 1)
        try await VolantisStore.declareCascade(root: root, participants: participants, declaration: edge)
        try await VolantisStore.declareCascade(root: root, participants: participants, declaration: edge)
        do {
            try await VolantisStore.declareCascade(
                root: root,
                participants: participants,
                declaration: CascadeDeclaration(parentIndex: 1, childIndex: 0, attributeID: 1)
            )
            XCTFail("cascade declaration accepted a cycle")
        } catch let error as VolantisError {
            XCTAssertEqual(error, .cascadeCycle)
        }
    }

    func testDeleteCascade() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let spec = cascadeSpec()
        let owner = DocumentID(high: 0xfedcba9876543210, low: 7)
        let parent = IngestDocument(id: owner, revision: 1, timestamp: 0, vector: [])
        try await seed(root, "parent", spec, [parent])
        try await seed(root, "child", spec, [document(2, attributes: [1: .id128(owner)])])
        try await seed(root, "leaf", spec, [document(3, attributes: [1: .id128(id(2))]), document(4)])
        let child = try await read(root, "child", spec, [id(2)])
        XCTAssertEqual(child.documents.first??.attributes[1], .id128(owner))
        let declarations = ["parent", "child", "leaf"].map { CascadeParticipant(name: $0, spec: spec) }
        for parentIndex: UInt32 in 0..<2 {
            try await VolantisStore.declareCascade(
                root: root,
                participants: declarations,
                declaration: CascadeDeclaration(parentIndex: parentIndex, childIndex: parentIndex + 1, attributeID: 1)
            )
        }
        let participants = [
            CascadeDeleteParticipant(name: "child", spec: spec),
            CascadeDeleteParticipant(name: "parent", spec: spec, deletes: [owner]),
            CascadeDeleteParticipant(name: "leaf", spec: spec)
        ]

        do {
            _ = try await VolantisStore.deleteCascade(root: root, participants: Array(participants.prefix(2)))
            XCTFail("cascade accepted a missing reachable namespace")
        } catch let error as VolantisError {
            XCTAssertEqual(error, .io)
        }
        let before = try await read(root, "parent", spec, [owner])
        XCTAssertEqual(before.documents.map { $0?.id }, [owner])

        let generations = try await VolantisStore.deleteCascade(root: root, participants: participants)
        let leaf = try await read(root, "leaf", spec, [id(3), id(4)])
        let afterChild = try await read(root, "child", spec, [id(2)])
        let afterParent = try await read(root, "parent", spec, [owner])
        XCTAssertEqual(leaf.documents.map { $0?.id }, [nil, id(4)])
        XCTAssertEqual(afterChild.missingCount, 1)
        XCTAssertEqual(afterParent.missingCount, 1)
        XCTAssertEqual(generations, [afterChild.generation, afterParent.generation, leaf.generation])
    }

    private func temporaryRoot() -> URL {
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/test-namespaces", isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
    }

    private func id(_ low: UInt64) -> DocumentID {
        DocumentID(high: 0, low: low)
    }

    private func document(
        _ low: UInt64,
        revision: UInt64 = 1,
        text: String? = nil,
        attributes: [UInt32: AttributeValue] = [:]
    ) -> IngestDocument {
        IngestDocument(id: id(low), revision: revision, timestamp: 0, vector: [], text: text, attributes: attributes)
    }

    private func cascadeSpec() -> NamespaceSpec {
        NamespaceSpec(
            attributes: [AttributeDefinition(id: 1, name: "owner", type: .id128, nullable: true)],
            vectorSpace: nil
        )
    }

    private func seed(_ root: URL, _ name: String, _ spec: NamespaceSpec, _ documents: [IngestDocument]) async throws {
        let store = try await VolantisStore.openNamespace(root: root, name: name, spec: spec)
        if !documents.isEmpty {
            _ = try await store.upsert(documents)
        }
        try await store.close()
    }

    private func read(
        _ root: URL, _ name: String, _ spec: NamespaceSpec, _ ids: [DocumentID]
    ) async throws -> GetResult {
        let store = try await VolantisStore.openNamespace(root: root, name: name, spec: spec, options: readOnly)
        let result = try await store.get(ids)
        try await store.close()
        return result
    }
}
