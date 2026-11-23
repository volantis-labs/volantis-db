import Foundation
import XCTest
import VolantisDB

@available(macOS 14.0, *)
final class QueryFilterSnippetTests: XCTestCase {
    func testQueryFiltersBeforeTopKAndIntersectsEligibleIDs() async throws {
        let root = fixturePath()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await VolantisStore.openNamespace(
            root: root,
            name: "filtered",
            spec: NamespaceSpec(
                attributes: [AttributeDefinition(id: 1, name: "kind", type: .rawString, nullable: false)],
                vectorSpace: VectorSpace(dimensions: 1, normalization: .none)
            )
        )
        let wrongKind = DocumentID(high: 0, low: 1)
        let atEnd = DocumentID(high: 0, low: 2)
        let atStart = DocumentID(high: 0, low: 3)
        _ = try await store.upsert([
            IngestDocument(
                id: wrongKind, revision: 1, timestamp: 10, vector: [0], text: "harbour",
                attributes: [1: .string("draft")]
            ),
            IngestDocument(
                id: atEnd, revision: 1, timestamp: 20, vector: [1], text: "harbour",
                attributes: [1: .string("note")]
            ),
            IngestDocument(
                id: atStart, revision: 1, timestamp: 10, vector: [2], text: "harbour",
                attributes: [1: .string("note")]
            )
        ])

        let options = QueryOptions(k: 1, tier: .exact)
        let filter = Filter.eq(1, .string("note"))
        let unfiltered = try await store.query(vector: [0], options: options)
        XCTAssertEqual(unfiltered.hits.map(\.documentID), [wrongKind])
        let filtered = try await store.query(
            vector: [0], filter: filter, timestampRange: 10..<20, options: options
        )
        XCTAssertEqual(filtered.hits.map(\.documentID), [atStart])
        let hybrid = try await store.query(
            vector: [0], text: "harbour", eligibleIds: [wrongKind, atStart],
            filter: filter, timestampRange: 10..<20, options: options
        )
        XCTAssertEqual(hybrid.mode, .hybrid)
        XCTAssertEqual(hybrid.hits.map(\.documentID), [atStart])
        let lexicalOptions = QueryOptions(k: 1)
        let excluded = try await store.query(
            text: "harbour", eligibleIds: [wrongKind, atEnd],
            filter: filter, timestampRange: 10..<20, options: lexicalOptions
        )
        XCTAssertTrue(excluded.hits.isEmpty)

        try await assertUnknownFilterRejected(store)
        try await store.close()
    }

    private func assertUnknownFilterRejected(_ store: VolantisStore) async throws {
        do {
            _ = try await store.query(text: "harbour", filter: .exists(99), options: QueryOptions(k: 1))
            XCTFail("query accepted an unknown attribute")
        } catch let error as VolantisError {
            XCTAssertEqual(error, .invalidArgument)
        }
    }

    func testQuerySnippetsExposeUTF8HighlightsAndSourceRanges() async throws {
        let root = fixturePath()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await VolantisStore.openNamespace(
            root: root,
            name: "snippets",
            spec: NamespaceSpec(attributes: [], vectorSpace: VectorSpace(dimensions: 1, normalization: .none))
        )
        let matched = DocumentID(high: 0, low: 1)
        let vectorOnly = DocumentID(high: 0, low: 2)
        _ = try await store.upsert([
            IngestDocument(id: matched, revision: 1, timestamp: 10, vector: [1], text: "前 🚀 café 🐚 harbour tail"),
            IngestDocument(id: vectorOnly, revision: 1, timestamp: 10, vector: [0])
        ])

        let plain = try await store.query(text: "cafe harbour", options: QueryOptions(k: 1))
        XCTAssertEqual(plain.hits.map(\.documentID), [matched])
        XCTAssertNil(plain.hits.first?.snippet)
        let options = QueryOptions(k: 2, snippetBytes: 18)
        let excerpt = try await store.query(text: "cafe harbour", options: options)
        let filtered = try await store.query(text: "cafe harbour", timestampRange: 10..<20, options: options)
        let hybrid = try await store.query(
            vector: [0], text: "cafe harbour", eligibleIds: [matched, vectorOnly],
            timestampRange: 10..<20, options: options
        )
        XCTAssertEqual(Set(hybrid.hits.compactMap(\.documentID)), Set([matched, vectorOnly]))
        XCTAssertNil(hybrid.hits.first { $0.documentID == vectorOnly }?.snippet)
        // Check the copied results after the native query allocations and store are gone.
        try await assertInvalidSnippetRequests(store)
        try await store.close()
        for result in [excerpt, filtered, hybrid] {
            let hit = try XCTUnwrap(result.hits.first { $0.documentID == matched })
            let snippet = try XCTUnwrap(hit.snippet)
            XCTAssertEqual(snippet.text, "café 🐚 harbour")
            XCTAssertEqual(snippet.sourceByteRange, 9..<27)
            XCTAssertEqual(snippet.highlights.map(\.byteRange), [0..<5, 11..<18])
            XCTAssertEqual(snippet.highlights.map(\.sourceByteRange), [9..<14, 20..<27])
            XCTAssertTrue(snippet.truncatedStart)
            XCTAssertTrue(snippet.truncatedEnd)
        }
    }

    private func assertInvalidSnippetRequests(_ store: VolantisStore) async throws {
        do {
            _ = try await store.query(vector: [0], options: QueryOptions(k: 1, snippetBytes: 18))
            XCTFail("vector-only query accepted a snippet request")
        } catch let error as VolantisError {
            XCTAssertEqual(error, .invalidArgument)
        }
        for invalidLength in [0, -1] {
            do {
                _ = try await store.query(text: "cafe", options: QueryOptions(k: 1, snippetBytes: invalidLength))
                XCTFail("query accepted a nonpositive snippet length")
            } catch let error as VolantisError {
                XCTAssertEqual(error, .invalidArgument)
            }
        }
    }

    private func fixturePath() -> URL {
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/query-tests/\(UUID().uuidString)", isDirectory: true)
    }
}
