import Foundation
import XCTest

#if ZE_TEXT
import CryptoKit
#endif

import ZeppelinEmbed

final class EmbeddedNamespaceTests: XCTestCase {
    func testEmbeddedNamespaceOpenRejectsExplicitVectorSpace() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        var spec = NamespaceSpec(attributes: [], embeddedModel: .default)
        spec.vectorSpace = VectorSpace(dimensions: 2, normalization: .none)
        do {
            _ = try await ZeppelinStore.openNamespace(root: root, name: "notes", spec: spec)
            XCTFail("embedded namespace accepted an explicit vector space")
        } catch let error as ZeppelinError {
            #if ZE_TEXT
            XCTAssertEqual(error, .invalidArgument)
            #else
            XCTAssertEqual(error, .unsupported)
            #endif
        }
    }

    func testQueryEmbeddedWithoutEmbedder() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await ZeppelinStore.openNamespace(
            root: root, name: "records", spec: NamespaceSpec(attributes: [], vectorSpace: nil))
        do {
            _ = try await store.queryEmbedded("bronze", leg: .dense, k: 1)
            XCTFail("embedded query succeeded without an embedder")
        } catch let error as ZeppelinError {
            #if ZE_TEXT
            XCTAssertEqual(error, .missingEmbedder)
            #else
            XCTAssertEqual(error, .unsupported)
            #endif
        }
        try await store.close()
    }

    func testNearestChunksWithoutEmbedder() async throws {
        let root = temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await ZeppelinStore.openNamespace(
            root: root, name: "records", spec: NamespaceSpec(attributes: [], vectorSpace: nil))
        let source = NeighbourSource(id: DocumentID(high: 0, low: 1))
        #if ZE_TEXT
        let expected = ZeppelinError.missingEmbedder
        #else
        let expected = ZeppelinError.unsupported
        #endif
        do {
            _ = try await store.nearestChunks(source)
            XCTFail("nearest-chunk search succeeded without an embedded namespace")
        } catch let error as ZeppelinError {
            XCTAssertEqual(error, expected)
        }
        do {
            _ = try await store.nearestChunksBatch([source, source])
            XCTFail("batch nearest-chunk search succeeded without an embedded namespace")
        } catch let error as ZeppelinError {
            XCTAssertEqual(error, expected)
        }
        try await store.close()
    }

    private func temporaryRoot() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("ze586-\(UUID())")
    }

    #if ZE_TEXT
    func testEmbeddedNamespaceOpenAndReopen() async throws {
        try await withCachedModel { root, spec in
            let store = try await ZeppelinStore.openNamespace(root: root, name: "notes", spec: spec)
            let id = DocumentID(high: 1 << 56, low: 42)
            _ = try await store.upsert([
                IngestDocument(id: id, revision: 3, timestamp: 10, vector: [],
                    text: "bronze zeppelin", attributes: [1: .u64(7)])
            ])
            try await store.close()

            let reopened = try await ZeppelinStore.openNamespace(root: root, name: "notes", spec: spec)
            let result = try await reopened.get([id], fields: [.text, .attributes])
            let document = try XCTUnwrap(result.documents.first ?? nil)
            XCTAssertEqual(document.text, "bronze zeppelin")
            XCTAssertEqual(document.attributes, [1: .u64(7)])
            try await reopened.close()
        }
    }

    func testQueryEmbeddedReturnsFilteredChunks() async throws {
        try await withCachedModel { root, spec in
            let store = try await ZeppelinStore.openNamespace(root: root, name: "notes", spec: spec)
            let id = DocumentID(high: 1 << 56, low: 42)
            let other = DocumentID(high: id.high, low: 43)
            let source = String(repeating: "café bronze zeppelin ", count: 600)
            let sourceBytes = Array(source.utf8)
            _ = try await store.upsert([
                IngestDocument(id: id, revision: 3, timestamp: 10, vector: [],
                    text: source, attributes: [1: .u64(7)]),
                IngestDocument(id: other, revision: 1, timestamp: 20, vector: [],
                    text: "bronze zeppelin", attributes: [1: .u64(8)])
            ])
            for leg in [TextLegs.dense, .lexical, .hybrid] {
                let hits = try await store.queryEmbedded(
                    "bronze", leg: leg, k: 2, eligibleIds: [id, other],
                    filter: .eq(1, .u64(7)), timestampRange: 10..<11
                )
                XCTAssertEqual(hits.count, 2)
                XCTAssertEqual(Set(hits.map(\.chunk)).count, 2)
                for hit in hits {
                    XCTAssertEqual(hit.id, id)
                    XCTAssertEqual(hit.revision, 3)
                    XCTAssertEqual(hit.timestamp, 10)
                    XCTAssertEqual(hit.attributes, [1: .u64(7)])
                    XCTAssertTrue(hit.score.isFinite)
                    guard hit.byteStart >= 0, hit.byteStart < hit.byteEnd,
                        hit.byteEnd <= sourceBytes.count else {
                        XCTFail("invalid UTF-8 span")
                        continue
                    }
                    XCTAssertEqual(String(bytes: sourceBytes[hit.byteStart..<hit.byteEnd], encoding: .utf8),
                        hit.text)
                }
            }
            let noEligible = try await store.queryEmbedded("bronze", eligibleIds: [])
            XCTAssertTrue(noEligible.isEmpty)
            let noMatchingAttribute = try await store.queryEmbedded("bronze", filter: .eq(1, .u64(9)))
            XCTAssertTrue(noMatchingAttribute.isEmpty)
            let noMatchingTimestamp = try await store.queryEmbedded("bronze", timestampRange: 11..<20)
            XCTAssertTrue(noMatchingTimestamp.isEmpty)
            try await store.close()
        }
    }

    func testNearestChunksPreservesBatchGroups() async throws {
        try await withCachedModel { root, spec in
            let store = try await ZeppelinStore.openNamespace(root: root, name: "notes", spec: spec)
            let id = DocumentID(high: 1 << 56, low: 42)
            let other = DocumentID(high: id.high, low: 43)
            _ = try await store.upsert([
                IngestDocument(id: id, revision: 1, timestamp: 10, vector: [],
                    text: "bronze zeppelin", attributes: [1: .u64(7)]),
                IngestDocument(id: other, revision: 1, timestamp: 20, vector: [],
                    text: "bronze zeppelin", attributes: [1: .u64(8)])
            ])
            let document = NeighbourSource(id: id)
            let chunk = NeighbourSource(id: other, chunk: 0)
            let options = NeighbourOptions(k: 1, eligibleIds: [id, other],
                filter: .eq(1, .u64(7)), timestampRange: 10..<11)
            let hits = try await store.nearestChunks(chunk, options: options)
            XCTAssertEqual(hits.map(\.id), [id])
            XCTAssertEqual(hits.map(\.chunk), [0])
            XCTAssertTrue(hits.allSatisfy { $0.score.isFinite && $0.score <= 0 })
            let groups = try await store.nearestChunksBatch([document, chunk, chunk], options: options)
            XCTAssertEqual(groups, [[], hits, hits])
            let includeSource = try await store.nearestChunks(document,
                options: NeighbourOptions(threshold: -0.1, includeSource: true, eligibleIds: [id]))
            XCTAssertEqual(includeSource.map(\.id), [id])
            XCTAssertEqual(includeSource.map(\.score), [0])
            let aboveZero = try await store.nearestChunksBatch([document, chunk],
                options: NeighbourOptions(threshold: 0, includeSource: true))
            XCTAssertEqual(aboveZero, [[], []])
            do {
                _ = try await store.nearestChunksBatch([document, NeighbourSource(id: other, chunk: 100)])
                XCTFail("nearest-chunk search accepted a missing source chunk")
            } catch let error as ZeppelinError {
                XCTAssertEqual(error, .notFound)
            }
            try await store.close()
        }
    }

    private func withCachedModel(_ body: (URL, NamespaceSpec) async throws -> Void) async throws {
        let root = temporaryRoot()
        let cache = root.appendingPathComponent("cache")
        let revision = "0123456789abcdef0123456789abcdef01234567"
        let entry = cache.appendingPathComponent("fixture%2Ftiny").appendingPathComponent(revision)
        try FileManager.default.createDirectory(at: entry, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let bytes = try Data(contentsOf: repository.appendingPathComponent(
            "crates/zeppelin-embed-text/tests/fixtures/ze115/gte.zem"))
        let manifest = Data("""
        {"format":1,"name":"tiny","bundle":"tiny.zem","query_coreml":null,
         "files":[{"path":"tiny.zem","size":\(bytes.count),"sha256":"\(digest(bytes))"}]}
        """.utf8)
        try bytes.write(to: entry.appendingPathComponent("tiny.zem"))
        try manifest.write(to: entry.appendingPathComponent("zeppelin-model.json"))

        let environment = ProcessInfo.processInfo.environment
        setenv("ZE_MODEL_CACHE", cache.path, 1)
        setenv("ZE_MODEL_OFFLINE", "1", 1)
        defer {
            for name in ["ZE_MODEL_CACHE", "ZE_MODEL_OFFLINE"] {
                if let value = environment[name] {
                    setenv(name, value, 1)
                } else {
                    unsetenv(name)
                }
            }
        }
        let model = TextModel(repo: "fixture/tiny", revision: revision, manifestSHA256: digest(manifest))
        let spec = NamespaceSpec(
            attributes: [AttributeDefinition(id: 1, name: "speaker", type: .u64, nullable: false)],
            embeddedModel: model
        )
        try await body(root, spec)
    }

    private func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    #endif
}
