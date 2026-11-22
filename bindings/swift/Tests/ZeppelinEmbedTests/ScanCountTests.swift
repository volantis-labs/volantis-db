import Foundation
import XCTest
import ZeppelinEmbed

@available(macOS 14.0, *)
final class ScanCountTests: XCTestCase {
    func testAttributeOrderedScanPaginatesAndRejectsDifferentOrder() async throws {
        let root = try namespaceRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await ZeppelinStore.openNamespace(
            root: root,
            name: "ordered",
            spec: NamespaceSpec(
                attributes: [
                    AttributeDefinition(id: 1, name: "priority", type: .i64, nullable: true)
                ],
                vectorSpace: nil
            )
        )
        _ = try await store.upsert([
            document(3, timestamp: 10, attributes: [1: .i64(9)]),
            document(4, timestamp: 20, attributes: [1: .null]),
        ])
        _ = try await store.seal()
        let write = try await store.upsert([
            document(1, timestamp: 30, attributes: [1: .i64(9)]),
            document(2, timestamp: 40, attributes: [1: .i64(-2)]),
        ])

        let order = AttributeScanOrder(attributeID: 1, direction: .descending)
        let first = try await store.scan(limit: 1, order: order, fields: [])
        XCTAssertEqual(first.documents.map(\.id.low), [1])
        XCTAssertEqual(first.generation, write.generation)
        let cursor = try XCTUnwrap(first.next)

        let second = try await store.scan(after: cursor, limit: 3, order: order, fields: [])
        XCTAssertEqual(second.documents.map(\.id.low), [3, 2, 4])
        XCTAssertEqual(second.generation, first.generation)
        XCTAssertNil(second.next)

        let ascending = try await store.scan(order: .init(attributeID: 1, direction: .ascending))
        XCTAssertEqual(ascending.documents.map(\.id.low), [2, 1, 3, 4])

        do {
            _ = try await store.scan(after: cursor, order: .timestampDescending)
            XCTFail("a timestamp scan accepted an attribute-order cursor")
        } catch let error as ZeppelinError {
            XCTAssertEqual(error, .invalidArgument)
        }
        try await store.close()
    }

    func testGroupedCountHonorsFilterAndRangeAndRejectsExceededBudget() async throws {
        let root = try namespaceRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = try await ZeppelinStore.openNamespace(
            root: root,
            name: "grouped",
            spec: NamespaceSpec(
                attributes: [
                    AttributeDefinition(id: 1, name: "label", type: .rawString, nullable: true),
                    AttributeDefinition(id: 2, name: "priority", type: .i64, nullable: true),
                    AttributeDefinition(id: 3, name: "scope", type: .u64, nullable: false),
                ],
                vectorSpace: nil
            )
        )
        _ = try await store.upsert([
            document(1, timestamp: 10, attributes: [1: .string("éclair"), 2: .i64(-5), 3: .u64(7)]),
            document(2, timestamp: 20, attributes: [1: .string(""), 2: .i64(8), 3: .u64(7)]),
        ])
        _ = try await store.seal()
        let write = try await store.upsert([
            document(3, timestamp: 30, attributes: [1: .string("éclair"), 2: .i64(-5), 3: .u64(7)]),
            document(4, timestamp: 40, attributes: [1: .null, 2: .null, 3: .u64(7)]),
            document(5, timestamp: 50, attributes: [1: .string("outside"), 2: .i64(20), 3: .u64(7)]),
            document(6, timestamp: 25, attributes: [1: .string("other"), 2: .i64(2), 3: .u64(8)]),
        ])

        let filter = Filter.eq(3, .u64(7))
        let range: Range<Int64> = 10..<50
        let labels = try await store.count(
            groupBy: CountGroupBy(attributeID: 1, limit: 2), filter: filter, timestampRange: range)
        XCTAssertEqual(labels.groups.map(\.value), [.string(""), .string("éclair")])
        XCTAssertEqual(labels.groups.map(\.count), [1, 2])
        XCTAssertEqual(labels.missingCount, 1)
        XCTAssertEqual(labels.count, 4)
        XCTAssertEqual(labels.generation, write.generation)

        let priorities = try await store.count(
            groupBy: CountGroupBy(attributeID: 2), filter: filter, timestampRange: range)
        XCTAssertEqual(priorities.groups.map(\.value), [.i64(-5), .i64(8)])
        XCTAssertEqual(priorities.groups.map(\.count), [2, 1])

        let scopes = try await store.count(groupBy: CountGroupBy(attributeID: 3))
        XCTAssertEqual(scopes.groups.map(\.value), [.u64(7), .u64(8)])
        XCTAssertEqual(scopes.groups.map(\.count), [5, 1])

        do {
            _ = try await store.count(
                groupBy: CountGroupBy(attributeID: 1, limit: 1),
                filter: filter,
                timestampRange: range
            )
            XCTFail("grouped count accepted more distinct values than its limit")
        } catch let error as ZeppelinError {
            XCTAssertEqual(error, .budgetExceeded)
        }
        try await store.close()
    }

    private func namespaceRoot() throws -> URL {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".build/test-namespaces", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.appendingPathComponent(UUID().uuidString, isDirectory: true)
    }

    private func document(
        _ id: UInt64, timestamp: Int64, attributes: [UInt32: AttributeValue]
    ) -> IngestDocument {
        IngestDocument(
            id: DocumentID(high: 0, low: id),
            revision: 1,
            timestamp: timestamp,
            vector: [],
            attributes: attributes
        )
    }
}
