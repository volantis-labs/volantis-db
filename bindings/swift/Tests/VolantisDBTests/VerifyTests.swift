import Foundation
import XCTest
import VolantisDB

final class VerifyTests: XCTestCase {
    func testVerifyReportsCleanStoreWithoutChangingFiles() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let path = try await closedNamespace(in: root)
        let before = try contents(of: path)

        let report = try await VolantisStore.verify(at: path)

        XCTAssertTrue(report.ok)
        XCTAssertGreaterThan(report.generation, 0)
        XCTAssertEqual(report.segmentsChecked, 1)
        XCTAssertEqual(report.walRecordsChecked, 1)
        XCTAssertEqual(report.findings, [])
        XCTAssertEqual(try contents(of: path), before)
    }

    func testVerifyReturnsDamageAsFindings() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let path = try await closedNamespace(in: root)
        let segment = try XCTUnwrap(contents(of: path).keys.first { $0.hasSuffix(".zseg") })
        try FileManager.default.removeItem(at: path.appendingPathComponent(segment))
        let wal = path.appendingPathComponent("wal.ze")
        try Data(contentsOf: wal).prefix(1).write(to: wal)
        let before = try contents(of: path)

        let report = try await VolantisStore.verify(at: path)

        XCTAssertFalse(report.ok)
        XCTAssertEqual(report.segmentsChecked, 1)
        XCTAssertEqual(report.walRecordsChecked, 0)
        XCTAssertEqual(report.findings.map(\.kind), [.segmentMissing, .walHeaderCorrupt])
        XCTAssertEqual(report.findings.map(\.file), [segment, "wal.ze"])
        XCTAssertEqual(report.findings.map(\.offset), [nil, 0])
        XCTAssertTrue(report.findings.allSatisfy { !$0.detail.isEmpty })
        XCTAssertEqual(try contents(of: path), before)
    }

    func testVerifyMissingDirectoryThrowsNotFound() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let path = root.appendingPathComponent("missing")

        do {
            _ = try await VolantisStore.verify(at: path)
            XCTFail("Verification succeeded for a missing directory")
        } catch {
            XCTAssertEqual(error as? VolantisError, .notFound)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: path.path))
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(
            "volantis-verify-\(UUID().uuidString)", isDirectory: true
        )
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return root
    }

    private func closedNamespace(in root: URL) async throws -> URL {
        let store = try await VolantisStore.openNamespace(
            root: root,
            name: "notes",
            spec: NamespaceSpec(attributes: [], vectorSpace: nil),
            options: OpenOptions(durabilityMode: .durable)
        )
        _ = try await store.upsert([note(1)])
        _ = try await store.seal()
        _ = try await store.upsert([note(2)])
        try await store.close()
        return root.appendingPathComponent("notes", isDirectory: true)
    }

    private func note(_ id: UInt64) -> IngestDocument {
        IngestDocument(
            id: DocumentID(high: 0, low: id),
            revision: 1,
            timestamp: 1,
            vector: [],
            text: "Verification note"
        )
    }

    private func contents(of path: URL) throws -> [String: Data] {
        let files = try FileManager.default.contentsOfDirectory(atPath: path.path)
        return try Dictionary(
            uniqueKeysWithValues: files.map {
                ($0, try Data(contentsOf: path.appendingPathComponent($0)))
            })
    }
}
