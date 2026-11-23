import XCTest
import VolantisDB

final class GraphMaintenanceTests: XCTestCase {
    #if ZE_GRAPH
    func testRelationshipTypesPersistAcrossResume() async throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: path) }
        let rule = GraphRelationshipType(type: "DEPENDS_ON", onDelete: .cascade)
        do {
            _ = try await VolantisStore.open(at: path, relationshipTypes: [rule, rule])
            XCTFail("Duplicate declarations should fail")
        } catch let error as VolantisError {
            XCTAssertEqual(error, .invalidArgument)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: path.path))

        let store = try await VolantisStore.open(at: path, relationshipTypes: [rule])
        _ = try await store.cypher(
            "CREATE (:Leaf)-[:DEPENDS_ON]->(:Child)-[:DEPENDS_ON]->(:Parent), (:Unrelated)"
        )
        try await store.quiesce()
        try await store.resume()
        _ = try await store.cypher("MATCH (p:Parent) DELETE p")
        let nodes = try await store.cypher("MATCH (n) RETURN count(n)")
        let unrelated = try await store.cypher("MATCH (n:Unrelated) RETURN count(n)")
        XCTAssertEqual(nodes.rows, [[.integer(1)]])
        XCTAssertEqual(unrelated.rows, [[.integer(1)]])
        try await store.close()
    }

    func testGraphMaintainCompletesCycleAndHonorsCancellation() async throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: path) }
        let store = try await VolantisStore.open(at: path)
        _ = try await store.enableGraph()
        let write = try await store.cypher("CREATE (:Kept {name: 'fixture'})")
        let generation = try XCTUnwrap(write.metadata.changedGeneration)

        let token = try GraphCancellationToken()
        try token.cancel()
        do {
            _ = try await store.graphMaintain(interruption: .cancellation(token))
            XCTFail("A cancelled maintenance step should fail")
        } catch let error as VolantisError {
            XCTAssertEqual(error, .cancelled)
        }

        var complete = false
        for _ in 0..<8 {
            let report = try await store.graphMaintain()
            XCTAssertGreaterThanOrEqual(report.generation, generation)
            if report.cycleComplete {
                complete = true
                break
            }
        }
        XCTAssertTrue(complete)
        let result = try await store.cypher("MATCH (n:Kept) RETURN n.name")
        XCTAssertEqual(result.rows, [[.string("fixture")]])
        try await store.close()
    }

    func testGraphSetMaintenancePolicyRejectsBelowMinimum() async throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: path) }
        let store = try await VolantisStore.open(at: path)
        _ = try await store.enableGraph()
        try await store.graphSetMaintenancePolicy(
            GraphMaintenancePolicy(autoReclaim: false, reclaimAfterBytes: 1_048_576)
        )
        do {
            try await store.graphSetMaintenancePolicy(
                GraphMaintenancePolicy(autoReclaim: true, reclaimAfterBytes: 1_048_575)
            )
            XCTFail("A threshold below 1 MiB should fail")
        } catch let error as VolantisError {
            XCTAssertEqual(error, .invalidArgument)
        }
        try await store.close()
    }
    #else
    func testGraphOperationsRejectUnsupportedBuild() async throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: path) }
        do {
            _ = try await VolantisStore.open(at: path, relationshipTypes: [])
            XCTFail("Graph creation requires the graph build")
        } catch let error as VolantisError {
            XCTAssertEqual(error, .graphUnsupportedBuild)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: path.path))

        let store = try await VolantisStore.open(at: path)
        do {
            _ = try await store.graphMaintain()
            XCTFail("Graph maintenance requires the graph build")
        } catch let error as VolantisError {
            XCTAssertEqual(error, .graphUnsupportedBuild)
        }
        do {
            try await store.graphSetMaintenancePolicy(GraphMaintenancePolicy())
            XCTFail("Graph policy requires the graph build")
        } catch let error as VolantisError {
            XCTAssertEqual(error, .graphUnsupportedBuild)
        }
        try await store.close()
    }
    #endif
}
