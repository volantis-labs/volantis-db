import Foundation
import XCTest
import VolantisDB

final class RequiredAPITests: XCTestCase {
    private struct Fixture: Decodable {
        struct Row: Decodable {
            let build: String
            let swift: [String]
        }
        let rows: [String: Row]
        let allowed_gaps: [String: [String: String]]
    }

    func testRequiredAPI() throws {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // VolantisDBTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // swift
            .deletingLastPathComponent() // bindings
            .appendingPathComponent("fixtures/required_api_v1.json")
        let table = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: fixture))
        let gaps = try XCTUnwrap(table.allowed_gaps["swift"])
        XCTAssertTrue(gaps.isEmpty, "E17 allows no new Swift API gaps")
        let api = publicAPI()
        var builds: Set<String> = ["core"]
        #if ZE_GRAPH
        builds.insert("graph")
        #endif
        #if ZE_TEXT
        builds.insert("text")
        #endif
        for (id, row) in table.rows.sorted(by: { $0.key < $1.key }) where builds.contains(row.build) {
            XCTAssertFalse(row.swift.isEmpty, "\(id): no Swift API specified")
            for name in row.swift {
                XCTAssertTrue(api.keys.contains(name), "\(id): missing \(name)")
            }
        }
    }

    // Use references from a public import. XCTest can check the fixture names
    // without opening a store or loading a model.
    private func publicAPI() -> [String: Any] {
        typealias AttributeScan = (isolated VolantisStore) ->
            (ScanCursor?, Int, AttributeScanOrder, DocumentFields, Range<Int64>?, Filter?,
             VolantisCancellationToken?, UInt64) async throws -> ScanPage
        let scan: AttributeScan =
            VolantisStore.scan(after:limit:order:fields:timestampRange:filter:cancellationToken:deadlineNanoseconds:)
        var api: [String: Any] = [
            "VolantisStore.open(at:options:)": VolantisStore.open(at:options:),
            "VolantisStore.close()": VolantisStore.close,
            "VolantisStore.state()": VolantisStore.state,
            "VolantisStore.stats()": VolantisStore.stats,
            "VolantisStore.ingest(_:)": VolantisStore.ingest(_:),
            "IngestDocument.text": \IngestDocument.text,
            "IngestDocument.metadata": \IngestDocument.metadata,
            "IngestDocument.timestamp": \IngestDocument.timestamp,
            "VolantisStore.upsert(_:)": VolantisStore.upsert(_:),
            "IngestDocument.attributes": \IngestDocument.attributes,
            "VolantisStore.upsert(conditional:)": VolantisStore.upsert(conditional:),
            "VolantisStore.delete(_:)": VolantisStore.delete(_:),
            "VolantisStore.delete(conditional:)": VolantisStore.delete(conditional:),
            "VolantisStore.deleteWhere(_:)": VolantisStore.deleteWhere(_:),
            "VolantisStore.get(_:fields:)": VolantisStore.get(_:fields:),
            "VolantisStore.scan(after:limit:order:fields:timestampRange:filter:cancellationToken:deadlineNanoseconds:)": scan,
            "AttributeScanOrder": AttributeScanOrder.self,
            "VolantisStore.count(filter:timestampRange:)": VolantisStore.count(filter:timestampRange:),
            "VolantisStore.count(groupBy:filter:timestampRange:)": VolantisStore.count(groupBy:filter:timestampRange:),
            "VolantisStore.search(vector:options:)": VolantisStore.search(vector:options:),
            "VolantisStore.search(vector:filter:options:)": VolantisStore.search(vector:filter:options:),
            "VolantisStore.query(vector:text:eligibleIds:filter:timestampRange:options:)": VolantisStore.query(vector:text:eligibleIds:filter:timestampRange:options:),
            "QueryOptions.tier": \QueryOptions.tier,
            "QueryOptions.lastAsPrefix": \QueryOptions.lastAsPrefix,
            "QueryOptions.snippetBytes": \QueryOptions.snippetBytes,
            "QueryHit.snippet": \QueryHit.snippet,
            "Snippet.highlights": \Snippet.highlights,
            "SnippetHighlight.sourceByteRange": \SnippetHighlight.sourceByteRange,
            "Filter.eq": Filter.eq,
            "Filter.in": Filter.in,
            "Filter.range": Filter.range,
            "Filter.not": Filter.not,
            "AttributeValue": AttributeValue.self,
            "QueryOptions": QueryOptions.self,
            "VolantisStore.openNamespace(root:name:spec:options:)": VolantisStore.openNamespace(root:name:spec:options:),
            "NamespaceSpec.attributes": \NamespaceSpec.attributes,
            "VolantisStore.schema()": VolantisStore.schema,
            "AttributeDefinition": AttributeDefinition.self,
            "VolantisStore.seal(cancellationToken:)": VolantisStore.seal(cancellationToken:),
            "VolantisStore.merge(cancellationToken:)": VolantisStore.merge(cancellationToken:),
            "VolantisStore.maintain(wallTimeNanoseconds:bytes:)": VolantisStore.maintain(wallTimeNanoseconds:bytes:),
            "VolantisStore.purge(_:)": VolantisStore.purge(_:),
            "VolantisStore.awaitPhysicalPurge(_:)": VolantisStore.awaitPhysicalPurge(_:),
            "VolantisStore.applyRetention(window:now:)": VolantisStore.applyRetention(window:now:),
            "VolantisStore.dropPartition(from:to:)": VolantisStore.dropPartition(from:to:),
            "VolantisStore.verify(at:)": VolantisStore.verify(at:),
            "VerifyReport": VerifyReport.self,
            "VolantisCancellationToken.create()": VolantisCancellationToken.create,
            "VolantisCancellationToken.cancel()": VolantisCancellationToken.cancel,
            "VolantisCancellationToken.free()": VolantisCancellationToken.free,
            "QueryOptions.cancellationToken": \QueryOptions.cancellationToken,
            "QueryOptions.deadlineNanoseconds": \QueryOptions.deadlineNanoseconds,
            "VolantisError.rawValue": \VolantisError.rawValue,
            "VolantisStore.lastErrorMessage()": VolantisStore.lastErrorMessage,
            "VolantisStore.abiVersion": VolantisStore.abiVersion,
            "NamespaceSpec.tokenizerProfile": \NamespaceSpec.tokenizerProfile,
            "VolantisStore.listNamespaces(root:)": VolantisStore.listNamespaces(root:),
            "VolantisStore.namespaceBatch(root:participants:)": VolantisStore.namespaceBatch(root:participants:),
            "VolantisStore.namespaceBatchLive(root:participants:)": VolantisStore.namespaceBatchLive(root:participants:),
            "VolantisStore.declareCascade(root:participants:declaration:)":
                VolantisStore.declareCascade(root:participants:declaration:),
            "VolantisStore.deleteCascade(root:participants:)": VolantisStore.deleteCascade(root:participants:),
        ]
        #if ZE_TEXT
        api.merge([
            "NamespaceSpec.embeddedModel": \NamespaceSpec.embeddedModel,
            "VolantisStore.queryEmbedded(_:leg:k:eligibleIds:filter:timestampRange:cancellationToken:deadlineNanoseconds:)": VolantisStore.queryEmbedded(_:leg:k:eligibleIds:filter:timestampRange:cancellationToken:deadlineNanoseconds:),
            "VolantisStore.nearestChunks(_:options:)": VolantisStore.nearestChunks(_:options:),
            "VolantisStore.nearestChunksBatch(_:options:)": VolantisStore.nearestChunksBatch(_:options:),
            "TextStore.open(at:bundle:options:mlxCacheLimitBytes:)": TextStore.open(at:bundle:options:mlxCacheLimitBytes:),
            "TextStore.open(at:model:cacheDirectory:offline:options:mlxCacheLimitBytes:progress:)": TextStore.open(at:model:cacheDirectory:offline:options:mlxCacheLimitBytes:progress:),
            "TextModel.default": TextModel.default,
            "TextStore.ingest(_:options:deadlineNanoseconds:)": TextStore.ingest(_:options:deadlineNanoseconds:),
            "TextIngestOptions.maintenanceWallTimeNanoseconds": \TextIngestOptions.maintenanceWallTimeNanoseconds,
            "TextIngestOptions.maintenanceBytes": \TextIngestOptions.maintenanceBytes,
            "TextStore.query(_:legs:k:tier:deadlineNanoseconds:)": TextStore.query(_:legs:k:tier:deadlineNanoseconds:),
            "TextQueryResult.hits": \TextQueryResult.hits,
            "TextStore.delete(_:)": TextStore.delete(_:),
            "TextStore.maintain(wallTimeNanoseconds:byteBudget:)": TextStore.maintain(wallTimeNanoseconds:byteBudget:),
            "TextStore.health()": TextStore.health,
            "TextStore.observedComputeUnits()": TextStore.observedComputeUnits,
        ]) { current, _ in current }
        #endif
        #if ZE_GRAPH
        api.merge([
            "VolantisStore.enableGraph()": VolantisStore.enableGraph,
            "VolantisStore.open(at:relationshipTypes:maxResidentBytes:readerDrainTimeoutMilliseconds:)": VolantisStore.open(at:relationshipTypes:maxResidentBytes:readerDrainTimeoutMilliseconds:),
            "VolantisStore.graphApply(_:interruption:)": VolantisStore.graphApply(_:interruption:),
            "GraphBatch": GraphBatch.self,
            "VolantisStore.cypher(_:parameters:controls:options:)": VolantisStore.cypher(_:parameters:controls:options:),
            "GraphControls.rowLimit": \GraphControls.rowLimit,
            "VolantisStore.graphQuery(_:parameters:options:interruption:)": VolantisStore.graphQuery(_:parameters:options:interruption:),
            "GraphPlan": GraphPlan.self,
            "VolantisStore.getNodes(_:fields:limits:interruption:)": VolantisStore.getNodes(_:fields:limits:interruption:),
            "VolantisStore.getRelationships(_:limits:interruption:)": VolantisStore.getRelationships(_:limits:interruption:),
            "VolantisStore.graphResources()": VolantisStore.graphResources,
            "VolantisStore.graphMaintain(interruption:)": VolantisStore.graphMaintain(interruption:),
            "VolantisStore.graphSetMaintenancePolicy(_:)": VolantisStore.graphSetMaintenancePolicy(_:),
            "GraphNodeImage.text": \GraphNodeImage.text,
        ]) { current, _ in current }
        #endif
        return api
    }
}
