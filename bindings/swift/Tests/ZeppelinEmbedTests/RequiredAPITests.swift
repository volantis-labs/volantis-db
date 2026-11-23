import Foundation
import XCTest
import ZeppelinEmbed

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
            .deletingLastPathComponent() // ZeppelinEmbedTests
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
        typealias AttributeScan = (isolated ZeppelinStore) ->
            (ScanCursor?, Int, AttributeScanOrder, DocumentFields, Range<Int64>?, Filter?,
             ZeppelinCancellationToken?, UInt64) async throws -> ScanPage
        let scan: AttributeScan =
            ZeppelinStore.scan(after:limit:order:fields:timestampRange:filter:cancellationToken:deadlineNanoseconds:)
        var api: [String: Any] = [
            "ZeppelinStore.open(at:options:)": ZeppelinStore.open(at:options:),
            "ZeppelinStore.close()": ZeppelinStore.close,
            "ZeppelinStore.state()": ZeppelinStore.state,
            "ZeppelinStore.stats()": ZeppelinStore.stats,
            "ZeppelinStore.ingest(_:)": ZeppelinStore.ingest(_:),
            "IngestDocument.text": \IngestDocument.text,
            "IngestDocument.metadata": \IngestDocument.metadata,
            "IngestDocument.timestamp": \IngestDocument.timestamp,
            "ZeppelinStore.upsert(_:)": ZeppelinStore.upsert(_:),
            "IngestDocument.attributes": \IngestDocument.attributes,
            "ZeppelinStore.upsert(conditional:)": ZeppelinStore.upsert(conditional:),
            "ZeppelinStore.delete(_:)": ZeppelinStore.delete(_:),
            "ZeppelinStore.delete(conditional:)": ZeppelinStore.delete(conditional:),
            "ZeppelinStore.deleteWhere(_:)": ZeppelinStore.deleteWhere(_:),
            "ZeppelinStore.get(_:fields:)": ZeppelinStore.get(_:fields:),
            "ZeppelinStore.scan(after:limit:order:fields:timestampRange:filter:cancellationToken:deadlineNanoseconds:)": scan,
            "AttributeScanOrder": AttributeScanOrder.self,
            "ZeppelinStore.count(filter:timestampRange:)": ZeppelinStore.count(filter:timestampRange:),
            "ZeppelinStore.count(groupBy:filter:timestampRange:)": ZeppelinStore.count(groupBy:filter:timestampRange:),
            "ZeppelinStore.search(vector:options:)": ZeppelinStore.search(vector:options:),
            "ZeppelinStore.search(vector:filter:options:)": ZeppelinStore.search(vector:filter:options:),
            "ZeppelinStore.query(vector:text:eligibleIds:filter:timestampRange:options:)": ZeppelinStore.query(vector:text:eligibleIds:filter:timestampRange:options:),
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
            "ZeppelinStore.openNamespace(root:name:spec:options:)": ZeppelinStore.openNamespace(root:name:spec:options:),
            "NamespaceSpec.attributes": \NamespaceSpec.attributes,
            "ZeppelinStore.schema()": ZeppelinStore.schema,
            "AttributeDefinition": AttributeDefinition.self,
            "ZeppelinStore.seal(cancellationToken:)": ZeppelinStore.seal(cancellationToken:),
            "ZeppelinStore.merge(cancellationToken:)": ZeppelinStore.merge(cancellationToken:),
            "ZeppelinStore.maintain(wallTimeNanoseconds:bytes:)": ZeppelinStore.maintain(wallTimeNanoseconds:bytes:),
            "ZeppelinStore.purge(_:)": ZeppelinStore.purge(_:),
            "ZeppelinStore.awaitPhysicalPurge(_:)": ZeppelinStore.awaitPhysicalPurge(_:),
            "ZeppelinStore.applyRetention(window:now:)": ZeppelinStore.applyRetention(window:now:),
            "ZeppelinStore.dropPartition(from:to:)": ZeppelinStore.dropPartition(from:to:),
            "ZeppelinStore.verify(at:)": ZeppelinStore.verify(at:),
            "VerifyReport": VerifyReport.self,
            "ZeppelinCancellationToken.create()": ZeppelinCancellationToken.create,
            "ZeppelinCancellationToken.cancel()": ZeppelinCancellationToken.cancel,
            "ZeppelinCancellationToken.free()": ZeppelinCancellationToken.free,
            "QueryOptions.cancellationToken": \QueryOptions.cancellationToken,
            "QueryOptions.deadlineNanoseconds": \QueryOptions.deadlineNanoseconds,
            "ZeppelinError.rawValue": \ZeppelinError.rawValue,
            "ZeppelinStore.lastErrorMessage()": ZeppelinStore.lastErrorMessage,
            "ZeppelinStore.abiVersion": ZeppelinStore.abiVersion,
            "NamespaceSpec.tokenizerProfile": \NamespaceSpec.tokenizerProfile,
            "ZeppelinStore.listNamespaces(root:)": ZeppelinStore.listNamespaces(root:),
            "ZeppelinStore.namespaceBatch(root:participants:)": ZeppelinStore.namespaceBatch(root:participants:),
            "ZeppelinStore.namespaceBatchLive(root:participants:)": ZeppelinStore.namespaceBatchLive(root:participants:),
            "ZeppelinStore.declareCascade(root:participants:declaration:)":
                ZeppelinStore.declareCascade(root:participants:declaration:),
            "ZeppelinStore.deleteCascade(root:participants:)": ZeppelinStore.deleteCascade(root:participants:),
        ]
        #if ZE_TEXT
        api.merge([
            "NamespaceSpec.embeddedModel": \NamespaceSpec.embeddedModel,
            "ZeppelinStore.queryEmbedded(_:leg:k:eligibleIds:filter:timestampRange:cancellationToken:deadlineNanoseconds:)": ZeppelinStore.queryEmbedded(_:leg:k:eligibleIds:filter:timestampRange:cancellationToken:deadlineNanoseconds:),
            "ZeppelinStore.nearestChunks(_:options:)": ZeppelinStore.nearestChunks(_:options:),
            "ZeppelinStore.nearestChunksBatch(_:options:)": ZeppelinStore.nearestChunksBatch(_:options:),
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
            "ZeppelinStore.enableGraph()": ZeppelinStore.enableGraph,
            "ZeppelinStore.open(at:relationshipTypes:maxResidentBytes:readerDrainTimeoutMilliseconds:)": ZeppelinStore.open(at:relationshipTypes:maxResidentBytes:readerDrainTimeoutMilliseconds:),
            "ZeppelinStore.graphApply(_:interruption:)": ZeppelinStore.graphApply(_:interruption:),
            "GraphBatch": GraphBatch.self,
            "ZeppelinStore.cypher(_:parameters:controls:options:)": ZeppelinStore.cypher(_:parameters:controls:options:),
            "GraphControls.rowLimit": \GraphControls.rowLimit,
            "ZeppelinStore.graphQuery(_:parameters:options:interruption:)": ZeppelinStore.graphQuery(_:parameters:options:interruption:),
            "GraphPlan": GraphPlan.self,
            "ZeppelinStore.getNodes(_:fields:limits:interruption:)": ZeppelinStore.getNodes(_:fields:limits:interruption:),
            "ZeppelinStore.getRelationships(_:limits:interruption:)": ZeppelinStore.getRelationships(_:limits:interruption:),
            "ZeppelinStore.graphResources()": ZeppelinStore.graphResources,
            "ZeppelinStore.graphMaintain(interruption:)": ZeppelinStore.graphMaintain(interruption:),
            "ZeppelinStore.graphSetMaintenancePolicy(_:)": ZeppelinStore.graphSetMaintenancePolicy(_:),
            "GraphNodeImage.text": \GraphNodeImage.text,
        ]) { current, _ in current }
        #endif
        return api
    }
}
