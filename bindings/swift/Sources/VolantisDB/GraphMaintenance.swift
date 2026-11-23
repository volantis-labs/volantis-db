import Foundation

#if ZE_GRAPH
import CVolantisDB
#endif

/// An immutable incoming-reference rule for a relationship type.
public struct GraphRelationshipType: Sendable {
    public enum OnDelete: UInt32, Sendable {
        case restrict = 1
        case cascade = 2
    }

    public let type: String
    public let onDelete: OnDelete

    public init(type: String, onDelete: OnDelete) {
        self.type = type
        self.onDelete = onDelete
    }
}

public struct GraphMaintenancePolicy: Sendable {
    public var autoReclaim: Bool
    public var reclaimAfterBytes: UInt64

    public init(autoReclaim: Bool = true, reclaimAfterBytes: UInt64 = 64 * 1_024 * 1_024) {
        self.autoReclaim = autoReclaim
        self.reclaimAfterBytes = reclaimAfterBytes
    }
}

public struct GraphMaintenanceReport: Sendable {
    public let generation: UInt64
    public let replacedPhysicalRefs: UInt64
    public let newPackBytes: UInt64
    public let relocatedBytes: UInt64
    public let drainedPacks: UInt64
    public let reclaimedBytes: UInt64
    public let removedBytes: UInt64
    public let cycleComplete: Bool
}

@available(macOS 14.0, iOS 17.0, *)
extension VolantisStore {
    /// Creates a graph store with persisted relationship rules. The path must be new.
    public static func open(
        at path: URL,
        relationshipTypes: [GraphRelationshipType],
        maxResidentBytes: UInt64 = 256 * 1_024 * 1_024,
        readerDrainTimeoutMilliseconds: UInt64 = 5_000
    ) async throws -> VolantisStore {
        #if ZE_GRAPH
        let handle = try await runGraphBlocking {
            try withRelationshipTypes(relationshipTypes) { rules in
                try Array(path.path.utf8).withUnsafeBufferPointer { pathBytes in
                    var request = ZeGraphOpenRequest()
                    request.abi_size = graphSize(ZeGraphOpenRequest.self)
                    request.path = ZeGraphBytes(data: pathBytes.baseAddress, count: pathBytes.count)
                    request.max_resident_bytes = maxResidentBytes
                    request.reader_drain_timeout_ms = readerDrainTimeoutMilliseconds
                    var handle: UInt64 = 0
                    try checkVolantis(ze_store_create_with_relationship_types(
                        &request, rules.baseAddress, rules.count, &handle
                    ))
                    return handle
                }
            }
        }
        let options = OpenOptions(
            durabilityMode: .durable,
            commitTier: .durable,
            readerDrainTimeoutMilliseconds: readerDrainTimeoutMilliseconds,
            maxResidentBytes: maxResidentBytes,
            maxTemporaryBytes: UInt64.max,
            excludeFromBackup: false
        )
        return VolantisStore(
            handle: handle,
            configuration: ReopenConfiguration(path: path, options: options, epoch: nil)
        )
        #else
        throw VolantisError.graphUnsupportedBuild
        #endif
    }

    /// Sets this open writer's in-memory maintenance policy.
    public func graphSetMaintenancePolicy(_ policy: GraphMaintenancePolicy) async throws {
        #if ZE_GRAPH
        let current = try openHandle()
        try await Self.runGraphBlocking {
            var native = ZeGraphMaintenancePolicy()
            native.abi_size = graphSize(ZeGraphMaintenancePolicy.self)
            native.automatic = policy.autoReclaim ? 1 : 0
            native.reclaim_after_bytes = policy.reclaimAfterBytes
            try checkVolantis(ze_store_set_graph_maintenance_policy(current, &native))
        }
        #else
        throw VolantisError.graphUnsupportedBuild
        #endif
    }

    /// Performs one bounded step. Call again while the report's cycleComplete is false.
    public func graphMaintain(
        interruption: GraphInterruption = .none
    ) async throws -> GraphMaintenanceReport {
        #if ZE_GRAPH
        let current = try openHandle()
        return try await Self.runGraphBlocking {
            try GraphControls(interruption: interruption).withControl { control in
                var report = ZeGraphMaintainReport()
                report.abi_size = graphSize(ZeGraphMaintainReport.self)
                try checkVolantis(ze_store_graph_maintain(current, control, &report))
                return GraphMaintenanceReport(
                    generation: report.generation,
                    replacedPhysicalRefs: report.replaced_physical_refs,
                    newPackBytes: report.new_pack_bytes,
                    relocatedBytes: report.relocated_bytes,
                    drainedPacks: report.drained_packs,
                    reclaimedBytes: report.reclaimed_bytes,
                    removedBytes: report.removed_bytes,
                    cycleComplete: try graphFlag(report.cycle_complete)
                )
            }
        }
        #else
        throw VolantisError.graphUnsupportedBuild
        #endif
    }
}

#if ZE_GRAPH
private func withRelationshipTypes<Result>(
    _ rules: [GraphRelationshipType],
    _ body: (UnsafeBufferPointer<ZeGraphRelationshipType>) throws -> Result
) throws -> Result {
    // Match the ABI's encoded declaration limits before allocating a byte buffer.
    guard rules.count <= 16_384 else { throw VolantisError.invalidArgument }
    let names = rules.map { $0.type.utf8 }
    let lengths = names.map(\.count)
    let maximumBytes = 8 * 1_024 * 1_024
    var encodedBytes = 0
    for length in lengths {
        guard length <= maximumBytes - encodedBytes - 9 else {
            throw VolantisError.invalidArgument
        }
        encodedBytes += 9 + length
    }
    var bytes: [UInt8] = []
    bytes.reserveCapacity(encodedBytes - 9 * rules.count)
    for name in names {
        bytes.append(contentsOf: name)
    }
    return try bytes.withUnsafeBufferPointer { buffer in
        var offset = 0
        let native = zip(rules, lengths).map { rule, length in
            var descriptor = ZeGraphRelationshipType()
            descriptor.abi_size = graphSize(ZeGraphRelationshipType.self)
            descriptor.name = ZeGraphBytes(data: buffer.baseAddress?.advanced(by: offset), count: length)
            descriptor.on_delete = rule.onDelete.rawValue
            offset += length
            return descriptor
        }
        return try native.withUnsafeBufferPointer(body)
    }
}
#endif
