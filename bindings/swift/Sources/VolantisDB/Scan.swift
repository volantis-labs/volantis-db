import CVolantisDB
import Foundation

public enum ScanOrder: Int32, Sendable {
    case storage = 0
    case timestampAscending = 1
    case timestampDescending = 2
}

public struct AttributeScanOrder: Sendable {
    public enum Direction: Sendable {
        case ascending
        case descending
    }

    public let attributeID: UInt32
    public let direction: Direction

    public init(attributeID: UInt32, direction: Direction) {
        self.attributeID = attributeID
        self.direction = direction
    }
}

public struct ScanCursor: Sendable {
    let generation: UInt64
    let segmentID: [UInt8]
    let row: UInt32
    let phase: UInt32
    let order: (code: Int32, attributeID: UInt32)
}

public struct ScanPage: Sendable {
    public let documents: [StoredDocument]
    public let generation: UInt64
    public let next: ScanCursor?
}

public struct CountResult: Sendable {
    public let count: UInt64
    public let generation: UInt64
}

public struct CountGroupBy: Sendable {
    public let attributeID: UInt32
    public let limit: Int

    public init(attributeID: UInt32, limit: Int = 1_024) {
        self.attributeID = attributeID
        self.limit = limit
    }
}

public struct CountGroup: Sendable {
    public let value: AttributeValue
    public let count: UInt64
}

public struct GroupedCountResult: Sendable {
    public let count: UInt64
    public let generation: UInt64
    public let missingCount: UInt64
    public let groups: [CountGroup]
}

private actor DocumentStreamPager {
    let store: VolantisStore
    let order: ScanOrder
    let fields: DocumentFields
    let filter: Filter?
    var cursor: ScanCursor?
    var documents: [StoredDocument] = []
    var position = 0
    var finished = false

    init(store: VolantisStore, order: ScanOrder, fields: DocumentFields, filter: Filter?) {
        self.store = store
        self.order = order
        self.fields = fields
        self.filter = filter
    }

    func next() async throws -> StoredDocument? {
        while position == documents.count {
            guard !finished else {
                return nil
            }
            let page = try await store.scan(
                after: cursor,
                limit: 1_024,
                order: order,
                fields: fields,
                filter: filter
            )
            documents = page.documents
            position = 0
            cursor = page.next
            finished = page.next == nil
        }
        defer { position += 1 }
        return documents[position]
    }
}

@available(macOS 14.0, iOS 17.0, *)
extension VolantisStore {
    public func scan(
        after cursor: ScanCursor? = nil,
        limit: Int = 1_024,
        order: ScanOrder = .storage,
        fields: DocumentFields = .all,
        timestampRange: Range<Int64>? = nil,
        filter: Filter? = nil,
        cancellationToken: VolantisCancellationToken? = nil,
        deadlineNanoseconds: UInt64 = 0
    ) async throws -> ScanPage {
        try await scan(
            after: cursor,
            limit: limit,
            order: (order.rawValue, 0),
            fields: fields,
            timestampRange: timestampRange,
            filter: filter,
            cancellationToken: cancellationToken,
            deadlineNanoseconds: deadlineNanoseconds
        )
    }

    public func scan(
        after cursor: ScanCursor? = nil,
        limit: Int = 1_024,
        order: AttributeScanOrder,
        fields: DocumentFields = .all,
        timestampRange: Range<Int64>? = nil,
        filter: Filter? = nil,
        cancellationToken: VolantisCancellationToken? = nil,
        deadlineNanoseconds: UInt64 = 0
    ) async throws -> ScanPage {
        try await scan(
            after: cursor,
            limit: limit,
            order: (order.direction == .ascending ? 3 : 4, order.attributeID),
            fields: fields,
            timestampRange: timestampRange,
            filter: filter,
            cancellationToken: cancellationToken,
            deadlineNanoseconds: deadlineNanoseconds
        )
    }

    private func scan(
        after cursor: ScanCursor?,
        limit: Int,
        order: (code: Int32, attributeID: UInt32),
        fields: DocumentFields,
        timestampRange: Range<Int64>?,
        filter: Filter?,
        cancellationToken: VolantisCancellationToken?,
        deadlineNanoseconds: UInt64
    ) async throws -> ScanPage {
        let current = try openHandle()
        let token = try cancellationToken?.rawValue() ?? 0
        return try await Self.runBlocking {
            try withUnsafeFilter(filter) { filterPointer in
                var request = ZeScanOrderedRequest()
                request.abi_size = abiSize(ZeScanOrderedRequest.self)
                request.scan.abi_size = abiSize(ZeScanRequest.self)
                if let cursor {
                    request.scan.cursor_generation = cursor.generation
                    withUnsafeMutableBytes(of: &request.scan.cursor_segment_id) { bytes in
                        bytes.copyBytes(from: cursor.segmentID)
                    }
                    request.scan.cursor_next_row = cursor.row
                    request.scan.cursor_phase = cursor.phase
                    request.cursor_order = cursor.order.code
                    request.cursor_order_attribute_id = cursor.order.attributeID
                }
                request.scan.limit = limit
                request.scan.order = order.code
                request.order_attribute_id = order.attributeID
                request.scan.include_vector = fields.contains(.vector) ? 1 : 0
                request.scan.include_text = fields.contains(.text) ? 1 : 0
                request.scan.include_metadata = fields.contains(.metadata) ? 1 : 0
                request.scan.include_attributes = fields.contains(.attributes) ? 1 : 0
                if let timestampRange {
                    request.scan.has_timestamp_range = 1
                    request.scan.start_ts = timestampRange.lowerBound
                    request.scan.end_ts = timestampRange.upperBound
                }
                request.scan.filter = filterPointer
                request.scan.cancel_token = token
                request.scan.deadline_ns = deadlineNanoseconds

                var result = ZeScanResult()
                result.abi_size = abiSize(ZeScanResult.self)
                try checkVolantis(ze_scan_ordered(current, &request, &result))
                defer { _ = ze_scan_result_free(&result) }
                let copied = try Self.copyStoredDocuments(
                    result.documents,
                    count: result.document_count,
                    fields: fields
                ).map { document -> StoredDocument in
                    guard let document else {
                        throw VolantisError.internalError
                    }
                    return document
                }
                let next: ScanCursor?
                switch result.has_more {
                case 0:
                    next = nil
                case 1:
                    var segmentID = result.next_segment_id
                    let bytes = withUnsafeBytes(of: &segmentID) { Array($0.prefix(16)) }
                    next = ScanCursor(
                        generation: result.generation,
                        segmentID: bytes,
                        row: result.next_row,
                        phase: result.next_phase,
                        order: order
                    )
                default:
                    throw VolantisError.internalError
                }
                return ScanPage(documents: copied, generation: result.generation, next: next)
            }
        }
    }

    public nonisolated func documents(
        order: ScanOrder = .storage,
        fields: DocumentFields = .all,
        filter: Filter? = nil
    ) -> AsyncThrowingStream<StoredDocument, Error> {
        let pager = DocumentStreamPager(store: self, order: order, fields: fields, filter: filter)
        return AsyncThrowingStream(unfolding: {
            try await pager.next()
        })
    }

    public func count(
        filter: Filter? = nil,
        timestampRange: Range<Int64>? = nil
    ) async throws -> CountResult {
        let current = try openHandle()
        return try await Self.runBlocking {
            try withUnsafeFilter(filter) { filterPointer in
                var request = ZeCountRequest()
                request.abi_size = abiSize(ZeCountRequest.self)
                request.filter = filterPointer
                if let timestampRange {
                    request.has_timestamp_range = 1
                    request.start_ts = timestampRange.lowerBound
                    request.end_ts = timestampRange.upperBound
                }
                var result = ZeCountResult()
                result.abi_size = abiSize(ZeCountResult.self)
                try checkVolantis(ze_count(current, &request, &result))
                return CountResult(count: result.count, generation: result.generation)
            }
        }
    }

    public func count(
        groupBy: CountGroupBy,
        filter: Filter? = nil,
        timestampRange: Range<Int64>? = nil
    ) async throws -> GroupedCountResult {
        let current = try openHandle()
        return try await Self.runBlocking {
            try withUnsafeFilter(filter) { filterPointer in
                var request = ZeCountGroupedRequest()
                request.abi_size = abiSize(ZeCountGroupedRequest.self)
                request.count.abi_size = abiSize(ZeCountRequest.self)
                request.count.filter = filterPointer
                if let timestampRange {
                    request.count.has_timestamp_range = 1
                    request.count.start_ts = timestampRange.lowerBound
                    request.count.end_ts = timestampRange.upperBound
                }
                request.group_attribute_id = groupBy.attributeID
                request.group_limit = groupBy.limit

                var result = ZeCountGroupedResult()
                result.abi_size = abiSize(ZeCountGroupedResult.self)
                try checkVolantis(ze_count_grouped(current, &request, &result))
                defer { _ = ze_count_grouped_result_free(&result) }

                guard let base = result.groups, result.group_count >= 0 else {
                    throw VolantisError.internalError
                }
                let groups = try UnsafeBufferPointer(start: base, count: result.group_count).map { group in
                    let value = try Self.attributeValue(group.value)
                    switch value {
                    case .u64, .i64, .string:
                        return CountGroup(value: value, count: group.count)
                    default:
                        throw VolantisError.internalError
                    }
                }
                return GroupedCountResult(
                    count: result.count,
                    generation: result.generation,
                    missingCount: result.missing_count,
                    groups: groups
                )
            }
        }
    }
}
