import CVolantisDB

public struct EmbeddedHit: Sendable {
    public let id: DocumentID
    public let revision: UInt64
    public let timestamp: Int64
    public let attributes: [UInt32: AttributeValue]
    public let chunk: Int
    /// Inclusive UTF-8 offset in the original document.
    public let byteStart: Int
    /// Exclusive UTF-8 offset in the original document.
    public let byteEnd: Int
    public let text: String
    public let score: Double
}

public struct NeighbourSource: Sendable {
    public var id: DocumentID
    /// A zero-based chunk index, or nil to use every chunk of the source.
    public var chunk: UInt32?

    public init(id: DocumentID, chunk: UInt32? = nil) {
        self.id = id
        self.chunk = chunk
    }
}

public struct NeighbourOptions: Sendable {
    /// Maximum per source (1...4096). Defaults to 10 when threshold is nil.
    public var k: Int? // swiftlint:disable:this identifier_name
    /// Exclusive lower score bound. With only a threshold, there is no limit.
    public var threshold: Double?
    public var includeSource: Bool
    public var eligibleIds: [DocumentID]?
    public var filter: Filter?
    public var timestampRange: Range<Int64>?
    public var cancellationToken: VolantisCancellationToken?
    public var deadlineNanoseconds: UInt64

    public init(
        k limit: Int? = nil,
        threshold: Double? = nil,
        includeSource: Bool = false,
        eligibleIds: [DocumentID]? = nil,
        filter: Filter? = nil,
        timestampRange: Range<Int64>? = nil,
        cancellationToken: VolantisCancellationToken? = nil,
        deadlineNanoseconds: UInt64 = 0
    ) {
        self.k = limit
        self.threshold = threshold
        self.includeSource = includeSource
        self.eligibleIds = eligibleIds
        self.filter = filter
        self.timestampRange = timestampRange
        self.cancellationToken = cancellationToken
        self.deadlineNanoseconds = deadlineNanoseconds
    }
}

public struct ChunkNeighbour: Equatable, Sendable {
    public let id: DocumentID
    public let chunk: Int
    /// Negative squared L2 distance to the closest selected source chunk.
    public let score: Double
}

@available(macOS 14.0, iOS 17.0, *)
extension VolantisStore {
    /// Searches the chunks of an embedded namespace. The limit is 1...4096.
    public func queryEmbedded(
        _ text: String,
        leg: TextLegs = .hybrid,
        k limit: Int = 10,
        eligibleIds: [DocumentID]? = nil,
        filter: Filter? = nil,
        timestampRange: Range<Int64>? = nil,
        cancellationToken: VolantisCancellationToken? = nil,
        deadlineNanoseconds: UInt64 = 0
    ) async throws -> [EmbeddedHit] {
        #if ZE_TEXT
        let handle = try openHandle()
        guard (1...4096).contains(limit) else { throw VolantisError.invalidArgument }
        let token = try cancellationToken?.rawValue() ?? 0
        return try await Self.runBlocking {
            try withExtendedLifetime(cancellationToken) {
                try Array(text.utf8).withUnsafeBufferPointer { text in
                    try (eligibleIds ?? []).map(Self.cDocumentID).withUnsafeBufferPointer { ids in
                        try withChunkConstraints(filter, timestampRange) { constraints in
                            var request = ZeEmbeddedQueryRequest()
                            request.abi_size = abiSize(ZeEmbeddedQueryRequest.self)
                            request.text = text.baseAddress
                            request.text_len = text.count
                            request.leg = leg.rawValue
                            request.has_eligible = eligibleIds == nil ? 0 : 1
                            request.eligible_ids = ids.isEmpty ? nil : ids.baseAddress
                            request.eligible_count = ids.count
                            request.constraints = constraints
                            request.k = limit
                            request.cancel_token = token
                            request.deadline_ns = deadlineNanoseconds

                            return try Self.embeddedHits(handle, request: &request)
                        }
                    }
                }
            }
        }
        #else
        throw VolantisError.unsupported
        #endif
    }

    public func nearestChunks(
        _ source: NeighbourSource,
        options: NeighbourOptions = NeighbourOptions()
    ) async throws -> [ChunkNeighbour] {
        let groups = try await nearestChunksBatch([source], options: options)
        guard let hits = groups.first else { throw VolantisError.internalError }
        return hits
    }

    /// Returns a group for every source, in input order, including duplicates and empty groups.
    public func nearestChunksBatch(
        _ sources: [NeighbourSource],
        options: NeighbourOptions = NeighbourOptions()
    ) async throws -> [[ChunkNeighbour]] {
        #if ZE_TEXT
        let handle = try openHandle()
        if let limit = options.k, !(1...4096).contains(limit) {
            throw VolantisError.invalidArgument
        }
        let token = try options.cancellationToken?.rawValue() ?? 0
        return try await Self.runBlocking {
            let rawSources = sources.map { source in
                ZeNeighbourSource(doc_id: Self.cDocumentID(source.id),
                    has_chunk: source.chunk == nil ? 0 : 1, chunk: source.chunk ?? 0)
            }
            return try withExtendedLifetime(options.cancellationToken) {
                try rawSources.withUnsafeBufferPointer { sources in
                    try (options.eligibleIds ?? []).map(Self.cDocumentID).withUnsafeBufferPointer { ids in
                        try withChunkConstraints(options.filter, options.timestampRange) { constraints in
                            var request = ZeNeighbourRequest()
                            request.abi_size = abiSize(ZeNeighbourRequest.self)
                            request.sources = sources.baseAddress
                            request.source_count = sources.count
                            request.constraints = constraints
                            request.has_eligible = options.eligibleIds == nil ? 0 : 1
                            request.eligible_ids = ids.isEmpty ? nil : ids.baseAddress
                            request.eligible_count = ids.count
                            request.k = options.k ?? 0
                            request.has_threshold = options.threshold == nil ? 0 : 1
                            request.threshold = options.threshold ?? 0
                            request.include_source = options.includeSource ? 1 : 0
                            request.cancel_token = token
                            request.deadline_ns = options.deadlineNanoseconds

                            return try Self.chunkNeighbours(handle, request: &request)
                        }
                    }
                }
            }
        }
        #else
        throw VolantisError.unsupported
        #endif
    }

    #if ZE_TEXT
    private static func embeddedHits(
        _ handle: UInt64,
        request: inout ZeEmbeddedQueryRequest
    ) throws -> [EmbeddedHit] {
        var result = ZeGetResult()
        result.abi_size = abiSize(ZeGetResult.self)
        defer { _ = ze_get_result_free(&result) }
        var ranks = Array(repeating: ZeChunkRank(), count: request.k)
        try ranks.withUnsafeMutableBufferPointer { ranks in
            try checkVolantis(ze_query_embedded(handle, &request,
                ranks.baseAddress, ranks.count, &result))
        }
        guard (0...request.k).contains(result.document_count) else {
            throw VolantisError.internalError
        }
        let documents = try Self.copyStoredDocuments(result.documents,
            count: result.document_count, fields: [.text, .attributes])
        return try zip(documents, ranks).map { document, rank in
            guard let document, rank.byte_start >= 0,
                rank.byte_end >= rank.byte_start else {
                throw VolantisError.internalError
            }
            let text = document.text ?? ""
            guard text.utf8.count == rank.byte_end - rank.byte_start else {
                throw VolantisError.internalError
            }
            return EmbeddedHit(id: document.id, revision: document.revision,
                timestamp: document.timestamp, attributes: document.attributes,
                chunk: rank.chunk, byteStart: rank.byte_start, byteEnd: rank.byte_end,
                text: text, score: rank.score)
        }
    }

    private static func chunkNeighbours(
        _ handle: UInt64,
        request: inout ZeNeighbourRequest
    ) throws -> [[ChunkNeighbour]] {
        var result = ZeNeighbourResult()
        result.abi_size = abiSize(ZeNeighbourResult.self)
        defer { _ = ze_neighbour_result_free(&result) }
        try checkVolantis(ze_nearest_chunks(handle, &request, &result))
        guard result.hit_count >= 0, result.hit_count == 0 || result.hits != nil else {
            throw VolantisError.internalError
        }
        var groups = Array(repeating: [ChunkNeighbour](), count: request.source_count)
        for hit in UnsafeBufferPointer(start: result.hits, count: result.hit_count) {
            guard groups.indices.contains(hit.source_index) else {
                throw VolantisError.internalError
            }
            let id = DocumentID(high: hit.doc_id.high, low: hit.doc_id.low)
            groups[hit.source_index].append(
                ChunkNeighbour(id: id, chunk: hit.chunk, score: hit.score))
        }
        return groups
    }
    #endif
}

#if ZE_TEXT
private func withChunkConstraints<Result>(
    _ filter: Filter?,
    _ timestampRange: Range<Int64>?,
    _ body: (UnsafePointer<ZeQueryFilter>?) throws -> Result
) throws -> Result {
    guard filter != nil || timestampRange != nil else { return try body(nil) }
    return try withUnsafeFilter(filter) { filter in
        var constraints = ZeQueryFilter()
        constraints.abi_size = abiSize(ZeQueryFilter.self)
        constraints.filter = filter
        if let timestampRange {
            constraints.has_timestamp_range = 1
            constraints.start_ts = timestampRange.lowerBound
            constraints.end_ts = timestampRange.upperBound
        }
        return try withUnsafePointer(to: &constraints) { try body($0) }
    }
}
#endif
