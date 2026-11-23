#if ZE_TEXT
import CZeppelinEmbed
import Foundation

public struct TextDocument: Sendable {
    /// Caller identity must fit in 96 bits, as required by the text ABI.
    public var id: DocumentID
    public var revision: UInt64
    public var text: String
    public init(id: DocumentID, revision: UInt64, text: String) {
        self.id = id; self.revision = revision; self.text = text
    }
}

public enum TextChunkPolicy: Sendable {
    case rejectOverlong
    case tokens(max: Int, overlap: Int)
}

public struct TextIngestOptions: Sendable {
    public var embedBatchSize = 32
    public var sealEvery = 4096
    public var channelCapacity = 2
    public var chunkPolicy: TextChunkPolicy = .rejectOverlong
    public var maintenanceWallTimeNanoseconds: UInt64 = 0
    public var maintenanceBytes: UInt64 = 0
    public init() {}
}

public struct TextIngestReport: Sendable {
    public let generation, documents, chunks, tokens, embedBatches, seals: UInt64
    public let maxInFlightBatches: UInt64
    public let allThreadsJoined: Bool
    public let elapsedNanoseconds: UInt64
}

public struct TextHit: Sendable {
    public let id: DocumentID
    public let revision: UInt64
    public let chunk: UInt32
    public let text: String
    public let score: Double
    public let vectorSquaredL2: Double?
    public let lexicalBM25: Double?
}

public struct TextQueryResult: Sendable {
    public let hits: [TextHit]
    public let embeddingEpoch: UInt64
    public let tokenizerEpoch: UInt64
}
#endif

public enum TextLegs: Int32, Sendable { case dense = 0, lexical = 1, hybrid = 2 }

/// Immutable model reference: repository, 40-hex revision, and 64-hex manifest pin.
public struct TextModel: Sendable {
    /// Published immutable default model pair.
    public static let `default` = TextModel(
        repo: "volantis-labs/volantis-db-clean71-arctic-m-v2",
        revision: "d64b817c4021a3f1b9bded3152d9afdf9ac813e7",
        manifestSHA256: "123c6308e00f769c68498f9df60dfb9efc688f58c4ff2c4da266f978fcb215dc")
    public let repo: String
    public let revision: String
    public let manifestSHA256: String
    public init(repo: String, revision: String, manifestSHA256: String) {
        self.repo = repo; self.revision = revision; self.manifestSHA256 = manifestSHA256
    }
}

#if ZE_TEXT
extension TextModel {
    func withUnsafeModelRef<Result>(_ body: (ZeTextModelRef) throws -> Result) rethrows -> Result {
        try Array(repo.utf8).withUnsafeBufferPointer { repo in
            try Array(revision.utf8).withUnsafeBufferPointer { revision in
                try Array(manifestSHA256.utf8).withUnsafeBufferPointer { pin in
                    var model = ZeTextModelRef()
                    model.abi_size = abiSize(ZeTextModelRef.self)
                    model.repo = repo.baseAddress
                    model.repo_len = repo.count
                    model.revision = revision.baseAddress
                    model.revision_len = revision.count
                    model.manifest_sha256 = pin.baseAddress
                    model.manifest_sha256_len = pin.count
                    return try body(model)
                }
            }
        }
    }
}

public struct TextDownloadProgress: Sendable {
    public let completedBytes: UInt64
    /// Zero while the manifest is being downloaded.
    public let totalBytes: UInt64
}

/// Preferred device counts from a CoreML compute plan, not an execution trace.
public struct TextObservedComputeUnits: Sendable {
    public let cpu, gpu, neuralEngine, other: UInt64
}
#endif
