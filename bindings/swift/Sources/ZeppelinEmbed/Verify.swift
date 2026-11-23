import CZeppelinEmbed
import Foundation

/// The kind of damage found by store verification.
public enum VerifyFindingKind: Equatable, Sendable {
    case manifestMissing
    case manifestCorrupt
    case manifestAheadOfWal
    case segmentMissing
    case segmentCorrupt
    case segmentMismatch
    case segmentRegionCorrupt
    case segmentIndexInvalid
    case walMissing
    case walHeaderCorrupt
    case walRecordCorrupt
    case walRecordInvalid
    case unreadable
    case purgeIntentCorrupt
    case graphObjectMissing
    case graphObjectCorrupt
    case graphInventoryInvalid

    fileprivate init(code: UInt32) throws {
        switch code {
        case UInt32(ZE_VERIFY_MANIFEST_MISSING): self = .manifestMissing
        case UInt32(ZE_VERIFY_MANIFEST_CORRUPT): self = .manifestCorrupt
        case UInt32(ZE_VERIFY_MANIFEST_AHEAD_OF_WAL): self = .manifestAheadOfWal
        case UInt32(ZE_VERIFY_SEGMENT_MISSING): self = .segmentMissing
        case UInt32(ZE_VERIFY_SEGMENT_CORRUPT): self = .segmentCorrupt
        case UInt32(ZE_VERIFY_SEGMENT_MISMATCH): self = .segmentMismatch
        case UInt32(ZE_VERIFY_SEGMENT_REGION_CORRUPT): self = .segmentRegionCorrupt
        case UInt32(ZE_VERIFY_SEGMENT_INDEX_INVALID): self = .segmentIndexInvalid
        case UInt32(ZE_VERIFY_WAL_MISSING): self = .walMissing
        case UInt32(ZE_VERIFY_WAL_HEADER_CORRUPT): self = .walHeaderCorrupt
        case UInt32(ZE_VERIFY_WAL_RECORD_CORRUPT): self = .walRecordCorrupt
        case UInt32(ZE_VERIFY_WAL_RECORD_INVALID): self = .walRecordInvalid
        case UInt32(ZE_VERIFY_UNREADABLE): self = .unreadable
        case UInt32(ZE_VERIFY_PURGE_INTENT_CORRUPT): self = .purgeIntentCorrupt
        case UInt32(ZE_VERIFY_GRAPH_OBJECT_MISSING): self = .graphObjectMissing
        case UInt32(ZE_VERIFY_GRAPH_OBJECT_CORRUPT): self = .graphObjectCorrupt
        case UInt32(ZE_VERIFY_GRAPH_INVENTORY_INVALID): self = .graphInventoryInvalid
        default: throw ZeppelinError.internalError
        }
    }
}

/// One damaged artifact.
public struct VerifyFinding: Equatable, Sendable {
    public let kind: VerifyFindingKind
    /// File name relative to the store directory.
    public let file: String
    /// Byte offset, when the decoder can locate the damage.
    public let offset: UInt64?
    public let detail: String

    fileprivate init(_ finding: ZeVerifyFinding) throws {
        guard finding.has_offset <= 1 else {
            throw ZeppelinError.internalError
        }
        kind = try VerifyFindingKind(code: finding.kind)
        file = try verifyString(finding.file, count: finding.file_len)
        offset = finding.has_offset == 1 ? finding.offset : nil
        detail = try verifyString(finding.detail, count: finding.detail_len)
    }
}

/// The counters and findings from a read-only store verification.
public struct VerifyReport: Equatable, Sendable {
    public var ok: Bool { findings.isEmpty }
    /// Generation of the decoded manifest, or zero without one.
    public let generation: UInt64
    /// Segments referenced by the manifest.
    public let segmentsChecked: UInt64
    /// WAL records that passed checksum and sequence validation.
    public let walRecordsChecked: UInt64
    /// Damaged artifacts in the order returned by the verifier.
    public let findings: [VerifyFinding]

    fileprivate init(_ result: ZeVerifyResult) throws {
        guard result.finding_count >= 0,
            (result.finding_count == 0) == (result.findings == nil)
        else {
            throw ZeppelinError.internalError
        }
        generation = result.generation
        segmentsChecked = result.segments_checked
        walRecordsChecked = result.wal_records_checked
        findings = try UnsafeBufferPointer(start: result.findings, count: result.finding_count)
            .map(VerifyFinding.init)
    }
}

@available(macOS 14.0, iOS 17.0, *)
extension ZeppelinStore {
    /// Verifies a store directory without opening or modifying the store.
    /// Damage is returned as findings. The call throws if the walk cannot start.
    public static func verify(at path: URL) async throws -> VerifyReport {
        let host = path.host(percentEncoded: false)?.lowercased() ?? ""
        guard path.isFileURL, host.isEmpty || host == "localhost" else {
            throw ZeppelinError.invalidArgument
        }

        return try await runBlocking {
            let bytes = Array(path.path(percentEncoded: false).utf8)
            return try bytes.withUnsafeBufferPointer { buffer in
                var request = ZeVerifyRequest()
                request.abi_size = abiSize(ZeVerifyRequest.self)
                request.path = buffer.baseAddress
                request.path_len = buffer.count
                var result = ZeVerifyResult()
                result.abi_size = abiSize(ZeVerifyResult.self)
                try checkZeppelin(ze_verify(&request, &result))

                // Release the arena even if decoding a finding fails.
                let report = Result { try VerifyReport(result) }
                try checkZeppelin(ze_verify_result_free(&result))
                return try report.get()
            }
        }
    }
}

private func verifyString(_ bytes: UnsafePointer<UInt8>?, count: Int) throws -> String {
    guard count >= 0, count == 0 || bytes != nil,
        let string = String(bytes: UnsafeBufferPointer(start: bytes, count: count), encoding: .utf8)
    else {
        throw ZeppelinError.internalError
    }
    return string
}
