import CVolantisDB

/// A requirement on the current live revision of a document.
public enum ExpectedRevision: Equatable, Sendable {
    case exactly(UInt64)
    case absent
}

public struct ConditionalUpsertDocument: Sendable {
    public let document: IngestDocument
    public let expectedRevision: ExpectedRevision?

    public init(document: IngestDocument, expectedRevision: ExpectedRevision? = nil) {
        self.document = document
        self.expectedRevision = expectedRevision
    }
}

public struct DeleteTarget: Sendable {
    public let id: DocumentID
    public let expectedRevision: ExpectedRevision?

    public init(id: DocumentID, expectedRevision: ExpectedRevision? = nil) {
        self.id = id
        self.expectedRevision = expectedRevision
    }
}

/// The first failed condition in a conditional write. The batch was not applied.
public struct RevisionConflict: Error, Equatable, Sendable {
    public var code: VolantisError { .revisionConflict }
    public let index: UInt64
    public let id: DocumentID
    public let expectedRevision: ExpectedRevision
    public let currentRevision: UInt64?
}

func cRevisionCondition(_ expected: ExpectedRevision?) -> ZeRevisionCondition {
    var condition = ZeRevisionCondition()
    switch expected {
    case .none:
        condition.kind = UInt32(ZE_REVISION_CONDITION_NONE)
    case .exactly(let revision):
        condition.kind = UInt32(ZE_REVISION_CONDITION_EXACTLY)
        condition.revision = revision
    case .absent:
        condition.kind = UInt32(ZE_REVISION_CONDITION_ABSENT)
    }
    return condition
}

func checkConditionalMutation(_ code: Int32, conflict: ZeRevisionConflict) throws {
    guard code == VolantisError.revisionConflict.rawValue else {
        return try checkVolantis(code)
    }
    let expected: ExpectedRevision
    switch conflict.expected_kind {
    case UInt32(ZE_REVISION_CONDITION_EXACTLY):
        expected = .exactly(conflict.expected_revision)
    case UInt32(ZE_REVISION_CONDITION_ABSENT):
        expected = .absent
    default:
        throw VolantisError.internalError
    }
    throw RevisionConflict(
        index: conflict.index,
        id: DocumentID(high: conflict.doc_id.high, low: conflict.doc_id.low),
        expectedRevision: expected,
        currentRevision: conflict.has_current == 1 ? conflict.current_revision : nil
    )
}

public struct DeleteWhereReport: Equatable, Sendable {
    public let deleted: UInt64
    public let generation: UInt64
}
