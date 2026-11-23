import CVolantisDB
import Foundation

public struct NamespaceUpsert: Sendable {
    public enum ExpectedRevision: Sendable {
        case exactly(UInt64)
        case absent
    }

    public var document: IngestDocument
    public var expectedRevision: ExpectedRevision?

    public init(_ document: IngestDocument, expectedRevision: ExpectedRevision? = nil) {
        self.document = document
        self.expectedRevision = expectedRevision
    }
}

public struct NamespaceMutation: Sendable {
    public var name: String
    public var spec: NamespaceSpec
    public var upserts: [NamespaceUpsert]
    public var deletes: [DocumentID]
    public var deleteWhere: Filter?

    public init(
        name: String,
        spec: NamespaceSpec,
        upserts: [NamespaceUpsert] = [],
        deletes: [DocumentID] = [],
        deleteWhere: Filter? = nil
    ) {
        self.name = name
        self.spec = spec
        self.upserts = upserts
        self.deletes = deletes
        self.deleteWhere = deleteWhere
    }
}

@available(macOS 14.0, iOS 17.0, *)
public struct LiveNamespaceMutation: Sendable {
    public var store: VolantisStore
    public var mutation: NamespaceMutation

    public init(store: VolantisStore, mutation: NamespaceMutation) {
        self.store = store
        self.mutation = mutation
    }
}

public struct CascadeParticipant: Sendable {
    public var name: String
    public var spec: NamespaceSpec

    public init(name: String, spec: NamespaceSpec) {
        self.name = name
        self.spec = spec
    }
}

public struct CascadeDeclaration: Sendable {
    public var parentIndex: UInt32
    public var childIndex: UInt32
    public var attributeID: UInt32

    public init(parentIndex: UInt32, childIndex: UInt32, attributeID: UInt32) {
        self.parentIndex = parentIndex
        self.childIndex = childIndex
        self.attributeID = attributeID
    }
}

public struct CascadeDeleteParticipant: Sendable {
    public var name: String
    public var spec: NamespaceSpec
    public var deletes: [DocumentID]

    public init(name: String, spec: NamespaceSpec, deletes: [DocumentID] = []) {
        self.name = name
        self.spec = spec
        self.deletes = deletes
    }
}

@available(macOS 14.0, iOS 17.0, *)
extension VolantisStore {
    public static func namespaceBatch(root: URL, participants: [NamespaceMutation]) async throws -> [UInt64] {
        try await runBlocking {
            try withNamespaceBatch(root: root, participants: participants, minimumCount: 2) { request in
                try checkVolantis(ze_namespace_batch(&request))
            }
        }
    }

    public static func namespaceBatchLive(root: URL, participants: [LiveNamespaceMutation]) async throws -> [UInt64] {
        guard (2...128).contains(participants.count) else {
            throw VolantisError.invalidArgument
        }
        var handles: [UInt64] = []
        for participant in participants {
            handles.append(try await participant.store.openHandle())
        }
        let batchHandles = handles
        return try await runBlocking {
            defer { withExtendedLifetime(participants) {} }
            return try batchHandles.withUnsafeBufferPointer { handleBuffer in
                let mutations = participants.map(\.mutation)
                return try withNamespaceBatch(root: root, participants: mutations, minimumCount: 2) { batch in
                    var request = ZeNamespaceBatchLiveRequest()
                    request.abi_size = abiSize(ZeNamespaceBatchLiveRequest.self)
                    request.batch = batch
                    request.handles = handleBuffer.baseAddress
                    try checkVolantis(ze_namespace_batch_live(&request))
                }
            }
        }
    }

    public static func declareCascade(
        root: URL,
        participants: [CascadeParticipant],
        declaration: CascadeDeclaration
    ) async throws {
        try await runBlocking {
            let mutations = participants.map { NamespaceMutation(name: $0.name, spec: $0.spec) }
            _ = try withNamespaceBatch(root: root, participants: mutations, minimumCount: 1) { request in
                var raw = ZeCascadeDeclaration()
                raw.abi_size = abiSize(ZeCascadeDeclaration.self)
                raw.parent_index = declaration.parentIndex
                raw.child_index = declaration.childIndex
                raw.attribute_id = declaration.attributeID
                try checkVolantis(ze_namespace_declare_cascade(&request, &raw))
            }
        }
    }

    public static func deleteCascade(root: URL, participants: [CascadeDeleteParticipant]) async throws -> [UInt64] {
        try await runBlocking {
            let mutations = participants.map { NamespaceMutation(name: $0.name, spec: $0.spec, deletes: $0.deletes) }
            return try withNamespaceBatch(root: root, participants: mutations, minimumCount: 1) { request in
                try checkVolantis(ze_namespace_delete_cascade(&request))
            }
        }
    }

    private static func withNamespaceBatch(
        root: URL,
        participants: [NamespaceMutation],
        minimumCount: Int,
        _ body: (inout ZeNamespaceBatchRequest) throws -> Void
    ) throws -> [UInt64] {
        // Bound the nesting of the participant buffers by the ABI transaction limit.
        guard (minimumCount...128).contains(participants.count) else {
            throw VolantisError.invalidArgument
        }
        var generations = [UInt64](repeating: 0, count: participants.count)
        var rawParticipants = [ZeNamespaceMutation](repeating: ZeNamespaceMutation(), count: participants.count)
        try rawParticipants.withUnsafeMutableBufferPointer { participantBuffer in
            // Each participant's buffers remain in scope until the native call returns.
            func marshal(_ remaining: ArraySlice<NamespaceMutation>) throws {
                if let participant = remaining.first {
                    try withNamespaceMutation(participant) { raw in
                        participantBuffer[remaining.startIndex] = raw
                        try marshal(remaining.dropFirst())
                    }
                    return
                }
                try Array(root.path.utf8).withUnsafeBufferPointer { rootBuffer in
                    try generations.withUnsafeMutableBufferPointer { generationBuffer in
                        var request = ZeNamespaceBatchRequest()
                        request.abi_size = abiSize(ZeNamespaceBatchRequest.self)
                        request.root = rootBuffer.baseAddress
                        request.root_len = rootBuffer.count
                        request.participants = UnsafeBufferPointer(participantBuffer).baseAddress
                        request.participant_count = participantBuffer.count
                        request.generations = generationBuffer.baseAddress
                        try body(&request)
                    }
                }
            }
            try marshal(participants[...])
        }
        return generations
    }

    private static func withNamespaceMutation(
        _ mutation: NamespaceMutation,
        _ body: (ZeNamespaceMutation) throws -> Void
    ) throws {
        var raw = ZeNamespaceMutation()
        raw.abi_size = abiSize(ZeNamespaceMutation.self)
        raw.tokenizer_profile = mutation.spec.tokenizerProfile.rawValue
        raw.upserts.abi_size = abiSize(ZeConditionalUpsertRequest.self)

        let conditions = mutation.upserts.map { upsert in
            var condition = ZeRevisionCondition()
            switch upsert.expectedRevision {
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

        try Array(mutation.name.utf8).withUnsafeBufferPointer { nameBuffer in
            raw.name = nameBuffer.baseAddress
            raw.name_len = nameBuffer.count
            try withNamespaceSpec(mutation.spec) { spec in
                raw.spec = spec
                try withUpsertRequest(mutation.upserts.map(\.document)) { upserts in
                    raw.upserts.batch = upserts
                    try conditions.withUnsafeBufferPointer { conditionBuffer in
                        raw.upserts.conditions = conditionBuffer.baseAddress
                        raw.upserts.condition_count = conditionBuffer.count
                        try mutation.deletes.map(cDocumentID).withUnsafeBufferPointer { deleteBuffer in
                            raw.deletes = deleteBuffer.baseAddress
                            raw.delete_count = deleteBuffer.count
                            try withUnsafeFilter(mutation.deleteWhere) { filter in
                                raw.filter = filter
                                try body(raw)
                            }
                        }
                    }
                }
            }
        }
    }
}
