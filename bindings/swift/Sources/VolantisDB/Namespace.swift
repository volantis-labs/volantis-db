import CVolantisDB
import Foundation

public struct VectorSpace: Sendable {
    public var dimensions: UInt32
    public var normalization: VectorNormalization

    public init(dimensions: UInt32, normalization: VectorNormalization) {
        self.dimensions = dimensions
        self.normalization = normalization
    }
}

public enum TokenizerProfile: Int32, Sendable {
    case textDefault = 0
    case code = 1
    case voice = 2
}

public struct NamespaceSpec: Sendable {
    public var attributes: [AttributeDefinition]
    public var vectorSpace: VectorSpace?
    public var epoch: Epoch?
    public var embeddedModel: TextModel?
    public var tokenizerProfile: TokenizerProfile

    public init(
        attributes: [AttributeDefinition],
        vectorSpace: VectorSpace?,
        epoch: Epoch? = nil,
        tokenizerProfile: TokenizerProfile = .textDefault
    ) {
        self.attributes = attributes
        self.vectorSpace = vectorSpace
        self.epoch = epoch
        self.embeddedModel = nil
        self.tokenizerProfile = tokenizerProfile
    }

    /// The pinned model supplies the vector space and embedding epoch.
    public init(
        attributes: [AttributeDefinition],
        embeddedModel: TextModel,
        tokenizerProfile: TokenizerProfile = .textDefault
    ) {
        self.init(attributes: attributes, vectorSpace: nil, tokenizerProfile: tokenizerProfile)
        self.embeddedModel = embeddedModel
    }
}

struct NamespaceReopenConfiguration: Sendable {
    var root: URL
    var name: String
    var spec: NamespaceSpec
}

@available(macOS 14.0, iOS 17.0, *)
extension VolantisStore {
    public static func openNamespace(
        root: URL,
        name: String,
        spec: NamespaceSpec,
        options: OpenOptions = OpenOptions()
    ) async throws -> VolantisStore {
        if spec.embeddedModel != nil {
            #if ZE_TEXT
            guard spec.vectorSpace == nil, spec.epoch == nil else {
                throw VolantisError.invalidArgument
            }
            #else
            throw VolantisError.unsupported
            #endif
        }
        let namespace = NamespaceReopenConfiguration(root: root, name: name, spec: spec)
        return try await open(
            configuration: ReopenConfiguration(
                path: root.appendingPathComponent(name, isDirectory: true),
                options: options,
                epoch: spec.epoch,
                namespace: namespace
            )
        )
    }

    public static func listNamespaces(root: URL) async throws -> [String] {
        try await runBlocking {
            let rootBytes = Array(root.path.utf8)
            return try rootBytes.withUnsafeBufferPointer { rootBuffer in
                var request = ZeNamespaceListRequest()
                request.abi_size = abiSize(ZeNamespaceListRequest.self)
                request.root = rootBuffer.baseAddress
                request.root_len = rootBytes.count
                var result = ZeNamespaceListResult()
                result.abi_size = abiSize(ZeNamespaceListResult.self)
                defer { _ = ze_namespace_list_result_free(&result) }
                try checkVolantis(ze_namespace_list(&request, &result))
                guard result.entry_count > 0 else {
                    return []
                }
                guard let entries = result.entries else {
                    throw VolantisError.internalError
                }
                return try UnsafeBufferPointer(start: entries, count: result.entry_count).map {
                    entry in
                    guard entry.name_len == 0 || entry.name != nil,
                        let name = String(
                            bytes: UnsafeBufferPointer(start: entry.name, count: entry.name_len),
                            encoding: .utf8
                        )
                    else {
                        throw VolantisError.internalError
                    }
                    return name
                }
            }
        }
    }

    static func openNamespaceHandle(
        _ configuration: NamespaceReopenConfiguration,
        options: OpenOptions
    ) throws -> UInt64 {
        let rootBytes = Array(configuration.root.path.utf8)
        let nameBytes = Array(configuration.name.utf8)
        return try rootBytes.withUnsafeBufferPointer { rootBuffer in
            try nameBytes.withUnsafeBufferPointer { nameBuffer in
                try withNamespaceSpec(configuration.spec) { specPointer in
                    var open = ZeOpenRequest()
                    open.abi_size = abiSize(ZeOpenRequest.self)
                    open.access_mode = options.accessMode.rawValue
                    open.durability_mode = options.durabilityMode.rawValue
                    open.commit_tier = options.commitTier.rawValue
                    open.reader_drain_timeout_ms = options.readerDrainTimeoutMilliseconds
                    open.max_resident_bytes = options.maxResidentBytes
                    open.max_temp_bytes = options.maxTemporaryBytes

                    var request = ZeNamespaceOpenRequest()
                    request.abi_size = abiSize(ZeNamespaceOpenRequest.self)
                    request.root = rootBuffer.baseAddress
                    request.root_len = rootBuffer.count
                    request.name = nameBuffer.baseAddress
                    request.name_len = nameBuffer.count
                    request.open = open
                    request.spec = specPointer
                    var handle: UInt64 = 0
                    try checkVolantis(
                        ze_namespace_open_with_tokenizer(
                            &request, configuration.spec.tokenizerProfile.rawValue, &handle
                        )
                    )
                    return handle
                }
            }
        }
    }

    nonisolated static func withNamespaceSpec<Result>(
        _ spec: NamespaceSpec,
        _ body: (UnsafePointer<ZeNamespaceSpec>) throws -> Result
    ) throws -> Result {
        let nameFields = spec.attributes.map { Array($0.name.utf8) }
        let nameOffsets = offsets(nameFields.map(\.count))
        let attributeNameBytes = nameFields.flatMap { $0 }
        return try attributeNameBytes.withUnsafeBufferPointer { nameBuffer in
            let definitions = spec.attributes.enumerated().map { index, definition in
                var raw = ZeAttributeDefinition()
                raw.attribute_id = definition.id
                raw.name = pointer(nameBuffer.baseAddress, offset: nameOffsets[index], count: nameFields[index].count)
                raw.name_len = nameFields[index].count
                raw.attribute_type = definition.type.rawValue
                raw.nullable = definition.nullable ? 1 : 0
                return raw
            }
            return try definitions.withUnsafeBufferPointer { definitionBuffer in
                func invoke(_ epoch: UnsafePointer<ZeEpochRequest>?) throws -> Result {
                    var raw = ZeNamespaceSpec()
                    raw.abi_size = abiSize(ZeNamespaceSpec.self)
                    raw.attributes = definitionBuffer.baseAddress
                    raw.attribute_count = definitionBuffer.count
                    raw.has_vector_space = spec.vectorSpace == nil ? 0 : 1
                    raw.dimensions = spec.vectorSpace?.dimensions ?? 0
                    raw.normalization = spec.vectorSpace?.normalization.rawValue ?? 0
                    raw.epoch = epoch
                    return try withUnsafeNamespaceSpec(&raw, model: spec.embeddedModel, body)
                }

                if let epoch = spec.epoch {
                    return try withEpochRequest(epoch) { try invoke($0) }
                }
                return try invoke(nil)
            }
        }
    }
}

private func withUnsafeNamespaceSpec<Result>(
    _ spec: inout ZeNamespaceSpec,
    model: TextModel?,
    _ body: (UnsafePointer<ZeNamespaceSpec>) throws -> Result
) throws -> Result {
    if let model {
        #if ZE_TEXT
        guard spec.has_vector_space == 0, spec.epoch == nil else {
            throw VolantisError.invalidArgument
        }
        return try model.withUnsafeModelRef { modelRef in
            var embedded = ZeEmbeddedNamespaceSpec()
            embedded.spec = spec
            embedded.spec.abi_size = abiSize(ZeEmbeddedNamespaceSpec.self)
            embedded.spec.has_vector_space = 2
            embedded.model = modelRef
            // The C entry point reads the sized extension from its prefix.
            return try withUnsafePointer(to: &embedded) { pointer in
                try body(UnsafeRawPointer(pointer).assumingMemoryBound(to: ZeNamespaceSpec.self))
            }
        }
        #else
        throw VolantisError.unsupported
        #endif
    }
    return try withUnsafePointer(to: &spec, body)
}
