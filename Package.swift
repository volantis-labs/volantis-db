// swift-tools-version: 5.10

import Foundation
import PackageDescription

let repositoryRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let environment = ProcessInfo.processInfo.environment
let ffiArchive = environment["ZE_LOCAL_FFI_ARCHIVE"] ?? repositoryRoot
    .appendingPathComponent("target/release/libvolantis_db_ffi.a")
    .standardizedFileURL.path
let useLocalFFI = environment["ZE_USE_LOCAL_FFI"] == "1"
let graph = environment["ZE_ENABLE_GRAPH"] == "1"
let useLocalXCFramework = environment["ZE_USE_LOCAL_XCFRAMEWORK"] == "1"
// The checksum has to be a literal. When a consumer depends on this package by
// URL, SwiftPM compiles the manifest in a sandbox where `#filePath` is
// `/Package.swift` and the rest of the repository is not reachable, so any
// attempt to read the checksum from a file next to this one fails and takes the
// consumer's whole dependency resolution down with it. The value is CI's, not
// a local build's (BL-167), and the `swift-release` workflow is the gate: it
// fails the release when this literal disagrees with the archive it attaches,
// and prints the value to pin. `scripts/xcframework/build.sh` only reports a
// local mismatch.
let binaryChecksum = "d9d2b3d6816edc3d7ddba38a3d882c225a602399c42e85e8d3d3d162946bf346" // ze:xcframework-checksum

let graphBinaryChecksum = "9772761ecea6983e3f0904c3e8f78cb6aac77ee7d684273193b028934ed1a8b8" // ze:graph-xcframework-checksum

// Framework discovery uses the framework's name; preserve the public C import
// through a source re-export when selecting the dynamic full artifact.
let binaryTargetName = graph && !useLocalFFI ? "VolantisDBGraph" : "CVolantisDB"
let cTarget: Target
if useLocalFFI {
    cTarget = .systemLibrary(
        name: "CVolantisDB",
        path: "bindings/swift/Sources/CVolantisDB"
    )
} else if useLocalXCFramework {
    cTarget = .binaryTarget(
        name: binaryTargetName,
        path: graph
            ? "target/xcframework-graph-cypher/VolantisDBGraph.xcframework"
            : "target/xcframework/VolantisDB.xcframework"
    )
} else {
    cTarget = .binaryTarget(
        name: binaryTargetName,
        url: graph
            ? "https://github.com/volantis-labs/volantis-db/releases/download/v0.1.0/VolantisDBGraph.xcframework.zip"
            : "https://github.com/volantis-labs/volantis-db/releases/download/v0.1.0/VolantisDB.xcframework.zip",
        checksum: graph ? graphBinaryChecksum : binaryChecksum
    )
}

let localLinkerSettings: [LinkerSetting]? = useLocalFFI
    ? [
        .unsafeFlags([
            "-Xlinker", "-force_load",
            "-Xlinker", ffiArchive,
        ]),
    ]
    : nil

// Packaged graph headers already expose the complete C module. Only the
// source shim needs its graph declarations enabled through the C compiler.
let graphSwiftSettings: [SwiftSetting]? = graph
    ? [.define("ZE_GRAPH")] + (useLocalFFI
        ? [.unsafeFlags(["-Xcc", "-DZE_GRAPH"])]
        : [.define("ZE_TEXT")])
    : nil

let package = Package(
    name: "VolantisDB",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "VolantisDB", targets: ["VolantisDB"]),
    ],
    targets: [
        cTarget,
    ] + (graph && !useLocalFFI ? [
        .target(name: "CVolantisDB", dependencies: ["VolantisDBGraph"],
                path: "bindings/swift/Sources/CVolantisDBGraph"),
    ] : []) + [
        .target(
            name: "VolantisDB",
            dependencies: ["CVolantisDB"],
            path: "bindings/swift/Sources/VolantisDB",
            swiftSettings: graphSwiftSettings,
            linkerSettings: localLinkerSettings
        ),
        .testTarget(
            name: "VolantisDBTests",
            dependencies: ["VolantisDB"],
            path: "bindings/swift/Tests/VolantisDBTests",
            swiftSettings: graphSwiftSettings
        ),
    ] + (graph ? [
        .executableTarget(name: "GraphWorkload", dependencies: ["VolantisDB"],
            path: "bindings/swift/Examples", exclude: ["InstalledGraphConsumer", "FiveVectorsSearch", "RecordStore"],
            sources: ["GraphWorkload.swift"]),
    ] : [])
)
