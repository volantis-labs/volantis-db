"""Pythonic Store wrapper over the frozen Zeppelin Embed C ABI."""

from __future__ import annotations

import ctypes as ct
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from enum import IntEnum
from pathlib import Path
from typing import Any, NoReturn, Self, TypeVar, overload

import numpy as np

from . import _ffi
from . import _structures as s
from ._errors import (
    ERROR_TYPES,
    RevisionConflict,
    TextUnavailable,
    UnsupportedBuild,
    ZeppelinError,
    invalid_argument,
    last_error_message,
    raise_for_status,
)
from ._library import LIBRARY
from ._types import (
    AccessMode,
    AttributeDefinition,
    AttributeScanOrder,
    AttributeType,
    AttributeValue,
    CascadeDeclaration,
    CascadeDeleteParticipant,
    CascadeParticipant,
    ChunkNeighbour,
    CommitTier,
    ComputeUnits,
    CountGroup,
    CountGroupBy,
    CountResult,
    DeleteTarget,
    DeleteWhereReport,
    DurabilityMode,
    EmbeddedHit,
    EmbeddedModelRef,
    EmbeddingEpoch,
    EmbeddingRuntime,
    EpochAliasReport,
    EpochDropReport,
    EpochIdentity,
    FusionMethod,
    FusionReport,
    GenerationReport,
    GetResult,
    GraphProfile,
    GroupedCountResult,
    LiveNamespaceMutation,
    MaintainReport,
    MaintenanceStatus,
    MutationReport,
    NamespaceMutation,
    NamespaceSpec,
    NeighbourSource,
    Normalization,
    ObservedComputeUnits,
    PartitionReport,
    PurgeReport,
    PurgeTokenReport,
    QueryHit,
    QueryMode,
    QueryResult,
    RevisionConflictDetails,
    ScanCursor,
    ScanPage,
    SearchHit,
    SearchResult,
    Snippet,
    SnippetHighlight,
    StateReport,
    StatsReport,
    StoredDocument,
    StoreState,
    TextHealthReport,
    TextHealthStatus,
    TextHit,
    TextIngestReport,
    TextLegs,
    TextQueryResult,
    Tier,
    TokenizerProfile,
    UpsertDocument,
    VectorSpace,
    VerifyFinding,
    VerifyFindingKind,
    VerifyReport,
)


def _require_text() -> None:
    if not _ffi.TEXT_AVAILABLE:
        raise TextUnavailable("text API is available on macOS arm64 only")


EnumValue = TypeVar("EnumValue", bound=IntEnum)


def _enum_value(value: EnumValue | str, enum_type: type[EnumValue], field: str) -> EnumValue:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        normalized = value.upper().replace("-", "_")
        try:
            return enum_type[normalized]
        except KeyError:
            invalid_argument(f"{field} value {value!r} is not supported")
    invalid_argument(f"{field} must be a {enum_type.__name__} or string")


def _bytes_pointer(data: bytes) -> tuple[Any, object | None]:
    if not data:
        return s.UInt8Pointer(), None
    owner = (ct.c_uint8 * len(data)).from_buffer_copy(data)
    return ct.cast(owner, s.UInt8Pointer), owner


def _text_pointer(text: str) -> tuple[Any, int, object | None]:
    if not isinstance(text, str):
        invalid_argument("text must be a string")
    try:
        encoded = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        invalid_argument(f"text is not valid UTF-8: {error}")
    pointer, owner = _bytes_pointer(encoded)
    return pointer, len(encoded), owner


def _doc_id(value: int | tuple[int, int]) -> s.ZeDocId:
    if isinstance(value, tuple):
        if len(value) != 2:
            invalid_argument("document-id tuple must contain high and low u64 values")
        high, low = value
    elif isinstance(value, int):
        if value < 0 or value >= 1 << 128:
            invalid_argument("document id must fit an unsigned 128-bit integer")
        high, low = value >> 64, value & ((1 << 64) - 1)
    else:
        invalid_argument("document id must be an integer or (high, low) tuple")
    if not isinstance(high, int) or not isinstance(low, int):
        invalid_argument("document-id halves must be integers")
    if high < 0 or high >= 1 << 64 or low < 0 or low >= 1 << 64:
        invalid_argument("document-id halves must fit unsigned 64-bit integers")
    return s.ZeDocId(high=high, low=low)


def _python_doc_id(value: s.ZeDocId) -> int:
    return (int(value.high) << 64) | int(value.low)


def _u64_revision(value: int, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 1 << 64:
        invalid_argument(f"{field} must be an unsigned 64-bit integer")
    return value


def _u32(value: int, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 1 << 32:
        invalid_argument(f"{field} must be an unsigned 32-bit integer")
    return value


def _revision_condition(expected_revision: int | None) -> s.ZeRevisionCondition:
    if expected_revision is None:
        return s.ZeRevisionCondition(kind=s.REVISION_CONDITION_ABSENT)
    return s.ZeRevisionCondition(
        kind=s.REVISION_CONDITION_EXACTLY,
        revision=_u64_revision(expected_revision, "expected_revision"),
    )


def _raise_for_conditional_status(
    status: int, handle: int, conflict: s.ZeRevisionConflict
) -> None:
    if status == RevisionConflict.code:
        raise RevisionConflict(
            last_error_message(handle),
            conflict=RevisionConflictDetails(
                index=int(conflict.index),
                doc_id=_python_doc_id(conflict.doc_id),
                expected_revision=(
                    int(conflict.expected_revision)
                    if conflict.expected_kind == s.REVISION_CONDITION_EXACTLY
                    else None
                ),
                current_revision=int(conflict.current_revision) if conflict.has_current else None,
            ),
        )
    raise_for_status(status, handle)


def _vector(value: np.ndarray[Any, Any], *, dimensions: int) -> np.ndarray[Any, Any]:
    if not isinstance(value, np.ndarray):
        invalid_argument("vector must be a NumPy array")
    if value.dtype != np.dtype(np.float32):
        invalid_argument("vector dtype must be float32")
    if value.ndim != dimensions:
        invalid_argument(f"vector must have exactly {dimensions} dimension(s)")
    if not value.flags.c_contiguous:
        invalid_argument("vector must be C-contiguous")
    if int(value.ctypes.data) % ct.alignment(ct.c_float) != 0:
        invalid_argument("vector buffer must be aligned for float32")
    return value


def _same_length(name: str, values: Sequence[object], expected: int) -> None:
    if len(values) != expected:
        invalid_argument(f"{name} length {len(values)} does not match document count {expected}")


def _bounded_int(value: object, low: int, high: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        invalid_argument(f"{field} must be an integer in {low}..{high}")
    return value


def _tower(value: Any, owners: list[object]) -> s.ZeEmbeddingTower:
    model_id, model_id_len, owner = _text_pointer(value.model_id)
    if owner is not None:
        owners.append(owner)
    model_version, model_version_len, owner = _text_pointer(value.model_version)
    if owner is not None:
        owners.append(owner)
    weights, owner = _bytes_pointer(value.weights_digest)
    if owner is not None:
        owners.append(owner)
    prompt, prompt_len, owner = _text_pointer(value.prompt_prefix)
    if owner is not None:
        owners.append(owner)
    if value.os_build is None:
        os_build, os_build_len, has_os_build = s.UInt8Pointer(), 0, 0
    else:
        os_build, os_build_len, owner = _text_pointer(value.os_build)
        if owner is not None:
            owners.append(owner)
        has_os_build = 1
    return s.ZeEmbeddingTower(
        model_id=model_id,
        model_id_len=model_id_len,
        model_version=model_version,
        model_version_len=model_version_len,
        weights_digest=weights,
        weights_digest_len=len(value.weights_digest),
        dims=value.dims,
        normalization=int(_enum_value(value.normalization, Normalization, "normalization")),
        prompt_prefix=prompt,
        prompt_prefix_len=prompt_len,
        max_tokens=value.max_tokens,
        runtime=int(_enum_value(value.runtime, EmbeddingRuntime, "runtime")),
        compute_units=int(_enum_value(value.compute_units, ComputeUnits, "compute_units")),
        has_os_build=has_os_build,
        os_build=os_build,
        os_build_len=os_build_len,
    )


def _epoch_request(value: EmbeddingEpoch) -> tuple[s.ZeEpochRequest, list[object]]:
    owners: list[object] = []
    document = _tower(value.document, owners)
    query = _tower(value.query, owners)
    alignment, owner = _bytes_pointer(value.alignment_digest)
    if owner is not None:
        owners.append(owner)
    request = s.ZeEpochRequest(
        abi_size=ct.sizeof(s.ZeEpochRequest),
        abi_reserved=0,
        embedding=s.ZeEmbeddingEpoch(
            document=document,
            query=query,
            alignment_digest=alignment,
            alignment_digest_len=len(value.alignment_digest),
        ),
        tokenizer_profile=value.tokenizer_profile,
        reserved=0,
    )
    return request, owners


def _open_request(
    path: str | os.PathLike[str],
    access_mode: AccessMode | str,
    durability: DurabilityMode | str,
    commit_tier: CommitTier | str,
    reader_drain_timeout_ms: int,
    max_resident_bytes: int,
    max_temp_bytes: int,
) -> tuple[s.ZeOpenRequest, object]:
    encoded = os.fspath(path).encode("utf-8", errors="strict")
    pointer, owner = _bytes_pointer(encoded)
    if owner is None:
        invalid_argument("store path must not be empty")
    request = s.ZeOpenRequest(
        abi_size=ct.sizeof(s.ZeOpenRequest),
        abi_reserved=0,
        path=pointer,
        path_len=len(encoded),
        access_mode=int(_enum_value(access_mode, AccessMode, "access_mode")),
        durability_mode=int(_enum_value(durability, DurabilityMode, "durability")),
        commit_tier=int(_enum_value(commit_tier, CommitTier, "commit_tier")),
        reader_drain_timeout_ms=reader_drain_timeout_ms,
        max_resident_bytes=max_resident_bytes,
        max_temp_bytes=max_temp_bytes,
    )
    return request, owner


def _attribute_value(
    attribute: AttributeValue,
    owners: list[object],
    *,
    allow_null: bool = True,
) -> s.ZeAttributeValue:
    attribute_type = _enum_value(
        attribute.attribute_type, AttributeType, "attribute_type"
    )
    value = attribute.value
    result = s.ZeAttributeValue(attribute_id=_u32(attribute.attribute_id, "attribute_id"))
    if value is None:
        if not allow_null:
            invalid_argument("filter values cannot be null")
        result.value_type = 0
    elif attribute_type is AttributeType.U64:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << 64:
            invalid_argument("U64 attribute value must fit an unsigned 64-bit integer")
        result.value_type = 1
        result.u64_value = value
    elif attribute_type is AttributeType.I64:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not -(1 << 63) <= value < 1 << 63
        ):
            invalid_argument("I64 attribute value must fit a signed 64-bit integer")
        result.value_type = 2
        result.i64_value = value
    elif attribute_type is AttributeType.F64:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            invalid_argument("F64 attribute value must be numeric")
        result.value_type = 3
        result.f64_value = float(value)
    elif attribute_type is AttributeType.BOOL:
        if not isinstance(value, bool):
            invalid_argument("Bool attribute value must be bool")
        result.value_type = 4
        result.bool_value = int(value)
    elif attribute_type is AttributeType.ID128:
        if not isinstance(value, int) or isinstance(value, bool):
            invalid_argument("ID128 attribute value must be an unsigned 128-bit integer")
        doc_id = _doc_id(value)
        result.value_type = 6
        result.u64_value = doc_id.low
        result.i64_value = ct.c_int64(doc_id.high).value
    else:
        if not isinstance(value, str):
            invalid_argument("string attribute value must be str")
        pointer, length, owner = _text_pointer(value)
        if owner is not None:
            owners.append(owner)
        result.value_type = 5
        result.string_value = pointer
        result.string_len = length
    return result


def _filter_field(
    field: str | int,
    attribute_ids: dict[str, int],
    attribute_types: dict[int, AttributeType],
    value: object | None,
) -> tuple[int, AttributeType]:
    if isinstance(field, str):
        if field == "ts":
            attribute_id = 0
        else:
            try:
                attribute_id = attribute_ids[field]
            except KeyError:
                invalid_argument(f"unknown filter field {field!r}")
    elif isinstance(field, int) and not isinstance(field, bool) and 0 <= field < 1 << 32:
        attribute_id = field
    else:
        invalid_argument("filter field must be an attribute name or u32 id")
    attribute_type = attribute_types.get(attribute_id)
    if attribute_type is None:
        if attribute_id == 0 or (isinstance(value, int) and value < 0):
            attribute_type = AttributeType.I64
        elif isinstance(value, bool):
            attribute_type = AttributeType.BOOL
        elif isinstance(value, int):
            attribute_type = AttributeType.U64
        elif isinstance(value, float):
            attribute_type = AttributeType.F64
        elif isinstance(value, str):
            attribute_type = AttributeType.RAW_STRING
        else:
            invalid_argument(f"cannot infer the type of filter field {field!r}")
    return attribute_id, attribute_type


_FILTER_FIELDS = {
    "eq": {"field", "value"},
    "not_eq": {"field", "value"},
    "in": {"field", "values"},
    "not_in": {"field", "values"},
    "range": {"field", "gt", "gte", "lt", "lte"},
    "exists": {"field"},
    "is_null": {"field"},
    "and": {"filters"},
    "or": {"filters"},
    "not": {"filter"},
}


def _flatten_filter(
    value: dict[str, Any],
    attribute_ids: dict[str, int],
    attribute_types: dict[int, AttributeType],
    *,
    strict: bool = False,
) -> tuple[s.ZeFilter, list[object]]:
    if not isinstance(value, dict):
        invalid_argument("filter must be built with Filter")
    slots: list[s.ZeFilterNode | None] = [None]
    owners: list[object] = []
    active: set[int] = set()
    operations = {
        "eq": 1,
        "not_eq": 2,
        "in": 3,
        "not_in": 4,
        "range": 5,
        "exists": 6,
        "is_null": 7,
        "and": 8,
        "or": 9,
        "not": 10,
    }

    def fill(node: dict[str, Any], index: int, depth: int) -> None:
        if not isinstance(node, dict):
            invalid_argument("filter children must be built with Filter")
        if depth > 32:
            invalid_argument("filter depth exceeds 32")
        identity = id(node)
        if identity in active:
            invalid_argument("filter tree contains a cycle")
        active.add(identity)
        try:
            operation = node.get("op")
            if not isinstance(operation, str) or operation not in operations:
                invalid_argument(f"filter operator {operation!r} is not supported")
            if strict:
                extra_fields = node.keys() - _FILTER_FIELDS[operation] - {"op"}
                if extra_fields:
                    invalid_argument(f"filter {operation} contains unsupported fields: {extra_fields}")
            op = operations[operation]
            if op >= 8:
                if op == 10:
                    if "filter" not in node:
                        invalid_argument("filter not requires exactly one child")
                    children: list[Any] = [node["filter"]]
                else:
                    raw_children = node.get("filters")
                    if not isinstance(raw_children, list):
                        invalid_argument(f"filter {operation} requires a filter list")
                    children = raw_children
                children_start = len(slots)
                slots.extend([None] * len(children))
                slots[index] = s.ZeFilterNode(
                    op=op,
                    children_start=children_start,
                    children_count=len(children),
                )
                for offset, child in enumerate(children):
                    fill(child, children_start + offset, depth + 1)
                return
            field = node.get("field")
            if field is None:
                invalid_argument(f"filter {operation} requires a field")
            if op in (1, 2):
                if "value" not in node:
                    invalid_argument(f"filter {operation} requires one value")
                node_values: tuple[Any, ...] = (node["value"],)
            elif op in (3, 4):
                raw_values = node.get("values")
                if not isinstance(raw_values, (list, tuple)):
                    invalid_argument(f"filter {operation} requires a value list")
                node_values = tuple(raw_values)
            else:
                node_values = ()
            if op == 5:
                if strict and any(
                    bound in node and node[bound] is None for bound in ("gt", "gte", "lt", "lte")
                ):
                    invalid_argument("filter range bounds must have values")
                if "gte" in node and "gt" in node:
                    invalid_argument("filter range cannot contain both gte and gt")
                if "lte" in node and "lt" in node:
                    invalid_argument("filter range cannot contain both lte and lt")
                lower = node.get("gte", node.get("gt"))
                upper = node.get("lte", node.get("lt"))
                if lower is None and upper is None:
                    invalid_argument("filter range requires at least one bound")
                lower_inclusive = "gte" in node
                upper_inclusive = "lte" in node
            else:
                lower = upper = None
                lower_inclusive = upper_inclusive = False
            sample = next((item for item in node_values if item is not None), None)
            if sample is None:
                sample = lower if lower is not None else upper
            attribute_id, attribute_type = _filter_field(
                field, attribute_ids, attribute_types, sample
            )
            values = (s.ZeAttributeValue * len(node_values))(
                *[
                    _attribute_value(
                        AttributeValue(attribute_id, attribute_type, item),
                        owners,
                        allow_null=False,
                    )
                    for item in node_values
                ]
            )
            values_pointer = (
                ct.cast(values, ct.POINTER(s.ZeAttributeValue))
                if node_values
                else ct.POINTER(s.ZeAttributeValue)()
            )
            if node_values:
                owners.append(values)
            lower_value = s.ZeAttributeValue()
            if lower is not None:
                lower_value = _attribute_value(
                    AttributeValue(attribute_id, attribute_type, lower),
                    owners,
                    allow_null=False,
                )
            upper_value = s.ZeAttributeValue()
            if upper is not None:
                upper_value = _attribute_value(
                    AttributeValue(attribute_id, attribute_type, upper),
                    owners,
                    allow_null=False,
                )
            slots[index] = s.ZeFilterNode(
                op=op,
                attribute_id=attribute_id,
                values=values_pointer,
                value_count=len(node_values),
                has_lower=int(lower is not None),
                lower=lower_value,
                lower_inclusive=int(lower_inclusive),
                has_upper=int(upper is not None),
                upper=upper_value,
                upper_inclusive=int(upper_inclusive),
            )
        finally:
            active.remove(identity)

    fill(value, 0, 1)
    filled: list[s.ZeFilterNode] = []
    for slot in slots:
        if slot is None:
            invalid_argument("filter tree contains an unfilled child index")
        filled.append(slot)
    nodes = (s.ZeFilterNode * len(filled))(*filled)
    owners.append(nodes)
    return (
        s.ZeFilter(
            abi_size=ct.sizeof(s.ZeFilter),
            abi_reserved=0,
            nodes=ct.cast(nodes, ct.POINTER(s.ZeFilterNode)),
            node_count=len(nodes),
            root=0,
        ),
        owners,
    )


def _returned_attribute(
    value: s.ZeAttributeValue,
    attribute_types: dict[int, AttributeType],
) -> AttributeValue:
    attribute_id = int(value.attribute_id)
    if value.value_type == 0:
        attribute_type = attribute_types.get(attribute_id, AttributeType.RAW_STRING)
        payload: int | float | bool | str | None = None
    elif value.value_type == 1:
        attribute_type = AttributeType.U64
        payload = int(value.u64_value)
    elif value.value_type == 2:
        attribute_type = AttributeType.I64
        payload = int(value.i64_value)
    elif value.value_type == 3:
        attribute_type = AttributeType.F64
        payload = float(value.f64_value)
    elif value.value_type == 4:
        attribute_type = AttributeType.BOOL
        payload = bool(value.bool_value)
    elif value.value_type == 5:
        attribute_type = attribute_types.get(attribute_id, AttributeType.RAW_STRING)
        payload = ct.string_at(value.string_value, value.string_len).decode("utf-8")
    elif value.value_type == 6:
        attribute_type = AttributeType.ID128
        payload = (ct.c_uint64(value.i64_value).value << 64) | int(value.u64_value)
    else:
        invalid_argument("returned attribute value type is invalid")
    return AttributeValue(attribute_id, attribute_type, payload)


def _stored_document(
    value: s.ZeStoredDocument,
    attribute_types: dict[int, AttributeType],
    *,
    include_vector: bool,
    include_text: bool,
    include_metadata: bool,
    include_attributes: bool,
) -> StoredDocument | None:
    if not value.has_document:
        return None
    vector = None
    if include_vector and value.vector and value.vector_len:
        vector = np.ctypeslib.as_array(value.vector, shape=(value.vector_len,)).copy()
    text = None
    if include_text:
        text = ct.string_at(value.text, value.text_len).decode("utf-8")
    metadata = None
    if include_metadata:
        metadata = ct.string_at(value.metadata, value.metadata_len)
    attributes = None
    if include_attributes:
        attributes = tuple(
            _returned_attribute(value.attributes[index], attribute_types)
            for index in range(value.attribute_count)
        )
    return StoredDocument(
        doc_id=_python_doc_id(value.doc_id),
        revision=int(value.revision),
        timestamp=int(value.timestamp),
        vector=vector,
        text=text,
        metadata=metadata,
        attributes=attributes,
    )


def epoch_identity(epoch: EmbeddingEpoch) -> EpochIdentity:
    """Compute the compact ABI identity of an epoch declaration."""

    request, owners = _epoch_request(epoch)
    report = s.sized(s.ZeEpochIdentity)
    status = LIBRARY.ze_epoch_identity(ct.byref(request), ct.byref(report))
    del owners
    raise_for_status(status)
    return EpochIdentity(int(report.embedding_epoch), int(report.tokenizer_epoch))


def _query_diagnostics(result: s.ZeQueryResult) -> QueryResult:
    return QueryResult(
        hits=(),
        generation=int(result.generation),
        mode=QueryMode(result.mode),
        approximate=bool(result.approximate),
        exact_rescore=bool(result.exact_rescore),
        budget_exhausted=bool(result.budget_exhausted),
        fusion=(
            FusionReport(
                FusionMethod(result.fusion_method),
                float(result.effective_alpha),
                int(result.fusion_rounds),
            )
            if result.has_fusion
            else None
        ),
        embedding_epoch=int(result.embedding_epoch)
        if result.has_embedding_epoch
        else None,
        tokenizer_epoch=int(result.tokenizer_epoch)
        if result.has_tokenizer_epoch
        else None,
        dims_touched=int(result.dims_touched),
        bytes_read=int(result.bytes_read),
        docs_evaluated=int(result.docs_evaluated),
        postings_decoded=int(result.postings_decoded),
    )


class CancelToken:
    """Generation-tagged cooperative cancellation token."""

    def __init__(self) -> None:
        token = ct.c_uint64()
        raise_for_status(LIBRARY.ze_cancel_token_create(ct.byref(token)))
        self._value = int(token.value)
        self._cancelled = False

    @property
    def value(self) -> int:
        if self._value == 0:
            raise ERROR_TYPES[3]("cancellation token is closed")
        return self._value

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def closed(self) -> bool:
        return self._value == 0

    def cancel(self) -> None:
        value = self.value
        raise_for_status(LIBRARY.ze_cancel_token_cancel(value))
        self._cancelled = True

    def close(self) -> None:
        if self._value == 0:
            return
        value = self._value
        self._value = 0
        raise_for_status(LIBRARY.ze_cancel_token_free(value))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


class Store:
    """One deterministic wrapper around a generation-tagged store handle."""

    def __init__(
        self,
        handle: int,
        *,
        text: bool = False,
        attribute_ids: dict[str, int] | None = None,
        attribute_types: dict[int, AttributeType] | None = None,
        vector_dimensions: int | None = None,
        record_only: bool = False,
        embedded: bool = False,
    ) -> None:
        self._handle = handle
        self._abi_call_count = 0
        self._text = text
        self._attribute_ids = {} if attribute_ids is None else dict(attribute_ids)
        self._attribute_types = {} if attribute_types is None else dict(attribute_types)
        self._vector_dimensions = vector_dimensions
        self._record_only = record_only
        self._embedded = embedded

    @property
    def closed(self) -> bool:
        return self._handle == 0

    @property
    def abi_call_count(self) -> int:
        """Number of store ABI calls, exposed for validation-boundary tests."""

        return self._abi_call_count

    def _live_handle(self) -> int:
        if self._handle == 0:
            raise ERROR_TYPES[3]("store is closed")
        return self._handle

    def _call(self, function: Any, *arguments: object) -> int:
        self._abi_call_count += 1
        return int(function(*arguments))

    def close(self) -> None:
        if self._handle == 0:
            return
        handle = self._handle
        status = self._call(LIBRARY.ze_close, handle)
        self._handle = 0
        raise_for_status(status, handle)

    def __enter__(self) -> Self:
        self._live_handle()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def enable_graph(self) -> NoReturn:
        """Raise UnsupportedBuild: Python wheels do not include graph support."""
        raise UnsupportedBuild("Python wheels do not include graph support")

    def graph_apply(
        self, items: Sequence[object], *,
        cancel_token: CancelToken | None = None, deadline_ns: int | None = None,
    ) -> NoReturn:
        """Refuse a graph batch without inspecting or applying its items."""
        raise UnsupportedBuild("Python wheels do not include graph support")

    def graph_query(
        self, plan: object, parameters: Mapping[str, object] | None = None, *,
        max_rows: int = 1024, cancel_token: CancelToken | None = None,
        deadline_ns: int | None = None,
    ) -> NoReturn:
        """Refuse a structured graph query without inspecting its plan."""
        raise UnsupportedBuild("Python wheels do not include graph support")

    def cypher(
        self, text: str, parameters: Mapping[str, object] | None = None, *,
        max_rows: int = 1024, cancel_token: CancelToken | None = None,
        deadline_ns: int | None = None,
    ) -> NoReturn:
        """Refuse a Cypher statement without compiling or executing it."""
        raise UnsupportedBuild("Python wheels do not include graph support")

    def get_nodes(
        self, ids: Sequence[int | tuple[int, int]], *,
        text: bool = False, vector: bool = False,
    ) -> NoReturn:
        """Refuse graph node reads without accessing the store."""
        raise UnsupportedBuild("Python wheels do not include graph support")

    def get_relationships(self, ids: Sequence[int | tuple[int, int]]) -> NoReturn:
        """Refuse graph relationship reads without accessing the store."""
        raise UnsupportedBuild("Python wheels do not include graph support")

    def graph_resources(self) -> NoReturn:
        """Refuse graph resource reads without accessing the store."""
        raise UnsupportedBuild("Python wheels do not include graph support")

    def state(self) -> StateReport:
        handle = self._live_handle()
        report = s.sized(s.ZeStateReport)
        raise_for_status(self._call(LIBRARY.ze_state, handle, ct.byref(report)), handle)
        return StateReport(StoreState(report.state))

    def stats(self) -> StatsReport:
        handle = self._live_handle()
        report = s.sized(s.ZeStatsReport)
        raise_for_status(self._call(LIBRARY.ze_stats, handle, ct.byref(report)), handle)
        return StatsReport(
            resident_owned_bytes=int(report.resident_owned_bytes),
            mapped_bytes=int(report.mapped_bytes),
            mapped_resident_bytes=int(report.mapped_resident_bytes),
            segment_bytes=int(report.segment_bytes),
            active_segment_bytes=int(report.active_segment_bytes),
            active_row_count=int(report.active_row_count),
            tombstone_count=int(report.tombstone_count),
            tombstone_bytes=int(report.tombstone_bytes),
            wal_bytes=int(report.wal_bytes),
            cache_bytes=int(report.cache_bytes),
            temporary_bytes=int(report.temporary_bytes),
            query_pool_bytes=int(report.query_pool_bytes),
            open_files=int(report.open_files),
            active_queries=int(report.active_queries),
            active_snapshot_leases=int(report.active_snapshot_leases),
            phys_footprint=int(report.phys_footprint) if report.has_phys_footprint else None,
        )

    def schema(self) -> tuple[AttributeDefinition, ...]:
        """Read the native user columns in schema order, excluding the implicit ts column."""
        handle = self._live_handle()
        result = s.sized(s.ZeSchemaColumnResult)
        columns: list[AttributeDefinition] = []
        while True:
            index = len(columns)
            raise_for_status(
                self._call(LIBRARY.ze_schema_column, handle, index, None, 0, ct.byref(result)),
                handle,
            )
            if index == result.column_count:
                return tuple(columns)

            # Even an empty name needs nonzero capacity for the copy call.
            name = (ct.c_uint8 * max(1, result.name_len))()
            raise_for_status(
                self._call(
                    LIBRARY.ze_schema_column, handle, index, name, len(name), ct.byref(result)
                ),
                handle,
            )
            try:
                attribute_type = AttributeType(result.attribute_type)
            except ValueError:
                invalid_argument(f"unknown schema attribute type {result.attribute_type}")
            columns.append(AttributeDefinition(
                attribute_id=result.attribute_id,
                name=ct.string_at(name, result.name_len).decode("utf-8"),
                attribute_type=attribute_type,
                nullable=bool(result.nullable),
            ))

    def ingest(
        self,
        doc_ids: Sequence[int | tuple[int, int]],
        vectors: np.ndarray[Any, Any],
        *,
        revisions: Sequence[int] | None = None,
        timestamps: Sequence[int] | None = None,
        texts: Sequence[str | None] | None = None,
        metadata: Sequence[bytes | None] | None = None,
        batch_size: int = 1_024,
    ) -> MutationReport:
        handle = self._live_handle()
        matrix = _vector(vectors, dimensions=2)
        count, dimension = matrix.shape
        _same_length("doc_ids", doc_ids, count)
        revisions = [1] * count if revisions is None else revisions
        timestamps = list(range(count)) if timestamps is None else timestamps
        texts = [None] * count if texts is None else texts
        metadata = [None] * count if metadata is None else metadata
        for name, values in (
            ("revisions", revisions),
            ("timestamps", timestamps),
            ("texts", texts),
            ("metadata", metadata),
        ):
            _same_length(name, values, count)
        if batch_size <= 0:
            invalid_argument("batch_size must be positive")
        last = MutationReport(0, 0)
        starts = range(0, count, batch_size) if count else (0,)
        for start in starts:
            end = min(start + batch_size, count)
            length = end - start
            documents = (s.ZeIngestDocument * length)()
            owners: list[object] = [matrix, documents]
            for local, index in enumerate(range(start, end)):
                text = texts[index]
                if text is None:
                    text_pointer, text_len = s.UInt8Pointer(), 0
                elif isinstance(text, str):
                    text_pointer, text_len, owner = _text_pointer(text)
                    if owner is not None:
                        owners.append(owner)
                else:
                    invalid_argument("texts values must be str or None")
                payload = metadata[index]
                if payload is None:
                    metadata_pointer, metadata_len = s.UInt8Pointer(), 0
                elif isinstance(payload, bytes):
                    metadata_pointer, owner = _bytes_pointer(payload)
                    metadata_len = len(payload)
                    if owner is not None:
                        owners.append(owner)
                else:
                    invalid_argument("metadata values must be bytes or None")
                row = matrix[index]
                documents[local] = s.ZeIngestDocument(
                    abi_size=ct.sizeof(s.ZeIngestDocument),
                    abi_reserved=0,
                    doc_id=_doc_id(doc_ids[index]),
                    revision=revisions[index],
                    timestamp=timestamps[index],
                    vector=row.ctypes.data_as(s.FloatPointer),
                    vector_len=dimension,
                    metadata=metadata_pointer,
                    metadata_len=metadata_len,
                    text=text_pointer,
                    text_len=text_len,
                )
            request = s.ZeIngestRequest(
                abi_size=ct.sizeof(s.ZeIngestRequest),
                abi_reserved=0,
                documents=ct.cast(documents, ct.POINTER(s.ZeIngestDocument)),
                document_count=length,
                dimension=dimension,
            )
            report = s.sized(s.ZeMutationReport)
            status = self._call(LIBRARY.ze_ingest, handle, ct.byref(request), ct.byref(report))
            del owners
            raise_for_status(status, handle)
            last = MutationReport(int(report.sequence), int(report.generation))
        return last

    def upsert(self, documents: Sequence[StoredDocument]) -> MutationReport:
        """Atomically upsert documents, checking conditions on UpsertDocument entries."""

        handle = self._live_handle()
        documents = tuple(documents)
        conditional, owners = self._marshal_upserts(
            documents,
            record_only=self._record_only,
            vector_dimensions=self._vector_dimensions,
            embedded=self._embedded,
        )
        report = s.sized(s.ZeMutationReport)
        if conditional.condition_count:
            conflict = s.sized(s.ZeRevisionConflict)
            status = self._call(
                LIBRARY.ze_upsert_conditional,
                handle,
                ct.byref(conditional),
                ct.byref(report),
                ct.byref(conflict),
            )
            _raise_for_conditional_status(status, handle, conflict)
        else:
            status = self._call(
                LIBRARY.ze_upsert, handle, ct.byref(conditional.batch), ct.byref(report)
            )
            raise_for_status(status, handle)
        del owners
        return MutationReport(int(report.sequence), int(report.generation))

    @staticmethod
    def _marshal_upserts(
        documents: Sequence[StoredDocument],
        *,
        record_only: bool,
        vector_dimensions: int | None,
        embedded: bool,
        require_conditions: bool = False,
    ) -> tuple[s.ZeConditionalUpsertRequest, list[object]]:
        if record_only or embedded:
            dimension = 0
        elif vector_dimensions is not None:
            dimension = vector_dimensions
        else:
            first_vector = next(
                (
                    document.vector
                    for document in documents
                    if isinstance(document, StoredDocument) and document.vector is not None
                ),
                None,
            )
            if first_vector is None:
                invalid_argument("vector namespace upsert requires document vectors")
            dimension = int(_vector(first_vector, dimensions=1).size)
        records = (s.ZeUpsertDocument * len(documents))()
        owners: list[object] = [records]
        for index, document in enumerate(documents):
            if not isinstance(document, StoredDocument):
                invalid_argument("documents must contain StoredDocument values")
            if (
                not isinstance(document.timestamp, int)
                or isinstance(document.timestamp, bool)
                or not -(1 << 63) <= document.timestamp < 1 << 63
            ):
                invalid_argument("timestamp must be a signed 64-bit integer")
            if record_only or embedded:
                if document.vector is not None:
                    if embedded:
                        invalid_argument("embedded upsert requires text without a vector")
                    invalid_argument("record-only namespace does not accept document vectors")
                vector_pointer, vector_len = s.FloatPointer(), 0
            else:
                if document.vector is None:
                    invalid_argument("vector namespace upsert requires document vectors")
                vector = _vector(document.vector, dimensions=1)
                if vector.size != dimension:
                    invalid_argument(
                        f"document vector length {vector.size} does not match {dimension}"
                    )
                owners.append(vector)
                vector_pointer = vector.ctypes.data_as(s.FloatPointer)
                vector_len = int(vector.size)
            if document.text is None:
                text_pointer, text_len = s.UInt8Pointer(), 0
            elif isinstance(document.text, str):
                text_pointer, text_len, owner = _text_pointer(document.text)
                if owner is not None:
                    owners.append(owner)
            else:
                invalid_argument("document text must be str or None")
            if document.metadata is None:
                metadata_pointer, metadata_len = s.UInt8Pointer(), 0
            elif isinstance(document.metadata, bytes):
                metadata_pointer, owner = _bytes_pointer(document.metadata)
                metadata_len = len(document.metadata)
                if owner is not None:
                    owners.append(owner)
            else:
                invalid_argument("document metadata must be bytes or None")
            attributes = () if document.attributes is None else document.attributes
            attribute_array = (s.ZeAttributeValue * len(attributes))(
                *[_attribute_value(attribute, owners) for attribute in attributes]
            )
            if attributes:
                owners.append(attribute_array)
                attribute_pointer = ct.cast(
                    attribute_array, ct.POINTER(s.ZeAttributeValue)
                )
            else:
                attribute_pointer = ct.POINTER(s.ZeAttributeValue)()
            records[index] = s.ZeUpsertDocument(
                abi_size=ct.sizeof(s.ZeUpsertDocument),
                abi_reserved=0,
                document=s.ZeIngestDocument(
                    abi_size=ct.sizeof(s.ZeIngestDocument),
                    abi_reserved=0,
                    doc_id=_doc_id(document.doc_id),
                    revision=_u64_revision(document.revision, "revision"),
                    timestamp=document.timestamp,
                    vector=vector_pointer,
                    vector_len=vector_len,
                    metadata=metadata_pointer,
                    metadata_len=metadata_len,
                    text=text_pointer,
                    text_len=text_len,
                ),
                attributes=attribute_pointer,
                attribute_count=len(attributes),
            )
        request = s.ZeUpsertRequest(
            abi_size=ct.sizeof(s.ZeUpsertRequest),
            abi_reserved=0,
            documents=ct.cast(records, ct.POINTER(s.ZeUpsertDocument)),
            document_count=len(records),
            dimension=dimension,
        )
        conditional = s.sized(s.ZeConditionalUpsertRequest)
        conditional.batch = request
        if require_conditions or any(isinstance(document, UpsertDocument) for document in documents):
            conditions = (s.ZeRevisionCondition * len(records))(
                *[
                    _revision_condition(document.expected_revision)
                    if isinstance(document, UpsertDocument)
                    else s.ZeRevisionCondition(kind=s.REVISION_CONDITION_NONE)
                    for document in documents
                ]
            )
            owners.append(conditions)
            conditional.conditions = ct.cast(conditions, ct.POINTER(s.ZeRevisionCondition))
            conditional.condition_count = len(conditions)
        return conditional, owners

    def get(
        self,
        ids: Sequence[int | tuple[int, int]],
        *,
        vector: bool = True,
        text: bool = True,
        metadata: bool = True,
        attributes: bool = True,
    ) -> GetResult:
        """Read requested ids in caller order from one pinned generation."""

        handle = self._live_handle()
        doc_ids = (s.ZeDocId * len(ids))(*[_doc_id(value) for value in ids])
        request = s.ZeGetRequest(
            abi_size=ct.sizeof(s.ZeGetRequest),
            abi_reserved=0,
            ids=ct.cast(doc_ids, ct.POINTER(s.ZeDocId)),
            id_count=len(doc_ids),
            include_vector=int(vector),
            include_text=int(text),
            include_metadata=int(metadata),
            include_attributes=int(attributes),
        )
        result = s.sized(s.ZeGetResult)
        primary: BaseException | None = None
        try:
            status = self._call(LIBRARY.ze_get, handle, ct.byref(request), ct.byref(result))
            raise_for_status(status, handle)
            returned = tuple(
                _stored_document(
                    result.documents[index],
                    self._attribute_types,
                    include_vector=vector and not self._record_only,
                    include_text=text,
                    include_metadata=metadata,
                    include_attributes=attributes,
                )
                for index in range(result.document_count)
            )
            return GetResult(returned, int(result.missing_count), int(result.generation))
        except BaseException as error:
            primary = error
            raise
        finally:
            del doc_ids
            free_status = self._call(LIBRARY.ze_get_result_free, ct.byref(result))
            if free_status != 0:
                if primary is None:
                    raise_for_status(free_status, handle)
                assert primary is not None
                primary.add_note(f"ze_get_result_free: {last_error_message(handle)}")

    def scan(
        self,
        *,
        cursor: ScanCursor | None = None,
        limit: int = 1_024,
        order: str | AttributeScanOrder = "storage",
        vector: bool = True,
        text: bool = True,
        metadata: bool = True,
        attributes: bool = True,
        timestamp_range: tuple[int, int] | None = None,
        filter: dict[str, Any] | None = None,
        cancel_token: CancelToken | None = None,
        deadline_ns: int = 0,
    ) -> ScanPage:
        """Enumerate one ordered and optionally filtered document page."""

        handle = self._live_handle()
        orders = {
            "storage": 0,
            "timestamp_ascending": 1,
            "timestamp-ascending": 1,
            "timestamp_descending": 2,
            "timestamp-descending": 2,
        }
        order_attribute_id = 0
        if isinstance(order, AttributeScanOrder):
            order_attribute_id = order.attribute_id
            if (
                not isinstance(order_attribute_id, int)
                or isinstance(order_attribute_id, bool)
                or not 1 <= order_attribute_id < 1 << 32
            ):
                invalid_argument("scan order attribute_id must be an integer in 1..=4294967295")
            if order.direction == "ascending":
                order_value = 3
            elif order.direction == "descending":
                order_value = 4
            else:
                invalid_argument("attribute scan direction must be 'ascending' or 'descending'")
        elif isinstance(order, str) and order.lower() in orders:
            order_value = orders[order.lower()]
        else:
            invalid_argument(f"scan order {order!r} is not supported")
        if cursor is None:
            cursor_generation = cursor_next_row = cursor_phase = 0
            cursor_order = cursor_order_attribute_id = 0
            cursor_segment_id = (ct.c_uint8 * 16)()
        elif isinstance(cursor, ScanCursor):
            cursor_generation = cursor._generation
            cursor_next_row = cursor._next_row
            cursor_phase = cursor._phase
            cursor_order = cursor._order
            cursor_order_attribute_id = cursor._order_attribute_id
            cursor_segment_id = (ct.c_uint8 * 16).from_buffer_copy(cursor._segment_id)
        else:
            invalid_argument("cursor must be a ScanCursor or None")
        if timestamp_range is None:
            has_timestamp_range, start_ts, end_ts = 0, 0, 0
        elif isinstance(timestamp_range, tuple) and len(timestamp_range) == 2:
            has_timestamp_range = 1
            start_ts, end_ts = timestamp_range
        else:
            invalid_argument("timestamp_range must be a (start, end) tuple or None")
        if filter is None:
            filter_pointer = ct.POINTER(s.ZeFilter)()
            filter_owners: list[object] = []
        else:
            flat_filter, filter_owners = _flatten_filter(
                filter, self._attribute_ids, self._attribute_types
            )
            filter_pointer = ct.pointer(flat_filter)
            filter_owners.extend((flat_filter, filter_pointer))
        request = s.ZeScanRequest(
            abi_size=ct.sizeof(s.ZeScanRequest),
            abi_reserved=0,
            cursor_generation=cursor_generation,
            cursor_segment_id=cursor_segment_id,
            cursor_next_row=cursor_next_row,
            cursor_phase=cursor_phase,
            limit=limit,
            order=order_value,
            include_vector=int(vector),
            include_text=int(text),
            include_metadata=int(metadata),
            include_attributes=int(attributes),
            has_timestamp_range=has_timestamp_range,
            start_ts=start_ts,
            end_ts=end_ts,
            filter=filter_pointer,
            cancel_token=0 if cancel_token is None else cancel_token.value,
            deadline_ns=deadline_ns,
        )
        ordered = s.ZeScanOrderedRequest(
            abi_size=ct.sizeof(s.ZeScanOrderedRequest),
            scan=request,
            order_attribute_id=order_attribute_id,
            cursor_order=cursor_order,
            cursor_order_attribute_id=cursor_order_attribute_id,
        )
        result = s.sized(s.ZeScanResult)
        status = self._call(LIBRARY.ze_scan_ordered, handle, ct.byref(ordered), ct.byref(result))
        del filter_owners
        raise_for_status(status, handle)
        primary: BaseException | None = None
        try:
            returned: list[StoredDocument] = []
            for index in range(result.document_count):
                document = _stored_document(
                    result.documents[index],
                    self._attribute_types,
                    include_vector=vector and not self._record_only,
                    include_text=text,
                    include_metadata=metadata,
                    include_attributes=attributes,
                )
                if document is None:
                    raise ERROR_TYPES[22]("scan returned a missing document")
                returned.append(document)
            next_cursor = (
                ScanCursor(
                    int(result.generation),
                    bytes(result.next_segment_id),
                    int(result.next_row),
                    int(result.next_phase),
                    order_value,
                    order_attribute_id,
                )
                if result.has_more
                else None
            )
            return ScanPage(tuple(returned), next_cursor, int(result.generation))
        except BaseException as error:
            primary = error
            raise
        finally:
            free_status = self._call(LIBRARY.ze_scan_result_free, ct.byref(result))
            if free_status != 0:
                if primary is None:
                    raise_for_status(free_status, handle)
                assert primary is not None
                primary.add_note(f"ze_scan_result_free: {last_error_message(handle)}")

    def iter_documents(
        self,
        *,
        order: str | AttributeScanOrder = "storage",
        vector: bool = True,
        text: bool = True,
        metadata: bool = True,
        attributes: bool = True,
        filter: dict[str, Any] | None = None,
    ) -> Iterator[StoredDocument]:
        """Follow scan cursors until every matching live document is yielded."""

        cursor = None
        while True:
            page = self.scan(
                cursor=cursor,
                order=order,
                vector=vector,
                text=text,
                metadata=metadata,
                attributes=attributes,
                filter=filter,
            )
            yield from page.documents
            cursor = page.cursor
            if cursor is None:
                return

    @overload
    def count(
        self,
        *,
        filter: dict[str, Any] | None = None,
        timestamp_range: tuple[int, int] | None = None,
        group_by: None = None,
    ) -> CountResult: ...

    @overload
    def count(
        self,
        *,
        filter: dict[str, Any] | None = None,
        timestamp_range: tuple[int, int] | None = None,
        group_by: CountGroupBy,
    ) -> GroupedCountResult: ...

    def count(
        self,
        *,
        filter: dict[str, Any] | None = None,
        timestamp_range: tuple[int, int] | None = None,
        group_by: CountGroupBy | None = None,
    ) -> CountResult:
        """Count matching live documents, optionally grouping by one attribute."""

        handle = self._live_handle()
        if group_by is not None:
            if not isinstance(group_by, CountGroupBy):
                invalid_argument("group_by must be a CountGroupBy or None")
            if (
                not isinstance(group_by.attribute_id, int)
                or isinstance(group_by.attribute_id, bool)
                or not 0 <= group_by.attribute_id < 1 << 32
            ):
                invalid_argument("group_by.attribute_id must be an unsigned 32-bit integer")
            if (
                not isinstance(group_by.limit, int)
                or isinstance(group_by.limit, bool)
                or not 1 <= group_by.limit <= 1 << 16
            ):
                invalid_argument("group_by.limit must be an integer in 1..=65536")
        if timestamp_range is None:
            has_timestamp_range, start_ts, end_ts = 0, 0, 0
        elif isinstance(timestamp_range, tuple) and len(timestamp_range) == 2:
            has_timestamp_range = 1
            start_ts, end_ts = timestamp_range
        else:
            invalid_argument("timestamp_range must be a (start, end) tuple or None")
        if filter is None:
            filter_pointer = ct.POINTER(s.ZeFilter)()
            filter_owners: list[object] = []
        else:
            flat_filter, filter_owners = _flatten_filter(
                filter, self._attribute_ids, self._attribute_types
            )
            filter_pointer = ct.pointer(flat_filter)
            filter_owners.extend((flat_filter, filter_pointer))
        request = s.ZeCountRequest(
            abi_size=ct.sizeof(s.ZeCountRequest),
            abi_reserved=0,
            filter=filter_pointer,
            has_timestamp_range=has_timestamp_range,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if group_by is not None:
            grouped_request = s.ZeCountGroupedRequest(
                abi_size=ct.sizeof(s.ZeCountGroupedRequest),
                count=request,
                group_attribute_id=group_by.attribute_id,
                group_limit=group_by.limit,
            )
            grouped = s.sized(s.ZeCountGroupedResult)
            status = self._call(
                LIBRARY.ze_count_grouped, handle, ct.byref(grouped_request), ct.byref(grouped)
            )
            del filter_owners
            raise_for_status(status, handle)
            primary: BaseException | None = None
            try:
                groups: list[CountGroup] = []
                for index in range(grouped.group_count):
                    group = grouped.groups[index]
                    raw = group.value
                    value: int | str
                    if raw.value_type == 1:
                        value = int(raw.u64_value)
                    elif raw.value_type == 2:
                        value = int(raw.i64_value)
                    elif raw.value_type == 5:
                        value = ct.string_at(raw.string_value, raw.string_len).decode("utf-8")
                    else:
                        invalid_argument("returned count group value type is invalid")
                    groups.append(CountGroup(value, int(group.count)))
                return GroupedCountResult(
                    int(grouped.count), int(grouped.generation), tuple(groups), int(grouped.missing_count)
                )
            except BaseException as error:
                primary = error
                raise
            finally:
                free_status = self._call(LIBRARY.ze_count_grouped_result_free, ct.byref(grouped))
                if free_status != 0:
                    if primary is None:
                        raise_for_status(free_status, handle)
                    else:
                        primary.add_note(
                            f"ze_count_grouped_result_free: {last_error_message(handle)}"
                        )
        result = s.sized(s.ZeCountResult)
        status = self._call(LIBRARY.ze_count, handle, ct.byref(request), ct.byref(result))
        del filter_owners
        raise_for_status(status, handle)
        return CountResult(int(result.count), int(result.generation))

    def _require_text_store(self) -> None:
        _require_text()
        if not self._text:
            invalid_argument(
                "text operation requires a store opened with open_text or open_model"
            )

    def ingest_text(
        self,
        ids: Sequence[int | tuple[int, int]],
        texts: Sequence[str],
        *,
        revisions: Sequence[int] | None = None,
        embed_batch_size: int = 32,
        seal_every: int = 512,
        channel_capacity: int = 2,
        chunk_max: int = 0,
        chunk_overlap: int = 0,
        maintenance_wall_time_ns: int = 0,
        maintenance_bytes: int = 0,
        cancel_token: CancelToken | None = None,
        deadline_ns: int = 0,
    ) -> TextIngestReport:
        """Ingest text; chunk_max > 0 enables overlapping token windows.

        Cancellation or expiry may leave acknowledged partial writes.
        """
        self._require_text_store()
        _same_length("texts", texts, len(ids))
        revisions = [1] * len(ids) if revisions is None else revisions
        _same_length("revisions", revisions, len(ids))
        documents = (s.ZeTextDocument * len(ids))()
        owners: list[object] = [documents]
        for index, value in enumerate(texts):
            pointer, length, owner = _text_pointer(value)
            if owner is not None:
                owners.append(owner)
            documents[index] = s.ZeTextDocument(
                abi_size=ct.sizeof(s.ZeTextDocument),
                abi_reserved=0,
                doc_id=_doc_id(ids[index]),
                revision=revisions[index],
                text=pointer,
                text_len=length,
            )
        request = s.ZeTextIngestRequest(
            abi_size=ct.sizeof(s.ZeTextIngestRequest),
            abi_reserved=0,
            documents=ct.cast(documents, ct.POINTER(s.ZeTextDocument)),
            document_count=len(documents),
            embed_batch_size=embed_batch_size,
            seal_every=seal_every,
            channel_capacity=channel_capacity,
        )
        controlled = s.ZeTextIngestRequestV2(
            abi_size=ct.sizeof(s.ZeTextIngestRequestV2),
            batch=request,
            chunk_policy=int(chunk_max != 0),
            chunk_max=chunk_max,
            chunk_overlap=chunk_overlap,
            maintenance_wall_time_ns=maintenance_wall_time_ns,
            maintenance_bytes=maintenance_bytes,
            cancel_token=0 if cancel_token is None else cancel_token.value,
            deadline_ns=deadline_ns,
        )
        report = s.sized(s.ZeTextIngestReport)
        handle = self._live_handle()
        status = self._call(
            LIBRARY.ze_text_ingest_v2, handle, ct.byref(controlled), ct.byref(report)
        )
        del owners
        raise_for_status(status, handle)
        return TextIngestReport(
            generation=int(report.generation),
            documents=int(report.documents),
            chunks=int(report.chunks),
            tokens=int(report.tokens),
            embed_batches=int(report.embed_batches),
            seals=int(report.seals),
            max_in_flight_batches=int(report.max_in_flight_batches),
            all_threads_joined=bool(report.all_threads_joined),
            elapsed_ns=int(report.elapsed_ns),
        )

    def query_text(
        self,
        text: str,
        *,
        k: int = 10,
        legs: TextLegs | str = TextLegs.HYBRID,
        tier: Tier | str | None = None,
        graph_profile: GraphProfile | str = GraphProfile.SIFT_CLASS,
        graph_ef: int = 0,
        graph_seed: int = 0,
        scan_rescore: bool = False,
        oversample: int = 0,
        max_candidates_per_segment: int = 0,
        cancel_token: CancelToken | None = None,
        deadline_ns: int = 0,
    ) -> TextQueryResult:
        """Query text with core diagnostics and optional relative deadline."""
        self._require_text_store()
        pointer, length, owner = _text_pointer(text)
        request = s.ZeTextQueryRequest(
            abi_size=ct.sizeof(s.ZeTextQueryRequest),
            abi_reserved=0,
            text=pointer,
            text_len=length,
            k=k,
            legs=int(_enum_value(legs, TextLegs, "legs")),
            reserved=0,
        )
        controlled = s.ZeTextQueryRequestV2(
            abi_size=ct.sizeof(s.ZeTextQueryRequestV2),
            query=request,
            has_tier=int(tier is not None),
            tier=0 if tier is None else int(_enum_value(tier, Tier, "tier")),
            graph_profile=int(
                _enum_value(graph_profile, GraphProfile, "graph_profile")
            ),
            graph_ef=graph_ef,
            graph_seed=graph_seed,
            scan_rescore=int(scan_rescore),
            oversample=oversample,
            max_candidates_per_segment=max_candidates_per_segment,
            cancel_token=0 if cancel_token is None else cancel_token.value,
            deadline_ns=deadline_ns,
        )
        outcome = s.sized(s.ZeTextQueryOutcome)
        handle = self._live_handle()
        primary: BaseException | None = None
        try:
            status = self._call(
                LIBRARY.ze_text_query_v2,
                handle,
                ct.byref(controlled),
                ct.byref(outcome),
            )
            raise_for_status(status, handle)
            result = outcome.result
            hits = tuple(
                TextHit(
                    doc_id=_python_doc_id(hit.doc_id),
                    revision=int(hit.revision),
                    chunk=int(hit.chunk),
                    text=ct.string_at(hit.text, hit.text_len).decode("utf-8"),
                    score=float(hit.score),
                    vector_squared_l2=(
                        float(hit.vector_squared_l2) if hit.has_vector_score else None
                    ),
                    lexical_bm25=(
                        float(hit.lexical_bm25) if hit.has_lexical_score else None
                    ),
                )
                for hit in (result.hits[index] for index in range(result.hit_count))
            )
            return TextQueryResult(
                hits=hits,
                embedding_epoch=int(result.embedding_epoch),
                tokenizer_epoch=int(result.tokenizer_epoch),
                diagnostics=_query_diagnostics(outcome.diagnostics)
                if outcome.has_diagnostics
                else None,
                runtime_name=(
                    ct.string_at(outcome.runtime_name, outcome.runtime_name_len).decode(
                        "utf-8"
                    )
                    if outcome.has_backend
                    else None
                ),
                backend_gpu=bool(outcome.backend_gpu) if outcome.has_backend else None,
                requested_compute_units=(
                    ComputeUnits(outcome.requested_compute_units)
                    if outcome.has_backend
                    else None
                ),
                observed_compute_units=(
                    (
                        int(outcome.observed_cpu_ops),
                        int(outcome.observed_gpu_ops),
                        int(outcome.observed_neural_engine_ops),
                    )
                    if outcome.has_observed_compute_units
                    else None
                ),
                sequence_length=int(outcome.sequence_length)
                if outcome.has_sequence_length
                else None,
                query_tokens=int(outcome.query_tokens),
                embedding_calls=int(outcome.embedding_calls),
                timings_ns=tuple(outcome.timings_ns) if outcome.has_timings else None,
            )
        except BaseException as error:
            primary = error
            raise
        finally:
            del owner
            free_status = self._call(
                LIBRARY.ze_text_query_outcome_free, ct.byref(outcome)
            )
            if free_status != 0:
                if primary is None:
                    raise_for_status(free_status, handle)
                assert primary is not None
                primary.add_note(
                    f"ze_text_query_outcome_free: {last_error_message(handle)}"
                )

    def delete_text(self, doc_ids: Sequence[int | tuple[int, int]]) -> MutationReport:
        """Delete every chunk of the supplied caller document ids."""
        self._require_text_store()
        return self.delete(doc_ids)

    def observed_compute_units(self) -> ObservedComputeUnits | None:
        """Load CoreML planned placement on demand; None without a CoreML query model."""
        self._require_text_store()
        handle = self._live_handle()
        report = s.sized(s.ZeTextObservedComputeUnits)
        raise_for_status(
            self._call(LIBRARY.ze_text_observed_compute_units, handle, ct.byref(report)),
            handle,
        )
        if not report.has_observed_compute_units:
            return None
        return ObservedComputeUnits(
            cpu_ops=int(report.cpu_ops),
            gpu_ops=int(report.gpu_ops),
            neural_engine_ops=int(report.neural_engine_ops),
            other_ops=int(report.other_ops),
        )

    def health(self) -> TextHealthReport:
        self._require_text_store()
        handle = self._live_handle()
        report = s.sized(s.ZeTextHealthReport)
        raise_for_status(
            self._call(LIBRARY.ze_text_health, handle, ct.byref(report)), handle
        )
        return TextHealthReport(
            generation=int(report.generation),
            status=TextHealthStatus(report.status),
            pending_docs=int(report.pending_docs),
            graph_coverage=float(report.graph_coverage),
            segment_count=int(report.segment_count),
            unresolved_fault_count=int(report.unresolved_fault_count),
            retained_through=int(report.retained_through)
            if report.has_retained_through
            else None,
        )

    def delete(self, doc_ids: Sequence[int | tuple[int, int] | DeleteTarget]) -> MutationReport:
        """Atomically delete ids, checking conditions on DeleteTarget entries."""

        if self._text:
            self._require_text_store()
        handle = self._live_handle()
        # Marshal ids and their conditions from the same batch entries.
        doc_ids = tuple(doc_ids)
        ids = (s.ZeDocId * len(doc_ids))(
            *[_doc_id(value.doc_id if isinstance(value, DeleteTarget) else value) for value in doc_ids]
        )
        request = s.ZeDeleteRequest(
            abi_size=ct.sizeof(s.ZeDeleteRequest),
            abi_reserved=0,
            doc_ids=ct.cast(ids, ct.POINTER(s.ZeDocId)),
            doc_id_count=len(ids),
        )
        report = s.sized(s.ZeMutationReport)
        if any(isinstance(value, DeleteTarget) for value in doc_ids):
            conditions = (s.ZeRevisionCondition * len(ids))(
                *[
                    _revision_condition(value.expected_revision)
                    if isinstance(value, DeleteTarget)
                    else s.ZeRevisionCondition(kind=s.REVISION_CONDITION_NONE)
                    for value in doc_ids
                ]
            )
            conditional = s.sized(s.ZeConditionalDeleteRequest)
            conditional.batch = request
            conditional.conditions = ct.cast(conditions, ct.POINTER(s.ZeRevisionCondition))
            conditional.condition_count = len(conditions)
            conflict = s.sized(s.ZeRevisionConflict)
            status = self._call(
                LIBRARY.ze_delete_conditional,
                handle,
                ct.byref(conditional),
                ct.byref(report),
                ct.byref(conflict),
            )
            _raise_for_conditional_status(status, handle, conflict)
            return MutationReport(int(report.sequence), int(report.generation))
        raise_for_status(
            self._call(
                LIBRARY.ze_text_delete if self._text else LIBRARY.ze_delete,
                handle,
                ct.byref(request),
                ct.byref(report),
            ),
            handle,
        )
        return MutationReport(int(report.sequence), int(report.generation))

    def delete_where(self, filter: dict[str, Any]) -> DeleteWhereReport:
        """Atomically delete and physically purge live documents matching a filter."""

        handle = self._live_handle()
        flat_filter, owners = _flatten_filter(
            filter, self._attribute_ids, self._attribute_types, strict=True
        )
        request = s.sized(s.ZeDeleteWhereRequest)
        request.filter = ct.pointer(flat_filter)
        report = s.sized(s.ZeDeleteWhereReport)
        status = self._call(LIBRARY.ze_delete_where, handle, ct.byref(request), ct.byref(report))
        del owners
        raise_for_status(status, handle)
        return DeleteWhereReport(deleted=int(report.deleted_count), generation=int(report.generation))

    def search(
        self,
        vector: np.ndarray[Any, Any],
        *,
        k: int = 10,
        thread_budget: int = 0,
        tier: Tier | None = None,
        graph_profile: GraphProfile = GraphProfile.SIFT_CLASS,
        graph_ef: int = 0,
        graph_seed: int = 0,
        cancel_token: CancelToken | None = None,
        deadline_ns: int = 0,
        filter: dict[str, Any] | None = None,
    ) -> SearchResult:
        handle = self._live_handle()
        probe = _vector(vector, dimensions=1)
        request = s.ZeSearchRequest(
            abi_size=ct.sizeof(s.ZeSearchRequest),
            abi_reserved=0,
            vector=probe.ctypes.data_as(s.FloatPointer),
            vector_len=probe.size,
            dimension=probe.size,
            k=k,
            thread_budget=thread_budget,
            has_tier=0 if tier is None else 1,
            tier=0 if tier is None else int(_enum_value(tier, Tier, "tier")),
            graph_profile=int(_enum_value(graph_profile, GraphProfile, "graph_profile")),
            reserved=0,
            graph_ef=graph_ef,
            graph_seed=graph_seed,
            cancel_token=0 if cancel_token is None else cancel_token.value,
            deadline_ns=deadline_ns,
        )
        result = s.sized(s.ZeSearchResult)
        if filter is None:
            function = LIBRARY.ze_search
            call_request: ct.Structure = request
            filter_owners: list[object] = []
        else:
            flat_filter, filter_owners = _flatten_filter(
                filter, self._attribute_ids, self._attribute_types
            )
            filter_pointer = ct.pointer(flat_filter)
            filter_owners.extend((flat_filter, filter_pointer))
            call_request = s.ZeSearchFilteredRequest(
                abi_size=ct.sizeof(s.ZeSearchFilteredRequest),
                abi_reserved=0,
                search=request,
                filter=filter_pointer,
            )
            function = LIBRARY.ze_search_filtered
        primary: BaseException | None = None
        try:
            status = self._call(function, handle, ct.byref(call_request), ct.byref(result))
            raise_for_status(status, handle)
            hits = tuple(
                SearchHit(
                    source_kind=int(hit.source_kind),
                    segment_id=bytes(hit.segment_id),
                    local_row=int(hit.local_row),
                    doc_id=_python_doc_id(hit.doc_id) if hit.has_document else None,
                    revision=int(hit.revision) if hit.has_document else None,
                    score=float(hit.score),
                )
                for hit in (result.hits[index] for index in range(result.hit_count))
            )
            return SearchResult(
                hits=hits,
                generation=int(result.generation),
                dims_touched=int(result.dims_touched),
                bytes_read=int(result.bytes_read),
                threads_used=int(result.threads_used),
                graph_segments_traversed=int(result.graph_segments_traversed),
                graph_validations=int(result.graph_validations),
                graph_entry_seed_discoveries=int(result.graph_entry_seed_discoveries),
                graph_visited_epoch_clears=int(result.graph_visited_epoch_clears),
                graph_candidates_scored=int(result.graph_candidates_scored),
                graph_candidates_rescored=int(result.graph_candidates_rescored),
                graph_segments_pruned_by_bound=int(result.graph_segments_pruned_by_bound),
            )
        except BaseException as error:
            primary = error
            raise
        finally:
            del filter_owners
            free_status = self._call(LIBRARY.ze_search_result_free, ct.byref(result))
            if free_status != 0:
                if primary is None:
                    raise_for_status(free_status, handle)
                assert primary is not None
                primary.add_note(f"ze_search_result_free: {last_error_message(handle)}")

    def _embedded_constraints(
        self,
        filter: dict[str, Any] | None,
        timestamp_range: tuple[int, int] | None,
    ) -> tuple[Any, list[object]]:
        if filter is None and timestamp_range is None:
            return None, []
        constraints = s.sized(s.ZeQueryFilter)
        owners: list[object] = [constraints]
        if filter is not None:
            flat_filter, filter_owners = _flatten_filter(
                filter, self._attribute_ids, self._attribute_types
            )
            constraints.filter = ct.pointer(flat_filter)
            owners.extend((flat_filter, *filter_owners))
        if timestamp_range is not None:
            if not isinstance(timestamp_range, tuple) or len(timestamp_range) != 2:
                invalid_argument("timestamp_range must be a (start, end) tuple")
            start, end = timestamp_range
            constraints.has_timestamp_range = 1
            constraints.start_ts = _bounded_int(start, -(1 << 63), (1 << 63) - 1, "start_ts")
            constraints.end_ts = _bounded_int(end, -(1 << 63), (1 << 63) - 1, "end_ts")
        return ct.pointer(constraints), owners

    def query_embedded(
        self,
        text: str,
        *,
        leg: TextLegs | str = TextLegs.DENSE,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        timestamp_range: tuple[int, int] | None = None,
        eligible_ids: Sequence[int | tuple[int, int]] | None = None,
        cancel_token: CancelToken | None = None,
        deadline_ns: int = 0,
    ) -> tuple[EmbeddedHit, ...]:
        """Rank chunks in a model-bound namespace using text."""

        _require_text()
        handle = self._live_handle()
        _bounded_int(k, 1, 4096, "k")
        _bounded_int(deadline_ns, 0, (1 << 64) - 1, "deadline_ns")
        if not isinstance(text, str):
            invalid_argument("text must be a string")
        if cancel_token is not None and not isinstance(cancel_token, CancelToken):
            invalid_argument("cancel_token must be a CancelToken")
        query_text, text_len, text_owner = _text_pointer(text)
        constraints, filter_owners = self._embedded_constraints(filter, timestamp_range)
        ids = None if eligible_ids is None else (
            s.ZeDocId * len(eligible_ids)
        )(*[_doc_id(value) for value in eligible_ids])
        request = s.ZeEmbeddedQueryRequest(
            abi_size=ct.sizeof(s.ZeEmbeddedQueryRequest),
            text=query_text,
            text_len=text_len,
            leg=int(_enum_value(leg, TextLegs, "leg")),
            has_eligible=int(ids is not None),
            eligible_ids=ids,
            eligible_count=0 if ids is None else len(ids),
            constraints=constraints,
            k=k,
            cancel_token=0 if cancel_token is None else cancel_token.value,
            deadline_ns=deadline_ns,
        )
        ranks = (s.ZeChunkRank * k)()
        result = s.sized(s.ZeGetResult)
        status = self._call(
            LIBRARY.ze_query_embedded, handle, ct.byref(request), ranks, k, ct.byref(result)
        )
        del text_owner, filter_owners
        raise_for_status(status, handle)
        if result.document_count == 0 and not result.documents:
            return ()
        primary: BaseException | None = None
        try:
            hits = []
            for index in range(result.document_count):
                document = result.documents[index]
                rank = ranks[index]
                hits.append(EmbeddedHit(
                    doc_id=_python_doc_id(document.doc_id),
                    revision=int(document.revision),
                    timestamp=int(document.timestamp),
                    attributes=tuple(
                        _returned_attribute(document.attributes[i], self._attribute_types)
                        for i in range(document.attribute_count)
                    ),
                    chunk=int(rank.chunk),
                    byte_start=int(rank.byte_start),
                    byte_end=int(rank.byte_end),
                    text=ct.string_at(document.text, document.text_len).decode("utf-8"),
                    score=float(rank.score),
                ))
            return tuple(hits)
        except BaseException as error:
            primary = error
            raise
        finally:
            free_status = self._call(LIBRARY.ze_get_result_free, ct.byref(result))
            if free_status != 0:
                if primary is None:
                    raise_for_status(free_status)
                assert primary is not None
                primary.add_note(f"ze_get_result_free: {last_error_message()}")

    def nearest_chunks(
        self,
        source: int | tuple[int, int] | NeighbourSource,
        *,
        k: int | None = None,
        threshold: float | None = None,
        include_source: bool = False,
        filter: dict[str, Any] | None = None,
        timestamp_range: tuple[int, int] | None = None,
        eligible_ids: Sequence[int | tuple[int, int]] | None = None,
        cancel_token: CancelToken | None = None,
        deadline_ns: int = 0,
    ) -> tuple[ChunkNeighbour, ...]:
        """Rank candidate chunks against one stored document or chunk."""

        return self.nearest_chunks_batch(
            [source],
            k=k,
            threshold=threshold,
            include_source=include_source,
            filter=filter,
            timestamp_range=timestamp_range,
            eligible_ids=eligible_ids,
            cancel_token=cancel_token,
            deadline_ns=deadline_ns,
        )[0]

    def nearest_chunks_batch(
        self,
        sources: Sequence[int | tuple[int, int] | NeighbourSource],
        *,
        k: int | None = None,
        threshold: float | None = None,
        include_source: bool = False,
        filter: dict[str, Any] | None = None,
        timestamp_range: tuple[int, int] | None = None,
        eligible_ids: Sequence[int | tuple[int, int]] | None = None,
        cancel_token: CancelToken | None = None,
        deadline_ns: int = 0,
    ) -> tuple[tuple[ChunkNeighbour, ...], ...]:
        """Rank candidate chunks for each source, preserving input order."""

        _require_text()
        handle = self._live_handle()
        if k is not None:
            _bounded_int(k, 1, 4096, "k")
        _bounded_int(deadline_ns, 0, (1 << 64) - 1, "deadline_ns")
        if not isinstance(include_source, bool):
            invalid_argument("include_source must be a bool")
        if cancel_token is not None and not isinstance(cancel_token, CancelToken):
            invalid_argument("cancel_token must be a CancelToken")
        bound = 0.0
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                invalid_argument("threshold must be a number")
            try:
                bound = float(threshold)
            except OverflowError:
                invalid_argument("threshold must fit a finite double")
        records = (s.ZeNeighbourSource * len(sources))()
        for index, source in enumerate(sources):
            if isinstance(source, NeighbourSource):
                doc_id, chunk = source.doc_id, source.chunk
            else:
                doc_id, chunk = source, None
            records[index] = s.ZeNeighbourSource(
                doc_id=_doc_id(doc_id),
                has_chunk=int(chunk is not None),
                chunk=0 if chunk is None else _bounded_int(chunk, 0, (1 << 32) - 1, "chunk"),
            )
        constraints, filter_owners = self._embedded_constraints(filter, timestamp_range)
        ids = None if eligible_ids is None else (
            s.ZeDocId * len(eligible_ids)
        )(*[_doc_id(value) for value in eligible_ids])
        request = s.ZeNeighbourRequest(
            abi_size=ct.sizeof(s.ZeNeighbourRequest),
            sources=records,
            source_count=len(records),
            constraints=constraints,
            eligible_ids=ids,
            eligible_count=0 if ids is None else len(ids),
            k=0 if k is None else k,
            threshold=bound,
            include_source=int(include_source),
            has_threshold=int(threshold is not None),
            has_eligible=int(ids is not None),
            cancel_token=0 if cancel_token is None else cancel_token.value,
            deadline_ns=deadline_ns,
        )
        result = s.sized(s.ZeNeighbourResult)
        status = self._call(LIBRARY.ze_nearest_chunks, handle, ct.byref(request), ct.byref(result))
        del filter_owners
        raise_for_status(status, handle)
        primary: BaseException | None = None
        try:
            groups: list[list[ChunkNeighbour]] = [[] for _ in records]
            for index in range(result.hit_count):
                hit = result.hits[index]
                groups[hit.source_index].append(
                    ChunkNeighbour(_python_doc_id(hit.doc_id), int(hit.chunk), float(hit.score))
                )
            return tuple(tuple(group) for group in groups)
        except BaseException as error:
            primary = error
            raise
        finally:
            free_status = self._call(LIBRARY.ze_neighbour_result_free, ct.byref(result))
            if free_status != 0:
                if primary is None:
                    raise_for_status(free_status)
                assert primary is not None
                primary.add_note(f"ze_neighbour_result_free: {last_error_message()}")

    def query(
        self,
        *,
        vector: np.ndarray[Any, Any] | None = None,
        text: str | None = None,
        eligible_ids: Sequence[int | tuple[int, int]] | None = None,
        filter: dict[str, Any] | None = None,
        snippet_bytes: int | None = None,
        k: int = 10,
        thread_budget: int = 0,
        tier: Tier | None = None,
        graph_profile: GraphProfile = GraphProfile.SIFT_CLASS,
        graph_ef: int = 0,
        graph_seed: int = 0,
        alpha: float | None = None,
        rules_enabled: bool = False,
        max_rounds: int | None = None,
        quoted_phrase: bool = False,
        identifier_token: bool = False,
        rarest_exact_document_frequency: int | None = None,
        last_as_prefix: bool = False,
        cancel_token: CancelToken | None = None,
        deadline_ns: int = 0,
    ) -> QueryResult:
        """Run a structured query with optional attribute filters and stored-text excerpts."""
        handle = self._live_handle()
        snippets = None
        if snippet_bytes is not None:
            if (
                isinstance(snippet_bytes, bool)
                or not isinstance(snippet_bytes, int)
                or not 0 < snippet_bytes < 1 << (8 * ct.sizeof(ct.c_size_t))
            ):
                invalid_argument("snippet_bytes must be a positive integer that fits size_t")
            snippets = s.sized(s.ZeQuerySnippets)
        snippets_pointer = None if snippets is None else ct.byref(snippets)
        if vector is None:
            vector_pointer, vector_len = s.FloatPointer(), 0
            probe = None
        else:
            probe = _vector(vector, dimensions=1)
            vector_pointer = probe.ctypes.data_as(s.FloatPointer)
            vector_len = probe.size
        if text is None:
            text_pointer, text_len, text_owner = s.UInt8Pointer(), 0, None
        elif isinstance(text, str):
            text_pointer, text_len, text_owner = _text_pointer(text)
        else:
            invalid_argument("text must be str or None")
        request = s.ZeQueryRequest(
            abi_size=ct.sizeof(s.ZeQueryRequest),
            abi_reserved=0,
            vector=vector_pointer,
            vector_len=vector_len,
            dimension=vector_len,
            text=text_pointer,
            text_len=text_len,
            k=k,
            thread_budget=thread_budget,
            has_tier=0 if tier is None else 1,
            tier=0 if tier is None else int(_enum_value(tier, Tier, "tier")),
            graph_profile=int(_enum_value(graph_profile, GraphProfile, "graph_profile")),
            lexical_flags=int(last_as_prefix),
            graph_ef=graph_ef,
            graph_seed=graph_seed,
            has_alpha=0 if alpha is None else 1,
            rules_enabled=int(rules_enabled),
            alpha=0.0 if alpha is None else alpha,
            has_max_rounds=0 if max_rounds is None else 1,
            quoted_phrase=int(quoted_phrase),
            max_rounds=0 if max_rounds is None else max_rounds,
            identifier_token=int(identifier_token),
            has_rarest_exact_document_frequency=(
                0 if rarest_exact_document_frequency is None else 1
            ),
            rarest_exact_document_frequency=(
                0
                if rarest_exact_document_frequency is None
                else rarest_exact_document_frequency
            ),
            cancel_token=0 if cancel_token is None else cancel_token.value,
            deadline_ns=deadline_ns,
        )
        # Keep the caller-owned id array alive across the FFI call (which releases the GIL).
        ids_owner = (
            None if eligible_ids is None else
            (s.ZeDocId * len(eligible_ids))(*[_doc_id(value) for value in eligible_ids])
        )
        request_v2 = s.ZeQueryRequestV2(
            abi_size=ct.sizeof(s.ZeQueryRequestV2),
            query=request,
            eligible_ids=(
                ct.POINTER(s.ZeDocId)() if ids_owner is None else
                ct.cast(ids_owner, ct.POINTER(s.ZeDocId))
            ),
            eligible_count=0 if ids_owner is None else len(ids_owner),
            has_eligible=int(eligible_ids is not None),
        )
        constraints = None
        filter_owners: list[object] = []
        if filter is not None:
            flat_filter, filter_owners = _flatten_filter(
                filter, self._attribute_ids, self._attribute_types, strict=True
            )
            constraints = s.sized(s.ZeQueryFilter)
            constraints.filter = ct.pointer(flat_filter)
        constraints_pointer = None if constraints is None else ct.byref(constraints)
        result = s.sized(s.ZeQueryResult)
        primary: BaseException | None = None
        try:
            if eligible_ids is not None:
                status = self._call(
                    LIBRARY.ze_query_v2, handle, ct.byref(request_v2), constraints_pointer,
                    snippet_bytes or 0, ct.byref(result), snippets_pointer,
                )
            elif constraints is not None:
                status = self._call(
                    LIBRARY.ze_query_filtered, handle, ct.byref(request), constraints_pointer,
                    snippet_bytes or 0, ct.byref(result), snippets_pointer,
                )
            elif snippets is not None:
                status = self._call(
                    LIBRARY.ze_query_with_snippets, handle, ct.byref(request),
                    snippet_bytes, ct.byref(result), snippets_pointer,
                )
            else:
                status = self._call(LIBRARY.ze_query, handle, ct.byref(request), ct.byref(result))
            raise_for_status(status, handle)
            if snippets is not None and snippets.snippet_count != result.hit_count:
                raise ERROR_TYPES[22]("query snippets are not aligned with its hits")
            hits = tuple(
                QueryHit(
                    doc_id=_python_doc_id(hit.doc_id) if hit.has_document else None,
                    revision=int(hit.revision) if hit.has_revision else None,
                    score=float(hit.score),
                    vector_squared_l2=(
                        float(hit.vector_squared_l2) if hit.has_vector_score else None
                    ),
                    lexical_bm25=float(hit.lexical_bm25) if hit.has_lexical_score else None,
                    snippet=None if snippets is None else self._query_snippet(snippets, index),
                )
                for index, hit in enumerate(result.hits[i] for i in range(result.hit_count))
            )
            fusion = (
                FusionReport(
                    FusionMethod(result.fusion_method),
                    float(result.effective_alpha),
                    int(result.fusion_rounds),
                )
                if result.has_fusion
                else None
            )
            return QueryResult(
                hits=hits,
                generation=int(result.generation),
                mode=QueryMode(result.mode),
                approximate=bool(result.approximate),
                exact_rescore=bool(result.exact_rescore),
                budget_exhausted=bool(result.budget_exhausted),
                fusion=fusion,
                embedding_epoch=int(result.embedding_epoch) if result.has_embedding_epoch else None,
                tokenizer_epoch=int(result.tokenizer_epoch) if result.has_tokenizer_epoch else None,
                dims_touched=int(result.dims_touched),
                bytes_read=int(result.bytes_read),
                docs_evaluated=int(result.docs_evaluated),
                postings_decoded=int(result.postings_decoded),
            )
        except BaseException as error:
            primary = error
            raise
        finally:
            del probe, text_owner, filter_owners
            frees = [(LIBRARY.ze_query_result_free, ct.byref(result))]
            if snippets is not None:
                frees.append((LIBRARY.ze_query_snippets_free, ct.byref(snippets)))
            cleanup_error = None
            for free, pointer in frees:
                status = self._call(free, pointer)
                try:
                    raise_for_status(status)
                except ZeppelinError as error:
                    if primary is None:
                        primary = cleanup_error = error
                    else:
                        primary.add_note(f"{free.__name__}: {error}")
            if cleanup_error is not None:
                raise cleanup_error

    def _query_snippet(self, snippets: s.ZeQuerySnippets, index: int) -> Snippet | None:
        snippet = snippets.snippets[index]
        if not snippet.has_snippet:
            return None
        source = s.sized(s.ZeSnippetSourceRanges)
        status = self._call(
            LIBRARY.ze_query_snippet_source_ranges, ct.byref(snippets), index, ct.byref(source)
        )
        raise_for_status(status)
        if source.highlight_count != snippet.highlight_count:
            raise ERROR_TYPES[22]("snippet source highlights are not aligned")
        return Snippet(
            text=ct.string_at(snippet.text, snippet.text_len).decode("utf-8"),
            highlights=tuple(
                SnippetHighlight(
                    start=int(snippet.highlights[i].start),
                    end=int(snippet.highlights[i].end),
                    source_byte_start=int(source.highlights[i].start),
                    source_byte_end=int(source.highlights[i].end),
                )
                for i in range(snippet.highlight_count)
            ),
            truncated_start=bool(snippet.truncated_start),
            truncated_end=bool(snippet.truncated_end),
            source_byte_start=int(source.source_start),
            source_byte_end=int(source.source_end),
        )

    def warm_lexical(self, cancel_token: CancelToken | None = None, *, deadline_ns: int = 0) -> None:
        """Prepare exact and prefix lexical queries; mutations invalidate preparation."""
        handle = self._live_handle()
        request = s.ZeWarmLexicalRequest(
            abi_size=ct.sizeof(s.ZeWarmLexicalRequest),
            cancel_token=0 if cancel_token is None else cancel_token.value,
            deadline_ns=deadline_ns,
        )
        raise_for_status(self._call(LIBRARY.ze_warm_lexical, handle, ct.byref(request)), handle)

    def seal(self, cancel_token: CancelToken | None = None) -> GenerationReport:
        handle = self._live_handle()
        request = s.ZeSealRequest(
            abi_size=ct.sizeof(s.ZeSealRequest),
            abi_reserved=0,
            cancel_token=0 if cancel_token is None else cancel_token.value,
        )
        report = s.sized(s.ZeGenerationReport)
        raise_for_status(
            self._call(LIBRARY.ze_seal, handle, ct.byref(request), ct.byref(report)),
            handle,
        )
        return GenerationReport(int(report.generation))

    def merge(self, cancel_token: CancelToken | None = None) -> GenerationReport:
        """Merge small compatible sealed segments and return the publication generation."""
        handle = self._live_handle()
        if cancel_token is not None and not isinstance(cancel_token, CancelToken):
            invalid_argument("cancel_token must be a CancelToken or None")
        request = s.sized(s.ZeSealRequest)
        request.cancel_token = 0 if cancel_token is None else cancel_token.value
        report = s.sized(s.ZeGenerationReport)
        raise_for_status(
            self._call(LIBRARY.ze_merge_sealed, handle, ct.byref(request), ct.byref(report)),
            handle,
        )
        return GenerationReport(int(report.generation))

    def drop_partition(self, start_ts: int, end_ts: int) -> PartitionReport:
        request = s.ZeDropPartitionRequest(
            abi_size=ct.sizeof(s.ZeDropPartitionRequest),
            abi_reserved=0,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        return self._partition_call(LIBRARY.ze_drop_partition, request)

    def apply_retention(self, *, window: int, now_ts: int) -> PartitionReport:
        request = s.ZeRetentionRequest(
            abi_size=ct.sizeof(s.ZeRetentionRequest),
            abi_reserved=0,
            window=window,
            now_ts=now_ts,
        )
        return self._partition_call(LIBRARY.ze_apply_retention, request)

    def _partition_call(self, function: Any, request: ct.Structure) -> PartitionReport:
        handle = self._live_handle()
        report = s.sized(s.ZePartitionReport)
        raise_for_status(
            self._call(function, handle, ct.byref(request), ct.byref(report)),
            handle,
        )
        return PartitionReport(
            generation=int(report.generation),
            segments_dropped=int(report.segments_dropped),
            bytes_reclaimed=int(report.bytes_reclaimed),
            straddlers_skipped=int(report.straddlers_skipped),
            is_no_op=bool(report.is_no_op),
        )

    def purge(self, doc_ids: Sequence[int | tuple[int, int]]) -> PurgeTokenReport:
        handle = self._live_handle()
        ids = (s.ZeDocId * len(doc_ids))(*[_doc_id(value) for value in doc_ids])
        request = s.ZePurgeRequest(
            abi_size=ct.sizeof(s.ZePurgeRequest),
            abi_reserved=0,
            doc_ids=ct.cast(ids, ct.POINTER(s.ZeDocId)),
            doc_id_count=len(ids),
        )
        report = s.sized(s.ZePurgeTokenReport)
        raise_for_status(
            self._call(LIBRARY.ze_purge, handle, ct.byref(request), ct.byref(report)),
            handle,
        )
        return PurgeTokenReport(
            token_id=int(report.token_id),
            generation=int(report.generation),
            unknown_id_count=int(report.unknown_id_count),
            is_no_op=bool(report.is_no_op),
        )

    def await_physical_purge(self, token_id: int) -> PurgeReport:
        handle = self._live_handle()
        request = s.ZeAwaitPurgeRequest(
            abi_size=ct.sizeof(s.ZeAwaitPurgeRequest),
            abi_reserved=0,
            token_id=token_id,
        )
        report = s.sized(s.ZePurgeReport)
        raise_for_status(
            self._call(
                LIBRARY.ze_await_physical_purge,
                handle,
                ct.byref(request),
                ct.byref(report),
            ),
            handle,
        )
        return PurgeReport(
            generation=int(report.generation),
            segments_rewritten=int(report.segments_rewritten),
            unknown_id_count=int(report.unknown_id_count),
            wal_rewritten=bool(report.wal_rewritten),
            is_no_op=bool(report.is_no_op),
        )

    def maintain(self, *, wall_time_ns: int, bytes: int) -> MaintainReport:
        if self._text:
            self._require_text_store()
        handle = self._live_handle()
        request = s.ZeMaintainRequest(
            abi_size=ct.sizeof(s.ZeMaintainRequest),
            abi_reserved=0,
            wall_time_ns=wall_time_ns,
            bytes=bytes,
        )
        report = s.sized(s.ZeMaintainReport)
        raise_for_status(
            self._call(
                LIBRARY.ze_text_maintain_budgeted
                if self._text
                else LIBRARY.ze_maintain,
                handle,
                ct.byref(request),
                ct.byref(report),
            ),
            handle,
        )
        return MaintainReport(
            graphs_built=int(report.graphs_built),
            bytes_consumed=int(report.bytes_consumed),
            checkpoints_resumed=int(report.checkpoints_resumed),
            status=MaintenanceStatus(report.status),
        )

    def epoch_current(self) -> EpochIdentity:
        handle = self._live_handle()
        report = s.sized(s.ZeEpochIdentity)
        raise_for_status(
            self._call(LIBRARY.ze_epoch_current, handle, ct.byref(report)),
            handle,
        )
        return EpochIdentity(int(report.embedding_epoch), int(report.tokenizer_epoch))

    def epoch_switch_alias(self, epoch: EmbeddingEpoch) -> EpochAliasReport:
        handle = self._live_handle()
        request, owners = _epoch_request(epoch)
        report = s.sized(s.ZeEpochAliasReport)
        status = self._call(
            LIBRARY.ze_epoch_switch_alias,
            handle,
            ct.byref(request),
            ct.byref(report),
        )
        del owners
        raise_for_status(status, handle)
        return EpochAliasReport(
            generation=int(report.generation),
            previous_embedding_epoch=int(report.previous_embedding_epoch),
            previous_tokenizer_epoch=int(report.previous_tokenizer_epoch),
            published_embedding_epoch=int(report.published_embedding_epoch),
            published_tokenizer_epoch=int(report.published_tokenizer_epoch),
            manifest_committed=bool(report.manifest_committed),
        )

    def epoch_drop(self, epoch: EmbeddingEpoch) -> EpochDropReport:
        handle = self._live_handle()
        request, owners = _epoch_request(epoch)
        report = s.sized(s.ZeEpochDropReport)
        status = self._call(
            LIBRARY.ze_epoch_drop,
            handle,
            ct.byref(request),
            ct.byref(report),
        )
        del owners
        raise_for_status(status, handle)
        return EpochDropReport(
            generation=int(report.generation),
            segments_dropped=int(report.segments_dropped),
            bytes_reclaimed=int(report.bytes_reclaimed),
        )


def _namespace_spec(
    spec: NamespaceSpec, owners: list[object]
) -> tuple[ct._Pointer[s.ZeNamespaceSpec], dict[str, int], dict[int, AttributeType]]:
    if not isinstance(spec, NamespaceSpec):
        invalid_argument("spec must be a NamespaceSpec")
    model = spec.embedded_model
    if model is not None:
        if not isinstance(model, EmbeddedModelRef):
            invalid_argument("embedded_model must be an EmbeddedModelRef")
        if spec.vector_space is not None:
            invalid_argument("the model determines the embedded namespace vector space")
        _require_text()
    definitions = (s.ZeAttributeDefinition * len(spec.attributes))()
    attribute_ids: dict[str, int] = {}
    attribute_types: dict[int, AttributeType] = {0: AttributeType.I64}
    for index, definition in enumerate(spec.attributes):
        if not isinstance(definition, AttributeDefinition):
            invalid_argument("namespace attributes must be AttributeDefinition values")
        pointer, length, owner = _text_pointer(definition.name)
        if owner is not None:
            owners.append(owner)
        attribute_type = _enum_value(
            definition.attribute_type, AttributeType, "attribute_type"
        )
        if not isinstance(definition.nullable, bool):
            invalid_argument("nullable must be a bool")
        definitions[index] = s.ZeAttributeDefinition(
            attribute_id=_u32(definition.attribute_id, "attribute_id"),
            name=pointer,
            name_len=length,
            attribute_type=int(attribute_type),
            nullable=int(definition.nullable),
        )
        attribute_ids[definition.name] = definition.attribute_id
        attribute_types[definition.attribute_id] = attribute_type
    owners.append(definitions)
    vector_space = spec.vector_space
    if vector_space is None:
        has_vector_space = 0
        dimensions = 0
        normalization = Normalization.NONE
        epoch_pointer = ct.POINTER(s.ZeEpochRequest)()
    elif isinstance(vector_space, VectorSpace):
        has_vector_space = 1
        dimensions = _u32(vector_space.dimensions, "dimensions")
        normalization = _enum_value(
            vector_space.normalization, Normalization, "normalization"
        )
        if vector_space.epoch is None:
            epoch_pointer = ct.POINTER(s.ZeEpochRequest)()
        else:
            if not isinstance(vector_space.epoch, EmbeddingEpoch):
                invalid_argument("epoch must be an EmbeddingEpoch or None")
            profile = vector_space.epoch.tokenizer_profile
            if not isinstance(profile, int) or not -(1 << 31) <= profile < 1 << 31:
                invalid_argument("epoch tokenizer_profile must be a signed 32-bit integer")
            epoch_request, epoch_owners = _epoch_request(vector_space.epoch)
            epoch_pointer = ct.pointer(epoch_request)
            owners.extend((epoch_request, epoch_pointer, *epoch_owners))
    else:
        invalid_argument("vector_space must be a VectorSpace or None")
    namespace_spec = s.ZeNamespaceSpec(
        abi_size=ct.sizeof(s.ZeNamespaceSpec),
        abi_reserved=0,
        attributes=ct.cast(definitions, ct.POINTER(s.ZeAttributeDefinition)),
        attribute_count=len(definitions),
        has_vector_space=has_vector_space,
        dimensions=dimensions,
        normalization=int(normalization),
        epoch=epoch_pointer,
    )
    if model is None:
        return ct.pointer(namespace_spec), attribute_ids, attribute_types

    model_ref = s.sized(s.ZeTextModelRef)
    for field in ("repo", "revision", "manifest_sha256"):
        value = getattr(model, field)
        if not isinstance(value, str):
            invalid_argument(f"embedded model {field} must be a string")
        pointer, length, owner = _text_pointer(value)
        setattr(model_ref, field, pointer)
        setattr(model_ref, field + "_len", length)
        if owner is not None:
            owners.append(owner)
    namespace_spec.abi_size = ct.sizeof(s.ZeEmbeddedNamespaceSpec)
    namespace_spec.has_vector_space = 2
    embedded_spec = s.ZeEmbeddedNamespaceSpec(spec=namespace_spec, model=model_ref)
    owners.append(embedded_spec)
    prefix = ct.cast(ct.pointer(embedded_spec), ct.POINTER(s.ZeNamespaceSpec))
    return prefix, attribute_ids, attribute_types


def open_namespace(
    root: str | Path,
    name: str,
    spec: NamespaceSpec,
    *,
    access_mode: AccessMode | str = AccessMode.READ_WRITE,
    durability: DurabilityMode | str = DurabilityMode.DERIVED,
    commit_tier: CommitTier | str = CommitTier.ORDERED,
    reader_drain_timeout_ms: int = 250,
    max_resident_bytes: int = (1 << 64) - 1,
    max_temp_bytes: int = (1 << 64) - 1,
) -> Store:
    """Open or idempotently create one namespace under a database root."""

    open_request, open_owner = _open_request(
        root,
        access_mode,
        durability,
        commit_tier,
        reader_drain_timeout_ms,
        max_resident_bytes,
        max_temp_bytes,
    )
    root_bytes = os.fspath(root).encode("utf-8", errors="strict")
    root_pointer, root_owner = _bytes_pointer(root_bytes)
    name_pointer, name_len, name_owner = _text_pointer(name)
    owners: list[object] = [open_owner]
    if root_owner is not None:
        owners.append(root_owner)
    if name_owner is not None:
        owners.append(name_owner)
    namespace_spec_pointer, attribute_ids, attribute_types = _namespace_spec(spec, owners)
    namespace_kind = namespace_spec_pointer.contents.has_vector_space
    request = s.ZeNamespaceOpenRequest(
        abi_size=ct.sizeof(s.ZeNamespaceOpenRequest),
        abi_reserved=0,
        root=root_pointer,
        root_len=len(root_bytes),
        name=name_pointer,
        name_len=name_len,
        open=open_request,
        spec=namespace_spec_pointer,
    )
    handle = ct.c_uint64()
    if spec.tokenizer_profile is None:
        status = LIBRARY.ze_namespace_open(ct.byref(request), ct.byref(handle))
    else:
        profile = _enum_value(spec.tokenizer_profile, TokenizerProfile, "tokenizer_profile")
        status = LIBRARY.ze_namespace_open_with_tokenizer(
            ct.byref(request), int(profile), ct.byref(handle)
        )
    del owners
    raise_for_status(status, int(handle.value))
    return Store(
        int(handle.value),
        attribute_ids=attribute_ids,
        attribute_types=attribute_types,
        vector_dimensions=spec.vector_space.dimensions if spec.vector_space else None,
        record_only=namespace_kind == 0,
        embedded=namespace_kind == 2,
    )


def _namespace_batch_request(
    root: str | Path, participants: Sequence[NamespaceMutation], *, minimum: int
) -> tuple[s.ZeNamespaceBatchRequest, ct.Array[ct.c_uint64], list[object]]:
    count = len(participants)
    if not minimum <= count <= 128:
        invalid_argument(f"namespace operation requires {minimum}..128 participants")
    if not isinstance(root, (str, os.PathLike)):
        invalid_argument("root must be a path")
    root_pointer, root_len, root_owner = _text_pointer(os.fspath(root))
    mutations = (s.ZeNamespaceMutation * count)()
    generations = (ct.c_uint64 * count)()
    owners: list[object] = [root_owner, mutations]
    for index, participant in enumerate(participants):
        if not isinstance(participant, NamespaceMutation):
            invalid_argument("participants must contain NamespaceMutation values")
        spec_pointer, attribute_ids, attribute_types = _namespace_spec(participant.spec, owners)
        spec = spec_pointer.contents
        name, name_len, name_owner = _text_pointer(participant.name)
        owners.extend((spec_pointer, name_owner))
        profile = participant.spec.tokenizer_profile
        if profile is None:
            tokenizer = int(spec.epoch.contents.tokenizer_profile) if spec.epoch else 0
        else:
            tokenizer = int(_enum_value(profile, TokenizerProfile, "tokenizer_profile"))
        upserts, upsert_owners = Store._marshal_upserts(
            tuple(participant.upserts),
            record_only=not spec.has_vector_space,
            vector_dimensions=int(spec.dimensions),
            embedded=spec.has_vector_space == 2,
            require_conditions=True,
        )
        owners.extend(upsert_owners)
        deletes = (s.ZeDocId * len(participant.deletes))(
            *[_doc_id(doc_id) for doc_id in participant.deletes]
        )
        owners.append(deletes)
        filter_pointer = ct.POINTER(s.ZeFilter)()
        if participant.delete_where is not None:
            filter_request, filter_owners = _flatten_filter(
                participant.delete_where, attribute_ids, attribute_types, strict=True
            )
            filter_pointer = ct.pointer(filter_request)
            owners.extend((filter_request, *filter_owners))
        mutation = s.sized(s.ZeNamespaceMutation)
        mutation.name = name
        mutation.name_len = name_len
        mutation.spec = spec_pointer
        mutation.tokenizer_profile = tokenizer
        mutation.upserts = upserts
        mutation.deletes = deletes
        mutation.delete_count = len(deletes)
        mutation.filter = filter_pointer
        mutations[index] = mutation
    request = s.sized(s.ZeNamespaceBatchRequest)
    request.root = root_pointer
    request.root_len = root_len
    request.participants = mutations
    request.participant_count = count
    request.generations = generations
    return request, generations, owners


def namespace_batch(root: str | Path, participants: Sequence[NamespaceMutation]) -> list[int]:
    """Commit changes across 2..128 closed namespaces and return their generations."""

    request, generations, owners = _namespace_batch_request(root, tuple(participants), minimum=2)
    status = LIBRARY.ze_namespace_batch(ct.byref(request))
    del owners
    raise_for_status(status)
    return list(generations)


def namespace_batch_live(
    root: str | Path, participants: Sequence[LiveNamespaceMutation]
) -> list[int]:
    """Commit changes through 2..128 existing writable namespace handles."""

    participants = tuple(participants)
    batch, generations, owners = _namespace_batch_request(root, participants, minimum=2)
    handles = (ct.c_uint64 * len(participants))()
    for index, participant in enumerate(participants):
        if not isinstance(participant, LiveNamespaceMutation) or not isinstance(
            participant.store, Store
        ):
            invalid_argument("participants must contain LiveNamespaceMutation values with a Store")
        handles[index] = participant.store._live_handle()
    request = s.sized(s.ZeNamespaceBatchLiveRequest)
    request.batch = batch
    request.handles = handles
    status = LIBRARY.ze_namespace_batch_live(ct.byref(request))
    del owners
    raise_for_status(status)
    return list(generations)


def declare_cascade(
    root: str | Path,
    participants: Sequence[CascadeParticipant],
    declaration: CascadeDeclaration,
) -> None:
    """Persist a child-to-parent cascade rule between closed namespaces."""

    if not isinstance(declaration, CascadeDeclaration):
        invalid_argument("declaration must be a CascadeDeclaration")
    request = s.sized(s.ZeCascadeDeclaration)
    request.parent_index = _u32(declaration.parent_index, "parent_index")
    request.child_index = _u32(declaration.child_index, "child_index")
    request.attribute_id = _u32(declaration.attribute_id, "attribute_id")
    mutations: list[NamespaceMutation] = []
    for participant in participants:
        if not isinstance(participant, CascadeParticipant):
            invalid_argument("participants must contain CascadeParticipant values")
        mutations.append(NamespaceMutation(participant.name, participant.spec))
    batch, _, owners = _namespace_batch_request(root, mutations, minimum=1)
    status = LIBRARY.ze_namespace_declare_cascade(ct.byref(batch), ct.byref(request))
    del owners
    raise_for_status(status)


def delete_cascade(
    root: str | Path, participants: Sequence[CascadeDeleteParticipant]
) -> list[int]:
    """Atomically delete explicit IDs and all their declared dependants."""

    mutations: list[NamespaceMutation] = []
    for participant in participants:
        if not isinstance(participant, CascadeDeleteParticipant):
            invalid_argument("participants must contain CascadeDeleteParticipant values")
        mutations.append(
            NamespaceMutation(participant.name, participant.spec, deletes=participant.deletes)
        )
    batch, generations, owners = _namespace_batch_request(root, mutations, minimum=1)
    status = LIBRARY.ze_namespace_delete_cascade(ct.byref(batch))
    del owners
    raise_for_status(status)
    return list(generations)


def list_namespaces(root: str | Path) -> list[str]:
    """Return namespace names immediately below one database root."""

    root_bytes = os.fspath(root).encode("utf-8", errors="strict")
    root_pointer, root_owner = _bytes_pointer(root_bytes)
    request = s.ZeNamespaceListRequest(
        abi_size=ct.sizeof(s.ZeNamespaceListRequest),
        abi_reserved=0,
        root=root_pointer,
        root_len=len(root_bytes),
    )
    result = s.sized(s.ZeNamespaceListResult)
    primary: BaseException | None = None
    try:
        status = LIBRARY.ze_namespace_list(ct.byref(request), ct.byref(result))
        raise_for_status(status)
        return [
            ct.string_at(result.entries[index].name, result.entries[index].name_len).decode(
                "utf-8"
            )
            for index in range(result.entry_count)
        ]
    except BaseException as error:
        primary = error
        raise
    finally:
        del root_owner
        free_status = LIBRARY.ze_namespace_list_result_free(ct.byref(result))
        if free_status != 0:
            if primary is None:
                raise_for_status(free_status)
            assert primary is not None
            primary.add_note(
                f"ze_namespace_list_result_free: {last_error_message()}"
            )


def verify(path: str | os.PathLike[str]) -> VerifyReport:
    """Check a store directory for damage without opening or modifying it."""

    try:
        path_text = os.fspath(path)
    except TypeError as error:
        invalid_argument(f"invalid store path: {error}")
    if not isinstance(path_text, str):
        invalid_argument("store path must be a string or a string-valued path-like object")
    try:
        encoded = path_text.encode("utf-8")
    except UnicodeEncodeError as error:
        invalid_argument(f"store path is not valid UTF-8: {error}")

    pointer, _owner = _bytes_pointer(encoded)
    request = s.ZeVerifyRequest(
        abi_size=ct.sizeof(s.ZeVerifyRequest),
        abi_reserved=0,
        path=pointer,
        path_len=len(encoded),
    )
    result = s.sized(s.ZeVerifyResult)
    raise_for_status(LIBRARY.ze_verify(ct.byref(request), ct.byref(result)))
    try:
        findings: list[VerifyFinding] = []
        for index in range(result.finding_count):
            finding = result.findings[index]
            try:
                kind = VerifyFindingKind(finding.kind)
            except ValueError as error:
                raise ERROR_TYPES[22](f"verify returned unknown finding kind {finding.kind}") from error
            findings.append(
                VerifyFinding(
                    kind=kind,
                    file=ct.string_at(finding.file, finding.file_len).decode("utf-8"),
                    offset=finding.offset if finding.has_offset else None,
                    detail=ct.string_at(finding.detail, finding.detail_len).decode("utf-8"),
                )
            )
        return VerifyReport(
            ok=not findings,
            generation=result.generation,
            segments_checked=result.segments_checked,
            wal_records_checked=result.wal_records_checked,
            findings=tuple(findings),
        )
    finally:
        raise_for_status(LIBRARY.ze_verify_result_free(ct.byref(result)))


def open(
    path: str | Path,
    *,
    access_mode: AccessMode | str = AccessMode.READ_WRITE,
    durability: DurabilityMode | str = DurabilityMode.DERIVED,
    commit_tier: CommitTier | str = CommitTier.ORDERED,
    reader_drain_timeout_ms: int = 250,
    max_resident_bytes: int = (1 << 64) - 1,
    max_temp_bytes: int = (1 << 64) - 1,
    epoch: EmbeddingEpoch | None = None,
) -> Store:
    """Open one store directory, optionally declaring its complete epoch."""

    request, path_owner = _open_request(
        path,
        access_mode,
        durability,
        commit_tier,
        reader_drain_timeout_ms,
        max_resident_bytes,
        max_temp_bytes,
    )
    handle = ct.c_uint64()
    if epoch is None:
        status = LIBRARY.ze_open(ct.byref(request), ct.byref(handle))
    else:
        epoch_request, epoch_owners = _epoch_request(epoch)
        status = LIBRARY.ze_open_with_epoch(
            ct.byref(request),
            ct.byref(epoch_request),
            ct.byref(handle),
        )
        del epoch_owners
    del path_owner
    raise_for_status(status, int(handle.value))
    return Store(int(handle.value))


def open_with_epoch(
    path: str | Path,
    epoch: EmbeddingEpoch,
    *,
    access_mode: AccessMode | str = AccessMode.READ_WRITE,
    durability: DurabilityMode | str = DurabilityMode.DERIVED,
    commit_tier: CommitTier | str = CommitTier.ORDERED,
    reader_drain_timeout_ms: int = 250,
    max_resident_bytes: int = (1 << 64) - 1,
    max_temp_bytes: int = (1 << 64) - 1,
) -> Store:
    """Open one store while explicitly declaring its complete epoch."""

    return open(
        path,
        access_mode=access_mode,
        durability=durability,
        commit_tier=commit_tier,
        reader_drain_timeout_ms=reader_drain_timeout_ms,
        max_resident_bytes=max_resident_bytes,
        max_temp_bytes=max_temp_bytes,
        epoch=epoch,
    )


def _mlx_cache_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 1 << 64:
        invalid_argument("mlx_cache_limit_bytes must be an unsigned 64-bit integer")
    return value


def open_text(
    path: str | Path,
    bundle_path: str | Path,
    *,
    access_mode: AccessMode | str = AccessMode.READ_WRITE,
    durability: DurabilityMode | str = DurabilityMode.DERIVED,
    commit_tier: CommitTier | str = CommitTier.ORDERED,
    reader_drain_timeout_ms: int = 250,
    max_resident_bytes: int = (1 << 64) - 1,
    mlx_cache_limit_bytes: int | None = None,
    max_temp_bytes: int = (1 << 64) - 1,
) -> Store:
    """Open a text store and immutable model bundle through the text ABI."""
    _require_text()
    request, path_owner = _open_request(
        path,
        access_mode,
        durability,
        commit_tier,
        reader_drain_timeout_ms,
        max_resident_bytes,
        max_temp_bytes,
    )
    bundle_pointer, bundle_len, bundle_owner = _text_pointer(str(bundle_path))
    text_request = s.ZeTextOpenRequest(
        abi_size=ct.sizeof(s.ZeTextOpenRequest),
        abi_reserved=0,
        store=request,
        bundle_path=bundle_pointer,
        bundle_path_len=bundle_len,
    )
    handle = ct.c_uint64()
    if mlx_cache_limit_bytes is None:
        status = LIBRARY.ze_text_open(ct.byref(text_request), ct.byref(handle))
    else:
        extended = s.ZeTextOpenRequestV2(
            abi_size=ct.sizeof(s.ZeTextOpenRequestV2), open=text_request,
            mlx_cache_limit_bytes=_mlx_cache_limit(mlx_cache_limit_bytes),
        )
        status = LIBRARY.ze_text_open_v2(ct.byref(extended), ct.byref(handle))
    del path_owner, bundle_owner
    raise_for_status(status, int(handle.value))
    return Store(int(handle.value), text=True)


DEFAULT_MODEL_REF: dict[str, str] = {
    "repo": "volantis-labs/volantis-db-clean71-arctic-m-v2",
    "revision": "d64b817c4021a3f1b9bded3152d9afdf9ac813e7",
    "manifest_sha256": "123c6308e00f769c68498f9df60dfb9efc688f58c4ff2c4da266f978fcb215dc",
}


def open_model(
    path: str | Path,
    *,
    repo: str | None = None,
    revision: str | None = None,
    manifest_sha256: str | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    progress: Callable[[int, int], None] | None = None,
    access_mode: AccessMode | str = AccessMode.READ_WRITE,
    durability: DurabilityMode | str = DurabilityMode.DERIVED,
    commit_tier: CommitTier | str = CommitTier.ORDERED,
    reader_drain_timeout_ms: int = 250,
    max_resident_bytes: int = (1 << 64) - 1,
    mlx_cache_limit_bytes: int | None = None,
    max_temp_bytes: int = (1 << 64) - 1,
) -> Store:
    """Download/verify a pinned HF model, then open its text store.

    Omit all three pins to use DEFAULT_MODEL_REF; partial pins are invalid.
    Progress reports byte counts synchronously. Intermediate reports require both
    at least 64 KiB and at least 100 ms since the last report; each verified file
    also reports completion. Initial (0, 0) means fetching the manifest.
    The callback must not reenter model resolution.
    Callback exceptions are propagated after native resolution returns.
    """
    _require_text()
    pins = (repo, revision, manifest_sha256)
    if all(value is None for value in pins):
        repo = DEFAULT_MODEL_REF["repo"]
        revision = DEFAULT_MODEL_REF["revision"]
        manifest_sha256 = DEFAULT_MODEL_REF["manifest_sha256"]
    elif repo is None or revision is None or manifest_sha256 is None:
        raise ValueError("provide all three of repo, revision and manifest_sha256, or none")
    request, path_owner = _open_request(
        path,
        access_mode,
        durability,
        commit_tier,
        reader_drain_timeout_ms,
        max_resident_bytes,
        max_temp_bytes,
    )
    model = s.ZeTextOpenModelRequest(
        abi_size=ct.sizeof(s.ZeTextOpenModelRequest),
        store=request,
        offline=int(offline),
    )
    owners: list[object] = [path_owner]
    for field, value in (
        ("repo", repo),
        ("revision", revision),
        ("manifest_sha256", manifest_sha256),
        ("cache_dir", "" if cache_dir is None else str(cache_dir)),
    ):
        pointer, length, owner = _text_pointer(value)
        setattr(model, field, pointer)
        setattr(model, field + "_len", length)
        if owner is not None:
            owners.append(owner)
    callback_errors: list[BaseException] = []

    def on_progress(_user_data: object, done: int, total: int) -> None:
        if progress is not None and not callback_errors:
            try:
                progress(done, total)
            except BaseException as error:
                callback_errors.append(error)

    callback = (
        s.ModelProgressCallback(on_progress)
        if progress is not None
        else s.ModelProgressCallback()
    )
    model.progress = callback
    handle = ct.c_uint64()
    if mlx_cache_limit_bytes is None:
        status = LIBRARY.ze_text_open_model(ct.byref(model), ct.byref(handle))
    else:
        extended = s.ZeTextOpenModelRequestV2(
            abi_size=ct.sizeof(s.ZeTextOpenModelRequestV2), open=model,
            mlx_cache_limit_bytes=_mlx_cache_limit(mlx_cache_limit_bytes),
        )
        status = LIBRARY.ze_text_open_model_v2(ct.byref(extended), ct.byref(handle))
    if callback_errors:
        if handle.value:
            close_status = LIBRARY.ze_close(handle.value)
            if close_status:
                callback_errors[0].add_note(
                    f"ze_close: {last_error_message(int(handle.value))}"
                )
        raise callback_errors[0]
    raise_for_status(status, int(handle.value))
    return Store(int(handle.value), text=True)
