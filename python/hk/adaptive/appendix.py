"""
HK Adaptive Neural Framework: Appendix Region Manager
Provides reading, appending, cryptographic SHA-256 lineage verification, rollback,
and delta merging for version-chained neural evolution in HK binary containers.
"""

import os
import struct
import hashlib
import time
import zlib
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union


class AppendixEntryType(IntEnum):
    LORA_ADAPTER = 0x01
    DELTA_PATCH = 0x02
    NEW_LAYER = 0x03
    CODE_EVAL = 0x04
    KV_CACHE_SINK = 0x05
    TOPOLOGY_HEAD = 0x06

class AppendixFlags(IntEnum):
    ACTIVE = 0x01
    COMPRESSED = 0x02

@dataclass
class AppendixMetrics:
    loss: float = 0.0
    accuracy: float = 0.0
    pass_rate: float = 0.0
    custom: float = 0.0

@dataclass
class AppendixRecord:
    entry_type: AppendixEntryType
    name: str
    generation: int
    data: bytes
    target: str = ""
    flags: int = AppendixFlags.ACTIVE
    timestamp: int = 0
    parent_hash: bytes = b"\x00" * 32
    metrics: AppendixMetrics = field(default_factory=AppendixMetrics)

HEADER_SIZE = 128
HEADER_FLAGS_OFFSET = 8
HEADER_CHECKSUM_OFFSET = 80
HEADER_SHA256_OFFSET = 90
HEADER_SHA256_SIZE = 32
FLAG_HAS_SHA256_CHECKSUM = 1 << 9
REC_HEADER_FORMAT = "<BBHIQ32sffffHHIQ" # 80 bytes (entry_type, flags, name_len, generation, timestamp, parent_hash[32], loss, acc, pass, custom, target_len, reserved, crc32, data_size)
REC_HEADER_SIZE = 80

def verify_container_integrity(file_bytes: bytes) -> Optional[bool]:
    """Returns True/False for a sealed container, or None for legacy files."""
    if len(file_bytes) < HEADER_SIZE:
        raise ValueError("File is smaller than HK header size")
    flags = struct.unpack_from("<I", file_bytes, HEADER_FLAGS_OFFSET)[0]
    if not flags & FLAG_HAS_SHA256_CHECKSUM:
        return None

    expected_compact = struct.unpack_from("<Q", file_bytes, HEADER_CHECKSUM_OFFSET)[0]
    expected_full = file_bytes[HEADER_SHA256_OFFSET : HEADER_SHA256_OFFSET + HEADER_SHA256_SIZE]
    h = hashlib.sha256()
    h.update(file_bytes[:HEADER_CHECKSUM_OFFSET])
    h.update(b"\x00" * 8)
    h.update(file_bytes[HEADER_CHECKSUM_OFFSET + 8 : HEADER_SHA256_OFFSET])
    h.update(b"\x00" * HEADER_SHA256_SIZE)
    h.update(file_bytes[HEADER_SHA256_OFFSET + HEADER_SHA256_SIZE :])
    computed = h.digest()
    return (expected_compact == int.from_bytes(computed[:8], "little")
            and expected_full == computed)

def compute_parent_hash(data: bytes, name: str = "", target: str = "") -> bytes:
    """Computes SHA-256 digest of record payload, including optional name and target."""
    h = hashlib.sha256()
    if name:
        h.update(name.encode("utf-8"))
    if target:
        h.update(target.encode("utf-8"))
    h.update(data)
    return h.digest()

def read_appendix(file_path: str) -> List[AppendixRecord]:
    """Parses all appendix records from an HK container.

    Only the appendix region is held in memory; the seal (if present) is checked by streaming the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HK file not found: {file_path}")

    from ..native import is_native_available, native_verify_integrity

    with open(file_path, "rb") as f:
        header = f.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            raise ValueError("File is smaller than HK header size")
        if header[:4] != b"HKNT":
            raise ValueError(f"Invalid HK magic: {header[:4]!r}")
    if is_native_available():
        # Streamed in Zig: bounded memory regardless of model size.
        if native_verify_integrity(file_path) is False:
            raise ValueError("Container checksum mismatch")
    else:
        # No native library: verify in Python, which holds the whole file in memory (flagged limitation).
        with open(file_path, "rb") as f:
            if verify_container_integrity(f.read()) is False:
                raise ValueError("Container checksum mismatch")

    with open(file_path, "rb") as f:
        appendix_offset = struct.unpack_from("<Q", header, 72)[0]
        f.seek(0, os.SEEK_END)
        if appendix_offset == 0 or appendix_offset >= f.tell():
            return []
        f.seek(appendix_offset)
        file_bytes = f.read()

    records = []
    offset = 0

    while offset + REC_HEADER_SIZE <= len(file_bytes):
        hdr_tuple = struct.unpack_from(REC_HEADER_FORMAT, file_bytes, offset)
        (entry_type_raw, flags, name_len, gen, ts, parent_hash,
         loss, acc, pass_rate, custom, target_len, reserved, crc32, data_size) = hdr_tuple
        offset += REC_HEADER_SIZE

        name = file_bytes[offset : offset + name_len].decode("utf-8", errors="replace")
        offset += name_len

        target = file_bytes[offset : offset + target_len].decode("utf-8", errors="replace")
        offset += target_len

        payload = file_bytes[offset : offset + data_size]
        if len(payload) != data_size:
            raise ValueError("Truncated appendix record payload")
        # A zero CRC is the legacy v1 representation and deliberately remains
        # readable. New writers always emit CRC-32/ISO-HDLC for the payload.
        if crc32 != 0 and (zlib.crc32(payload) & 0xFFFFFFFF) != crc32:
            raise ValueError("Appendix record payload checksum mismatch")
        offset += data_size

        # Align to 8 bytes
        if offset % 8 != 0:
            offset += (8 - (offset % 8))

        records.append(AppendixRecord(
            entry_type=AppendixEntryType(entry_type_raw),
            name=name,
            target=target,
            flags=flags,
            generation=gen,
            timestamp=ts,
            parent_hash=parent_hash,
            metrics=AppendixMetrics(loss=loss, accuracy=acc, pass_rate=pass_rate, custom=custom),
            data=payload
        ))

    return records

def append_record(file_path: str, record: AppendixRecord) -> None:
    """Appends an evolutionary record to an HK container in place, in Zig.

    The native routine verifies the existing seal first, journals the edit so a crash rolls back to the
    previous sealed file (``hk repair``), appends O(record) bytes, and re-seals with one streamed hash pass.
    """
    from ..native import is_native_available, native_appendix_append

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HK file not found: {file_path}")
    if not is_native_available():
        raise RuntimeError("append_record needs the native hk library (it owns the container seal and edit journal).")

    ts = record.timestamp if record.timestamp > 0 else int(time.time())
    ok = native_appendix_append(
        file_path, int(record.entry_type), record.flags, record.name, record.target, record.generation,
        record.parent_hash,
        (record.metrics.loss, record.metrics.accuracy, record.metrics.pass_rate, record.metrics.custom),
        record.data, ts,
    )
    if not ok:
        raise RuntimeError(
            f"append_record({file_path!r}) was refused by the native appendix writer; the file was left "
            "unchanged (failed seal check or I/O error). Run `hk verify` / `hk repair` on it."
        )

write_appendix_record = append_record

def rollback_appendix(file_path: str, target_generation: int) -> None:
    """Rolls back the HK container to target_generation (truncates later records in place), in Zig."""
    from ..native import is_native_available, native_appendix_rollback

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HK file not found: {file_path}")
    if not is_native_available():
        raise RuntimeError("rollback_appendix needs the native hk library (it owns the container seal and edit journal).")
    if not native_appendix_rollback(file_path, max(int(target_generation), 0)):
        raise RuntimeError(
            f"rollback_appendix({file_path!r}) was refused by the native appendix writer; the file was left "
            "unchanged (failed seal check or I/O error). Run `hk verify` / `hk repair` on it."
        )

def verify_lineage(records_or_path: Union[List[AppendixRecord], str]) -> bool:
    """Cryptographically validates SHA-256 parent hash chaining across generations."""
    if isinstance(records_or_path, str):
        records = read_appendix(records_or_path)
    else:
        records = records_or_path

    if len(records) <= 1:
        return True
    for i in range(1, len(records)):
        prev = records[i - 1]
        curr = records[i]
        expected_full = compute_parent_hash(prev.data, prev.name, prev.target)
        expected_payload = compute_parent_hash(prev.data)
        if curr.parent_hash != expected_full and curr.parent_hash != expected_payload:
            return False
    return True


class AppendixManager:
    """High-level object interface for managing container appendix evolution."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_records(self) -> List[AppendixRecord]:
        return read_appendix(self.file_path)

    def append_lora_checkpoint(
        self,
        name: str,
        target: str,
        generation: int,
        adapter_bytes: bytes,
        metrics: Optional[Dict[str, float]] = None,
    ):
        m = AppendixMetrics(
            loss=metrics.get("loss", 0.0) if metrics else 0.0,
            accuracy=metrics.get("accuracy", 0.0) if metrics else 0.0,
            pass_rate=metrics.get("pass_rate", 0.0) if metrics else 0.0,
            custom=metrics.get("custom", 0.0) if metrics else 0.0,
        )
        existing_records = self.get_records()
        parent_hash = b"\x00" * 32
        if existing_records:
            prev = existing_records[-1]
            parent_hash = compute_parent_hash(prev.data, prev.name, prev.target)

        rec = AppendixRecord(
            entry_type=AppendixEntryType.LORA_ADAPTER,
            name=name,
            generation=generation,
            data=adapter_bytes,
            target=target,
            metrics=m,
            parent_hash=parent_hash,
        )
        append_record(self.file_path, rec)

    def rollback(self, target_generation: int):
        rollback_appendix(self.file_path, target_generation)

    def verify(self) -> bool:
        return verify_lineage(self.get_records())
