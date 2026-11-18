"""Build an Ultra-Optimized patched copy of a PE (exe or dll):
AVX2, BMI2, FMA3, F16C, BMI1, LZCNT, MOVBE (and optional POPCNT) sites lowered,
with CALL/RET stub linkage, Multi-Instruction Basic-Block Fusion, forward and
backward neighbor stealing, a read-only constant pool in .fxcpt, and
cross-platform PE Checksum calculation.

Writes to STAGING and never touches the input file.
"""
from __future__ import annotations

import bisect
import collections
import ctypes
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK / 'pylibs'))
import pefile
import iced_x86 as ix
import bmi2_lower
import avx2_lower
import unwind_builder
import sandy_lower

GAME = Path('.')
STAGING = WORK / 'staging'
STAGING.mkdir(exist_ok=True)


# Intel-recommended multi-byte NOPs (one decoded instruction each).
NOPS = {
    1: bytes.fromhex('90'), 2: bytes.fromhex('6690'), 3: bytes.fromhex('0f1f00'),
    4: bytes.fromhex('0f1f4000'), 5: bytes.fromhex('0f1f440000'), 6: bytes.fromhex('660f1f440000'),
    7: bytes.fromhex('0f1f8000000000'), 8: bytes.fromhex('0f1f840000000000'),
    9: bytes.fromhex('660f1f840000000000'),
}


def align(n: int, a: int) -> int:
    return (n + a - 1) & ~(a - 1)


def compute_pe_checksum(data: bytearray, checksum_offset: int) -> int:
    """Compute the standard Windows PE CheckSum (identical to imagehlp.CheckSumMappedFile).
    Uses imagehlp on Windows when available for speed, with a pure-Python fallback."""
    try:
        calculate = ctypes.WinDLL('imagehlp').CheckSumMappedFile
        calculate.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)
        ]
        calculate.restype = ctypes.c_void_p
        if len(data) <= 0xFFFFFFFF:
            buffer = (ctypes.c_ubyte * len(data)).from_buffer(data)
            header = ctypes.c_uint32()
            checksum = ctypes.c_uint32()
            if calculate(buffer, len(data), ctypes.byref(header), ctypes.byref(checksum)):
                return checksum.value
    except Exception:
        pass

    # Pure-Python standard PE CheckSum (16-bit ones' complement sum + file length)
    struct.pack_into('<I', data, checksum_offset, 0)
    mv = memoryview(data)
    total = 0
    # Process in 64 KB chunks to avoid large temporary allocations
    chunk_size = 65536
    limit = len(data) & ~1
    for pos in range(0, limit, chunk_size):
        chunk = mv[pos:min(pos + chunk_size, limit)].cast('H')
        total += sum(chunk)
        total = (total & 0xFFFFFFFF) + (total >> 32)
    if len(data) & 1:
        total += data[-1]
    total = (total & 0xFFFF) + (total >> 16)
    total = (total & 0xFFFF) + (total >> 16)
    return (total + len(data)) & 0xFFFFFFFF


class Builder:
    def __init__(self, path, hits, patches=(), enabled_features=None,
                 enable_fusion: bool = True, enable_neighbor_steal: bool = True):
        self.patches = list(patches)
        self.path = Path(path)
        self.original = self.path.read_bytes()
        self.data = bytearray(self.original)
        self.pe = pefile.PE(data=self.original, fast_load=True)
        self.pe.parse_data_directories(directories=[1, 3, 5])
        self.base = self.pe.OPTIONAL_HEADER.ImageBase
        self.old_image = self.pe.OPTIONAL_HEADER.SizeOfImage
        self.iat_rva = align(self.old_image, self.pe.OPTIONAL_HEADER.SectionAlignment)
        self.iat_raw = align(len(self.original), self.pe.OPTIONAL_HEADER.FileAlignment)
        self.iat_payload = b''
        self.rva = self.iat_rva + self.pe.OPTIONAL_HEADER.SectionAlignment
        self.raw = self.iat_raw + self.pe.OPTIONAL_HEADER.FileAlignment
        self.payload = bytearray(32)
        self.hits = hits
        self.records = []
        self.redirects = []
        self.new_relocs = []
        self.skipped = []
        self.targets = None
        self.caves = None
        self.pdata_ranges = None
        self.pdata_starts = None
        self.unwind_entries = []
        self.unwind_blobs = {}
        self.unwind_refusals = []
        self.fmt = ix.Formatter(ix.FormatterSyntax.INTEL)
        self.enable_fusion = enable_fusion
        self.enable_neighbor_steal = enable_neighbor_steal

        # Configure which CPUID features to translate
        default_feats = {
            ix.CpuidFeature.AVX2, ix.CpuidFeature.BMI2, ix.CpuidFeature.BMI1,
            ix.CpuidFeature.FMA, ix.CpuidFeature.F16C, ix.CpuidFeature.LZCNT,
            ix.CpuidFeature.MOVBE,
        }
        if enabled_features is not None:
            feat_map = {k: v for k, v in vars(ix.CpuidFeature).items() if isinstance(v, int)}
            self.target_features = {feat_map[name] for name in enabled_features if name in feat_map}
        else:
            self.target_features = default_feats
        self.include_popcnt = ix.CpuidFeature.POPCNT in self.target_features
        self._info = ix.InstructionInfoFactory()
        self._boundary_cache = {}
        self.prolog_end = {}

        # Read-only constant pool at the start of .fxcpt (sign masks for FMA
        # negations). Referenced RIP-relative, so it needs no relocations.
        self.const_pool = {key: self.base + self.append(blob, 32)
                           for key, blob in sandy_lower.SIGN_POOL_BYTES.items()}
        sandy_lower.set_constant_pool(self.const_pool)

    def append(self, data: bytes, a: int = 1) -> int:
        pos = align(len(self.payload), a)
        self.payload.extend(b'\0' * (pos - len(self.payload)))
        self.payload.extend(data)
        return self.rva + pos

    def store(self, rva: int, data: bytes):
        if rva >= self.rva:
            self.payload[rva - self.rva : rva - self.rva + len(data)] = data
        else:
            off = self.pe.get_offset_from_rva(rva)
            self.data[off : off + len(data)] = data

    def imports(self) -> int:
        dllname = self.append(b'fxcompat.dll\0')
        names = [
            self.append(b'\0\0' + n + b'\0', 2)
            for n in [b'FxCompatImportAnchor', b'FxCompatRegisterModule']
        ]
        thunks = struct.pack('<QQQ', *names, 0)
        ilt = self.append(thunks, 8)
        self.iat_payload = thunks
        iat = self.iat_rva
        old = []
        imp = self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[1]
        for p in range(imp.VirtualAddress, imp.VirtualAddress + imp.Size, 20):
            desc = self.pe.get_data(p, 20)
            if desc == b'\0' * 20:
                break
            old.append(desc)
        table = b''.join(old) + struct.pack('<IIIII', ilt, 0, 0, dllname, iat) + b'\0' * 20
        table_rva = self.append(table, 8)
        imp.VirtualAddress = table_rva
        imp.Size = len(table)
        return iat

    def prepare_caves(self):
        if self.caves is not None:
            return
        cache_dir = WORK / 'code_cache'
        cache_dir.mkdir(exist_ok=True)
        digest = hashlib.sha256(self.original).hexdigest()
        cache_file = cache_dir / f'{digest}.padding.json'
        targets_file = cache_dir / f'{digest}.targets.json'

        entries = getattr(self.pe, 'DIRECTORY_ENTRY_EXCEPTION', [])
        ranges = sorted((e.struct.BeginAddress, e.struct.EndAddress) for e in entries)
        for e in entries:
            info = getattr(e, 'unwindinfo', None)
            if info is not None and hasattr(info, 'SizeOfProlog'):
                self.prolog_end[e.struct.BeginAddress] = e.struct.BeginAddress + info.SizeOfProlog
        starts = [s for s, _ in ranges]
        self.pdata_ranges = ranges
        self.pdata_starts = starts

        if cache_file.exists() and targets_file.exists():
            self.caves = json.loads(cache_file.read_text())
            self.targets = set(json.loads(targets_file.read_text()))
            self.used_caves = set()
            return

        print('Preparing safe padding & branch target inventory for', self.path.name, flush=True)
        targets = set(starts)
        for _, end_rva in ranges:
            targets.add(end_rva)
        candidates = []
        for sec in self.pe.sections:
            if not sec.Characteristics & 0x20000000:
                continue
            raw = sec.get_data()[:sec.Misc_VirtualSize]
            for ins in ix.Decoder(64, raw, ip=self.base + sec.VirtualAddress):
                if ins.op0_kind in (ix.OpKind.NEAR_BRANCH16, ix.OpKind.NEAR_BRANCH32, ix.OpKind.NEAR_BRANCH64):
                    targets.add(ins.near_branch_target - self.base)
                elif ins.mnemonic == ix.Mnemonic.LEA and ins.is_ip_rel_memory_operand:
                    t_rva = ins.ip_rel_memory_address - self.base
                    if 0 <= t_rva < self.old_image:
                        targets.add(t_rva)
            for match in re.finditer(b'(?:\xcc{5,}|\x90{5,})', raw):
                for pos in range(sec.VirtualAddress + match.start(), sec.VirtualAddress + match.end() - 4):
                    idx = bisect.bisect_right(starts, pos) - 1
                    if idx >= 0 and pos < ranges[idx][1]:
                        continue
                    if idx + 1 < len(ranges) and pos + 5 > ranges[idx + 1][0]:
                        continue
                    candidates.append(pos)
        for block in getattr(self.pe, 'DIRECTORY_ENTRY_BASERELOC', []):
            for entry in block.entries:
                if entry.type == 10:
                    target = int.from_bytes(self.pe.get_data(entry.rva, 8), 'little') - self.base
                    if 0 <= target < self.old_image:
                        targets.add(target)
        self.targets = targets
        if cache_file.exists():
            self.caves = json.loads(cache_file.read_text())
        else:
            self.caves = sorted(p for p in candidates if not any(p + i in targets for i in range(5)))
            cache_file.write_text(json.dumps(self.caves))
        try:
            targets_file.write_text(json.dumps(sorted(targets)))
        except Exception:
            pass
        self.used_caves = set()

    def cave(self, rva: int) -> int | None:
        self.prepare_caves()
        idx = bisect.bisect_left(self.caves, rva - 126)
        while idx < len(self.caves) and self.caves[idx] <= rva + 129:
            p = self.caves[idx]
            idx += 1
            if any(p + i in self.used_caves for i in range(5)):
                continue
            off = self.pe.get_offset_from_rva(p)
            if self.data[off : off + 5] not in (b'\xcc' * 5, b'\x90' * 5):
                continue
            self.used_caves.update(range(p, p + 5))
            return p
        return None

    def lower(self, ins: ix.Instruction, ip: int, adjust: int = 0) -> bytes | None:
        fs = ins.cpuid_features()
        if not any(f in self.target_features for f in fs):
            return None
        if ix.CpuidFeature.BMI2 in fs and ix.CpuidFeature.BMI2 in self.target_features:
            return bmi2_lower.lower(ins, ip, external_stack_adjust=adjust)
        if ix.CpuidFeature.AVX2 in fs and ix.CpuidFeature.AVX2 in self.target_features:
            return avx2_lower.lower(ins, ip, external_stack_adjust=adjust)
        if sandy_lower.supports_features(fs, include_popcnt=self.include_popcnt):
            return sandy_lower.lower(ins, ip, external_stack_adjust=adjust, include_popcnt=self.include_popcnt)
        return None

    def _validate_and_register_body(self, body: bytes, body_start_rva: int):
        decoder = ix.Decoder(64, body, ip=self.base + body_start_rva)
        for item in decoder:
            if item.is_invalid:
                raise RuntimeError('Invalid instruction in generated body')
            if any(f in self.target_features or f == ix.CpuidFeature.AVX512F for f in item.cpuid_features()):
                raise RuntimeError(f'Unsupported instruction remains in generated body: {item}')
            if item.code == ix.Code.MOV_R64_IMM64 and self.base <= item.immediate64 < self.base + self.old_image:
                const = decoder.get_constant_offsets(item)
                self.new_relocs.append((item.ip - self.base + const.immediate_offset, 10))
        description = unwind_builder.inspect_unwind_intervals(body, self.base + body_start_rva)
        if description['intervals'] is None:
            self.unwind_refusals.append({'stub_rva': hex(body_start_rva), 'reason': description['reason']})
        else:
            for interval in description['intervals']:
                blob = interval['unwind']
                if blob not in self.unwind_blobs:
                    self.unwind_blobs[blob] = self.append(blob, 4)
                self.unwind_entries.append((
                    body_start_rva + interval['begin_offset'],
                    body_start_rva + interval['end_offset'],
                    self.unwind_blobs[blob]
                ))

    # ------------------------------------------------------------------ stubs
    #
    # Ultra 3.0 stub linkage. A patched site now reads `call stub` and the stub
    # ends with a plain `ret`, so every stub exit is predicted by the CPU's
    # return-stack buffer. The 2.x linkage (`jmp stub` + `push [lit]` ... `ret`)
    # mispredicted on EVERY exit (~15-20 cycles each on Sandy Bridge / FX).
    # The pushed return address (site+5) lies inside the original function, so
    # stack walks and C++ exceptions inside a stub unwind exactly as before;
    # lowered bodies still see an 8-byte external stack adjustment.
    #
    # 'pushret' linkage is kept where a CALL cannot be used: VEH redirects (the
    # exception handler jumps to the stub) and padding caves (a return address
    # inside inter-function padding would break unwinding).

    def _body(self, items, body_ip: int) -> bytes | None:
        out = bytearray()
        for kind, ins in items:
            ip = body_ip + len(out)
            part = self.lower(ins, ip, 8) if kind == 'lower' else self.relocate(ins, ip, 8)
            if part is None:
                return None
            out += part
        return bytes(out)

    def _check_body(self, body: bytes, body_ip: int):
        for item in ix.Decoder(64, body, ip=body_ip):
            if item.is_invalid:
                raise RuntimeError('Invalid instruction in generated body')
            if any(f in self.target_features or f == ix.CpuidFeature.AVX512F for f in item.cpuid_features()):
                raise RuntimeError(f'Unsupported instruction remains in generated body: {item}')

    def make_stub(self, items, next_ip: int, linkage: str = 'call') -> int | None:
        """Create a stub in .fxcpt. items: [('lower', ins) | ('reloc', ins), ...]."""
        start = self.rva + align(len(self.payload), 16)
        prefix = 0 if linkage == 'call' else 6
        body = self._body(items, self.base + start + prefix)
        if body is None:
            return None
        self._check_body(body, self.base + start + prefix)
        if linkage == 'call':
            code = body + b'\xc3'
        else:
            literal = start + 6 + len(body) + 1
            code = b'\xff\x35' + struct.pack('<i', literal - (start + 6)) + body + b'\xc3' + struct.pack('<Q', next_ip)
        actual = self.append(code, 16)
        assert actual == start
        if linkage != 'call':
            self.new_relocs.append((literal, 10))
        self._validate_and_register_body(body, start + prefix)
        return start

    def stack_neutral(self, body: bytes, ip: int) -> bool:
        for ins in ix.Decoder(64, body, ip=ip):
            if ins.stack_pointer_increment != 0:
                return False
            if ins.op0_kind == ix.OpKind.REGISTER and ins.op0_register in (ix.Register.RSP, ix.Register.ESP):
                return False
        return True

    # ------------------------------------------------------- neighbor moving

    _RSP_REGS = (ix.Register.RSP, ix.Register.ESP, ix.Register.SP, ix.Register.SPL)

    def _is_safe_neighbor(self, neighbor: ix.Instruction) -> bool:
        """An ordinary instruction that can run inside a stub after re-encoding.

        RIP-relative operands are re-targeted by the encoder; [rsp+disp] operands
        get +8 for the stub's return address. Anything else touching RSP, any
        control flow, and anything we translate ourselves is refused."""
        if neighbor.is_invalid or neighbor.len <= 0:
            return False
        if neighbor.flow_control != ix.FlowControl.NEXT:
            return False
        if neighbor.encoding in (ix.EncodingKind.EVEX, ix.EncodingKind.MVEX):
            return False
        if any(f in self.target_features or f == ix.CpuidFeature.AVX512F for f in neighbor.cpuid_features()):
            return False
        if neighbor.stack_pointer_increment != 0:
            return False
        for idx in range(neighbor.op_count):
            if neighbor.op_kind(idx) == ix.OpKind.REGISTER and neighbor.op_register(idx) in self._RSP_REGS:
                return False
        if neighbor.memory_index in self._RSP_REGS:
            return False
        rsp_base = neighbor.memory_base in self._RSP_REGS
        if rsp_base:
            if neighbor.memory_base != ix.Register.RSP or neighbor.memory_displacement >= 1 << 63:
                return False   # no 32-bit stack addressing, nothing below RSP
        reads = 0
        for used in self._info.info(neighbor).used_registers():
            if used.register in self._RSP_REGS:
                if used.access != ix.OpAccess.READ:
                    return False
                reads += 1
        return reads == (1 if rsp_base else 0)

    def relocate(self, ins: ix.Instruction, new_ip: int, stack_adjust: int) -> bytes | None:
        copy = ins.copy()
        if copy.memory_base == ix.Register.RSP and stack_adjust:
            disp = copy.memory_displacement + stack_adjust
            if disp >= 1 << 31:
                return None
            copy.memory_displacement = disp
            copy.memory_displ_size = 1 if disp <= 127 else 8
        encoder = ix.BlockEncoder(64)
        encoder.add(copy)
        try:
            return encoder.encode(new_ip)
        except ValueError:
            return None

    def _in_same_pdata_function(self, rva1: int, rva2_end: int) -> bool:
        self.prepare_caves()
        idx = bisect.bisect_right(self.pdata_starts, rva1) - 1
        if idx < 0:
            return False
        f_start, f_end = self.pdata_ranges[idx]
        return f_start <= rva1 and rva2_end <= f_end

    def _function_of(self, rva: int):
        idx = bisect.bisect_right(self.pdata_starts, rva) - 1
        if idx < 0:
            return None
        f_start, f_end = self.pdata_ranges[idx]
        return (f_start, f_end) if f_start <= rva < f_end else None

    def _previous_instruction(self, rva: int) -> ix.Instruction | None:
        """Instruction ending exactly at rva, by linear decode of its .pdata range."""
        func = self._function_of(rva)
        if func is None:
            return None
        f_start, f_end = func
        starts = self._boundary_cache.get(f_start)
        if starts is None:
            if f_end - f_start > 1 << 20:
                return None
            raw = self.pe.get_data(f_start, f_end - f_start)
            starts = [ins.ip - self.base for ins in ix.Decoder(64, raw, ip=self.base + f_start)]
            if len(self._boundary_cache) > 4096:
                self._boundary_cache.clear()
            self._boundary_cache[f_start] = starts
        idx = bisect.bisect_left(starts, rva)
        if idx >= len(starts) or starts[idx] != rva or idx == 0:
            return None
        p_rva = starts[idx - 1]
        if p_rva < self.prolog_end.get(f_start, f_start + 0xFF):
            return None
        raw = self.pe.get_data(p_rva, 15)
        prev = ix.Decoder(64, raw, ip=self.base + p_rva).decode()
        return prev if prev.next_ip == self.base + rva else None

    # --------------------------------------------------------------- patches

    @staticmethod
    def filler(n: int) -> bytes:
        """Bytes after a `call` that bring execution to the next instruction:
        one multi-byte NOP for short gaps, a short jump for long ones."""
        if n <= 0:
            return b''
        if n <= 9:
            return NOPS[n]
        if n - 2 <= 127:
            return b'\xeb' + bytes([n - 2]) + b'\x90' * (n - 2)
        return b'\xe9' + struct.pack('<i', n - 5) + b'\x90' * (n - 5)

    @staticmethod
    def nop_fill(n: int) -> bytes:
        out = b''
        while n > 0:
            k = min(n, 9)
            out += NOPS[k]
            n -= k
        return out

    def call_patch(self, site_rva: int, stub: int, length: int) -> bytes:
        return b'\xe8' + struct.pack('<i', stub - site_rva - 5) + self.filler(length - 5)

    def _record(self, rva, ins, orig, patch, stub, cave, method):
        self.records.append({
            'rva': hex(rva), 'asm': self.fmt.format(ins),
            'original': orig.hex(), 'patch': patch.hex(),
            'stub_rva': hex(stub) if stub is not None else None,
            'cave_rva': hex(cave) if cave is not None else None, 'method': method,
        })

    def instructions(self):
        self.prepare_caves()
        seen = set()
        decoded_sites = []
        patch_ranges = [(p['rva'], p['rva'] + len(bytes.fromhex(p['old']))) for p in self.patches]

        def overlaps_patch(rva_start: int, rva_end: int) -> bool:
            return any(rva_start < pe and pb < rva_end for pb, pe in patch_ranges)

        def targets_in(lo: int, hi: int) -> bool:
            return any(t in self.targets for t in range(lo, hi))

        for h in sorted(self.hits, key=lambda h: int(h['rva'], 16)):
            rva = int(h['rva'], 16)
            if rva in seen:
                continue
            seen.add(rva)
            raw = self.pe.get_data(rva, 15)
            ins = ix.Decoder(64, raw, ip=self.base + rva).decode()
            if ins.is_invalid:
                raise RuntimeError('Invalid original instruction')
            if not any(f in self.target_features for f in ins.cpuid_features()):
                continue
            if 'bytes' in h and raw[:ins.len].hex() != h['bytes']:
                raise RuntimeError('Input inventory bytes differ')
            decoded_sites.append((rva, ins, raw[:ins.len]))

        site_rvas = {rva for rva, _, _ in decoded_sites}
        consumed_rvas = set()
        high_water = 0          # every original byte below this may already be patched

        # Pass 1: Check which sites can be lowered IN-PLACE (zero jump overhead!)
        inplace_ready = {}
        for rva, ins, orig_bytes in decoded_sites:
            try:
                body = self.lower(ins, ins.ip, adjust=0)
            except Exception:
                body = None
            if body is not None and len(body) <= ins.len and self.stack_neutral(body, ins.ip):
                inplace_ready[rva] = body + self.nop_fill(ins.len - len(body))

        def needs_help(k: int) -> bool:
            """Site k is a short (<5 byte) non-in-place site that fusion could rescue."""
            return (0 <= k < n_sites
                    and decoded_sites[k][1].len < 5
                    and decoded_sites[k][0] not in inplace_ready
                    and decoded_sites[k][0] not in consumed_rvas)

        # Pass 2: Group remaining sites into contiguous basic-block clusters when fusion is enabled
        i = 0
        n_sites = len(decoded_sites)
        while i < n_sites:
            rva, ins, orig_bytes = decoded_sites[i]

            # Always prefer pure IN-PLACE replacement unless an adjacent short site needs this one to reach 5 bytes.
            if rva in inplace_ready:
                prev_needs_help = (
                    self.enable_fusion and needs_help(i - 1)
                    and decoded_sites[i - 1][0] + decoded_sites[i - 1][1].len == rva
                    and rva not in self.targets
                )
                next_needs_help = (
                    self.enable_fusion and needs_help(i + 1)
                    and rva + ins.len == decoded_sites[i + 1][0]
                    and decoded_sites[i + 1][0] not in self.targets
                )
                if not prev_needs_help and not next_needs_help:
                    patch = inplace_ready[rva]
                    self.store(rva, patch)
                    consumed_rvas.add(rva)
                    high_water = max(high_water, rva + ins.len)
                    self._record(rva, ins, orig_bytes, patch, None, None, 'in_place')
                    i += 1
                    continue

            # Build a contiguous cluster of instructions if fusion is enabled
            cluster = [(rva, ins, orig_bytes)]
            if self.enable_fusion:
                j = i + 1
                while j < n_sites and len(cluster) < 16:
                    prev_rva, prev_ins, _ = cluster[-1]
                    nxt_rva, nxt_ins, nxt_orig = decoded_sites[j]
                    if prev_rva + prev_ins.len != nxt_rva:
                        break
                    # Never fuse across a known branch/relocation/pdata target boundary!
                    if nxt_rva in self.targets:
                        break
                    # Stop once the cluster can hold a call if the next site can stay in place.
                    cluster_bytes = sum(x[1].len for x in cluster)
                    if cluster_bytes >= 5 and nxt_rva in inplace_ready:
                        after_needs_help = (
                            needs_help(j + 1)
                            and nxt_rva + nxt_ins.len == decoded_sites[j + 1][0]
                            and decoded_sites[j + 1][0] not in self.targets
                        )
                        if not after_needs_help:
                            break
                    cluster.append((nxt_rva, nxt_ins, nxt_orig))
                    j += 1

            total_cluster_len = sum(x[1].len for x in cluster)

            # Case 1: Multi-instruction cluster -> one call into one fused stub
            if len(cluster) > 1 and total_cluster_len >= 5:
                try:
                    stub = self.make_stub([('lower', x[1]) for x in cluster], cluster[-1][1].next_ip)
                except Exception:
                    stub = None
                if stub is not None:
                    first_rva = cluster[0][0]
                    patch = self.call_patch(first_rva, stub, total_cluster_len)
                    self.store(first_rva, patch)
                    high_water = max(high_water, first_rva + total_cluster_len)
                    for idx_c, (c_rva, c_ins, c_orig) in enumerate(cluster):
                        consumed_rvas.add(c_rva)
                        rel_off = c_rva - first_rva
                        self._record(c_rva, c_ins, c_orig, patch[rel_off: rel_off + c_ins.len], stub, None,
                                     'fused_block_call' if idx_c == 0 else 'fused_block_member')
                    i += len(cluster)
                    continue

            # Single instruction handling (if cluster has 1 item or multi-stub failed)
            try:
                body = self.lower(ins, ins.ip)
            except Exception as err:
                self.skipped.append({'rva': hex(rva), 'asm': str(ins), 'error': str(err)})
                i += 1
                continue
            if body is None:
                self.skipped.append({'rva': hex(rva), 'asm': str(ins), 'error': 'unsupported'})
                i += 1
                continue

            if len(body) <= ins.len and self.stack_neutral(body, ins.ip):
                patch = body + self.nop_fill(ins.len - len(body))
                self.store(rva, patch)
                consumed_rvas.add(rva)
                high_water = max(high_water, rva + ins.len)
                self._record(rva, ins, orig_bytes, patch, None, None, 'in_place')
                i += 1
                continue

            if ins.len >= 5:
                try:
                    stub = self.make_stub([('lower', ins)], ins.next_ip)
                except Exception as err:
                    self.skipped.append({'rva': hex(rva), 'asm': str(ins), 'error': str(err)})
                    i += 1
                    continue
                if stub is None:
                    i += 1
                    continue
                patch = self.call_patch(rva, stub, ins.len)
                self.store(rva, patch)
                consumed_rvas.add(rva)
                high_water = max(high_water, rva + ins.len)
                self._record(rva, ins, orig_bytes, patch, stub, None, 'native_call')
                i += 1
                continue

            # Instruction is < 5 bytes (4-byte VEX instruction).
            # 1) Forward neighbor stealing: move the next ordinary instruction into the stub.
            if self.enable_neighbor_steal and self._steal(rva, ins, orig_bytes, site_rvas, overlaps_patch,
                                                          targets_in, forward=True, high_water=high_water):
                consumed_rvas.add(rva)
                high_water = max(high_water, int(self.records[-1]['_end'], 16))
                del self.records[-1]['_end']
                i += 1
                continue
            # 2) Backward neighbor stealing: move the previous ordinary instruction into the stub.
            if self.enable_neighbor_steal and self._steal(rva, ins, orig_bytes, site_rvas, overlaps_patch,
                                                          targets_in, forward=False, high_water=high_water):
                consumed_rvas.add(rva)
                high_water = max(high_water, int(self.records[-1]['_end'], 16))
                del self.records[-1]['_end']
                i += 1
                continue

            # 3) Padding cave: short jump to a 5-byte jump in nearby inter-function padding.
            cave = self.cave(rva)
            if cave is not None:
                try:
                    stub = self.make_stub([('lower', ins)], ins.next_ip, linkage='pushret')
                except Exception as err:
                    self.skipped.append({'rva': hex(rva), 'asm': str(ins), 'error': str(err)})
                    i += 1
                    continue
                if stub is not None:
                    self.store(cave, b'\xe9' + struct.pack('<i', stub - cave - 5))
                    patch = b'\xeb' + struct.pack('<b', cave - rva - 2) + self.nop_fill(ins.len - 2)
                    self.store(rva, patch)
                    consumed_rvas.add(rva)
                    high_water = max(high_water, rva + ins.len)
                    self._record(rva, ins, orig_bytes, patch, stub, cave, 'padding_jump')
                    i += 1
                    continue

            # 4) Fallback: Vectored Exception Handler redirect via fxcompat.dll
            try:
                stub = self.make_stub([('lower', ins)], ins.next_ip, linkage='pushret')
            except Exception as err:
                self.skipped.append({'rva': hex(rva), 'asm': str(ins), 'error': str(err)})
                i += 1
                continue
            if stub is None:
                i += 1
                continue
            patch = orig_bytes
            self.redirects.append((rva, stub))
            self.store(rva, patch)
            consumed_rvas.add(rva)
            self._record(rva, ins, orig_bytes, patch, stub, None, 'native_exception_redirect')
            i += 1

        print(
            'Translated', len(self.records), 'skipped', len(self.skipped),
            dict(collections.Counter(r['method'] for r in self.records)), flush=True
        )

    def _steal(self, rva, ins, orig_bytes, site_rvas, overlaps_patch, targets_in, forward, high_water) -> bool:
        if forward:
            nb_rva = rva + ins.len
            if nb_rva in site_rvas:
                return False
            nb = ix.Decoder(64, self.pe.get_data(nb_rva, 15), ip=self.base + nb_rva).decode()
            first, end = rva, nb_rva + nb.len
            # the site's own bytes stay addressable only at rva; nothing may jump into the rest
            blocked = targets_in(rva + 1, end)
            items = [('lower', ins), ('reloc', nb)]
        else:
            nb = self._previous_instruction(rva)
            if nb is None:
                return False
            nb_rva = nb.ip - self.base
            if nb_rva in site_rvas or nb_rva < high_water:
                return False
            first, end = nb_rva, ins.next_ip - self.base
            blocked = targets_in(nb_rva + 1, end)
            items = [('reloc', nb), ('lower', ins)]
        if blocked or not self._is_safe_neighbor(nb):
            return False
        if not self._in_same_pdata_function(first, end) or overlaps_patch(first, end):
            return False
        if end - first < 5:
            return False
        try:
            stub = self.make_stub(items, self.base + end)
        except Exception:
            stub = None
        if stub is None:
            return False
        patch = self.call_patch(first, stub, end - first)
        self.store(first, patch)
        self._record(rva, ins, orig_bytes, patch[rva - first: rva - first + ins.len], stub, None,
                     'neighbor_steal_call' if forward else 'neighbor_steal_back_call')
        self.records[-1]['neighbor'] = {'rva': hex(nb_rva), 'asm': self.fmt.format(nb)}
        self.records[-1]['_end'] = hex(end)
        return True

    def aux_entry(self, iat: int):
        old = self.pe.OPTIONAL_HEADER.AddressOfEntryPoint
        start = self.rva + align(len(self.payload), 16)
        code = bytearray(bytes.fromhex('83fa01 7500'))
        code += bytes.fromhex('50 51 52 4150 4151 4152 4153 9c 4883ec28')
        callpos = start + len(code)
        code += b'\xff\x15' + struct.pack('<i', iat + 8 - callpos - 6)
        code += bytes.fromhex('85c0 7500')
        failure_branch = len(code) - 1
        restore = bytes.fromhex('4883c428 9d 415b 415a 4159 4158 5a 59 58')
        code += restore
        original_jmp = len(code)
        code += b'\xe9' + struct.pack('<i', old - (start + len(code) + 5))
        fail = len(code)
        code += restore + bytes.fromhex('31c0 c3')
        code[4] = original_jmp - 5
        code[failure_branch] = fail - (failure_branch + 1)
        self.append(code, 16)
        self.pe.OPTIONAL_HEADER.AddressOfEntryPoint = start

    def finish(self):
        iat = self.imports()
        self.instructions()
        for patch in self.patches:
            rva = patch['rva']
            old = bytes.fromhex(patch['old'])
            new = bytes.fromhex(patch['new'])
            if self.pe.get_data(rva, len(old)) != old:
                raise RuntimeError(f"Patch {patch['name']}: bytes changed")
            if any(int(r['rva'], 16) < rva + len(old) and rva < int(r['rva'], 16) + len(r['original']) // 2 for r in self.records):
                raise RuntimeError(f"Patch {patch['name']} overlaps a lowered instruction")
            self.store(rva, new)
        if self.pe.FILE_HEADER.Characteristics & 0x2000:
            self.aux_entry(iat)
        entries = self.append(b''.join(struct.pack('<II', *x) for x in self.redirects), 8)
        self.payload[:32] = struct.pack('<8sIIIIII', b'FXCPT001', 1, len(self.redirects), entries, 0, self.old_image, 0)
        if self.unwind_entries:
            directory = self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[3]
            old_table = self.pe.get_data(directory.VirtualAddress, directory.Size)
            new_table = old_table + b''.join(struct.pack('<III', *row) for row in sorted(self.unwind_entries))
            directory.VirtualAddress = self.append(new_table, 4)
            directory.Size = len(new_table)
        relocations = collections.defaultdict(set)
        for block in getattr(self.pe, 'DIRECTORY_ENTRY_BASERELOC', []):
            for e in block.entries:
                if e.type:
                    relocations[e.rva & ~4095].add((e.type << 12) | (e.rva & 4095))
        for rva, kind in self.new_relocs:
            relocations[rva & ~4095].add((kind << 12) | (rva & 4095))
        reloc = bytearray()
        for page, items in sorted(relocations.items()):
            rows = sorted(items)
            if len(rows) & 1:
                rows.append(0)
            reloc += struct.pack('<II', page, 8 + len(rows) * 2) + struct.pack('<' + 'H' * len(rows), *rows)
        reloc_rva = self.append(reloc, 4)
        self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[5].VirtualAddress = reloc_rva
        self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[5].Size = len(reloc)
        self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[4].VirtualAddress = 0
        self.pe.OPTIONAL_HEADER.DATA_DIRECTORY[4].Size = 0
        self.pe.OPTIONAL_HEADER.SizeOfImage = align(self.rva + len(self.payload), self.pe.OPTIONAL_HEADER.SectionAlignment)
        self.pe.OPTIONAL_HEADER.SizeOfCode += align(len(self.payload), self.pe.OPTIONAL_HEADER.FileAlignment)
        self.pe.OPTIONAL_HEADER.SizeOfInitializedData += self.pe.OPTIONAL_HEADER.FileAlignment
        self.pe.OPTIONAL_HEADER.CheckSum = 0
        header = self.pe.sections[-1].get_file_offset() + 40
        if header + 80 > self.pe.OPTIONAL_HEADER.SizeOfHeaders:
            raise RuntimeError('No room for added section headers')
        self.pe.FILE_HEADER.NumberOfSections += 2
        headers = self.pe.write()[:self.pe.OPTIONAL_HEADER.SizeOfHeaders]
        self.data[:len(headers)] = headers
        rawsize = align(len(self.payload), self.pe.OPTIONAL_HEADER.FileAlignment)
        iat_section = struct.pack('<8sIIIIIIHHI', b'.fxiat\0\0', len(self.iat_payload), self.iat_rva, self.pe.OPTIONAL_HEADER.FileAlignment, self.iat_raw, 0, 0, 0, 0, 0xc0000040)
        section = struct.pack('<8sIIIIIIHHI', b'.fxcpt\0\0', len(self.payload), self.rva, rawsize, self.raw, 0, 0, 0, 0, 0x60000020)
        self.data[header : header + 80] = iat_section + section
        self.data.extend(b'\0' * (self.iat_raw - len(self.data)))
        self.data.extend(self.iat_payload)
        self.data.extend(b'\0' * (self.raw - len(self.data)))
        self.data.extend(self.payload)
        self.data.extend(b'\0' * (rawsize - len(self.payload)))
        chk_offset = self.pe.OPTIONAL_HEADER.get_field_absolute_offset('CheckSum')
        checksum = compute_pe_checksum(self.data, chk_offset)
        struct.pack_into('<I', self.data, chk_offset, checksum)
        output = STAGING / self.path.name
        temporary = STAGING / (self.path.name + '.building')
        temporary.write_bytes(self.data)
        temporary.replace(output)
        report = {
            'source': str(self.path),
            'source_sha256': hashlib.sha256(self.original).hexdigest(),
            'output': str(output),
            'sha256': hashlib.sha256(self.data).hexdigest(),
            'section_rva': hex(self.rva),
            'section_size': len(self.payload),
            'redirects': len(self.redirects),
            'methods': dict(collections.Counter(r['method'] for r in self.records)),
            'translated': len(self.records),
            'skipped': self.skipped,
            'records': self.records,
            'unwind_intervals': len(self.unwind_entries),
            'patches': [{**q, 'rva': hex(q['rva'])} for q in self.patches],
            'unwind_refusals': self.unwind_refusals,
            'engine': 'FX Reborn Toolkit Ultra 3.0',
            'optimizations': {
                'call_ret_stub_linkage': True,
                'basic_block_fusion': self.enable_fusion,
                'neighbor_stealing_forward': self.enable_neighbor_steal,
                'neighbor_stealing_backward': self.enable_neighbor_steal,
                'constant_pool': True,
                'fma_exact_zero_scratch': True,
                'lane_split_dst_scratch': avx2_lower.LANES_USE_DST,
                'f16c_simd_pooled': True,
                'andn_in_place': True,
            },
            'limitations': [
                'FMA3 is emulated as a rounded multiply followed by a rounded add/sub (two roundings)',
                'VEH and padding-cave stubs use a synthetic PUSH return address (call-linked stubs unwind through the original function)',
                'Fault-address translation and partial-instruction fault recovery are not implemented',
                'Further runtime coverage and gameplay testing required'
            ]
        }
        (STAGING / (self.path.name + '.manifest.json')).write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'skipped')}, indent=2), flush=True)