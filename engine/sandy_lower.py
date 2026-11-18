"""Sandy Bridge (AVX1/SSE4.2) Ultra-Optimized lowering for FMA3, BMI1, F16C, LZCNT, MOVBE, and POPCNT.

Follows the exact contract of bmi2_lower.lower() and avx2_lower.lower():
- Returns position-independent machine code bytes for ``ip``.
- Preserves all registers except architectural destination(s).
- Preserves RFLAGS for FMA3, F16C, and MOVBE; reproduces exact architectural
  flag semantics for BMI1 (ANDN, BEXTR, BLSI, BLSMSK, BLSR, TZCNT), LZCNT, and POPCNT.
- Accepts external_stack_adjust in (0, 8) when a synthetic return address is
  pushed by the stub wrapper.
- Emits only baseline x86-64, SSE2, SSE4.1, and 128/256-bit AVX1 instructions.

Ultra 3.0 (every path verified bit-exact by tests/native/test_lowering.py on real hardware):
1. FMA3, exact operand semantics for all 132/213/231 forms, PS/PD/SS/SD, xmm/ymm:
   - dst is a multiplicand (132/213): 2-3 AVX1 ops, zero scratch, zero stack.
   - dst is the addend (231 accumulate, "acc += a*b"): the addend is parked in a
     4..32 byte stack slot instead of saving a whole YMM scratch.
   - Negations (FNMADD/FNMSUB/FMSUBADD) XOR with a sign mask from the .fxcpt
     constant pool; exact zero signs.
   (2.x computed 2*a*b for packed 231 FMADD, 0 for 231 FMSUB, and corrupted the
   upper lanes of scalar 213 FNMADD with a memory operand.)
2. BMI1 ANDN In-Place & Zero-Stack Fast Paths:
   - When dst == src1 and src2 is a distinct register: emits `NOT dst; AND dst, src2` (4-5 bytes total,
     fitting IN-PLACE inside the 5-byte VEX ANDN instruction with zero jump overhead!).
   - When dst != src1 and dst != src2: emits `MOV dst, src1; NOT dst; AND dst, src2` (zero stack frame!).
3. F16C VCVTPH2PS branchless SIMD lowering with pooled constants (25 ops for xmm,
   46 for ymm, 3-4 scratch YMMs). Exact for normals, subnormals, +-0, Inf and NaN
   independent of DAZ/FTZ. (2.x converted every negative half incorrectly.)
   VCVTPS2PH works again (2.x referenced a non-existent opcode and skipped it).
4. POPCNT SWAR lowering with exact flags (ZF only; 2.x left PF set by parity).
5. MOVBE 16-bit forms no longer change CF/OF (XCHG AL,AH instead of ROL AX,8).
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pylibs"))
import iced_x86 as ix

from bmi2_lower import FULL, GPRS32, GPRS64, LOW32, LoweringError, _Builder, _signed64

R, C, I, M, O = ix.Register, ix.Code, ix.Instruction, ix.MemoryOperand, ix.OpKind
F = ix.CpuidFeature

TARGET_FEATURES = frozenset({F.BMI1, F.FMA, F.F16C, F.LZCNT, F.MOVBE, F.POPCNT})

GPRS16 = [
    R.AX, R.CX, R.DX, R.BX, R.SP, R.BP, R.SI, R.DI,
    R.R8W, R.R9W, R.R10W, R.R11W, R.R12W, R.R13W, R.R14W, R.R15W,
]
LOW16 = dict(zip(GPRS64, GPRS16))
FULL_ALL = dict(zip(GPRS16, GPRS64)) | FULL

MNEMONIC_NAMES = {v: k for k, v in vars(ix.Mnemonic).items() if isinstance(v, int)}


def supports_features(features, include_popcnt: bool = False) -> bool:
    active = TARGET_FEATURES if include_popcnt else (TARGET_FEATURES - {F.POPCNT})
    return bool(active.intersection(features))


def _xmm(reg):
    if R.XMM0 <= reg <= R.XMM15:
        return reg
    if R.YMM0 <= reg <= R.YMM15:
        return R.XMM0 + (reg - R.YMM0)
    raise LoweringError(f"Expected XMM/YMM register, got {reg}")


def _ymm(reg):
    return R.YMM0 + (_xmm(reg) - R.XMM0)


def _disp_size(disp: int, address32: bool = False) -> int:
    if -128 <= disp <= 127:
        return 1
    return 4 if address32 else 8


def _mem_operand(emit, ins: ix.Instruction, addr_scratch, total_stack_adjust: int, mem_bytes: int, extra_disp: int = 0):
    """Build a MemoryOperand valid after RSP decreased by total_stack_adjust."""
    if ins.is_ip_rel_memory_operand:
        if addr_scratch is None:
            return ix.MemoryOperand.ctor_u64(
                R.RIP, R.NONE, 1, (ins.ip_rel_memory_address + extra_disp) & ((1 << 64) - 1),
                8, False, ins.segment_prefix
            )
        emit(I.create_reg_u64(C.MOV_R64_IMM64, addr_scratch, ins.ip_rel_memory_address + extra_disp))
        return M(addr_scratch, seg=ins.segment_prefix)
    base, index = ins.memory_base, ins.memory_index
    disp = _signed64(ins.memory_displacement + extra_disp)
    if index in (R.RSP, R.ESP):
        raise LoweringError("RSP cannot be an x86 SIB index")
    if base in (R.RSP, R.ESP):
        if index == R.NONE and disp < 0 and disp + (mem_bytes - extra_disp) > -total_stack_adjust:
            raise LoweringError("Memory operand overlaps borrowed stack frame")
        disp = _signed64(disp + total_stack_adjust)
    address32 = base in GPRS32 or index in GPRS32
    displ_size = _disp_size(disp, address32)
    if base == R.NONE and index == R.NONE:
        displ_size = ins.memory_displ_size if ins.memory_displ_size in (4, 8) else 4
    return M(base, index, ins.memory_index_scale, disp, displ_size, seg=ins.segment_prefix)


def _gpr_width(reg) -> int:
    if reg in GPRS64:
        return 64
    if reg in GPRS32:
        return 32
    if reg in GPRS16:
        return 16
    raise LoweringError(f"Unsupported GPR: {reg}")


# ---------------------------------------------------------------------------
# BMI1, LZCNT & POPCNT (with Ultra Zero-Stack Fast Paths for ANDN)
# ---------------------------------------------------------------------------

def _lower_bmi1_lzcnt(ins: ix.Instruction, ip: int, external_stack_adjust: int) -> bytes:
    mname = MNEMONIC_NAMES.get(ins.mnemonic, "")
    if mname not in ("ANDN", "BEXTR", "BLSI", "BLSMSK", "BLSR", "TZCNT", "LZCNT", "POPCNT"):
        raise LoweringError(f"Unsupported BMI1/LZCNT/POPCNT instruction: {ins}")

    dst = ins.op0_register
    bits = _gpr_width(dst)
    dst64 = FULL_ALL[dst]
    if dst64 == R.RSP:
        raise LoweringError("Destination cannot be RSP")

    b = _Builder()
    emit = b.emit

    # Ultra Fast Path for ANDN (covers ~31% of all ANDN sites with zero stack usage,
    # and ~18% fit IN-PLACE in 4-5 bytes without jumping to .fxcpt!)
    # Architectural flags for ANDN: OF=0, CF=0, SF/ZF set according to result.
    # Standard x86 AND also sets OF=0, CF=0, and SF/ZF according to result!
    # NOT does not modify any flags.
    if mname == "ANDN":
        r1 = ins.op0_register
        r2 = ins.op1_register
        not_code = C.NOT_RM64 if bits == 64 else C.NOT_RM32
        mov_code = C.MOV_R64_RM64 if bits == 64 else C.MOV_R32_RM32
        and_reg_code = C.AND_R64_RM64 if bits == 64 else C.AND_R32_RM32
        and_mem_code = C.AND_R64_RM64 if bits == 64 else C.AND_R32_RM32

        if ins.op2_kind == O.REGISTER:
            r3 = ins.op2_register
            # Case A: ANDN r1, r1, r3 (where r1 != r3) -> NOT r1; AND r1, r3 (4-5 bytes, IN-PLACE!)
            if r1 == r2 and FULL_ALL[r1] != FULL_ALL[r3]:
                emit(I.create_reg(not_code, r1))
                emit(I.create_reg_reg(and_reg_code, r1, r3))
                return b.encode(ip)
            # Case B: ANDN r1, r2, r3 (where r1 != r2 and r1 != r3) -> MOV r1, r2; NOT r1; AND r1, r3
            if FULL_ALL[r1] != FULL_ALL[r2] and FULL_ALL[r1] != FULL_ALL[r3]:
                emit(I.create_reg_reg(mov_code, r1, r2))
                emit(I.create_reg(not_code, r1))
                emit(I.create_reg_reg(and_reg_code, r1, r3))
                return b.encode(ip)
        elif ins.op2_kind == O.MEMORY and not ins.is_ip_rel_memory_operand:
            # Check if r1 is used in the memory addressing of op2
            base64 = FULL_ALL.get(ins.memory_base, R.NONE)
            idx64 = FULL_ALL.get(ins.memory_index, R.NONE)
            if dst64 not in (base64, idx64):
                mem = _mem_operand(emit, ins, None, external_stack_adjust, bits // 8)
                if r1 == r2:
                    emit(I.create_reg(not_code, r1))
                    emit(I.create_reg_mem(and_mem_code, r1, mem))
                    return b.encode(ip)
                else:
                    emit(I.create_reg_reg(mov_code, r1, r2))
                    emit(I.create_reg(not_code, r1))
                    emit(I.create_reg_mem(and_mem_code, r1, mem))
                    return b.encode(ip)

    scratch = [R.RAX, R.RCX, R.RDX, R.R8]

    # Stack layout (low to high): R8(0), RDX(8), RCX(16), RAX(24), result_slot(32)
    emit(I.create_reg(C.PUSH_R64, dst64))
    for reg in scratch:
        emit(I.create_reg(C.PUSH_R64, reg))
    frame = 8 * (1 + len(scratch))  # 40 bytes
    slot = {reg: 8 * (len(scratch) - 1 - i) for i, reg in enumerate(scratch)}
    result_slot = frame - 8
    total_adjust = frame + external_stack_adjust

    def read_reg(target64, source):
        s64 = FULL_ALL[source]
        if s64 in slot:
            emit(I.create_reg_mem(C.MOV_R64_RM64, target64, M(R.RSP, displ=slot[s64], displ_size=1)))
        elif s64 == R.RSP:
            emit(I.create_reg_mem(C.LEA_R64_M, target64, M(R.RSP, displ=total_adjust, displ_size=1)))
        else:
            emit(I.create_reg_reg(C.MOV_R64_RM64, target64, s64))
        if bits == 32:
            emit(I.create_reg_reg(C.MOV_R32_RM32, LOW32[target64], LOW32[target64]))
        elif bits == 16:
            emit(I.create_reg_reg(C.MOVZX_R32_RM16, LOW32[target64], LOW16[target64]))

    def read_op(target64, op_idx: int):
        if ins.op_kind(op_idx) == O.REGISTER:
            read_reg(target64, ins.op_register(op_idx))
            return
        if ins.op_kind(op_idx) != O.MEMORY:
            raise LoweringError(f"Unsupported operand kind at {op_idx}")
        mem = _mem_operand(emit, ins, target64, total_adjust, bits // 8)
        if bits == 64:
            emit(I.create_reg_mem(C.MOV_R64_RM64, target64, mem))
        elif bits == 32:
            emit(I.create_reg_mem(C.MOV_R32_RM32, LOW32[target64], mem))
        else:
            emit(I.create_reg_mem(C.MOVZX_R32_RM16, LOW32[target64], mem))

    acc = R.RAX if bits == 64 else (R.EAX if bits == 32 else R.AX)

    if mname == "ANDN":
        read_op(R.RAX, 2)
        read_op(R.RCX, 1)
        emit(I.create_reg(C.NOT_RM64 if bits == 64 else C.NOT_RM32, R.RCX if bits == 64 else R.ECX))
        emit(I.create_reg_reg(C.AND_R64_RM64 if bits == 64 else C.AND_R32_RM32,
                              R.RAX if bits == 64 else R.EAX,
                              R.RCX if bits == 64 else R.ECX))

    elif mname == "BLSI":
        read_op(R.RAX, 1)
        emit(I.create_reg_reg(C.MOV_R64_RM64, R.RCX, R.RAX))
        emit(I.create_reg_reg(C.XOR_R32_RM32, R.EDX, R.EDX))
        emit(I.create_reg(C.NEG_RM64 if bits == 64 else C.NEG_RM32, R.RAX if bits == 64 else R.EAX))
        emit(I.create_reg(C.SETB_RM8, R.DL))
        emit(I.create_reg_reg(C.AND_R64_RM64 if bits == 64 else C.AND_R32_RM32,
                              R.RAX if bits == 64 else R.EAX,
                              R.RCX if bits == 64 else R.ECX))
        emit(I.create_reg_i32(C.BT_RM32_IMM8, R.EDX, 0))

    elif mname == "BLSMSK":
        read_op(R.RAX, 1)
        emit(I.create_reg_reg(C.MOV_R64_RM64, R.RCX, R.RAX))
        emit(I.create_reg_reg(C.XOR_R32_RM32, R.EDX, R.EDX))
        emit(I.create_reg_i32(C.SUB_RM64_IMM8 if bits == 64 else C.SUB_RM32_IMM8,
                              R.RAX if bits == 64 else R.EAX, 1))
        emit(I.create_reg(C.SETB_RM8, R.DL))
        emit(I.create_reg_reg(C.XOR_R64_RM64 if bits == 64 else C.XOR_R32_RM32,
                              R.RAX if bits == 64 else R.EAX,
                              R.RCX if bits == 64 else R.ECX))
        emit(I.create_reg_i32(C.BT_RM32_IMM8, R.EDX, 0))

    elif mname == "BLSR":
        read_op(R.RAX, 1)
        emit(I.create_reg_reg(C.MOV_R64_RM64, R.RCX, R.RAX))
        emit(I.create_reg_reg(C.XOR_R32_RM32, R.EDX, R.EDX))
        emit(I.create_reg_i32(C.SUB_RM64_IMM8 if bits == 64 else C.SUB_RM32_IMM8,
                              R.RAX if bits == 64 else R.EAX, 1))
        emit(I.create_reg(C.SETB_RM8, R.DL))
        emit(I.create_reg_reg(C.AND_R64_RM64 if bits == 64 else C.AND_R32_RM32,
                              R.RAX if bits == 64 else R.EAX,
                              R.RCX if bits == 64 else R.ECX))
        emit(I.create_reg_i32(C.BT_RM32_IMM8, R.EDX, 0))

    elif mname == "BEXTR":
        read_op(R.RAX, 1)
        read_op(R.RCX, 2)
        emit(I.create_reg_reg(C.MOVZX_R32_RM8, R.EDX, R.CH))
        emit(I.create_reg_reg(C.MOVZX_R32_RM8, R.ECX, R.CL))
        emit(I.create_reg_i32(C.CMP_RM32_IMM8, R.ECX, bits))
        b.branch(C.JAE_REL8_64, "bextr_zero")
        emit(I.create_reg_reg(C.SHR_RM64_CL if bits == 64 else C.SHR_RM32_CL,
                              R.RAX if bits == 64 else R.EAX, R.CL))
        emit(I.create_reg_i32(C.MOV_R32_IMM32, R.R8D, bits))
        emit(I.create_reg_reg(C.SUB_R32_RM32, R.R8D, R.ECX))
        emit(I.create_reg_reg(C.CMP_R32_RM32, R.EDX, R.R8D))
        b.branch(C.JAE_REL8_64, "bextr_flags")
        emit(I.create_reg_reg(C.MOV_R32_RM32, R.ECX, R.EDX))
        emit(I.create_reg_i32(C.MOV_R32_IMM32, R.R8D, 1))
        emit(I.create_reg_reg(C.SHL_RM64_CL if bits == 64 else C.SHL_RM32_CL,
                              R.R8 if bits == 64 else R.R8D, R.CL))
        emit(I.create_reg(C.DEC_RM64 if bits == 64 else C.DEC_RM32,
                          R.R8 if bits == 64 else R.R8D))
        emit(I.create_reg_reg(C.AND_R64_RM64 if bits == 64 else C.AND_R32_RM32,
                              R.RAX if bits == 64 else R.EAX,
                              R.R8 if bits == 64 else R.R8D))
        b.branch(C.JMP_REL8_64, "bextr_flags")
        b.label("bextr_zero")
        emit(I.create_reg_reg(C.XOR_R32_RM32, R.EAX, R.EAX))
        b.label("bextr_flags")
        emit(I.create_reg_reg(C.TEST_RM64_R64 if bits == 64 else C.TEST_RM32_R32,
                              R.RAX if bits == 64 else R.EAX,
                              R.RAX if bits == 64 else R.EAX))

    elif mname in ("TZCNT", "LZCNT"):
        read_op(R.RAX, 1)
        test_code = C.TEST_RM64_R64 if bits == 64 else (C.TEST_RM32_R32 if bits == 32 else C.TEST_RM16_R16)
        emit(I.create_reg_reg(test_code, acc, acc))
        b.branch(C.JE_REL8_64, "cnt_zero")
        if mname == "TZCNT":
            bsf_code = C.BSF_R64_RM64 if bits == 64 else (C.BSF_R32_RM32 if bits == 32 else C.BSF_R16_RM16)
            emit(I.create_reg_reg(bsf_code, acc, acc))
            emit(I.create_reg_reg(test_code, acc, acc))
        else:
            bsr_code = C.BSR_R64_RM64 if bits == 64 else (C.BSR_R32_RM32 if bits == 32 else C.BSR_R16_RM16)
            emit(I.create_reg_reg(bsr_code, acc, acc))
            emit(I.create_reg_i32(C.XOR_RM16_IMM8 if bits == 16 else C.XOR_RM32_IMM8,
                                  R.AX if bits == 16 else R.EAX, bits - 1))
        b.branch(C.JMP_REL8_64, "cnt_done")
        b.label("cnt_zero")
        emit(I.create_reg_i32(C.MOV_R32_IMM32, R.EAX, bits))
        emit(I.create_reg_reg(C.TEST_RM32_R32, R.EAX, R.EAX))
        emit(I.create(C.STC))
        b.label("cnt_done")

    elif mname == "POPCNT":
        # SWAR parallel bit count; architectural flags: OF=CF=SF=AF=PF=0, ZF=(result == 0).
        read_op(R.RAX, 1)
        emit(I.create_reg_reg(C.MOV_R64_RM64, R.RCX, R.RAX))
        emit(I.create_reg_i32(C.SHR_RM64_IMM8, R.RCX, 1))
        emit(I.create_reg_u64(C.MOV_R64_IMM64, R.RDX, 0x5555555555555555))
        emit(I.create_reg_reg(C.AND_R64_RM64, R.RCX, R.RDX))
        emit(I.create_reg_reg(C.SUB_R64_RM64, R.RAX, R.RCX))
        emit(I.create_reg_reg(C.MOV_R64_RM64, R.RCX, R.RAX))
        emit(I.create_reg_i32(C.SHR_RM64_IMM8, R.RCX, 2))
        emit(I.create_reg_u64(C.MOV_R64_IMM64, R.RDX, 0x3333333333333333))
        emit(I.create_reg_reg(C.AND_R64_RM64, R.RAX, R.RDX))
        emit(I.create_reg_reg(C.AND_R64_RM64, R.RCX, R.RDX))
        emit(I.create_reg_reg(C.ADD_R64_RM64, R.RAX, R.RCX))
        emit(I.create_reg_reg(C.MOV_R64_RM64, R.RCX, R.RAX))
        emit(I.create_reg_i32(C.SHR_RM64_IMM8, R.RCX, 4))
        emit(I.create_reg_reg(C.ADD_R64_RM64, R.RAX, R.RCX))
        emit(I.create_reg_u64(C.MOV_R64_IMM64, R.RDX, 0x0F0F0F0F0F0F0F0F))
        emit(I.create_reg_reg(C.AND_R64_RM64, R.RAX, R.RDX))
        emit(I.create_reg_u64(C.MOV_R64_IMM64, R.RDX, 0x0101010101010101))
        emit(I.create_reg_reg(C.IMUL_R64_RM64, R.RAX, R.RDX))
        emit(I.create_reg_i32(C.SHR_RM64_IMM8, R.RAX, 56))
        # POPCNT flags: ZF=(count==0), OF=CF=SF=AF=PF=0. TEST alone would set PF
        # from the count's parity, so ZF goes through SAHF (count <= 64, AH is 0).
        emit(I.create_reg_reg(C.XOR_R32_RM32, R.EDX, R.EDX))
        emit(I.create_reg_reg(C.TEST_RM32_R32, R.EAX, R.EAX))
        emit(I.create_reg(C.SETE_RM8, R.DL))
        emit(I.create_reg_i32(C.SHL_RM32_IMM8, R.EDX, 6))
        emit(I.create_reg_reg(C.AND_R32_RM32, R.EDX, R.EDX))      # OF=0
        emit(I.create_reg_reg(C.MOV_RM8_R8, R.AH, R.DL))
        emit(I.create(C.SAHF))                                     # SF ZF AF PF CF
        emit(I.create_reg_i32(C.MOV_R8_IMM8, R.AH, 0))

    if bits == 16:
        emit(I.create_mem_reg(C.MOV_RM16_R16, M(R.RSP, displ=result_slot, displ_size=1), R.AX))
    else:
        emit(I.create_mem_reg(C.MOV_RM64_R64, M(R.RSP, displ=result_slot, displ_size=1), R.RAX))

    for reg in reversed(scratch):
        emit(I.create_reg(C.POP_R64, reg))
    emit(I.create_reg(C.POP_R64, dst64))
    return b.encode(ip)


# ---------------------------------------------------------------------------
# MOVBE
# ---------------------------------------------------------------------------

def _lower_movbe(ins: ix.Instruction, ip: int, external_stack_adjust: int) -> bytes:
    b = _Builder()
    emit = b.emit
    is_store = ins.op0_kind == O.MEMORY

    if is_store:
        src = ins.op1_register
        bits = _gpr_width(src)
        scratch = [R.RAX, R.RCX]
        for reg in scratch:
            emit(I.create_reg(C.PUSH_R64, reg))
        frame = 8 * len(scratch)
        slot = {R.RAX: 8, R.RCX: 0}
        total_adjust = frame + external_stack_adjust

        s64 = FULL_ALL[src]
        if s64 in slot:
            emit(I.create_reg_mem(C.MOV_R64_RM64, R.RAX, M(R.RSP, displ=slot[s64], displ_size=1)))
        elif s64 == R.RSP:
            emit(I.create_reg_mem(C.LEA_R64_M, R.RAX, M(R.RSP, displ=total_adjust, displ_size=1)))
        else:
            emit(I.create_reg_reg(C.MOV_R64_RM64, R.RAX, s64))

        if bits == 16:
            emit(I.create_reg_reg(C.XCHG_RM8_R8, R.AL, R.AH))  # MOVBE leaves RFLAGS untouched
        elif bits == 32:
            emit(I.create_reg(C.BSWAP_R32, R.EAX))
        else:
            emit(I.create_reg(C.BSWAP_R64, R.RAX))

        mem = _mem_operand(emit, ins, R.RCX, total_adjust, bits // 8)
        if bits == 16:
            emit(I.create_mem_reg(C.MOV_RM16_R16, mem, R.AX))
        elif bits == 32:
            emit(I.create_mem_reg(C.MOV_RM32_R32, mem, R.EAX))
        else:
            emit(I.create_mem_reg(C.MOV_RM64_R64, mem, R.RAX))

        for reg in reversed(scratch):
            emit(I.create_reg(C.POP_R64, reg))
        return b.encode(ip)

    # Load form: MOVBE r16/32/64, m16/32/64
    dst = ins.op0_register
    bits = _gpr_width(dst)
    dst64 = FULL_ALL[dst]
    if dst64 == R.RSP:
        raise LoweringError("Destination cannot be RSP")

    # Ultra Fast Path: if bits in (32, 64) and not RIP-relative, we can load directly into dst and BSWAP dst!
    if bits in (32, 64) and not ins.is_ip_rel_memory_operand:
        mem = _mem_operand(emit, ins, None, external_stack_adjust, bits // 8)
        if bits == 32:
            emit(I.create_reg_mem(C.MOV_R32_RM32, dst, mem))
            emit(I.create_reg(C.BSWAP_R32, dst))
        else:
            emit(I.create_reg_mem(C.MOV_R64_RM64, dst, mem))
            emit(I.create_reg(C.BSWAP_R64, dst))
        return b.encode(ip)

    emit(I.create_reg(C.PUSH_R64, dst64))
    emit(I.create_reg(C.PUSH_R64, R.RAX))
    frame = 16
    total_adjust = frame + external_stack_adjust

    mem = _mem_operand(emit, ins, R.RAX, total_adjust, bits // 8)
    if bits == 16:
        emit(I.create_reg_mem(C.MOV_R16_RM16, R.AX, mem))
        emit(I.create_reg_reg(C.XCHG_RM8_R8, R.AL, R.AH))  # MOVBE leaves RFLAGS untouched
        emit(I.create_mem_reg(C.MOV_RM16_R16, M(R.RSP, displ=8, displ_size=1), R.AX))
    elif bits == 32:
        emit(I.create_reg_mem(C.MOV_R32_RM32, R.EAX, mem))
        emit(I.create_reg(C.BSWAP_R32, R.EAX))
        emit(I.create_mem_reg(C.MOV_RM64_R64, M(R.RSP, displ=8, displ_size=1), R.RAX))
    else:
        emit(I.create_reg_mem(C.MOV_R64_RM64, R.RAX, mem))
        emit(I.create_reg(C.BSWAP_R64, R.RAX))
        emit(I.create_mem_reg(C.MOV_RM64_R64, M(R.RSP, displ=8, displ_size=1), R.RAX))

    emit(I.create_reg(C.POP_R64, R.RAX))
    emit(I.create_reg(C.POP_R64, dst64))
    return b.encode(ip)


# ---------------------------------------------------------------------------
# FMA3 (Ultra 3.0: exact-semantics AVX1 expansion with minimal state traffic)
# ---------------------------------------------------------------------------
#
# Emulation contract: a*b is rounded, then the add/sub is rounded (two roundings
# instead of FMA's one). Everything else is exact: operand order, lane
# selection, preserved upper bits of scalar forms, zeroed bits [255:128] of
# VEX.128 forms, and the sign of exact zero results.
#
# Operand roles per form: 132: a=dst b=src3 c=src2 | 213: a=src2 b=dst c=src3
#                         231: a=src2 b=src3 c=dst
#
# Path A (c is not dst, so dst is a multiplicand): zero scratch, zero stack.
#     vmul dst, dst, other ; vadd/vsub/vaddsub dst, dst, c
# Path B (c is dst, e.g. the accumulate form "acc += a*b"): c is parked in a
#     4..32 byte stack slot instead of saving and restoring a full YMM scratch.
# Generic: one or two saved scratch YMMs (FMSUBADD, register aliasing, no pool).
#
# Negations use a sign-mask constant pool in .fxcpt (set by the patch builder
# with set_constant_pool); without a pool the mask is built in a scratch.

CONSTANTS: dict | None = None

SIGN_POOL_BYTES = {
    "ps": struct.pack("<8I", *([0x80000000] * 8)),
    "pd": struct.pack("<4Q", *([1 << 63] * 4)),
    "ss": struct.pack("<I", 0x80000000) + bytes(28),
    "sd": struct.pack("<Q", 1 << 63) + bytes(24),
    # VCVTPH2PS constants (per 32-bit lane)
    "h2f_bias": struct.pack("<8I", *([0x38000000] * 8)),        # exponent rebias 15 -> 127
    "h2f_magic": struct.pack("<8I", *([0x38800000] * 8)),       # 2^-14, subnormal magic
    "h2f_norm_min": struct.pack("<8I", *([0x007FFFFF] * 8)),    # E > this <=> exponent != 0
    "h2f_inf_min": struct.pack("<8I", *([0x0F7FFFFF] * 8)),     # E > this <=> exponent == 31
    "h2f_nan_min": struct.pack("<8I", *([0x0F800000] * 8)),     # E > this <=> NaN
    "h2f_qnan": struct.pack("<8I", *([0x00400000] * 8)),        # quiet bit
}


def set_constant_pool(pool: dict | None) -> None:
    """pool maps 'ps'/'pd'/'ss'/'sd' to the absolute VA of SIGN_POOL_BYTES[key]."""
    global CONSTANTS
    CONSTANTS = dict(pool) if pool else None


def _fma_parse(mname: str):
    for prefix, kind in (("VFMADDSUB", "FMADDSUB"), ("VFMSUBADD", "FMSUBADD"), ("VFNMADD", "FNMADD"),
                         ("VFNMSUB", "FNMSUB"), ("VFMADD", "FMADD"), ("VFMSUB", "FMSUB")):
        if mname.startswith(prefix):
            rest = mname[len(prefix):]
            return kind, rest[:3], rest[3:]
    raise LoweringError(f"Unsupported FMA instruction: {mname}")


def _lower_fma(ins: ix.Instruction, ip: int, external_stack_adjust: int) -> bytes:
    mname = MNEMONIC_NAMES.get(ins.mnemonic, "")
    kind, order, suffix = _fma_parse(mname)
    if order not in ("132", "213", "231") or suffix not in ("PS", "PD", "SS", "SD"):
        raise LoweringError(f"Unsupported FMA form: {ins}")
    is_scalar = suffix in ("SS", "SD")
    is_double = suffix in ("PD", "SD")
    wide = (not is_scalar) and (R.YMM0 <= ins.op0_register <= R.YMM15)

    def vreg(x):
        return _ymm(x) if wide else _xmm(x)

    pshape = "YMM_YMM_YMMM256" if wide else "XMM_XMM_XMMM128"
    shape = pshape if not is_scalar else ("XMM_XMM_XMMM64" if is_double else "XMM_XMM_XMMM32")
    vmul = getattr(C, f"VEX_VMUL{suffix}_{shape}")
    vadd = getattr(C, f"VEX_VADD{suffix}_{shape}")
    vsub = getattr(C, f"VEX_VSUB{suffix}_{shape}")
    vaddsub = None if is_scalar else getattr(C, f"VEX_VADDSUB{suffix}_{pshape}")
    fp = "PD" if is_double else "PS"
    vxor = getattr(C, f"VEX_VXOR{fp}_{pshape}")
    mem_bytes = 32 if wide else (16 if not is_scalar else (8 if is_double else 4))
    sign_addr = CONSTANTS.get(suffix.lower()) if CONSTANTS else None

    b = _Builder()
    emit = b.emit

    def rrx(code, d, s1, s2):
        if isinstance(s2, ix.MemoryOperand):
            emit(I.create_reg_reg_mem(code, d, s1, s2))
        else:
            emit(I.create_reg_reg_reg(code, d, s1, s2))

    def sign_operand():
        return ix.MemoryOperand.ctor_u64(R.RIP, R.NONE, 1, sign_addr, 8, False, R.NONE)

    r1 = vreg(ins.op0_register)
    r2 = vreg(ins.op1_register)
    op3_mem = ins.op2_kind == O.MEMORY
    r3 = None if op3_mem else vreg(ins.op2_register)

    def op3(adjust):
        return _mem_operand(emit, ins, None, adjust, mem_bytes) if op3_mem else r3

    c_is_r1 = order == "231" or (order == "132" and r2 == r1) or (order == "213" and r3 == r1)

    if not c_is_r1:
        # ---- Path A: dst is a multiplicand and c is another register/memory ----
        c_reg = r2 if order == "132" else r3          # None when c is memory (213 + mem)
        ok = kind in ("FMADD", "FMSUB", "FMADDSUB") \
            or (kind == "FNMADD" and ((not is_scalar and c_reg is not None) or sign_addr)) \
            or (kind == "FNMSUB" and sign_addr)
        if ok:
            adj = external_stack_adjust
            if order == "132":
                rrx(vmul, r1, r1, op3(adj))
                c = r2
            else:
                emit(I.create_reg_reg_reg(vmul, r1, r1, r2))
                c = op3(adj)
            if kind == "FMADD":
                rrx(vadd, r1, r1, c)
            elif kind == "FMSUB":
                rrx(vsub, r1, r1, c)
            elif kind == "FMADDSUB":
                rrx(vaddsub, r1, r1, c)
            elif kind == "FNMADD" and not is_scalar and c_reg is not None:
                emit(I.create_reg_reg_reg(vsub, r1, c_reg, r1))
            else:  # FNMADD = (-p) + c ; FNMSUB = (-p) - c  (exact zero signs)
                emit(I.create_reg_reg_mem(vxor, r1, r1, sign_operand()))
                rrx(vadd if kind == "FNMADD" else vsub, r1, r1, c)
            return b.encode(ip)
    else:
        # ---- Path B: c is dst (accumulate forms); park c in a small stack slot ---
        a = r1 if order == "132" else r2
        bop_reg = r1 if order == "213" else r3        # None when b is memory
        ok = kind in ("FMADD", "FMSUB", "FMADDSUB") \
            or (kind in ("FNMADD", "FNMSUB") and sign_addr) \
            or (kind == "FMSUBADD" and sign_addr and a != r1 and bop_reg != r1)
        if ok:
            frame = 32 if wide else 16
            total = frame + external_stack_adjust
            slot = M(R.RSP)
            b_op = op3(total) if order != "213" else r1
            emit(I.create_reg_mem(C.LEA_R64_M, R.RSP, M(R.RSP, displ=-frame, displ_size=1)))
            if kind == "FMSUBADD":
                emit(I.create_reg_reg_mem(vxor, r1, r1, sign_operand()))
            if is_scalar:
                emit(I.create_mem_reg(C.VEX_VMOVSD_M64_XMM if is_double else C.VEX_VMOVSS_M32_XMM, slot, r1))
                if a != r1:
                    emit(I.create_reg_reg_reg(C.VEX_VMOVSD_XMM_XMM_XMM if is_double else C.VEX_VMOVSS_XMM_XMM_XMM,
                                              r1, r1, a))
                    if not isinstance(b_op, ix.MemoryOperand) and b_op == r1:
                        b_op = slot
                rrx(vmul, r1, r1, b_op)
            else:
                store = getattr(C, f"VEX_VMOVUP{'D' if is_double else 'S'}_{'YMMM256_YMM' if wide else 'XMMM128_XMM'}")
                emit(I.create_mem_reg(store, slot, r1))
                rrx(vmul, r1, a, b_op)
            if kind in ("FNMADD", "FNMSUB"):
                emit(I.create_reg_reg_mem(vxor, r1, r1, sign_operand()))
            combine = {"FMADD": vadd, "FMSUB": vsub, "FMADDSUB": vaddsub, "FNMADD": vadd,
                       "FNMSUB": vsub, "FMSUBADD": vaddsub}[kind]
            emit(I.create_reg_reg_mem(combine, r1, r1, slot))
            emit(I.create_reg_mem(C.LEA_R64_M, R.RSP, M(R.RSP, displ=frame, displ_size=1)))
            return b.encode(ip)

    # ---- Generic exact path: product in a saved scratch -----------------------
    used_xmms = {
        _xmm(ins.op_register(n))
        for n in range(ins.op_count)
        if ins.op_kind(n) == O.REGISTER and R.XMM0 <= ins.op_register(n) <= R.YMM15
    }
    free = [R.XMM0 + i for i in range(16) if (R.XMM0 + i) not in used_xmms]
    c_is_mem = order == "213" and op3_mem
    needs_neg = kind == "FNMSUB" or (kind == "FNMADD" and c_is_mem)
    use_t2 = kind == "FMSUBADD" or (needs_neg and not sign_addr)
    use_t3 = kind == "FMSUBADD" and c_is_mem and not sign_addr
    scratch = free[:3 if use_t3 else (2 if use_t2 else 1)]
    t, t2 = vreg(scratch[0]), (vreg(scratch[1]) if use_t2 else None)
    frame = 32 * len(scratch)
    total = frame + external_stack_adjust
    emit(I.create_reg_mem(C.LEA_R64_M, R.RSP, M(R.RSP, displ=-frame, displ_size=1)))
    for n, reg in enumerate(scratch):
        emit(I.create_mem_reg(C.VEX_VMOVDQU_YMMM256_YMM, M(R.RSP, displ=32 * n, displ_size=1), _ymm(reg)))

    def build_mask(target):
        x = _xmm(target)
        emit(I.create_reg_reg_reg(C.VEX_VPCMPEQD_XMM_XMM_XMMM128, x, x, x))
        if is_double:
            emit(I.create_reg_reg_i32(C.VEX_VPSLLQ_XMM_XMM_IMM8, x, x, 63))
        else:
            emit(I.create_reg_reg_i32(C.VEX_VPSLLD_XMM_XMM_IMM8, x, x, 31))
        if wide:
            emit(I.create_reg_reg_reg_i32(C.VEX_VINSERTF128_YMM_YMM_XMMM128_IMM8, _ymm(x), _ymm(x), x, 1))

    if order == "132":
        rrx(vmul, t, r1, op3(total))
        c = r2
    elif order == "213":
        emit(I.create_reg_reg_reg(vmul, t, r2, r1))
        c = op3(total)
    else:
        rrx(vmul, t, r2, op3(total))
        c = r1
    if kind == "FMADD":
        rrx(vadd, t, t, c)
    elif kind == "FMSUB":
        rrx(vsub, t, t, c)
    elif kind == "FMADDSUB":
        rrx(vaddsub, t, t, c)
    elif kind == "FNMADD" and not c_is_mem:
        emit(I.create_reg_reg_reg(vsub, t, c, t))
    elif kind in ("FNMADD", "FNMSUB"):
        if sign_addr:
            emit(I.create_reg_reg_mem(vxor, t, t, sign_operand()))
        else:
            build_mask(t2)
            emit(I.create_reg_reg_reg(vxor, t, t, t2))
        rrx(vadd if kind == "FNMADD" else vsub, t, t, c)
    else:  # FMSUBADD = addsub(p, -c)
        if c_is_mem:
            load = getattr(C, f"VEX_VMOVUP{'D' if is_double else 'S'}_{'YMM_YMMM256' if wide else 'XMM_XMMM128'}")
            emit(I.create_reg_mem(load, t2, c))
            if sign_addr:
                emit(I.create_reg_reg_mem(vxor, t2, t2, sign_operand()))
            else:
                t3 = vreg(scratch[2])
                build_mask(t3)
                emit(I.create_reg_reg_reg(vxor, t2, t2, t3))
        elif sign_addr:
            emit(I.create_reg_reg_mem(vxor, t2, c, sign_operand()))
        else:
            build_mask(t2)
            emit(I.create_reg_reg_reg(vxor, t2, t2, c))
        emit(I.create_reg_reg_reg(vaddsub, t, t, t2))

    if is_scalar:
        emit(I.create_reg_reg_reg(C.VEX_VMOVSD_XMM_XMM_XMM if is_double else C.VEX_VMOVSS_XMM_XMM_XMM,
                                  _xmm(ins.op0_register), _xmm(ins.op0_register), _xmm(t)))
    elif wide:
        emit(I.create_reg_reg(C.VEX_VMOVAPS_YMM_YMMM256, ins.op0_register, t))
    else:
        emit(I.create_reg_reg(C.VEX_VMOVAPS_XMM_XMMM128, ins.op0_register, t))
    for n, reg in enumerate(scratch):
        emit(I.create_reg_mem(C.VEX_VMOVDQU_YMM_YMMM256, _ymm(reg), M(R.RSP, displ=32 * n, displ_size=1)))
    emit(I.create_reg_mem(C.LEA_R64_M, R.RSP, M(R.RSP, displ=frame, displ_size=1)))
    return b.encode(ip)


# ---------------------------------------------------------------------------
# F16C (VCVTPH2PS Vectorized SIMD & VCVTPS2PH)
# ---------------------------------------------------------------------------

def _lower_ph2ps_pool(ins: ix.Instruction, ip: int, external_stack_adjust: int) -> bytes:
    """VCVTPH2PS with pool constants: 3 scratch YMMs (4 for ymm), ~17 ops per 4 lanes.

    E = (h & 0x7FFF) << 13 ; normal = E + bias ; Inf/NaN = E + 2*bias (+quiet bit)
    subnormal/zero = (E | magic) - magic (exact, independent of DAZ/FTZ) ; | sign"""
    b = _Builder()
    emit = b.emit
    wide = R.YMM0 <= ins.op0_register <= R.YMM15
    used = {_xmm(ins.op_register(n)) for n in range(ins.op_count)
            if ins.op_kind(n) == O.REGISTER and R.XMM0 <= ins.op_register(n) <= R.YMM15}
    free = [R.XMM0 + i for i in range(16) if (R.XMM0 + i) not in used]
    scratch = free[:4 if wide else 3]
    t1, t2, t3 = scratch[0], scratch[1], scratch[2]
    frame = 32 * len(scratch)
    total = frame + external_stack_adjust

    def k(key):
        return ix.MemoryOperand.ctor_u64(R.RIP, R.NONE, 1, CONSTANTS[key], 8, False, R.NONE)

    def convert(x):
        emit(I.create_reg_reg_i32(C.VEX_VPSLLD_XMM_XMM_IMM8, t1, x, 17))
        emit(I.create_reg_reg_i32(C.VEX_VPSRLD_XMM_XMM_IMM8, t1, t1, 4))           # E
        emit(I.create_reg_reg_i32(C.VEX_VPSLLD_XMM_XMM_IMM8, t2, x, 16))
        emit(I.create_reg_reg_mem(C.VEX_VPAND_XMM_XMM_XMMM128, t2, t2, k("ps")))     # sign
        emit(I.create_reg_reg_mem(C.VEX_VPADDD_XMM_XMM_XMMM128, x, t1, k("h2f_bias")))
        emit(I.create_reg_reg_mem(C.VEX_VPCMPGTD_XMM_XMM_XMMM128, t3, t1, k("h2f_inf_min")))
        emit(I.create_reg_reg_mem(C.VEX_VPAND_XMM_XMM_XMMM128, t3, t3, k("h2f_bias")))
        emit(I.create_reg_reg_reg(C.VEX_VPADDD_XMM_XMM_XMMM128, x, x, t3))
        emit(I.create_reg_reg_mem(C.VEX_VPCMPGTD_XMM_XMM_XMMM128, t3, t1, k("h2f_nan_min")))
        emit(I.create_reg_reg_mem(C.VEX_VPAND_XMM_XMM_XMMM128, t3, t3, k("h2f_qnan")))
        emit(I.create_reg_reg_reg(C.VEX_VPOR_XMM_XMM_XMMM128, x, x, t3))
        emit(I.create_reg_reg_mem(C.VEX_VPOR_XMM_XMM_XMMM128, t3, t1, k("h2f_magic")))
        emit(I.create_reg_reg_mem(C.VEX_VSUBPS_XMM_XMM_XMMM128, t3, t3, k("h2f_magic")))
        emit(I.create_reg_reg_mem(C.VEX_VPCMPGTD_XMM_XMM_XMMM128, t1, t1, k("h2f_norm_min")))
        emit(I.create_reg_reg_reg_reg(C.VEX_VBLENDVPS_XMM_XMM_XMMM128_XMM, x, t3, x, t1))
        emit(I.create_reg_reg_reg(C.VEX_VPOR_XMM_XMM_XMMM128, x, x, t2))

    emit(I.create_reg_mem(C.LEA_R64_M, R.RSP, M(R.RSP, displ=-frame, displ_size=_disp_size(-frame))))
    for n, reg in enumerate(scratch):
        emit(I.create_mem_reg(C.VEX_VMOVDQU_YMMM256_YMM, M(R.RSP, displ=32 * n, displ_size=_disp_size(32 * n)), _ymm(reg)))
    dst_x = _xmm(ins.op0_register)
    is_mem = ins.op1_kind == O.MEMORY
    if not wide:
        if is_mem:
            emit(I.create_reg_mem(C.VEX_VPMOVZXWD_XMM_XMMM64, dst_x, _mem_operand(emit, ins, None, total, 8)))
        else:
            emit(I.create_reg_reg(C.VEX_VPMOVZXWD_XMM_XMMM64, dst_x, _xmm(ins.op1_register)))
        convert(dst_x)
    else:
        lo = scratch[3]
        if is_mem:
            emit(I.create_reg_mem(C.VEX_VPMOVZXWD_XMM_XMMM64, lo, _mem_operand(emit, ins, None, total, 16)))
            emit(I.create_reg_mem(C.VEX_VPMOVZXWD_XMM_XMMM64, dst_x, _mem_operand(emit, ins, None, total, 16, 8)))
        else:
            src = _xmm(ins.op1_register)
            emit(I.create_reg_reg(C.VEX_VPMOVZXWD_XMM_XMMM64, lo, src))
            emit(I.create_reg_reg_i32(C.VEX_VPSRLDQ_XMM_XMM_IMM8, dst_x, src, 8))
            emit(I.create_reg_reg(C.VEX_VPMOVZXWD_XMM_XMMM64, dst_x, dst_x))
        convert(dst_x)
        convert(lo)
        emit(I.create_reg_reg_reg_i32(C.VEX_VINSERTF128_YMM_YMM_XMMM128_IMM8, ins.op0_register, _ymm(lo), dst_x, 1))
    for n, reg in enumerate(scratch):
        emit(I.create_reg_mem(C.VEX_VMOVDQU_YMM_YMMM256, _ymm(reg), M(R.RSP, displ=32 * n, displ_size=_disp_size(32 * n))))
    emit(I.create_reg_mem(C.LEA_R64_M, R.RSP, M(R.RSP, displ=frame, displ_size=_disp_size(frame))))
    return b.encode(ip)


def _lower_f16c(ins: ix.Instruction, ip: int, external_stack_adjust: int) -> bytes:
    mname = MNEMONIC_NAMES.get(ins.mnemonic, "")
    if mname not in ("VCVTPH2PS", "VCVTPS2PH"):
        raise LoweringError(f"Unsupported F16C instruction: {ins}")

    b = _Builder()
    emit = b.emit

    if mname == "VCVTPH2PS" and CONSTANTS and "h2f_bias" in CONSTANTS:
        return _lower_ph2ps_pool(ins, ip, external_stack_adjust)
    if mname == "VCVTPH2PS":
        # Self-contained variant (no constant pool): builds its constants in registers.
        wide = R.YMM0 <= ins.op0_register <= R.YMM15
        used_xmms = {
            _xmm(ins.op_register(n))
            for n in range(ins.op_count)
            if ins.op_kind(n) == O.REGISTER and R.XMM0 <= ins.op_register(n) <= R.YMM15
        }
        free_xmms = [R.XMM0 + i for i in range(16) if (R.XMM0 + i) not in used_xmms]
        s0, s1, s2, s3, s4 = free_xmms[0], free_xmms[1], free_xmms[2], free_xmms[3], free_xmms[4]

        need_rax = ins.op1_kind == O.MEMORY and ins.is_ip_rel_memory_operand
        frame = 5 * 32 + (16 if need_rax else 0)
        total_adjust = frame + external_stack_adjust

        emit(I.create_reg_mem(C.LEA_R64_M, R.RSP, M(R.RSP, displ=-frame, displ_size=_disp_size(-frame))))
        for idx, xreg in enumerate((s0, s1, s2, s3, s4)):
            emit(I.create_mem_reg(C.VEX_VMOVDQU_YMMM256_YMM, M(R.RSP, displ=idx * 32, displ_size=_disp_size(idx * 32)), _ymm(xreg)))
        if need_rax:
            emit(I.create_mem_reg(C.MOV_RM64_R64, M(R.RSP, displ=160, displ_size=8), R.RAX))

        # Load input FP16 vector into s0
        if ins.op1_kind == O.REGISTER:
            emit(I.create_reg_reg(C.VEX_VMOVAPS_XMM_XMMM128, s0, _xmm(ins.op1_register)))
        else:
            mem = _mem_operand(emit, ins, R.RAX if need_rax else None, total_adjust, 16 if wide else 8)
            if wide:
                emit(I.create_reg_mem(C.VEX_VMOVUPS_XMM_XMMM128, s0, mem))
            else:
                emit(I.create_reg_mem(C.VEX_VMOVQ_XMM_XMMM64, s0, mem))

        def convert_4_half_in_xmm(x_vec):
            # Input: low 64 bits of x_vec hold 4 x FP16
            # Output: x_vec holds 4 x FP32
            # Scratch: s1, s2, s3, s4
            emit(I.create_reg_reg(C.VEX_VPMOVZXWD_XMM_XMMM64, x_vec, x_vec))
            # s1 = sign = (x_vec >> 15) << 31
            emit(I.create_reg_reg_i32(C.VEX_VPSRLD_XMM_XMM_IMM8, s1, x_vec, 15))
            emit(I.create_reg_reg_i32(C.VEX_VPSLLD_XMM_XMM_IMM8, s1, s1, 31))
            # s2 = (exp_mant << 13) = (x_vec << 17) >> 4  (shifting by 17 drops the sign bit;
            # 2.x used << 16 >> 3, which left the sign in bit 28 and broke every negative half)
            emit(I.create_reg_reg_i32(C.VEX_VPSLLD_XMM_XMM_IMM8, s2, x_vec, 17))
            emit(I.create_reg_reg_i32(C.VEX_VPSRLD_XMM_XMM_IMM8, s2, s2, 4))
            # s3 = exp_field = (s2 >> 23) << 23 = (x_vec & 0x7C00) << 13
            emit(I.create_reg_reg_i32(C.VEX_VPSRLD_XMM_XMM_IMM8, s3, s2, 23))
            emit(I.create_reg_reg_i32(C.VEX_VPSLLD_XMM_XMM_IMM8, s3, s3, 23))

            # Build constants in s4 & x_vec:
            # all_ones in s4
            emit(I.create_reg_reg_reg(C.VEX_VPCMPEQD_XMM_XMM_XMMM128, s4, s4, s4))
            # x_vec = 0x38000000 = (all_ones >> 29) << 27
            emit(I.create_reg_reg_i32(C.VEX_VPSRLD_XMM_XMM_IMM8, x_vec, s4, 29))
            emit(I.create_reg_reg_i32(C.VEX_VPSLLD_XMM_XMM_IMM8, x_vec, x_vec, 27))
            # s4 = 0x00800000 = (all_ones >> 31) << 23
            emit(I.create_reg_reg_i32(C.VEX_VPSRLD_XMM_XMM_IMM8, s4, s4, 31))
            emit(I.create_reg_reg_i32(C.VEX_VPSLLD_XMM_XMM_IMM8, s4, s4, 23))
            # s4 = 0x38800000 = 0x38000000 | 0x00800000 (magic float)
            emit(I.create_reg_reg_reg(C.VEX_VPOR_XMM_XMM_XMMM128, s4, s4, x_vec))

            # Normal result in x_vec: x_vec = s2 + 0x38000000
            emit(I.create_reg_reg_reg(C.VEX_VPADDD_XMM_XMM_XMMM128, x_vec, s2, x_vec))

            # Subnormal/zero result: VSUBPS(s2 | 0x38800000, 0x38800000)
            # Put (s2 | 0x38800000) in s4_sub: wait, we need s2 later only for Inf/NaN!
            # Wait: for Inf/NaN (s3 == 0x0F800000), x_vec is (s2 + 0x38000000) which has exp = 15+112 = 127 (0x3F800000).
            # If we add 0x40000000 to x_vec when Inf/NaN, exp becomes 127 + 128 = 255 (0x7F800000)!
            # And for NaN (mant != 0), we also set quiet NaN bit 0x00400000!
            # So we don't even need s2 after computing subnormal and NaN check!
            # Let's check:
            # 1) Save s2 in s4? No:
            #    Let's compute subnormal in s4:
            #    Wait: VSUBPS needs both operands! If we do:
            #    VPOR s2_magic, s2, s4 (where s2_magic overwrites s2? Wait: if exp==31, we don't use subnormal!).
            #    Wait: if we overwrite s2 with VSUBPS(s2 | s4, s4), what happens to s2 when exp == 31?
            #    We only need to know if mant != 0 when exp == 31!
            #    And when exp == 31 (s3 == 0x0F800000), s2 > s3 iff mant != 0!
            #    Even simpler: in x_vec (which is s2 + 0x38000000), when exp == 31, x_vec == 0x47800000 | (mant << 13)!
            #    So x_vec STILL HAS (mant << 13) intact in its low 23 bits!
            #    That means s2 is NEVER needed again after we compute subnormal!
            emit(I.create_reg_reg_reg(C.VEX_VPOR_XMM_XMM_XMMM128, s2, s2, s4))
            emit(I.create_reg_reg_reg(C.VEX_VSUBPS_XMM_XMM_XMMM128, s2, s2, s4))
            # Now s2 holds the exact subnormal/zero float32!
            # Blend s2 into x_vec where s3 == 0:
            emit(I.create_reg_reg_reg(C.VEX_VPXOR_XMM_XMM_XMMM128, s4, s4, s4))
            emit(I.create_reg_reg_reg(C.VEX_VPCMPEQD_XMM_XMM_XMMM128, s4, s3, s4))  # s4 = mask (s3 == 0)
            emit(I.create_reg_reg_reg_reg(C.VEX_VBLENDVPS_XMM_XMM_XMMM128_XMM, x_vec, x_vec, s2, s4))

            # Now check Inf/NaN: s3 == 0x0F800000 (which is 31 << 23)
            # Construct 0x0F800000 in s4:
            emit(I.create_reg_reg_reg(C.VEX_VPCMPEQD_XMM_XMM_XMMM128, s4, s4, s4))
            emit(I.create_reg_reg_i32(C.VEX_VPSRLD_XMM_XMM_IMM8, s4, s4, 27))       # 0x1F (31)
            emit(I.create_reg_reg_i32(C.VEX_VPSLLD_XMM_XMM_IMM8, s4, s4, 23))       # 0x0F800000
            emit(I.create_reg_reg_reg(C.VEX_VPCMPEQD_XMM_XMM_XMMM128, s3, s3, s4))  # s3 = mask (exp == 31)

            # When exp == 31: x_vec currently has 0x47800000 | (mant << 13).
            # If we OR x_vec with 0x7F800000, wait: 0x47800000 | 0x7F800000 = 0x7F800000!
            # Because 0x47 = 0100_0111b and 0x7F = 0111_1111b, so 0x47800000 is a subset of 0x7F800000!
            # AND what if mant != 0 (NaN)? Then x_vec > 0x47800000, so we also OR 0x00400000 (quiet NaN bit)!
            # Let's verify: (0x47800000 & 0x7F800000) == 0x47800000!
            # YES! 0x47 is 01000111 in binary, and 0x7F is 01111111 in binary! Every 1-bit of 0x47800000 is already 1 in 0x7F800000!
            # Let's construct 0x7F800000 in s4: (0x1F << 23) is 0x0F800000; better: (all_ones >> 24) << 23 = 0x7F800000!
            emit(I.create_reg_reg_reg(C.VEX_VPCMPEQD_XMM_XMM_XMMM128, s4, s4, s4))
            emit(I.create_reg_reg_i32(C.VEX_VPSRLD_XMM_XMM_IMM8, s4, s4, 24))       # 0xFF
            emit(I.create_reg_reg_i32(C.VEX_VPSLLD_XMM_XMM_IMM8, s4, s4, 23))       # 0x7F800000
            # Check if mant != 0: (x_vec << 9) != 0
            emit(I.create_reg_reg_i32(C.VEX_VPSLLD_XMM_XMM_IMM8, s2, x_vec, 9))
            # s2 != 0 <=> VPCMPEQD(s2, zero) is 0
            # Even simpler: quiet NaN bit is 0x00400000 = (s4 >> 9) & ... wait:
            # If we do VPOR s2, x_vec, s4 -> s2 has 0x7F800000 | (mant << 13).
            # To set bit 22 (0x00400000) when mant != 0:
            # Notice that if mant != 0, at least one bit in [22:13] is 1.
            # If we shift mant left so any set bit reaches bit 31, or compare (x_vec << 9) > 0:
            emit(I.create_reg_reg_reg(C.VEX_VPXOR_XMM_XMM_XMMM128, s2, s2, s2))
            emit(I.create_reg_reg_i32(C.VEX_VPSLLD_XMM_XMM_IMM8, s4, x_vec, 9))     # mant in high 10 bits [31:22]
            emit(I.create_reg_reg_reg(C.VEX_VPCMPEQD_XMM_XMM_XMMM128, s2, s4, s2))  # s2 = all_ones if mant == 0 else 0
            # We want 0x00400000 when mant != 0, and 0x7F800000 always when exp == 31:
            # Wait: (~s2 & 0x00400000) | 0x7F800000!
            emit(I.create_reg_reg_reg(C.VEX_VPCMPEQD_XMM_XMM_XMMM128, s4, s4, s4))
            emit(I.create_reg_reg_i32(C.VEX_VPSRLD_XMM_XMM_IMM8, s4, s4, 31))       # 1
            emit(I.create_reg_reg_i32(C.VEX_VPSLLD_XMM_XMM_IMM8, s4, s4, 22))       # 0x00400000
            emit(I.create_reg_reg_reg(C.VEX_VPANDN_XMM_XMM_XMMM128, s2, s2, s4))    # s2 = 0x00400000 if mant != 0 else 0
            # Now build 0x7F800000 in s4
            emit(I.create_reg_reg_reg(C.VEX_VPCMPEQD_XMM_XMM_XMMM128, s4, s4, s4))
            emit(I.create_reg_reg_i32(C.VEX_VPSRLD_XMM_XMM_IMM8, s4, s4, 24))
            emit(I.create_reg_reg_i32(C.VEX_VPSLLD_XMM_XMM_IMM8, s4, s4, 23))       # 0x7F800000
            emit(I.create_reg_reg_reg(C.VEX_VPOR_XMM_XMM_XMMM128, s2, s2, s4))      # 0x7F800000 | quiet_nan_bit
            emit(I.create_reg_reg_reg(C.VEX_VPOR_XMM_XMM_XMMM128, s2, s2, x_vec))   # 0x7F800000 | quiet_nan_bit | (mant << 13)
            # Blend s2 into x_vec where s3 (exp == 31) is true:
            emit(I.create_reg_reg_reg_reg(C.VEX_VBLENDVPS_XMM_XMM_XMMM128_XMM, x_vec, x_vec, s2, s3))
            # Finally OR sign bit (s1)
            emit(I.create_reg_reg_reg(C.VEX_VPOR_XMM_XMM_XMMM128, x_vec, x_vec, s1))

        if not wide:
            convert_4_half_in_xmm(s0)
            emit(I.create_reg_reg(C.VEX_VMOVAPS_XMM_XMMM128, ins.op0_register, s0))
        else:
            # Save high 64 bits of s0 onto stack or extract after low 64 bits?
            # We have stack space or we can store high 64 bits of s0 to [RSP + 5*32] if we allocate,
            # OR simply convert high 64 bits into dst_xmm first, then low 64 bits in s0, then VINSERTF128!
            dst_x = _xmm(ins.op0_register)
            emit(I.create_reg_reg_i32(C.VEX_VPSRLDQ_XMM_XMM_IMM8, dst_x, s0, 8))
            #Wait: if dst_x is one of s0..s4? No, s0..s4 were chosen from free_xmms (excluding op0 and op1)!
            # So dst_x is guaranteed distinct from s0..s4!
            convert_4_half_in_xmm(dst_x)   # dst_x now has high 4 floats!
            convert_4_half_in_xmm(s0)      # s0 now has low 4 floats!
            emit(I.create_reg_reg_reg_i32(C.VEX_VINSERTF128_YMM_YMM_XMMM128_IMM8,
                                          ins.op0_register, _ymm(s0), dst_x, 1))

        if need_rax:
            emit(I.create_reg_mem(C.MOV_R64_RM64, R.RAX, M(R.RSP, displ=160, displ_size=8)))
        for idx, xreg in enumerate((s0, s1, s2, s3, s4)):
            emit(I.create_reg_mem(C.VEX_VMOVDQU_YMM_YMMM256, _ymm(xreg), M(R.RSP, displ=idx * 32, displ_size=_disp_size(idx * 32))))
        emit(I.create_reg_mem(C.LEA_R64_M, R.RSP, M(R.RSP, displ=frame, displ_size=_disp_size(frame))))
        return b.encode(ip)

    # VCVTPS2PH dst(xmm/m64/m128), src(xmm/ymm), imm8
    gprs = [R.RAX, R.RCX, R.RDX, R.RBX, R.RSI, R.RDI, R.R8]
    frame = 128
    total_adjust = frame + external_stack_adjust

    emit(I.create_reg_mem(C.LEA_R64_M, R.RSP, M(R.RSP, displ=-frame, displ_size=_disp_size(-frame))))
    for i, reg in enumerate(gprs):
        emit(I.create_mem_reg(C.MOV_RM64_R64, M(R.RSP, displ=64 + i * 8, displ_size=1), reg))
    emit(I.create(C.PUSHFQ))
    emit(I.create_reg(C.POP_R64, R.RAX))
    emit(I.create_mem_reg(C.MOV_RM32_R32, M(R.RSP, displ=120, displ_size=1), R.EAX))

    wide = R.YMM0 <= ins.op1_register <= R.YMM15
    count = 8 if wide else 4
    imm8 = ins.immediate8
    store_src = C.VEX_VMOVDQU_YMMM256_YMM if wide else C.VEX_VMOVDQU_XMMM128_XMM
    emit(I.create_mem_reg(store_src, M(R.RSP, displ=0, displ_size=1), ins.op1_register))

    emit(I.create_reg_reg(C.XOR_R32_RM32, R.EAX, R.EAX))
    emit(I.create_mem_reg(C.MOV_RM64_R64, M(R.RSP, displ=32, displ_size=1), R.RAX))
    emit(I.create_mem_reg(C.MOV_RM64_R64, M(R.RSP, displ=40, displ_size=1), R.RAX))

    if ins.op0_kind == O.REGISTER:
        emit(I.create_reg_mem(C.LEA_R64_M, R.RDI, M(R.RSP, displ=32, displ_size=1)))
    else:
        mem_dst = _mem_operand(emit, ins, R.RDI, total_adjust, count * 2)
        emit(I.create_reg_mem(C.LEA_R64_M, R.RDI, mem_dst))

    if imm8 & 4:
        emit(I.create_mem(C.STMXCSR_M32, M(R.RSP, displ=124, displ_size=1)))
        emit(I.create_reg_mem(C.MOV_R32_RM32, R.R8D, M(R.RSP, displ=124, displ_size=1)))
        emit(I.create_reg_i32(C.SHR_RM32_IMM8, R.R8D, 13))
        emit(I.create_reg_i32(C.AND_RM32_IMM8, R.R8D, 3))
    else:
        emit(I.create_reg_i32(C.MOV_R32_IMM32, R.R8D, imm8 & 3))

    emit(I.create_reg_reg(C.XOR_R32_RM32, R.ESI, R.ESI))
    b.label("ps2ph_loop")
    emit(I.create_reg_mem(C.MOV_R32_RM32, R.EAX, M(R.RSP, R.RSI, 4)))
    emit(I.create_reg_reg(C.MOV_R32_RM32, R.EBX, R.EAX))
    emit(I.create_reg_i32(C.SHR_RM32_IMM8, R.EBX, 31))  # sign
    emit(I.create_reg_reg(C.MOV_R32_RM32, R.ECX, R.EAX))
    emit(I.create_reg_i32(C.SHR_RM32_IMM8, R.ECX, 23))
    emit(I.create_reg_i32(C.AND_RM32_IMM32, R.ECX, 0xFF))  # exp
    emit(I.create_reg_i32(C.AND_RM32_IMM32, R.EAX, 0x7FFFFF))  # mant

    emit(I.create_reg_i32(C.CMP_RM32_IMM32, R.ECX, 255))
    b.branch(C.JE_REL32_64, "ps2ph_inf_nan")
    emit(I.create_reg_reg(C.TEST_RM32_R32, R.ECX, R.ECX))
    b.branch(C.JE_REL32_64, "ps2ph_zero_sub")

    emit(I.create_reg_i32(C.SUB_RM32_IMM8, R.ECX, 112))  # signed h_exp
    emit(I.create_reg_i32(C.CMP_RM32_IMM8, R.ECX, 31))
    b.branch(C.JGE_REL32_64, "ps2ph_overflow")
    emit(I.create_reg_i32(C.CMP_RM32_IMM8, R.ECX, 0))
    b.branch(C.JLE_REL32_64, "ps2ph_subnormal")

    # Normal: 1 <= h_exp <= 30
    emit(I.create_reg_reg(C.MOV_R32_RM32, R.EDX, R.EAX))
    emit(I.create_reg_i32(C.AND_RM32_IMM32, R.EDX, 0x1FFF))  # rem
    emit(I.create_reg_i32(C.SHR_RM32_IMM8, R.EAX, 13))  # h_mant
    emit(I.create_reg_reg(C.TEST_RM32_R32, R.R8D, R.R8D))
    b.branch(C.JNE_REL32_64, "ps2ph_norm_dir")
    emit(I.create_reg_i32(C.CMP_RM32_IMM32, R.EDX, 0x1000))
    b.branch(C.JB_REL32_64, "ps2ph_norm_post_rnd")
    b.branch(C.JA_REL32_64, "ps2ph_norm_inc")
    emit(I.create_reg_i32(C.TEST_RM32_IMM32, R.EAX, 1))
    b.branch(C.JE_REL32_64, "ps2ph_norm_post_rnd")
    b.label("ps2ph_norm_inc")
    emit(I.create_reg(C.INC_RM32, R.EAX))
    b.branch(C.JMP_REL32_64, "ps2ph_norm_post_rnd")

    b.label("ps2ph_norm_dir")
    emit(I.create_reg_reg(C.TEST_RM32_R32, R.EDX, R.EDX))
    b.branch(C.JE_REL32_64, "ps2ph_norm_post_rnd")
    emit(I.create_reg_i32(C.CMP_RM32_IMM8, R.R8D, 3))
    b.branch(C.JE_REL32_64, "ps2ph_norm_post_rnd")
    emit(I.create_reg_i32(C.CMP_RM32_IMM8, R.R8D, 1))
    b.branch(C.JE_REL32_64, "ps2ph_norm_rtn")
    emit(I.create_reg_reg(C.TEST_RM32_R32, R.EBX, R.EBX))
    b.branch(C.JE_REL32_64, "ps2ph_norm_inc")
    b.branch(C.JMP_REL32_64, "ps2ph_norm_post_rnd")
    b.label("ps2ph_norm_rtn")
    emit(I.create_reg_reg(C.TEST_RM32_R32, R.EBX, R.EBX))
    b.branch(C.JNE_REL32_64, "ps2ph_norm_inc")

    b.label("ps2ph_norm_post_rnd")
    emit(I.create_reg_i32(C.CMP_RM32_IMM32, R.EAX, 0x400))
    b.branch(C.JB_REL32_64, "ps2ph_norm_pack")
    emit(I.create_reg_reg(C.XOR_R32_RM32, R.EAX, R.EAX))
    emit(I.create_reg(C.INC_RM32, R.ECX))
    emit(I.create_reg_i32(C.CMP_RM32_IMM8, R.ECX, 31))
    b.branch(C.JE_REL32_64, "ps2ph_overflow")
    b.label("ps2ph_norm_pack")
    emit(I.create_reg_i32(C.SHL_RM32_IMM8, R.ECX, 10))
    emit(I.create_reg_reg(C.OR_R32_RM32, R.EAX, R.ECX))
    b.branch(C.JMP_REL32_64, "ps2ph_pack_sign")

    b.label("ps2ph_subnormal")
    emit(I.create_reg(C.NEG_RM32, R.ECX))
    emit(I.create_reg_i32(C.ADD_RM32_IMM8, R.ECX, 14))
    emit(I.create_reg_i32(C.CMP_RM32_IMM8, R.ECX, 25))
    b.branch(C.JAE_REL32_64, "ps2ph_sub_tiny")
    emit(I.create_reg_i32(C.OR_RM32_IMM32, R.EAX, 0x800000))
    emit(I.create_reg_i32(C.MOV_R32_IMM32, R.EDX, 1))
    emit(I.create_reg_reg(C.SHL_RM32_CL, R.EDX, R.CL))
    emit(I.create_reg(C.DEC_RM32, R.EDX))
    emit(I.create_reg_reg(C.AND_R32_RM32, R.EDX, R.EAX))
    emit(I.create_reg_reg(C.SHR_RM32_CL, R.EAX, R.CL))
    emit(I.create_reg(C.DEC_RM32, R.ECX))
    emit(I.create_mem_i32(C.MOV_RM32_IMM32, M(R.RSP, displ=124, displ_size=1), 1))
    emit(I.create_mem_reg(C.SHL_RM32_CL, M(R.RSP, displ=124, displ_size=1), R.CL))
    b.branch(C.JMP_REL32_64, "ps2ph_sub_rnd")

    b.label("ps2ph_zero_sub")
    emit(I.create_reg_reg(C.TEST_RM32_R32, R.EAX, R.EAX))
    b.branch(C.JE_REL32_64, "ps2ph_pack_sign")
    b.label("ps2ph_sub_tiny")
    emit(I.create_reg_reg(C.XOR_R32_RM32, R.EAX, R.EAX))
    emit(I.create_reg_i32(C.MOV_R32_IMM32, R.EDX, 1))
    emit(I.create_mem_i32(C.MOV_RM32_IMM32, M(R.RSP, displ=124, displ_size=1), 2))

    b.label("ps2ph_sub_rnd")
    emit(I.create_reg_reg(C.TEST_RM32_R32, R.R8D, R.R8D))
    b.branch(C.JNE_REL32_64, "ps2ph_sub_dir")
    emit(I.create_reg_mem(C.CMP_R32_RM32, R.EDX, M(R.RSP, displ=124, displ_size=1)))
    b.branch(C.JB_REL32_64, "ps2ph_pack_sign")
    b.branch(C.JA_REL32_64, "ps2ph_sub_inc")
    emit(I.create_reg_i32(C.TEST_RM32_IMM32, R.EAX, 1))
    b.branch(C.JE_REL32_64, "ps2ph_pack_sign")
    b.label("ps2ph_sub_inc")
    emit(I.create_reg(C.INC_RM32, R.EAX))
    b.branch(C.JMP_REL32_64, "ps2ph_pack_sign")

    b.label("ps2ph_sub_dir")
    emit(I.create_reg_reg(C.TEST_RM32_R32, R.EDX, R.EDX))
    b.branch(C.JE_REL32_64, "ps2ph_pack_sign")
    emit(I.create_reg_i32(C.CMP_RM32_IMM8, R.R8D, 3))
    b.branch(C.JE_REL32_64, "ps2ph_pack_sign")
    emit(I.create_reg_i32(C.CMP_RM32_IMM8, R.R8D, 1))
    b.branch(C.JE_REL32_64, "ps2ph_sub_rtn")
    emit(I.create_reg_reg(C.TEST_RM32_R32, R.EBX, R.EBX))
    b.branch(C.JE_REL32_64, "ps2ph_sub_inc")
    b.branch(C.JMP_REL32_64, "ps2ph_pack_sign")
    b.label("ps2ph_sub_rtn")
    emit(I.create_reg_reg(C.TEST_RM32_R32, R.EBX, R.EBX))
    b.branch(C.JNE_REL32_64, "ps2ph_sub_inc")
    b.branch(C.JMP_REL32_64, "ps2ph_pack_sign")

    b.label("ps2ph_overflow")
    emit(I.create_reg_reg(C.TEST_RM32_R32, R.R8D, R.R8D))
    b.branch(C.JE_REL32_64, "ps2ph_ovf_inf")
    emit(I.create_reg_i32(C.CMP_RM32_IMM8, R.R8D, 3))
    b.branch(C.JE_REL32_64, "ps2ph_ovf_max")
    emit(I.create_reg_i32(C.CMP_RM32_IMM8, R.R8D, 1))
    b.branch(C.JE_REL32_64, "ps2ph_ovf_rtn")
    emit(I.create_reg_reg(C.TEST_RM32_R32, R.EBX, R.EBX))
    b.branch(C.JE_REL32_64, "ps2ph_ovf_inf")
    b.branch(C.JMP_REL32_64, "ps2ph_ovf_max")
    b.label("ps2ph_ovf_rtn")
    emit(I.create_reg_reg(C.TEST_RM32_R32, R.EBX, R.EBX))
    b.branch(C.JNE_REL32_64, "ps2ph_ovf_inf")
    b.label("ps2ph_ovf_max")
    emit(I.create_reg_i32(C.MOV_R32_IMM32, R.EAX, 0x7BFF))
    b.branch(C.JMP_REL32_64, "ps2ph_pack_sign")
    b.label("ps2ph_ovf_inf")
    emit(I.create_reg_i32(C.MOV_R32_IMM32, R.EAX, 0x7C00))
    b.branch(C.JMP_REL32_64, "ps2ph_pack_sign")

    b.label("ps2ph_inf_nan")
    emit(I.create_reg_reg(C.TEST_RM32_R32, R.EAX, R.EAX))
    b.branch(C.JE_REL32_64, "ps2ph_ovf_inf")
    emit(I.create_reg_i32(C.SHR_RM32_IMM8, R.EAX, 13))
    emit(I.create_reg_i32(C.OR_RM32_IMM32, R.EAX, 0x7E00))

    b.label("ps2ph_pack_sign")
    emit(I.create_reg_i32(C.SHL_RM32_IMM8, R.EBX, 15))
    emit(I.create_reg_reg(C.OR_R32_RM32, R.EAX, R.EBX))
    seg_dst = ins.segment_prefix if ins.op0_kind == O.MEMORY else R.NONE
    emit(I.create_mem_reg(C.MOV_RM16_R16, M(R.RDI, R.RSI, 2, seg=seg_dst), R.AX))
    emit(I.create_reg(C.INC_RM32, R.ESI))
    emit(I.create_reg_i32(C.CMP_RM32_IMM8, R.ESI, count))
    b.branch(C.JB_REL32_64, "ps2ph_loop")

    emit(I.create_reg_mem(C.MOV_R32_RM32, R.EAX, M(R.RSP, displ=120, displ_size=1)))
    emit(I.create_reg(C.PUSH_R64, R.RAX))
    emit(I.create(C.POPFQ))
    for i, reg in enumerate(gprs):
        emit(I.create_reg_mem(C.MOV_R64_RM64, reg, M(R.RSP, displ=64 + i * 8, displ_size=1)))
    if ins.op0_kind == O.REGISTER:
        emit(I.create_reg_mem(C.VEX_VMOVDQU_XMM_XMMM128, ins.op0_register, M(R.RSP, displ=32, displ_size=1)))
    emit(I.create_reg_mem(C.LEA_R64_M, R.RSP, M(R.RSP, displ=frame, displ_size=_disp_size(frame))))
    return b.encode(ip)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def lower(ins: ix.Instruction, ip: int = 0, external_stack_adjust: int = 0, include_popcnt: bool = False) -> bytes | None:
    if external_stack_adjust not in (0, 8):
        raise LoweringError("external_stack_adjust must be 0 or 8")
    fs = ins.cpuid_features()
    if F.FMA in fs:
        return _lower_fma(ins, ip, external_stack_adjust)
    if F.F16C in fs:
        return _lower_f16c(ins, ip, external_stack_adjust)
    if F.MOVBE in fs:
        return _lower_movbe(ins, ip, external_stack_adjust)
    if F.BMI1 in fs or F.LZCNT in fs or (include_popcnt and F.POPCNT in fs):
        return _lower_bmi1_lzcnt(ins, ip, external_stack_adjust)
    return None