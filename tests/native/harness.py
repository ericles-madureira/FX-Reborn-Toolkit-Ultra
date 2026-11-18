"""Native execution harness for generated lowering code (Windows x64, AVX1 CPU).

Runs a block of machine code with a fully controlled register/flag/stack/memory
state and returns the resulting state, so lowered sequences can be checked
against a Python reference model or against another lowering, bit for bit.

The generated code under test only needs AVX1 (it is what an AVX-only CPU
runs), so this works on Sandy Bridge / Ivy Bridge / AMD FX machines.

State layout (bytes):
    0     ymm0..ymm15          16 x 32
    512   rax..r15             16 x 8   (rsp slot receives rsp delta)
    640   rflags
    648   rsp delta after the test (must be 0)
    656   mxcsr
    1024  stack image          1024 bytes, [rsp+0] .. [rsp+1023] during the test
    2048  data memory          1024 bytes, point a GPR at STATE+2048 to use it
"""
from __future__ import annotations

import ctypes
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'engine' / 'pylibs'))
sys.path.insert(0, str(ROOT / 'engine'))
import iced_x86 as ix  # noqa: E402

R, C, I, M = ix.Register, ix.Code, ix.Instruction, ix.MemoryOperand

OFF_YMM, OFF_GPR, OFF_FLAGS, OFF_RSPD, OFF_MXCSR = 0, 512, 640, 648, 656
OFF_STACK, OFF_MEM, STATE_SIZE = 1024, 2048, 3072
GPRS64 = [R.RAX, R.RCX, R.RDX, R.RBX, R.RSP, R.RBP, R.RSI, R.RDI,
          R.R8, R.R9, R.R10, R.R11, R.R12, R.R13, R.R14, R.R15]
YMMS = [R.YMM0 + n for n in range(16)]
XMMS = [R.XMM0 + n for n in range(16)]
STATUS_FLAGS = 0x8D5  # CF PF AF ZF SF OF

_k32 = ctypes.WinDLL('kernel32', use_last_error=True)
_k32.VirtualAlloc.restype = ctypes.c_void_p
_k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
_k32.FlushInstructionCache.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]

CODE_SIZE = 1 << 16
_region = _k32.VirtualAlloc(None, CODE_SIZE + 4 * STATE_SIZE + 4096, 0x3000, 0x40)
if not _region:
    raise OSError('VirtualAlloc failed')
CODE = _region
SLOTS = _region + CODE_SIZE              # saved state ptr, saved rsp, tmp, test rsp, saved mxcsr
STATE = _region + CODE_SIZE + 4096       # 4 KiB aligned
DATA = STATE + OFF_MEM
POOL = SLOTS + 0x100                     # read-only constants for lowerings (sign masks)


def _mem(base, disp=0):
    return M(base, displ=disp, displ_size=8 if disp else 0)


def _rip(addr):
    return M(R.RIP, displ=addr, displ_size=8)


SLOT_STATE, SLOT_RSP, SLOT_TMP, SLOT_TEST_RSP, SLOT_MXCSR = (SLOTS + 8 * n for n in range(5))


def _prefix():
    out = []
    for reg in (R.RBX, R.RBP, R.RSI, R.RDI, R.R12, R.R13, R.R14, R.R15):
        out.append(I.create_reg(C.PUSH_R64, reg))
    out.append(I.create_reg_i32(C.SUB_RM64_IMM32, R.RSP, 168))
    for n in range(6, 16):
        out.append(I.create_mem_reg(C.MOVDQU_XMMM128_XMM, _mem(R.RSP, (n - 6) * 16), XMMS[n]))
    out.append(I.create_mem_reg(C.MOV_RM64_R64, _rip(SLOT_STATE), R.RCX))
    out.append(I.create_mem_reg(C.MOV_RM64_R64, _rip(SLOT_RSP), R.RSP))
    out.append(I.create_mem(C.STMXCSR_M32, _rip(SLOT_MXCSR)))
    out.append(I.create_reg_mem(C.LEA_R64_M, R.RSP, M(R.RSP, displ=-4096, displ_size=8)))
    out.append(I.create_mem_reg(C.MOV_RM64_R64, _rip(SLOT_TEST_RSP), R.RSP))
    for k in range(32):
        out.append(I.create_reg_mem(C.VEX_VMOVDQU_YMM_YMMM256, R.YMM0, _mem(R.RCX, OFF_STACK + 32 * k)))
        out.append(I.create_mem_reg(C.VEX_VMOVDQU_YMMM256_YMM, _mem(R.RSP, 32 * k), R.YMM0))
    for n in range(16):
        out.append(I.create_reg_mem(C.VEX_VMOVDQU_YMM_YMMM256, YMMS[n], _mem(R.RCX, OFF_YMM + 32 * n)))
    out.append(I.create_mem(C.LDMXCSR_M32, _mem(R.RCX, OFF_MXCSR)))
    out.append(I.create_mem(C.PUSH_RM64, _mem(R.RCX, OFF_FLAGS)))
    out.append(I.create(C.POPFQ))
    for n, reg in enumerate(GPRS64):
        if reg in (R.RSP, R.RCX):
            continue
        out.append(I.create_reg_mem(C.MOV_R64_RM64, reg, _mem(R.RCX, OFF_GPR + 8 * n)))
    out.append(I.create_reg_mem(C.MOV_R64_RM64, R.RCX, _mem(R.RCX, OFF_GPR + 8)))
    return out


def _postfix():
    out = [I.create_mem_reg(C.MOV_RM64_R64, _rip(SLOT_TMP), R.RAX),
           I.create_reg_mem(C.MOV_R64_RM64, R.RAX, _rip(SLOT_STATE))]
    for n, reg in enumerate(GPRS64):
        if reg in (R.RSP, R.RAX):
            continue
        out.append(I.create_mem_reg(C.MOV_RM64_R64, _mem(R.RAX, OFF_GPR + 8 * n), reg))
    out += [I.create_reg_mem(C.MOV_R64_RM64, R.RCX, _rip(SLOT_TMP)),
            I.create_mem_reg(C.MOV_RM64_R64, _mem(R.RAX, OFF_GPR), R.RCX),
            I.create(C.PUSHFQ),
            I.create_reg(C.POP_R64, R.RCX),
            I.create_mem_reg(C.MOV_RM64_R64, _mem(R.RAX, OFF_FLAGS), R.RCX),
            I.create_reg_reg(C.MOV_R64_RM64, R.RCX, R.RSP),
            I.create_reg_mem(C.SUB_R64_RM64, R.RCX, _rip(SLOT_TEST_RSP)),
            I.create_mem_reg(C.MOV_RM64_R64, _mem(R.RAX, OFF_RSPD), R.RCX),
            I.create_mem(C.STMXCSR_M32, _mem(R.RAX, OFF_MXCSR))]
    for n in range(16):
        out.append(I.create_mem_reg(C.VEX_VMOVDQU_YMMM256_YMM, _mem(R.RAX, OFF_YMM + 32 * n), YMMS[n]))
    out.append(I.create_reg_mem(C.MOV_R64_RM64, R.RCX, _rip(SLOT_TEST_RSP)))
    for k in range(32):
        out.append(I.create_reg_mem(C.VEX_VMOVDQU_YMM_YMMM256, R.YMM0, _mem(R.RCX, 32 * k)))
        out.append(I.create_mem_reg(C.VEX_VMOVDQU_YMMM256_YMM, _mem(R.RAX, OFF_STACK + 32 * k), R.YMM0))
    out += [I.create_reg_mem(C.MOV_R64_RM64, R.RSP, _rip(SLOT_RSP)),
            I.create_mem(C.LDMXCSR_M32, _rip(SLOT_MXCSR)),
            I.create(C.CLD)]
    for n in range(6, 16):
        out.append(I.create_reg_mem(C.MOVDQU_XMM_XMMM128, XMMS[n], _mem(R.RSP, (n - 6) * 16)))
    out.append(I.create_reg_i32(C.ADD_RM64_IMM32, R.RSP, 168))
    for reg in reversed((R.RBX, R.RBP, R.RSI, R.RDI, R.R12, R.R13, R.R14, R.R15)):
        out.append(I.create_reg(C.POP_R64, reg))
    out += [I.create(C.VEX_VZEROUPPER), I.create(C.RETNQ)]
    return out


def _install_pool():
    import sandy_lower
    pool, off = {}, 0
    for key, blob in sandy_lower.SIGN_POOL_BYTES.items():
        ctypes.memmove(POOL + off, blob, len(blob))
        pool[key] = POOL + off
        off += 32
    return pool


SIGN_POOL = _install_pool()


def _encode(instructions, ip):
    enc = ix.BlockEncoder(64)
    enc.add_many(instructions)
    return enc.encode(ip)


_PREFIX = _encode(_prefix(), CODE)
TEST_IP = CODE + ((len(_PREFIX) + 63) & ~63)
_RUN = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(CODE)


def install(test_bytes: bytes, tail_gap: int = 0):
    """Place test bytes at TEST_IP (they must have been encoded for TEST_IP)."""
    pad = TEST_IP - CODE - len(_PREFIX)
    body = _PREFIX + b'\x90' * pad + test_bytes
    post_ip = CODE + len(body)
    body += _encode(_postfix(), post_ip)
    if len(body) > CODE_SIZE:
        raise ValueError('test code too large')
    ctypes.memmove(CODE, body, len(body))
    _k32.FlushInstructionCache(ctypes.c_void_p(-1), ctypes.c_void_p(CODE), len(body))


def run(state: bytes | bytearray) -> bytearray:
    if len(state) != STATE_SIZE:
        raise ValueError('bad state size')
    ctypes.memmove(STATE, bytes(state), STATE_SIZE)
    _RUN(STATE)
    return bytearray(ctypes.string_at(STATE, STATE_SIZE))


def stub_wrapper(lowered_for: 'callable', with_ret: bool = True) -> bytes:
    """Build `call S; jmp END; S: <body(adjust=8)>; ret; END:` at TEST_IP.

    lowered_for(ip) must return body bytes encoded for that ip with
    external_stack_adjust=8 (a return address is on the stack)."""
    stub_ip = TEST_IP + 10
    body = lowered_for(stub_ip)
    end = stub_ip + len(body) + 1
    call = b'\xe8' + struct.pack('<i', stub_ip - (TEST_IP + 5))
    jmp = b'\xe9' + struct.pack('<i', end - (TEST_IP + 10))
    return call + jmp + body + b'\xc3'


# ---------------------------------------------------------------- state helpers

def new_state(rng, *, floats: str | None = None, flags: int | None = None) -> bytearray:
    st = bytearray(rng.randbytes(STATE_SIZE))
    if floats == 'ps':
        for k in range(128):
            struct.pack_into('<f', st, OFF_YMM + 4 * k, rand_f32(rng))
    elif floats == 'pd':
        for k in range(64):
            struct.pack_into('<d', st, OFF_YMM + 8 * k, rand_f64(rng))
    if floats == 'ps':
        for k in range(256):
            struct.pack_into('<f', st, OFF_MEM + 4 * k, rand_f32(rng))
            struct.pack_into('<f', st, OFF_STACK + 4 * k, rand_f32(rng))
    elif floats == 'pd':
        for k in range(128):
            struct.pack_into('<d', st, OFF_MEM + 8 * k, rand_f64(rng))
            struct.pack_into('<d', st, OFF_STACK + 8 * k, rand_f64(rng))
    fl = rng.getrandbits(12) & STATUS_FLAGS if flags is None else flags
    struct.pack_into('<Q', st, OFF_FLAGS, 0x202 | fl)
    struct.pack_into('<Q', st, OFF_RSPD, 0)
    struct.pack_into('<I', st, OFF_MXCSR, 0x1F80)
    struct.pack_into('<I', st, OFF_MXCSR + 4, 0)
    return st


def rand_f32(rng):
    e = rng.randint(-20, 20)
    return rng.choice((-1, 1)) * rng.random() * (2.0 ** e)


def rand_f64(rng):
    e = rng.randint(-60, 60)
    return rng.choice((-1, 1)) * rng.random() * (2.0 ** e)


def set_gpr(st, reg, value):
    struct.pack_into('<Q', st, OFF_GPR + 8 * GPRS64.index(reg), value & ((1 << 64) - 1))


def gpr(st, reg):
    return struct.unpack_from('<Q', st, OFF_GPR + 8 * GPRS64.index(reg))[0]


def ymm_bytes(st, n):
    return bytes(st[OFF_YMM + 32 * n: OFF_YMM + 32 * n + 32])


def flags(st):
    return struct.unpack_from('<Q', st, OFF_FLAGS)[0]


def rsp_delta(st):
    return struct.unpack_from('<q', st, OFF_RSPD)[0]


def decode_one(ins: ix.Instruction, ip: int = 0x140001000) -> ix.Instruction:
    """Encode then decode, giving a real decoded instruction (len, ip, flags)."""
    data = ix.Encoder(64)
    data.encode(ins, ip)
    raw = data.take_buffer()
    return ix.Decoder(64, raw, ip=ip).decode()
