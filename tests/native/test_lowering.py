"""Bit-exact native tests for the lowering engines.

Every generated sequence is executed on THIS CPU (AVX1 is enough) from a random
register/flag/stack/memory state and compared against an independent Python
reference model of the instruction being emulated. Each case runs both
in place (external_stack_adjust=0) and inside a call stub (adjust=8), the way
the patcher uses them.

    python\\python.exe tests\\native\\test_lowering.py            (all)
    python\\python.exe tests\\native\\test_lowering.py fma bmi1   (some groups)
"""
from __future__ import annotations

import ctypes
import itertools
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import harness as h  # noqa: E402
from harness import R, C, I, M  # noqa: E402
import iced_x86 as ix  # noqa: E402
import sandy_lower  # noqa: E402
import avx2_lower  # noqa: E402
import bmi2_lower  # noqa: E402

FAKE_IP = h.DATA - 0x20000
MASK64 = (1 << 64) - 1
failures: list[str] = []
counts = {'cases': 0, 'runs': 0}


def f32(x):
    return ctypes.c_float(x).value


# ---------------------------------------------------------------- runner

def check(name, ins, lower, expect, rng, *, floats=None, iterations=24, setup=None, flag_mask=h.STATUS_FLAGS):
    """Run lower(ins, ip, adjust) natively and compare with expect(state)->state."""
    counts['cases'] += 1
    for adjust in (0, 8):
        try:
            if adjust:
                code = h.stub_wrapper(lambda ip: lower(ins, ip, 8))
            else:
                code = lower(ins, h.TEST_IP, 0)
        except Exception as err:  # noqa: BLE001
            failures.append(f'{name} adjust={adjust}: lowering raised {err!r}')
            return
        if code is None:
            failures.append(f'{name} adjust={adjust}: lowering returned None')
            return
        h.install(code)
        for _ in range(iterations):
            st = h.new_state(rng, floats=floats)
            h.set_gpr(st, R.RBX, h.DATA + 256)
            h.set_gpr(st, R.RSI, rng.randrange(0, 16))
            if setup:
                setup(st, rng)
            exp = expect(bytearray(st))
            got = h.run(st)
            counts['runs'] += 1
            problem = _compare(exp, got, flag_mask)
            if problem:
                failures.append(f'{name} adjust={adjust}: {problem}')
                return


def _compare(exp, got, flag_mask):
    if h.rsp_delta(got) != 0:
        return f'rsp changed by {h.rsp_delta(got)}'
    for n in range(16):
        if h.ymm_bytes(exp, n) != h.ymm_bytes(got, n):
            return f'ymm{n} expected {h.ymm_bytes(exp, n).hex()} got {h.ymm_bytes(got, n).hex()}'
    for reg in h.GPRS64:
        if reg == R.RSP:
            continue
        if h.gpr(exp, reg) != h.gpr(got, reg):
            return f'{reg} expected {h.gpr(exp, reg):#x} got {h.gpr(got, reg):#x}'
    if (h.flags(exp) ^ h.flags(got)) & flag_mask:
        return f'flags expected {h.flags(exp):#x} got {h.flags(got):#x}'
    if exp[h.OFF_STACK:] != got[h.OFF_STACK:]:
        return 'stack image or data memory differs'
    return None


def mem_operand(kind, rng, size):
    """Returns (MemoryOperand, state offset of the operand)."""
    if kind == 'rbx':
        disp = rng.choice([0, 8, -64, 96])
        return M(R.RBX, displ=disp, displ_size=1 if disp else 0), h.OFF_MEM + 256 + disp
    if kind == 'rbx_rsi':
        return M(R.RBX, R.RSI, 8, displ=16, displ_size=1), None   # resolved in setup via fixed rsi
    if kind == 'rsp':
        disp = rng.choice([0, 16, 40, 0x100])
        return M(R.RSP, displ=disp, displ_size=1 if disp < 128 else 8), h.OFF_STACK + disp
    if kind == 'rip':
        off = rng.choice([0, 64, 512])
        return M(R.RIP, displ=h.DATA + off, displ_size=8), h.OFF_MEM + off
    raise ValueError(kind)


def decoded(ins):
    return h.decode_one(ins, FAKE_IP)


# ---------------------------------------------------------------- FMA

FMA_KINDS = ['FMADD', 'FMSUB', 'FNMADD', 'FNMSUB', 'FMADDSUB', 'FMSUBADD']


def fma_code(kind, order, suffix, wide, mem):
    shape = {'PS': 'YMM_YMM_YMMM256' if wide else 'XMM_XMM_XMMM128',
             'PD': 'YMM_YMM_YMMM256' if wide else 'XMM_XMM_XMMM128',
             'SS': 'XMM_XMM_XMMM32', 'SD': 'XMM_XMM_XMMM64'}[suffix]
    return getattr(C, f'VEX_V{kind}{order}{suffix}_{shape}', None)


def fma_expect(kind, order, suffix, wide, d, s2, op3, mem_off):
    double = suffix in ('PD', 'SD')
    scalar = suffix in ('SS', 'SD')
    fmt, esize = ('d', 8) if double else ('f', 4)
    lanes = 1 if scalar else ((32 if wide else 16) // esize)
    rnd = (lambda x: x) if double else f32

    def expect(st):
        def vec(n):
            return list(struct.unpack_from(f'<{32 // esize}{fmt}', st, h.OFF_YMM + 32 * n))
        a1 = vec(d)
        a2 = vec(s2)
        if mem_off is None:
            a3 = vec(op3)
        else:
            a3 = list(struct.unpack_from(f'<{lanes}{fmt}', st, mem_off)) + [0.0] * (32 // esize)
        out = list(a1)
        for i in range(lanes):
            if order == '132':
                a, b, c = a1[i], a3[i], a2[i]
            elif order == '213':
                a, b, c = a2[i], a1[i], a3[i]
            else:
                a, b, c = a2[i], a3[i], a1[i]
            p = rnd(a * b)
            sub = {'FMADD': False, 'FMSUB': True, 'FNMADD': None, 'FNMSUB': None,
                   'FMADDSUB': i % 2 == 0, 'FMSUBADD': i % 2 == 1}[kind]
            if kind == 'FNMADD':
                r = rnd(c - p)
            elif kind == 'FNMSUB':
                r = rnd(-p - c)
            else:
                r = rnd(p - c) if sub else rnd(p + c)
            out[i] = r
        if not wide:
            out[16 // esize:] = [0.0] * (16 // esize)
        packed = struct.pack(f'<{32 // esize}{fmt}', *out)
        if not wide:
            packed = packed[:16] + bytes(16)
        st[h.OFF_YMM + 32 * d: h.OFF_YMM + 32 * d + 32] = packed
        return st
    return expect


def test_fma(rng, full=False):
    for pool in (True, False):
        sandy_lower.set_constant_pool(h.SIGN_POOL if pool else None)
        _test_fma(rng, full)
    sandy_lower.set_constant_pool(h.SIGN_POOL)


def _test_fma(rng, full=False):
    regs = [0, 1, 2, 7, 15]
    for kind, order, suffix in itertools.product(FMA_KINDS, ('132', '213', '231'), ('PS', 'PD', 'SS', 'SD')):
        for wide in ((False, True) if suffix in ('PS', 'PD') else (False,)):
            for opnd in ('reg', 'rbx', 'rsp', 'rip'):
                code = fma_code(kind, order, suffix, wide, opnd != 'reg')
                if code is None:
                    continue
                combos = list(itertools.product(regs, regs, regs if opnd == 'reg' else [None]))
                if not full:
                    combos = [c for c in combos if len(set(x for x in c if x is not None)) == len([x for x in c if x is not None])][:6] + \
                             [c for c in combos if len(set(x for x in c if x is not None)) < len([x for x in c if x is not None])][:6]
                for d, s2, s3 in combos:
                    vr = (lambda n: h.YMMS[n]) if wide else (lambda n: h.XMMS[n])
                    if opnd == 'reg':
                        ins = I.create_reg_reg_reg(code, vr(d), vr(s2), vr(s3))
                        mem_off = None
                    else:
                        size = 32 if wide else (16 if suffix in ('PS', 'PD') else (8 if suffix == 'SD' else 4))
                        mop, mem_off = mem_operand(opnd, rng, size)
                        ins = I.create_reg_reg_mem(code, vr(d), vr(s2), mop)
                    dec = decoded(ins)
                    name = f'{dec} [{opnd}]'
                    floats = 'pd' if suffix in ('PD', 'SD') else 'ps'
                    check(name, dec, lambda i, ip, adj: sandy_lower.lower(i, ip, adj),
                          fma_expect(kind, order, suffix, wide, d, s2, s3, mem_off), rng,
                          floats=floats, iterations=6)


# ---------------------------------------------------------------- BMI1 / LZCNT / POPCNT / MOVBE

def _gpr_regs(bits):
    if bits == 64:
        return h.GPRS64
    if bits == 32:
        return [R.EAX + n for n in range(16)]
    return [R.AX + n for n in range(16)]


def _full(reg):
    for bank in (h.GPRS64, [R.EAX + n for n in range(16)], [R.AX + n for n in range(16)]):
        if reg in bank:
            return h.GPRS64[bank.index(reg)]
    raise ValueError(reg)


def _write(st, reg, bits, value):
    full = _full(reg)
    mask = (1 << bits) - 1
    if bits == 16:
        value = (h.gpr(st, full) & ~0xFFFF) | (value & 0xFFFF)
    else:
        value &= mask   # 32-bit writes zero-extend
    h.set_gpr(st, full, value)


def _read(st, reg, bits):
    return h.gpr(st, _full(reg)) & ((1 << bits) - 1)


def _set_flags(st, **f):
    bits = {'CF': 0, 'PF': 2, 'AF': 4, 'ZF': 6, 'SF': 7, 'OF': 11}
    v = h.flags(st)
    for k, on in f.items():
        v = (v | (1 << bits[k])) if on else (v & ~(1 << bits[k]))
    struct.pack_into('<Q', st, h.OFF_FLAGS, v)


BMI_FLAGMASK = {
    'ANDN': 0x8C1, 'BLSI': 0x8C1, 'BLSMSK': 0x8C1, 'BLSR': 0x8C1, 'BEXTR': 0x8C1 & ~0x80 | 0x40,
    'TZCNT': 0x41, 'LZCNT': 0x41, 'POPCNT': 0x8D5,
}


def bmi_model(name, bits, dst, src_val):
    mask = (1 << bits) - 1

    def expect(st, a=None, b=None):
        a = src_val[0](st)
        b = src_val[1](st) if len(src_val) > 1 else None
        fl = {}
        if name == 'ANDN':
            r = ~a & b & mask
            fl = dict(CF=0, OF=0, ZF=r == 0, SF=r >> (bits - 1))
        elif name == 'BLSI':
            r = a & (-a) & mask
            fl = dict(CF=a != 0, OF=0, ZF=r == 0, SF=r >> (bits - 1))
        elif name == 'BLSMSK':
            r = (a ^ (a - 1)) & mask
            fl = dict(CF=a == 0, OF=0, ZF=0, SF=r >> (bits - 1))
        elif name == 'BLSR':
            r = (a & (a - 1)) & mask
            fl = dict(CF=a == 0, OF=0, ZF=r == 0, SF=r >> (bits - 1))
        elif name == 'BEXTR':
            start, length = b & 0xFF, (b >> 8) & 0xFF
            r = (a >> start) & ((1 << length) - 1) if start < bits else 0
            r &= mask
            fl = dict(CF=0, OF=0, ZF=r == 0)
        elif name == 'TZCNT':
            r = bits if a == 0 else (a & -a).bit_length() - 1
            fl = dict(CF=a == 0, ZF=r == 0)
        elif name == 'LZCNT':
            r = bits - a.bit_length()
            fl = dict(CF=a == 0, ZF=r == 0)
        elif name == 'POPCNT':
            r = bin(a).count('1')
            fl = dict(CF=0, OF=0, SF=0, AF=0, PF=0, ZF=r == 0)
        _write(st, dst, bits, r)
        _set_flags(st, **fl)
        return st
    return expect


def test_bmi1(rng):
    vex = {'ANDN': 'VEX_ANDN_R{b}_R{b}_RM{b}', 'BEXTR': 'VEX_BEXTR_R{b}_RM{b}_R{b}',
           'BLSI': 'VEX_BLSI_R{b}_RM{b}', 'BLSMSK': 'VEX_BLSMSK_R{b}_RM{b}', 'BLSR': 'VEX_BLSR_R{b}_RM{b}',
           'TZCNT': 'TZCNT_R{b}_RM{b}', 'LZCNT': 'LZCNT_R{b}_RM{b}', 'POPCNT': 'POPCNT_R{b}_RM{b}'}
    picks = [0, 1, 2, 3, 6, 8, 11]   # rax rcx rdx rbx(=mem base) rsi(=index) r8 r11
    for name, pattern in vex.items():
        for bits in (32, 64) if name not in ('TZCNT', 'LZCNT', 'POPCNT') else (16, 32, 64):
            code = getattr(C, pattern.format(b=bits))
            regs = _gpr_regs(bits)
            three = name in ('ANDN', 'BEXTR')
            for opnd in ('reg', 'rbx', 'rsp', 'rip'):
                combos = itertools.product(picks, picks, picks if three else [None])
                combos = [c for c in combos if 3 not in c[:1] and 6 not in c[:1]]
                rng.shuffle(combos)
                for d, a, b in combos[:10]:
                    if name == 'ANDN':
                        if opnd == 'reg':
                            ins = I.create_reg_reg_reg(code, regs[d], regs[a], regs[b])
                            srcs = (lambda st, r=regs[a]: _read(st, r, bits), lambda st, r=regs[b]: _read(st, r, bits))
                        else:
                            mop, off = mem_operand(opnd, rng, bits // 8)
                            ins = I.create_reg_reg_mem(code, regs[d], regs[a], mop)
                            srcs = (lambda st, r=regs[a]: _read(st, r, bits),
                                    lambda st, o=off: int.from_bytes(st[o:o + bits // 8], 'little'))
                    elif name == 'BEXTR':
                        if opnd == 'reg':
                            ins = I.create_reg_reg_reg(code, regs[d], regs[a], regs[b])
                            srcs = (lambda st, r=regs[a]: _read(st, r, bits), lambda st, r=regs[b]: _read(st, r, bits))
                        else:
                            mop, off = mem_operand(opnd, rng, bits // 8)
                            ins = I.create_reg_mem_reg(code, regs[d], mop, regs[b])
                            srcs = (lambda st, o=off: int.from_bytes(st[o:o + bits // 8], 'little'),
                                    lambda st, r=regs[b]: _read(st, r, bits))
                    else:
                        if opnd == 'reg':
                            ins = I.create_reg_reg(code, regs[d], regs[a])
                            srcs = (lambda st, r=regs[a]: _read(st, r, bits),)
                        else:
                            mop, off = mem_operand(opnd, rng, bits // 8)
                            ins = I.create_reg_mem(code, regs[d], mop)
                            srcs = (lambda st, o=off: int.from_bytes(st[o:o + bits // 8], 'little'),)
                    dec = decoded(ins)

                    def setup(st, rng, name=name, b=b, bits=bits):
                        # make interesting values: zeros, single bits, small BEXTR controls
                        for reg in h.GPRS64:
                            if reg in (R.RBX, R.RSI, R.RSP):
                                continue
                            choice = rng.random()
                            if choice < 0.15:
                                h.set_gpr(st, reg, 0)
                            elif choice < 0.3:
                                h.set_gpr(st, reg, 1 << rng.randrange(64))
                        if name == 'BEXTR' and b is not None:
                            reg = h.GPRS64[b]
                            if reg not in (R.RBX, R.RSI):
                                h.set_gpr(st, reg, rng.randrange(0, 72) | (rng.randrange(0, 72) << 8))
                        if rng.random() < 0.2:
                            st[h.OFF_MEM + 248: h.OFF_MEM + 360] = bytes(112)
                    check(f'{dec} [{opnd}]', dec, lambda i, ip, adj: sandy_lower.lower(i, ip, adj, include_popcnt=True),
                          bmi_model(name, bits, dec.op0_register, srcs), rng, setup=setup,
                          flag_mask=BMI_FLAGMASK[name], iterations=24)


def test_movbe(rng):
    for bits in (16, 32, 64):
        regs = _gpr_regs(bits)
        load = getattr(C, f'MOVBE_R{bits}_M{bits}')
        store = getattr(C, f'MOVBE_M{bits}_R{bits}')
        for opnd in ('rbx', 'rsp', 'rip'):
            for r in (0, 1, 2, 8, 11):
                mop, off = mem_operand(opnd, rng, bits // 8)
                n = bits // 8

                def exp_load(st, r=regs[r], off=off, n=n, bits=bits):
                    v = int.from_bytes(st[off:off + n], 'big')
                    _write(st, r, bits, v)
                    return st

                def exp_store(st, r=regs[r], off=off, n=n, bits=bits):
                    v = _read(st, r, bits)
                    st[off:off + n] = v.to_bytes(n, 'big')
                    return st
                dec = decoded(I.create_reg_mem(load, regs[r], mop))
                check(f'{dec} [{opnd}]', dec, sandy_lower.lower, exp_load, rng)
                dec = decoded(I.create_mem_reg(store, mop, regs[r]))
                check(f'{dec} [{opnd}]', dec, sandy_lower.lower, exp_store, rng)


# ---------------------------------------------------------------- AVX2 (selected forms with simple models)

def _lanes(st, n, fmt):
    size = struct.calcsize('<' + fmt)
    return list(struct.unpack_from(f'<{32 // size}{fmt}', st, h.OFF_YMM + 32 * n))


def _put(st, n, fmt, values):
    st[h.OFF_YMM + 32 * n: h.OFF_YMM + 32 * n + 32] = struct.pack(f'<{len(values)}{fmt}', *values)


def test_avx2(rng):
    lower = avx2_lower.lower
    Y = h.YMMS
    # lane-split integer ops
    ops = {
        'VPADDD': (lambda a, b: (a + b) & 0xFFFFFFFF, 'I'),
        'VPSUBW': (lambda a, b: (a - b) & 0xFFFF, 'H'),
        'VPMULLD': (lambda a, b: (a * b) & 0xFFFFFFFF, 'I'),
        'VPMAXSD': (lambda a, b: a if a >= b else b, 'i'),
        'VPCMPGTD': (lambda a, b: 0xFFFFFFFF if a > b else 0, 'i'),
        'VPAND': (lambda a, b: a & b, 'Q'),
        'VPXOR': (lambda a, b: a ^ b, 'Q'),
    }
    for mn, (fn, fmt) in ops.items():
        code = getattr(C, f'VEX_{mn}_YMM_YMM_YMMM256')
        for d, a, b in [(0, 1, 2), (0, 0, 1), (0, 1, 0), (3, 3, 3), (5, 6, 6), (15, 14, 13)]:
            for opnd in ('reg', 'rbx', 'rsp'):
                if opnd == 'reg':
                    ins = I.create_reg_reg_reg(code, Y[d], Y[a], Y[b])
                    off = None
                else:
                    mop, off = mem_operand(opnd, rng, 32)
                    ins = I.create_reg_reg_mem(code, Y[d], Y[a], mop)
                dec = decoded(ins)

                def expect(st, d=d, a=a, b=b, off=off, fn=fn, fmt=fmt):
                    va = _lanes(st, a, fmt)
                    vb = _lanes(st, b, fmt) if off is None else list(struct.unpack_from(f'<{len(va)}{fmt}', st, off))
                    res = [fn(x, y) for x, y in zip(va, vb)]
                    if fmt == 'i':
                        res = [x if x < 0x80000000 else x - (1 << 32) for x in
                               [(v & 0xFFFFFFFF) for v in res]]
                    _put(st, d, fmt, res)
                    return st
                check(f'{dec} [{opnd}]', dec, lower, expect, rng, iterations=8)
    # all-ones idiom, broadcasts, permq
    for mn in ('VPCMPEQB', 'VPCMPEQD'):
        code = getattr(C, f'VEX_{mn}_YMM_YMM_YMMM256')
        for d, a in [(0, 0), (1, 2), (7, 7)]:
            dec = decoded(I.create_reg_reg_reg(code, Y[d], Y[a], Y[a]))

            def expect(st, d=d):
                st[h.OFF_YMM + 32 * d: h.OFF_YMM + 32 * d + 32] = b'\xff' * 32
                return st
            check(str(dec), dec, lower, expect, rng, iterations=4)
    for d, s in [(0, 1), (2, 2)]:
        dec = decoded(I.create_reg_reg(C.VEX_VPBROADCASTD_YMM_XMMM32, Y[d], h.XMMS[s]))

        def expect(st, d=d, s=s):
            v = _lanes(st, s, 'I')[0]
            _put(st, d, 'I', [v] * 8)
            return st
        check(str(dec), dec, lower, expect, rng, iterations=4)
        mop, off = mem_operand('rbx', rng, 4)
        dec = decoded(I.create_reg_mem(C.VEX_VPBROADCASTD_YMM_XMMM32, Y[d], mop))

        def expect_m(st, d=d, off=off):
            v = struct.unpack_from('<I', st, off)[0]
            _put(st, d, 'I', [v] * 8)
            return st
        check(str(dec) + ' [rbx]', dec, lower, expect_m, rng, iterations=4)
    for imm in (0x00, 0x4E, 0x1B, 0xD8, 0xFF, 0x44, 0xEE):
        for d, s in [(0, 1), (3, 3)]:
            dec = decoded(I.create_reg_reg_i32(C.VEX_VPERMQ_YMM_YMMM256_IMM8, Y[d], Y[s], imm))

            def expect(st, d=d, s=s, imm=imm):
                v = _lanes(st, s, 'Q')
                _put(st, d, 'Q', [v[(imm >> (2 * k)) & 3] for k in range(4)])
                return st
            check(str(dec), dec, lower, expect, rng, iterations=4)


def _half_to_f32_bits(hbits):
    sign, exp, man = hbits >> 15, (hbits >> 10) & 31, hbits & 0x3FF
    if exp == 31:   # Inf/NaN: quiet NaN keeps the payload's top bits (hardware behaviour)
        return (sign << 31) | 0x7F800000 | ((man | (0x200 if man else 0)) << 13)
    return struct.unpack('<I', struct.pack('<f', struct.unpack('<e', struct.pack('<H', hbits))[0]))[0]


def _f32_to_half_bits_rne(fbits):
    sign, exp, man = fbits >> 31, (fbits >> 23) & 0xFF, fbits & 0x7FFFFF
    if exp == 0xFF:
        return (sign << 15) | 0x7C00 | ((man >> 13) | (0x200 if man else 0))
    x = struct.unpack('<f', struct.pack('<I', fbits))[0]
    try:
        return struct.unpack('<H', struct.pack('<e', x))[0]
    except OverflowError:
        return (sign << 15) | 0x7C00


def test_f16c(rng):
    for pool in (True, False):
        sandy_lower.set_constant_pool(h.SIGN_POOL if pool else None)
        _test_f16c(rng)
    sandy_lower.set_constant_pool(h.SIGN_POOL)


def _test_f16c(rng):
    Y, X = h.YMMS, h.XMMS

    def setup(st, rng):
        # interesting halves: subnormals, zeros, Inf, NaN, normals
        for k in range(16 * 16):
            pick = rng.random()
            if pick < 0.1:
                v = rng.choice([0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0x0001, 0x03FF, 0x8001])
            elif pick < 0.3:
                v = rng.randrange(0, 0x400) | (rng.randrange(2) << 15)
            else:
                v = rng.randrange(0x10000)
            struct.pack_into('<H', st, h.OFF_YMM + 2 * k, v)
        for k in range(512):
            struct.pack_into('<H', st, h.OFF_MEM + 2 * k, rng.randrange(0x10000))

    for wide in (False, True):
        for d, s in ((0, 1), (3, 3), (15, 0)):
            code = C.VEX_VCVTPH2PS_YMM_XMMM128 if wide else C.VEX_VCVTPH2PS_XMM_XMMM64
            for opnd in ('reg', 'rbx', 'rip'):
                if opnd == 'reg':
                    dec = decoded(I.create_reg_reg(code, (Y if wide else X)[d], X[s]))
                    src = lambda st, s=s: st[h.OFF_YMM + 32 * s: h.OFF_YMM + 32 * s + 16]
                else:
                    mop, off = mem_operand(opnd, rng, 16)
                    dec = decoded(I.create_reg_mem(code, (Y if wide else X)[d], mop))
                    src = lambda st, off=off: st[off: off + 16]

                def expect(st, d=d, wide=wide, src=src):
                    raw = bytes(src(st))
                    n = 8 if wide else 4
                    vals = [_half_to_f32_bits(struct.unpack_from('<H', raw, 2 * i)[0]) for i in range(n)]
                    out = struct.pack(f'<{n}I', *vals) + bytes(32 - 4 * n)
                    st[h.OFF_YMM + 32 * d: h.OFF_YMM + 32 * d + 32] = out
                    return st
                check(f'{dec} [{opnd}]', dec, sandy_lower.lower, expect, rng, setup=setup, iterations=12)

    def setup_ps(st, rng):
        for k in range(128):
            pick = rng.random()
            if pick < 0.6:
                x = rng.choice((-1, 1)) * rng.random() * 2.0 ** rng.randint(-26, 17)
                struct.pack_into('<f', st, h.OFF_YMM + 4 * k, x)
            elif pick < 0.7:
                struct.pack_into('<I', st, h.OFF_YMM + 4 * k, rng.choice([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0x7F800010]))
            else:
                struct.pack_into('<I', st, h.OFF_YMM + 4 * k, rng.getrandbits(32))

    for wide in (False, True):
        for d, s in ((0, 1), (2, 2)):
            for imm in (0, 4, 8):          # RNE, MXCSR (RNE in the test), RNE + suppress
                code = C.VEX_VCVTPS2PH_XMMM128_YMM_IMM8 if wide else C.VEX_VCVTPS2PH_XMMM64_XMM_IMM8
                dec = decoded(I.create_reg_reg_i32(code, X[d], (Y if wide else X)[s], imm))

                def expect(st, d=d, s=s, wide=wide):
                    n = 8 if wide else 4
                    fb = struct.unpack_from(f'<{n}I', st, h.OFF_YMM + 32 * s)
                    hv = [_f32_to_half_bits_rne(b) for b in fb]
                    out = struct.pack(f'<{n}H', *hv) + bytes(32 - 2 * n)
                    st[h.OFF_YMM + 32 * d: h.OFF_YMM + 32 * d + 32] = out
                    return st
                check(f'{dec}', dec, sandy_lower.lower, expect, rng, setup=setup_ps, iterations=12)


def _lane_codes():
    out = []
    for name, code in vars(C).items():
        if not isinstance(code, int) or not name.startswith('VEX_'):
            continue
        mn = name[4:].split('_')[0]
        if mn in avx2_lower.LANE_MNEMONICS and '_YMM_' in name + '_' and name[4:].split('_')[1] == 'YMM':
            out.append((name, code))
    return sorted(out)


def _build_lane_ins(name, code, regs, mem, imm):
    """Create an instruction for a lane-op code from its operand-kind list."""
    info = ix.OpCodeInfo(code)
    kinds = [info.op_kind(i) for i in range(info.op_count)]
    K = ix.OpCodeOperandKind
    ops, reg_iter = [], iter(regs)
    for kd in kinds:
        if kd in (K.YMM_REG, K.YMM_VVVV, K.YMM_IS4):
            ops.append(('r', h.YMMS[next(reg_iter)]))
        elif kd == K.YMM_OR_MEM:
            ops.append(('m', mem) if mem is not None else ('r', h.YMMS[next(reg_iter)]))
        elif kd == K.XMM_OR_MEM:
            ops.append(('m', mem) if mem is not None else ('r', h.XMMS[next(reg_iter)]))
        elif kd == K.IMM8:
            ops.append(('i', imm))
        else:
            return None
    ins = I()
    ins.code = code
    for n, (kind, val) in enumerate(ops):
        if kind == 'r':
            ins.set_op_kind(n, ix.OpKind.REGISTER)
            ins.set_op_register(n, val)
        elif kind == 'm':
            ins.set_op_kind(n, ix.OpKind.MEMORY)
            ins.memory_base = val.base if hasattr(val, 'base') else R.RBX
        else:
            ins.set_op_kind(n, ix.OpKind.IMMEDIATE8)
            ins.immediate8 = val
    return ins


def test_lanes(rng):
    """Optimized lane split (destination as scratch) vs the original conservative split."""
    def lower_opt(i, ip, adj):
        avx2_lower.LANES_USE_DST = True
        return avx2_lower.lower(i, ip, adj)

    for name, code in _lane_codes():
        info = ix.OpCodeInfo(code)
        nregs = sum(1 for i in range(info.op_count) if info.op_kind(i) != ix.OpCodeOperandKind.IMM8)
        combos = [list(range(nregs)), [0] + [1] * (nregs - 1), [5] + [3, 4, 6][:nregs - 1],
                  [9] + [9] * (nregs - 1), [2] + [7, 2, 3][:nregs - 1]]
        for regs in combos:
            for use_mem in (False, True):
                has_mem = any(info.op_kind(i) in (ix.OpCodeOperandKind.YMM_OR_MEM, ix.OpCodeOperandKind.XMM_OR_MEM)
                              for i in range(info.op_count))
                if use_mem and not has_mem:
                    continue
                for base in (('rbx',) if not use_mem else ('rbx', 'rsp')):
                    imm = rng.randrange(256)
                    ins = _build_lane_ins(name, code, regs, object() if use_mem else None, imm)
                    if ins is None:
                        continue
                    if use_mem:
                        for n in range(ins.op_count):
                            if ins.op_kind(n) == ix.OpKind.MEMORY:
                                ins.memory_base = R.RBX if base == 'rbx' else R.RSP
                                ins.memory_displacement = 32 if base == 'rbx' else 64
                                ins.memory_displ_size = 1
                    try:
                        dec = decoded(ins)
                    except Exception:  # noqa: BLE001
                        continue
                    avx2_lower.LANES_USE_DST = False
                    try:
                        ref = avx2_lower.lower(dec, h.TEST_IP, 0)
                        ref8 = h.stub_wrapper(lambda ip: avx2_lower.lower(dec, ip, 8))
                    finally:
                        avx2_lower.LANES_USE_DST = True
                    if ref is None:
                        continue

                    _check_against_code(f'{dec} [{"mem-" + base if use_mem else "reg"}]', dec, lower_opt,
                                        ref, ref8, rng)
    avx2_lower.LANES_USE_DST = True


def _check_against_code(name, ins, lower, ref, ref8, rng, iterations=6):
    counts['cases'] += 1
    for adjust, ref_code in ((0, ref), (8, ref8)):
        code = lower(ins, h.TEST_IP, 0) if not adjust else h.stub_wrapper(lambda ip: lower(ins, ip, 8))
        states = []
        for _ in range(iterations):
            st = h.new_state(rng)
            h.set_gpr(st, R.RBX, h.DATA + 256)
            states.append(st)
        h.install(ref_code)
        exp = [h.run(bytearray(st)) for st in states]
        h.install(code)
        for st, e in zip(states, exp):
            got = h.run(st)
            counts['runs'] += 1
            problem = _compare(e, got, h.STATUS_FLAGS)
            if problem:
                failures.append(f'{name} adjust={adjust}: {problem}')
                return


GROUPS = {'fma': test_fma, 'bmi1': test_bmi1, 'movbe': test_movbe, 'avx2': test_avx2,
          'lanes': test_lanes, 'f16c': test_f16c, 'fma-full': lambda rng: test_fma(rng, full=True)}
DEFAULT_GROUPS = ['fma', 'bmi1', 'movbe', 'f16c', 'avx2', 'lanes']


def main(argv):
    rng = random.Random(0x3_0)
    sandy_lower.set_constant_pool(h.SIGN_POOL)
    groups = [a for a in argv if a in GROUPS] or DEFAULT_GROUPS
    for g in groups:
        before = len(failures)
        GROUPS[g](rng)
        print(f'[{g}] {len(failures) - before} failing case(s)', flush=True)
    print(f"cases={counts['cases']} runs={counts['runs']} failures={len(failures)}")
    for line in failures[:int(__import__("os").environ.get("NFAIL", "60"))]:
        print('  FAIL', line)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
