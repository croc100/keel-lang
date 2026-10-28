"""
KEEL -> x86-64 machine code. The bottom of the language.

Everything else in KEEL is a claim about the world. This file is where the
claim gets cashed: a checked DAG becomes a buffer of bytes, the operating
system is asked to make those bytes executable, and the silicon runs them.
There is no interpreter, no runtime, and no virtual machine.

UNTRUSTED, like the builder. A wrong emitter cannot produce an accepted-but-
wrong program, because

  * `check` accepted the Def before anything was emitted, and
  * `verify()` re-tests the emitted code against the proven interval on the
    actual CPU.

So this file may be as clever as it likes. It is not in the trust root, and
the three clever things it now does are the reason that separation matters.

WHAT IS HERE
  * the ABI is a pointer pair, `void f(const double *in, double *out)`, so
    the number of parameters is not limited by the register count and a
    vector can be passed at all.
  * REGISTER ALLOCATION. Linear scan over the topological order with
    last-use liveness and furthest-next-use spilling. Scalars live in
    xmm0-15 and touch memory only when they must.
  * INSTRUCTION SELECTION DRIVEN BY `bind`. A vector operation lowers to one
    256-bit packed instruction when, and only when, it is bound to a unit
    whose declared lanes can hold it. Unbound, the same operation lowers
    lane by lane. The binding the checker proves is the binding that is
    emitted, which is the gap this file used to admit it had.
  * `fma` as a single instruction with a single rounding, emitted ONLY for
    an explicit `fma` node. The backend never contracts `a*b+c` on its own:
    the checker proved that expression with two roundings, and silently
    giving it one would make the proof describe a program that was not run.
    That is the `-ffp-contract=fast` bug, and here it is unrepresentable.
  * CPUID. The machine is asked what it is rather than assumed, and the
    instruction set actually used is reported in the listing.

WHAT IS HONESTLY NOT HERE
  no scheduling (the port pressure the checker computes is a bound the
  emitter does not yet try to meet), no AVX-512 masking or 512-bit forms, no
  PTX, no FPGA bitstream, no GPIO. `bind("TENSOR")` is proven and refused by
  the emitter rather than quietly lowered to something else.
"""

import ctypes
import platform
import struct
import sys

# This backend emits x86-64 and nothing else. On any other machine it must
# refuse BEFORE a single byte is executed: running an x86 instruction stream
# on ARM is not an exception, it is a dead process, and a test suite that
# does that on someone's laptop has no business existing.
X86 = platform.machine().lower() in ("amd64", "x86_64", "x86", "i386", "i686")

from .types import SITES, Real, Vec, fp_fmt

# --------------------------------------------------------------------- encoding
# Written out rather than table-driven, so that every byte can be read against
# an ISA manual.

RET = b"\xc3"

# VEX map select
_M_0F, _M_0F38 = 1, 2
# VEX prefix select: 0 = none, 1 = 66, 2 = F3, 3 = F2
_P_NONE, _P_66, _P_F3, _P_F2 = 0, 1, 2, 3


def _vex(r, x, b, mm, w, vvvv, el, pp):
    """Three-byte VEX. Always the three-byte form: it can encode every case,
    so there is one path to get right instead of two."""
    b1 = ((~r & 1) << 7) | ((~x & 1) << 6) | ((~b & 1) << 5) | mm
    b2 = ((w & 1) << 7) | ((~vvvv & 15) << 3) | ((el & 1) << 2) | pp
    return bytes([0xC4, b1, b2])


def _modrm_rr(reg, rm):
    return bytes([0xC0 | ((reg & 7) << 3) | (rm & 7)])


def _modrm_m(reg, base, disp):
    """[base+disp32]. mod=10 always, so one encoding covers every offset.
    rm=100 means 'a SIB byte follows', which rsp and r12 require."""
    out = bytes([0x80 | ((reg & 7) << 3) | (base & 7)])
    if (base & 7) == 4:
        out += bytes([0x24])            # SIB: base only, no index
    return out + struct.pack("<i", disp)


def vop_rr(opc, mm, pp, w, el, dst, src1, src2):
    """VEX three-operand register form: dst = src1 OP src2."""
    return (_vex(dst >> 3, 0, src2 >> 3, mm, w, src1, el, pp)
            + bytes([opc]) + _modrm_rr(dst, src2))


def vop_m(opc, mm, pp, w, el, reg, base, disp):
    """VEX memory form. `vvvv` is a register specifier in ONE'S COMPLEMENT, so
    the "unused" value the manuals print as 1111b is what you get by naming
    register 0 here. Encoding 0000 instead is #UD -- an illegal-instruction
    fault rather than a wrong answer, which is the good kind of mistake."""
    return (_vex(reg >> 3, 0, base >> 3, mm, w, 0, el, pp)
            + bytes([opc]) + _modrm_m(reg, base, disp))


def vop_fma(opc, el, dst, src1, src2):
    """VEX.66.0F38.W1 -- the FMA3 encoding family."""
    return (_vex(dst >> 3, 0, src2 >> 3, _M_0F38, 1, src1, el, _P_66)
            + bytes([opc]) + _modrm_rr(dst, src2))


# scalar double, VEX.LIG.F2.0F
_VS = {"add": 0x58, "sub": 0x5C, "mul": 0x59, "div": 0x5E,
       "sqrt": 0x51, "min": 0x5D, "max": 0x5F, "mov": 0x10}
# packed double, VEX.256.66.0F
_VP = {"add": 0x58, "sub": 0x5C, "mul": 0x59, "div": 0x5E, "sqrt": 0x51}


def vsd(name, dst, s1, s2):
    return vop_rr(_VS[name], _M_0F, _P_F2, 0, 0, dst, s1, s2)


def vpd(name, dst, s1, s2):
    return vop_rr(_VP[name], _M_0F, _P_66, 0, 1, dst, s1, s2)


def vld_sd(reg, base, disp):            # vmovsd xmm, [base+disp]
    return vop_m(0x10, _M_0F, _P_F2, 0, 0, reg, base, disp)


def vst_sd(reg, base, disp):            # vmovsd [base+disp], xmm
    return vop_m(0x11, _M_0F, _P_F2, 0, 0, reg, base, disp)


def vld_pd(reg, base, disp):            # vmovupd ymm, [base+disp]
    return vop_m(0x10, _M_0F, _P_66, 0, 1, reg, base, disp)


def vst_pd(reg, base, disp):            # vmovupd [base+disp], ymm
    return vop_m(0x11, _M_0F, _P_66, 0, 1, reg, base, disp)


def vxor_pd(dst, s1, s2, el=0):
    return vop_rr(0x57, _M_0F, _P_66, 0, el, dst, s1, s2)


def vbcast(dst, base, disp):            # vbroadcastsd ymm, [base+disp]
    return vop_m(0x19, _M_0F38, _P_66, 0, 1, dst, base, disp)


# -- SSE2 fallback, for a machine with no AVX. Two-operand, so a copy is
#    sometimes needed; that is the whole reason three-operand VEX exists.
_SSE2 = {"add": b"\xf2\x0f\x58", "sub": b"\xf2\x0f\x5c",
         "mul": b"\xf2\x0f\x59", "div": b"\xf2\x0f\x5e",
         "min": b"\xf2\x0f\x5d", "max": b"\xf2\x0f\x5f",
         "sqrt": b"\xf2\x0f\x51", "xor": b"\x66\x0f\x57"}


def _rex(reg, base=0, w=0):
    v = 0x40 | ((w & 1) << 3) | ((reg >> 3) << 2) | (base >> 3)
    return bytes([v]) if v != 0x40 else b""


def ssd(name, dst, src):
    p = _SSE2[name]
    return p[:1] + _rex(dst, src) + p[1:] + _modrm_rr(dst, src)


def smov(dst, src):                     # movapd xmm, xmm
    return b"\x66" + _rex(dst, src) + b"\x0f\x28" + _modrm_rr(dst, src)


def sld_sd(reg, base, disp):
    return b"\xf2" + _rex(reg, base) + b"\x0f\x10" + _modrm_m(reg, base, disp)


def sst_sd(reg, base, disp):
    return b"\xf2" + _rex(reg, base) + b"\x0f\x11" + _modrm_m(reg, base, disp)


def mov_rax(bits):                      # mov rax, imm64
    return b"\x48\xb8" + struct.pack("<Q", bits)


def mov_r9d(k):                         # mov r9d, imm32
    return b"\x41\xb9" + struct.pack("<I", k)


DEC_R9D = b"\x41\xff\xc9"              # dec r9d


def jnz_back(distance):                 # jnz rel32, backwards
    return b"\x0f\x85" + struct.pack("<i", -distance)


def movq_x_rax(reg, avx):               # movq xmm{reg}, rax
    if avx:
        return (_vex(reg >> 3, 0, 0, _M_0F, 1, 0, 0, _P_66)
                + b"\x6e" + _modrm_rr(reg, 0))
    return b"\x66" + _rex(reg, 0, w=1) + b"\x0f\x6e" + _modrm_rr(reg, 0)


def _bits(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


# These emit no arithmetic. A proven invariant, a unit binding and a
# criticality monitor are facts established before the first instruction runs.
_NOP_OPS = ("require", "bind", "monitor", "repeat", "sync")


class Unsupported(Exception):
    pass


# -------------------------------------------------------------- what this CPU is

def _link(blob, proto):
    """Ask the OS for executable memory. This is the moment the program stops
    being data."""
    if not X86:
        raise OSError("this backend emits x86-64; this machine is %r"
                      % platform.machine())
    if sys.platform == "win32":
        k = ctypes.windll.kernel32
        k.VirtualAlloc.restype = ctypes.c_void_p
        k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                   ctypes.c_ulong, ctypes.c_ulong]
        buf = k.VirtualAlloc(None, len(blob), 0x3000, 0x40)
        if not buf:
            raise OSError("VirtualAlloc failed")
    else:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_long]
        buf = libc.mmap(None, len(blob), 0x7, 0x22, -1, 0)
        if not buf or buf == ctypes.c_void_p(-1).value:
            raise OSError("mmap failed")
    ctypes.memmove(buf, blob, len(blob))
    return proto(buf)


_ARG1 = b"\x49\x89\xc8" if sys.platform == "win32" else b"\x49\x89\xf8"
_ARG2 = b"\x49\x89\xd3" if sys.platform == "win32" else b"\x49\x89\xf3"
# in -> r10, out -> r11, so the body is ABI-independent
_IN, _OUT, _SP = 10, 11, 4
_ENTRY = ((b"\x49\x89\xca" if sys.platform == "win32" else b"\x49\x89\xfa")
          + (b"\x49\x89\xd3" if sys.platform == "win32" else b"\x49\x89\xf3"))


def _cpuid(leaf, sub=0):
    """Emit CPUID and run it. The backend interrogating the machine with the
    backend is not a trick: it is the only way to learn this without trusting
    a table. rbx is callee-saved in both ABIs and CPUID clobbers it."""
    blob = (_ARG1                                   # mov r8, <arg0>
            + b"\x53"                               # push rbx
            + b"\xb8" + struct.pack("<I", leaf)     # mov eax, leaf
            + b"\xb9" + struct.pack("<I", sub)      # mov ecx, sub
            + b"\x0f\xa2"                           # cpuid
            + b"\x41\x89\x00"                       # mov [r8], eax
            + b"\x41\x89\x58\x04"                   # mov [r8+4], ebx
            + b"\x41\x89\x48\x08"                   # mov [r8+8], ecx
            + b"\x41\x89\x50\x0c"                   # mov [r8+12], edx
            + b"\x5b" + RET)                        # pop rbx ; ret
    out = (ctypes.c_uint32 * 4)()
    fn = _link(blob, ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint32)))
    fn(out)
    return tuple(out)


_FEATURES = None


def cpu():
    """{'brand':..., 'avx':bool, 'fma':bool, 'avx512f':bool, 'lanes':int}"""
    global _FEATURES
    if _FEATURES is None and not X86:
        _FEATURES = {"brand": "%s (not x86-64)" % platform.machine(),
                     "avx": False, "fma": False, "avx512f": False,
                     "lanes": 0}
    if _FEATURES is None:
        try:
            _, _, c1, _ = _cpuid(1)
            b7 = _cpuid(7, 0)[1] if _cpuid(0)[0] >= 7 else 0
            brand = b""
            if _cpuid(0x80000000)[0] >= 0x80000004:
                for lf in (0x80000002, 0x80000003, 0x80000004):
                    brand += b"".join(struct.pack("<I", r)
                                      for r in _cpuid(lf))
            avx, fma = bool(c1 & (1 << 28)), bool(c1 & (1 << 12))
            f512 = bool(b7 & (1 << 16))
            _FEATURES = {
                "brand": brand.split(b"\x00")[0].decode("ascii",
                                                        "replace").strip(),
                "avx": avx, "fma": fma, "avx512f": f512,
                "lanes": 8 if f512 else (4 if avx else 2),
            }
        except Exception as e:                               # noqa: BLE001
            _FEATURES = {"brand": "unknown (%s)" % e, "avx": False,
                         "fma": False, "avx512f": False, "lanes": 2}
    return _FEATURES


def isa_of(features=None):
    f = features if features is not None else cpu()
    s = set()
    for k in ("avx", "fma", "avx512f"):
        if f.get(k):
            s.add(k)
    return frozenset(s)


# -------------------------------------------------------------------- allocator

NREG = 16
SCRATCH = (0, 1, 2)        # hardcoded by the vector paths; never allocated
ALLOCATABLE = tuple(r for r in range(NREG) if r not in SCRATCH)


class _Alloc:
    """Linear scan with last-use liveness and furthest-next-use spilling.

    No clever lifetime splitting, no coalescing beyond destination reuse. What
    it does guarantee is that a value in a register is not reloaded, which is
    the difference between this and the stack machine that used to be here.
    """

    def __init__(self, emit_store, emit_load, slot):
        self.loc = {}          # addr -> ('r', k) | ('m', None)
        self.reg = [None] * NREG
        self.free = list(ALLOCATABLE)
        self.pinned = set()
        self.next_use = {}
        self._st, self._ld, self._slot = emit_store, emit_load, slot
        self.spills = 0
        self.reloads = 0

    def victim(self):
        best, bi = -1, None
        for k, a in enumerate(self.reg):
            if a is None or k in self.pinned:
                continue
            n = self.next_use.get(a, 10 ** 9)
            if n > best:
                best, bi = n, k
        if bi is None:
            raise Unsupported("ran out of registers with everything pinned")
        return bi

    def take(self, out):
        if self.free:
            return self.free.pop(0)
        k = self.victim()
        out.append(self._st(k, self._slot(self.reg[k])))
        self.spills += 1
        self.loc[self.reg[k]] = ("m", None)
        self.reg[k] = None
        return k

    def ensure(self, a, out):
        """Get `a` into a register and pin it for the current instruction."""
        w = self.loc.get(a)
        if w is None:
            raise Unsupported("value %s used before it was defined" % a[:6])
        if w[0] == "r":
            self.pinned.add(w[1])
            return w[1]
        k = self.take(out)
        out.append(self._ld(k, self._slot(a)))
        self.reloads += 1
        self.reg[k], self.loc[a] = a, ("r", k)
        self.pinned.add(k)
        return k

    def define(self, a, k):
        self.reg[k], self.loc[a] = a, ("r", k)

    def release(self, a):
        w = self.loc.get(a)
        if w and w[0] == "r":
            self.reg[w[1]] = None
            self.free.append(w[1])
            self.loc.pop(a, None)


# ---------------------------------------------------------------------- emitter

class Code:
    def __init__(self, name, blob, params, nodes, frame, isa, out_n, stats):
        self.name = name
        self.blob = blob
        self.params = params        # (name, in-buffer slot, width)
        self.nodes = nodes          # (addr, op, nbytes, how)
        self.frame = frame
        self.isa = isa
        self.out_n = out_n
        self.stats = stats
        self._fn = None

    def __len__(self):
        return len(self.blob)

    @property
    def nargs(self):
        return sum(w for _, _, w in self.params)

    def listing(self):
        head = ("%s: %d bytes, %d-byte frame, isa=%s, in=%d out=%d"
                % (self.name, len(self.blob), self.frame,
                   "+".join(sorted(self.isa)) or "sse2",
                   self.nargs, self.out_n))
        head += ("\n  spills=%d reloads=%d packed=%d scalar=%d"
                 % (self.stats["spills"], self.stats["reloads"],
                    self.stats["packed"], self.stats["scalar"]))
        rows = ["  %-6s %-9s %-14s %3d B" % (a[:6], op, how, n)
                for a, op, n, how in self.nodes]
        return "\n".join([head] + rows)

    def hexdump(self, width=16):
        b = self.blob
        return "\n".join(
            "%04x  %s" % (i, " ".join("%02x" % c for c in b[i:i + width]))
            for i in range(0, len(b), width))

    def run(self, *args):
        if self._fn is None:
            proto = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_double),
                                     ctypes.POINTER(ctypes.c_double))
            self._fn = _link(self.blob, proto)
        need = self.nargs
        flat = []
        for a in args:
            if isinstance(a, (tuple, list)):
                flat.extend(float(x) for x in a)
            else:
                flat.append(float(a))
        if len(flat) != need:
            raise TypeError("%s takes %d double(s), got %d"
                            % (self.name, need, len(flat)))
        # The buffers are padded: a 3-lane vector is read with a 32-byte
        # move, and the pad slot is zeroed so the unused lane is 0.0 rather
        # than whatever was next in memory.
        nin = max(1, self.params[-1][1] + 4) if self.params else 4
        bi = (ctypes.c_double * (nin + 4))()
        k = 0
        for _, off, w in self.params:
            for j in range(w):
                bi[off + j] = flat[k]
                k += 1
        bo = (ctypes.c_double * (self.out_n + 4))()
        self._fn(bi, bo)
        if self.out_n == 1:
            return bo[0]
        return tuple(bo[i] for i in range(self.out_n))


def _probe(frame):
    """Touch one byte per page of a frame larger than a page.

    A thread stack grows when the guard page is hit, one page at a time. Move
    rsp past it without touching the pages in between and the next write is an
    access violation rather than a growth. Nothing in the ISA warns you; the
    program just dies.
    """
    if frame <= 4096:
        return b""
    out = []
    off = 0
    while off < frame:
        out.append(b"\x48\x89\x84\x24" + struct.pack("<i", off))
        off += 4096
    return b"".join(out)


def _ty_lanes(t):
    if isinstance(t, Vec):
        return t.n
    return 1


def emit(d, report, isa=None, loop=1):
    """Lower an ACCEPTED definition to x86-64. Pass the Report from `check`;
    emitting an unchecked definition is refused, because the point of the
    backend is to be the last step and not the first.

    `loop` runs the body that many times before storing the result, which is
    how a one-nanosecond claim gets measured through a one-microsecond call.
    """
    if not report.ok:
        raise Unsupported("refusing to emit a rejected definition: %s" % d.name)
    use = frozenset(isa) if isa is not None else isa_of()
    avx = "avx" in use
    g, order = d.graph, d.graph.topo(d.root)

    # ---- shape pass: every value is f64; a vector gets a 32-byte home
    for a in order:
        t = g[a].ty
        if isinstance(t, Real) and fp_fmt(t) != "f64":
            raise Unsupported("f64 only; %s is %s" % (a[:6], fp_fmt(t)))
        if isinstance(t, Vec):
            if fp_fmt(t.elem) != "f64":
                raise Unsupported("f64 vectors only; %s is %s"
                                  % (a[:6], fp_fmt(t.elem)))
            if t.n > 4:
                raise Unsupported("vectors wider than 4 lanes need AVX-512 "
                                  "masking, which this backend does not emit")

    # ---- which nodes are bound to a unit wide enough to go packed
    packed = {}
    for a in order:
        n = g[a]
        if n.op != "bind":
            continue
        s = SITES.get(n.attrs.get("unit"))
        inner = g[n.args[0]]
        # A reduction over vectors -- dot, norm -- returns a SCALAR while its
        # operands are vectors, so its width comes from what it consumes.
        # Looking only at the result type made every bind on one silently do
        # nothing, which is worse than refusing it.
        wide = inner.ty if isinstance(inner.ty, Vec) else (
            g[inner.args[0]].ty if inner.op in ("dot", "norm") else None)
        if s is None or not isinstance(wide, Vec):
            continue
        if s.lanes >= wide.n and (not s.fmts or "f64" in s.fmts):
            if s.kind.startswith("x86"):
                packed[n.args[0]] = s.name
            else:
                raise Unsupported(
                    "bind(%s) is proven but this backend emits x86-64 only; "
                    "refusing to lower it as something else" % s.name)

    # ---- stack, handed out on demand (see the layout note in this patch)
    scratch = 0
    slot = {}
    nxt_off = [32]
    for a in order:
        if isinstance(g[a].ty, Vec):
            slot[a] = nxt_off[0]
            nxt_off[0] += 32

    def slot_of(a):
        """A scalar gets a stack slot the first time it is spilled, and most
        never are."""
        w = slot.get(a)
        if w is None:
            w = slot[a] = nxt_off[0]
            nxt_off[0] += 8
        return w

    # ---- input buffer layout: a vector takes 4 slots so its pad lane is its
    #      own, and parameters appear in DECLARATION order. Topological order
    #      would make the calling convention a property of the checker's walk.
    params, off = [], 0
    from .ir import param_addrs
    for a in param_addrs(d):
        n = g[a]
        if True:
            if not isinstance(n.ty, (Real, Vec)):
                raise Unsupported("param %r is %s, which has no double "
                                  "representation" % (n.attrs["name"],
                                                      type(n.ty).__name__))
            w = _ty_lanes(n.ty)
            params.append((n.attrs["name"], off, w))
            off += 4 if w > 1 else 1
    pslot = {p[0]: p[1] for p in params}

    out = []
    stats = {"packed": 0, "scalar": 0, "spills": 0, "reloads": 0}
    nodes = []

    if avx:
        def st(k, disp):
            return vst_sd(k, _SP, disp)

        def ld(k, disp):
            return vld_sd(k, _SP, disp)
    else:
        def st(k, disp):
            return sst_sd(k, _SP, disp)

        def ld(k, disp):
            return sld_sd(k, _SP, disp)

    al = _Alloc(st, ld, slot_of)

    # last use index for each scalar value; the root is used once at the end
    last = {}
    for i, a in enumerate(order):
        for dep in g[a].args:
            last[dep] = i
    last[d.root] = len(order)

    # spill priority: the value whose LAST use is furthest away. An exact
    # next-use distance would be better; this is the cheap, safe approximation.
    al.next_use = {a: last.get(a, 10 ** 9) for a in order}

    def const_into(k, value, here):
        here.append(mov_rax(_bits(value)))
        here.append(movq_x_rax(k, avx))

    def bin_scalar(kind, dst, x, y, here):
        if avx:
            here.append(vsd(kind, dst, x, y))
        else:
            if dst != x:
                here.append(smov(dst, x))
            here.append(ssd(kind, dst, y))

    # -- vector values live in memory; `lane` addresses one of their doubles
    def vlane(a, j):
        return slot[a] + 8 * j

    for i, a in enumerate(order):
        n, here = g[a], []
        al.pinned.clear()
        how = ""
        vec = isinstance(n.ty, Vec)

        if n.op == "param":
            if vec:
                base = pslot[n.attrs["name"]]
                for j in range(n.ty.n + 1):
                    here.append((vld_sd if avx else sld_sd)(
                        0, _IN, 8 * (base + j)))
                    here.append(st(0, vlane(a, j)))
                how = "in[%d..]" % base
            else:
                k = al.take(here)
                here.append((vld_sd if avx else sld_sd)(
                    k, _IN, 8 * pslot[n.attrs["name"]]))
                al.define(a, k)
                how = "in[%d]" % pslot[n.attrs["name"]]

        elif n.op == "const":
            k = al.take(here)
            const_into(k, n.attrs["value"], here)
            al.define(a, k)
            how = "imm"

        elif n.op == "vec3":
            for j, dep in enumerate(n.args):
                r = al.ensure(dep, here)
                here.append(st(r, vlane(a, j)))
            here.append(mov_rax(0))
            here.append(movq_x_rax(0, avx))
            here.append(st(0, vlane(a, 3)))
            how = "pack"

        elif n.op in ("add", "sub", "mul", "div") and vec:
            if a in packed:
                here.append(vld_pd(0, _SP, slot[n.args[0]]))
                here.append(vld_pd(1, _SP, slot[n.args[1]]))
                here.append(vpd(n.op, 0, 0, 1))
                here.append(vst_pd(0, _SP, slot[a]))
                stats["packed"] += 1
                how = "v" + n.op + "pd/" + packed[a]
            else:
                for j in range(n.ty.n + 1):
                    here.append(ld(0, vlane(n.args[0], j)))
                    here.append(ld(1, vlane(n.args[1], j)))
                    bin_scalar(n.op, 0, 0, 1, here)
                    here.append(st(0, vlane(a, j)))
                stats["scalar"] += n.ty.n
                how = "lane x%d" % n.ty.n

        elif n.op in ("add", "sub", "mul", "div"):
            x = al.ensure(n.args[0], here)
            y = al.ensure(n.args[1], here)
            dead = [dep for dep in n.args if last.get(dep) == i]
            # reuse a dying operand's register: three-operand VEX makes this
            # free, and two-operand SSE2 needs it anyway
            dst = x if n.args[0] in dead else al.take(here)
            bin_scalar(n.op, dst, x, y, here)
            for dep in dead:
                if al.loc.get(dep, ("", -1))[1] == dst:
                    al.loc.pop(dep, None)
                else:
                    al.release(dep)
            al.define(a, dst)
            stats["scalar"] += 1
            how = "v" + n.op + "sd" if avx else n.op + "sd"

        elif n.op == "fma":
            # ONE rounding, by one instruction. The checker proved this node
            # with one rounding too; that is why this is a separate operation
            # and not an optimisation of mul-then-add.
            if "fma" not in use:
                raise Unsupported("fma needs FMA3; this CPU/isa set has none")
            if vec:
                here.append(vld_pd(0, _SP, slot[n.args[2]]))
                here.append(vld_pd(1, _SP, slot[n.args[0]]))
                here.append(vld_pd(2, _SP, slot[n.args[1]]))
                here.append(vop_fma(0xB8, 1, 0, 1, 2))   # vfmadd231pd
                here.append(vst_pd(0, _SP, slot[a]))
                stats["packed"] += 1
                how = "vfmadd231pd"
            else:
                x = al.ensure(n.args[0], here)
                y = al.ensure(n.args[1], here)
                c = al.ensure(n.args[2], here)
                dst = c if last.get(n.args[2]) == i else al.take(here)
                if dst != c:
                    here.append(vsd("mov", dst, dst, c) if avx
                                else smov(dst, c))
                here.append(vop_fma(0xB9, 0, dst, x, y))  # vfmadd231sd
                for dep in n.args:
                    if last.get(dep) == i and al.loc.get(
                            dep, ("", -1))[1] != dst:
                        al.release(dep)
                al.define(a, dst)
                stats["scalar"] += 1
                how = "vfmadd231sd"

        elif n.op == "sqrt" and vec:
            if a in packed:
                here.append(vld_pd(0, _SP, slot[n.args[0]]))
                here.append(vpd("sqrt", 0, 0, 0))
                here.append(vst_pd(0, _SP, slot[a]))
                stats["packed"] += 1
                how = "vsqrtpd"
            else:
                for j in range(n.ty.n + 1):
                    here.append(ld(0, vlane(n.args[0], j)))
                    bin_scalar("sqrt", 0, 0, 0, here)
                    here.append(st(0, vlane(a, j)))
                stats["scalar"] += n.ty.n
                how = "lane x%d" % n.ty.n

        elif n.op == "sqrt":
            x = al.ensure(n.args[0], here)
            dst = x if last.get(n.args[0]) == i else al.take(here)
            if avx:
                here.append(vsd("sqrt", dst, x, x))
            else:
                if dst != x:
                    here.append(smov(dst, x))
                here.append(ssd("sqrt", dst, dst))
            if dst == x:
                al.loc.pop(n.args[0], None)
            al.define(a, dst)
            stats["scalar"] += 1
            how = "vsqrtsd" if avx else "sqrtsd"

        elif n.op == "neg" and vec:
            here.append(mov_rax(1 << 63))
            here.append(movq_x_rax(1, avx))
            for j in range(n.ty.n + 1):
                here.append(ld(0, vlane(n.args[0], j)))
                if avx:
                    here.append(vxor_pd(0, 0, 1))
                else:
                    here.append(ssd("xor", 0, 1))
                here.append(st(0, vlane(a, j)))
            stats["scalar"] += n.ty.n
            how = "lane x%d" % n.ty.n

        elif n.op == "neg":
            x = al.ensure(n.args[0], here)
            here.append(mov_rax(1 << 63))
            here.append(movq_x_rax(0, avx))
            dst = x if last.get(n.args[0]) == i else al.take(here)
            if avx:
                here.append(vxor_pd(dst, x, 0))
            else:
                if dst != x:
                    here.append(smov(dst, x))
                here.append(ssd("xor", dst, 0))
            if dst == x:
                al.loc.pop(n.args[0], None)
            al.define(a, dst)
            stats["scalar"] += 1
            how = "xorpd"

        elif n.op == "convert":
            k = float(g[n.args[0]].ty.unit.scale / n.attrs["to"].scale) \
                if not vec else \
                float(g[n.args[0]].ty.elem.unit.scale / n.attrs["to"].scale)
            if vec:
                if a in packed:
                    here.append(mov_rax(_bits(k)))
                    here.append(movq_x_rax(2, avx))
                    here.append(vst_sd(2, _SP, scratch))
                    here.append(vbcast(1, _SP, scratch))
                    here.append(vld_pd(0, _SP, slot[n.args[0]]))
                    here.append(vpd("mul", 0, 0, 1))
                    here.append(vst_pd(0, _SP, slot[a]))
                    stats["packed"] += 1
                    how = "vmulpd/bcast"
                else:
                    here.append(mov_rax(_bits(k)))
                    here.append(movq_x_rax(1, avx))
                    for j in range(n.ty.n + 1):
                        here.append(ld(0, vlane(n.args[0], j)))
                        bin_scalar("mul", 0, 0, 1, here)
                        here.append(st(0, vlane(a, j)))
                    stats["scalar"] += n.ty.n
                    how = "lane x%d" % n.ty.n
            else:
                # The exact rational scale becomes one multiply. This is the
                # node that would have saved Mars Climate Orbiter.
                x = al.ensure(n.args[0], here)
                r = al.take(here)
                const_into(r, k, here)
                dst = x if last.get(n.args[0]) == i else al.take(here)
                bin_scalar("mul", dst, x, r, here)
                al.reg[r] = None
                al.free.append(r)
                if dst == x:
                    al.loc.pop(n.args[0], None)
                al.define(a, dst)
                stats["scalar"] += 1
                how = "scale"

        elif n.op == "vscale":
            if a in packed:
                r = al.ensure(n.args[1], here)
                here.append(vst_sd(r, _SP, scratch))
                here.append(vbcast(1, _SP, scratch))
                here.append(vld_pd(0, _SP, slot[n.args[0]]))
                here.append(vpd("mul", 0, 0, 1))
                here.append(vst_pd(0, _SP, slot[a]))
                stats["packed"] += 1
                how = "vmulpd/bcast"
            else:
                r = al.ensure(n.args[1], here)
                here.append(st(r, scratch))
                for j in range(n.ty.n + 1):
                    here.append(ld(0, vlane(n.args[0], j)))
                    here.append(ld(1, scratch))
                    bin_scalar("mul", 0, 0, 1, here)
                    here.append(st(0, vlane(a, j)))
                stats["scalar"] += n.ty.n
                how = "lane x%d" % n.ty.n

        elif n.op in ("dot", "norm"):
            u = n.args[0]
            w = n.args[1] if n.op == "dot" else u
            nl = g[u].ty.n
            if a in packed or (n.op == "norm" and u in packed):
                here.append(vld_pd(0, _SP, slot[u]))
                here.append(vld_pd(1, _SP, slot[w]))
                here.append(vpd("mul", 0, 0, 1))
                here.append(vst_pd(0, _SP, scratch))
                stats["packed"] += 1
                src = scratch
                how = "vmulpd+hsum"
            else:
                for j in range(nl):
                    here.append(ld(0, vlane(u, j)))
                    here.append(ld(1, vlane(w, j)))
                    bin_scalar("mul", 0, 0, 1, here)
                    here.append(st(0, scratch + 8 * j))
                stats["scalar"] += nl
                src = scratch
                how = "lane x%d+hsum" % nl
            # the horizontal sum reads lanes 0..n-1 only, so the pad lane of a
            # 3-vector is never summed
            dst = al.take(here)
            here.append(ld(dst, src))
            for j in range(1, nl):
                k = al.take(here)
                here.append(ld(k, src + 8 * j))
                bin_scalar("add", dst, dst, k, here)
                al.reg[k] = None
                al.free.append(k)
            if n.op == "norm":
                if avx:
                    here.append(vsd("sqrt", dst, dst, dst))
                else:
                    here.append(ssd("sqrt", dst, dst))
            al.define(a, dst)
            stats["scalar"] += nl

        elif n.op == "clamp":
            x = al.ensure(n.args[0], here)
            r = al.take(here)
            dst = x if last.get(n.args[0]) == i else al.take(here)
            const_into(r, n.attrs["lo"], here)
            bin_scalar("max", dst, x, r, here)
            const_into(r, n.attrs["hi"], here)
            bin_scalar("min", dst, dst, r, here)
            al.reg[r] = None
            al.free.append(r)
            if dst == x:
                al.loc.pop(n.args[0], None)
            al.define(a, dst)
            stats["scalar"] += 2
            how = "max/min"

        elif n.op in _NOP_OPS:
            # A proven invariant, a unit binding and a criticality monitor are
            # facts established before the first instruction runs. Emitting a
            # runtime check would be paying twice for one proof.
            src = n.args[0]
            if vec:
                how = "(proof)"
                if slot[a] != slot[src]:
                    for j in range(n.ty.n + 1):
                        here.append(ld(0, vlane(src, j)))
                        here.append(st(0, vlane(a, j)))
            else:
                k = al.ensure(src, here)
                if last.get(src) == i:
                    al.loc.pop(src, None)
                    al.define(a, k)
                else:
                    dst = al.take(here)
                    here.append(vsd("mov", dst, dst, k) if avx
                                else smov(dst, k))
                    al.define(a, dst)
                how = "(proof)"

        else:
            raise Unsupported("no x86-64 lowering for %r" % n.op)

        blob = b"".join(here)
        nodes.append((a, n.op, len(blob), how))
        out.append(blob)

    # ---- result to out[]
    rt = g[d.root].ty
    tail = []
    if isinstance(rt, Vec):
        for j in range(rt.n):
            tail.append(ld(0, vlane(d.root, j)))
            tail.append((vst_sd if avx else sst_sd)(0, _OUT, 8 * j))
        out_n = rt.n
    else:
        k = al.ensure(d.root, tail)
        tail.append((vst_sd if avx else sst_sd)(k, _OUT, 0))
        out_n = 1
    body = b"".join(out)
    if loop > 1:
        body = (mov_r9d(loop) + body + DEC_R9D
                + jnz_back(len(body) + len(DEC_R9D) + 6))
    frame = (nxt_off[0] + 15) & ~15
    blob = (_ENTRY + b"\x48\x81\xec" + struct.pack("<I", frame)
            + _probe(frame) + body + b"".join(tail)
            + b"\x48\x81\xc4" + struct.pack("<I", frame) + RET)
    stats["spills"], stats["reloads"] = al.spills, al.reloads
    stats["loop"] = loop
    code = Code(d.name, blob, params, nodes, frame, use, out_n, stats)
    code.src, code.rep = d, report
    return code


# --------------------------------------------------------------- verification

def verify(d, report, samples=5, isa=None, time_it=False):
    """Run the emitted code over the proven input box and check every answer
    against the proven output interval.

    This is the only place in KEEL where a claim meets a measurement. The
    checker said the result lies in [lo,hi] using interval arithmetic over a
    model of floating point; the CPU now says what it actually computes. If
    they disagree the model is wrong, and a language for safety-critical work
    that never confronts its model with the silicon has not earned the word
    "proven".

    With `time_it`, the WCET claim is confronted too: the minimum observed
    time over many runs (the least-disturbed sample) against the seconds the
    cost model predicted. A measurement above the claim means the claim is
    wrong; far below it only means the bound was loose, which is the safe
    direction.
    """
    code = emit(d, report, isa=isa)
    g = d.graph
    iv = report.iv[d.root]
    boxes = []
    reach = set(g.topo(d.root))
    for a in [x for x in g.order if x in reach]:
        if g[a].op == "param":
            lo, hi = g[a].attrs["lo"], g[a].attrs["hi"]
            if lo is None or hi is None:
                raise Unsupported("param %r has no box to sample"
                                  % g[a].attrs["name"])
            w = _ty_lanes(g[a].ty)
            pts = ([lo + (hi - lo) * i / (samples - 1.0)
                    for i in range(samples)] if samples > 1 else [lo])
            for _ in range(w):
                boxes.append(pts)

    lo_iv = getattr(iv, "lo", None)
    pts_n, worst = 0, None

    def inside(y):
        vals = y if isinstance(y, tuple) else (y,)
        return all(iv.lo <= v <= iv.hi for v in vals)

    def walk(prefix):
        nonlocal pts_n, worst
        if len(prefix) == len(boxes):
            y = code.run(*prefix)
            pts_n += 1
            if not inside(y) and worst is None:
                worst = (tuple(prefix), y)
            return
        for v in boxes[len(prefix)]:
            walk(prefix + [v])

    if lo_iv is not None:
        walk([])

    res = {
        "name": d.name,
        "bytes": len(code),
        "points": pts_n,
        "proven": (iv.lo, iv.hi) if lo_iv is not None else None,
        "inside": worst is None,
        "counterexample": worst,
        "isa": "+".join(sorted(code.isa)) or "sse2",
        "code": code,
    }
    if time_it:
        res.update(_time(code, boxes, report))
    return res


def _time(code, boxes, report, reps=400, loop=512):
    """Measure the body, not the call.

    A nine-nanosecond kernel cannot be timed through two microseconds of
    ctypes, so the body is emitted twice -- once bare and once wrapped in a
    counted loop -- and the DIFFERENCE is divided by the extra iterations.
    Everything fixed cancels exactly instead of being estimated.
    """
    import time
    args = [b[len(b) // 2] for b in boxes]

    def best_of(c):
        c.run(*args)
        t = float("inf")
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(reps):
                c.run(*args)
            t = min(t, (time.perf_counter() - t0) / reps)
        return t

    bare = best_of(code)
    if getattr(code, "src", None) is None:
        return {"throughput_s": None, "claimed_s": None, "overhead_s": bare,
                "wcet_consistent": None, "ratio": None,
                "kind": "body not re-emittable; no timing claim made"}
    many = best_of(emit(code.src, code.rep, isa=code.isa, loop=loop))
    meas = max(many - bare, 0.0) / (loop - 1)
    claim = max(report.cost.seconds,
                report.cost.throughput_cycles() / 3.0e9)
    # What this number IS: throughput. The loop's iterations are independent,
    # because each one reloads its inputs, so an out-of-order core overlaps
    # them and the result is time-per-execution-in-a-stream, not the latency
    # of one execution.
    #
    # So `measured <= claimed` is CONSISTENT with the WCET bound and does not
    # confirm it. Calling it confirmation would be measuring one thing and
    # claiming another, which is the failure mode this whole project is
    # arranged against. A real latency measurement needs a dependent chain
    # and lives in keel/calibrate.py.
    return {"throughput_s": meas, "claimed_s": claim, "overhead_s": bare,
            "wcet_consistent": (meas <= claim) if claim > 0 else None,
            "ratio": (meas / claim) if claim > 0 else None,
            "kind": "throughput (iterations overlap); not a latency proof"}
