"""
KEEL type system.

Everything in here exists because a human author would refuse to maintain it.
A machine author pays the cost for free, so it is mandatory rather than optional.

  * units carry a DIMENSION VECTOR *and* an EXACT SCALE.
    Dimension-only checking is not enough: Mars Climate Orbiter was lost to
    lbf*s vs N*s, which are DIMENSIONALLY IDENTICAL and differ only in scale.
    There is no implicit conversion anywhere; an explicit `convert` node is
    the only way to change scale, and it shows up in the cost model.

  * vectors and rotations carry REFERENCE FRAMES.
    Vec3<ECI> + Vec3<Body> does not typecheck. You must route through
    Rot<ECI,Body>.

  * arrays carry their MEMORY LEVEL.
    Arr(f64,[1024],DRAM) and Arr(f64,[1024],L1) are different types, because
    at exascale the cost difference between them is ~100x in energy.

  * reals carry a REPRODUCIBILITY flag and a PROVEN RELATIVE ERROR BOUND.
    f64 addition is not associative, so a reduction's answer depends on the
    tree shape. A value either is bitwise-reproducible or it is not, and that
    is part of its type.

  * every value carries a DAL (design assurance level) for mixed criticality.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple

# ----------------------------------------------------------------- dimensions
# 7 SI base dimensions, plus radian tracked as a pseudo-dimension so that an
# angle can never be silently used as a bare scalar.
DIM_NAMES = ("m", "kg", "s", "A", "K", "mol", "cd", "rad")
NDIM = len(DIM_NAMES)
_ZERO = (0,) * NDIM


class Unit:
    __slots__ = ("dim", "scale", "label")

    def __init__(self, dim=_ZERO, scale=1, label=None):
        self.dim = tuple(dim)
        assert len(self.dim) == NDIM
        self.scale = Fraction(scale)
        self.label = label

    # -- algebra
    def __mul__(self, o):
        return Unit(tuple(a + b for a, b in zip(self.dim, o.dim)), self.scale * o.scale)

    def __truediv__(self, o):
        return Unit(tuple(a - b for a, b in zip(self.dim, o.dim)), self.scale / o.scale)

    def __pow__(self, k):
        return Unit(tuple(a * k for a in self.dim), self.scale ** k)

    # -- identity: dimension AND scale must match exactly
    def __eq__(self, o):
        return isinstance(o, Unit) and self.dim == o.dim and self.scale == o.scale

    def __hash__(self):
        return hash((self.dim, self.scale))

    def same_dim(self, o):
        return self.dim == o.dim

    @property
    def dimensionless(self):
        return self.dim == _ZERO

    def __repr__(self):
        if self.label:
            return self.label
        if self.dimensionless and self.scale == 1:
            return "1"
        num = [n if e == 1 else "%s^%d" % (n, e)
               for n, e in zip(DIM_NAMES, self.dim) if e > 0]
        den = [n if e == -1 else "%s^%d" % (n, -e)
               for n, e in zip(DIM_NAMES, self.dim) if e < 0]
        core = "*".join(num) or "1"
        if den:
            core += "/" + "*".join(den)
        if self.scale != 1:
            core = "%s*%s" % (self.scale, core)
        return core

    def canon(self):
        return (self.dim, str(self.scale))


ONE = Unit(label="1")
M = Unit((1, 0, 0, 0, 0, 0, 0, 0), 1, "m")
KG = Unit((0, 1, 0, 0, 0, 0, 0, 0), 1, "kg")
S = Unit((0, 0, 1, 0, 0, 0, 0, 0), 1, "s")
RAD = Unit((0, 0, 0, 0, 0, 0, 0, 1), 1, "rad")

MPS = Unit((M / S).dim, 1, "m/s")
MPS2 = Unit((M / S ** 2).dim, 1, "m/s^2")
RADPS = Unit((RAD / S).dim, 1, "rad/s")
NEWTON = Unit((KG * M / S ** 2).dim, 1, "N")
NS = Unit((NEWTON * S).dim, 1, "N*s")

# 1 lbf == 0.45359237 kg * 9.80665 m/s^2 == 4.4482216152605 N  (exact)
_LBF_SCALE = Fraction("0.45359237") * Fraction("9.80665")
LBF = Unit(NEWTON.dim, _LBF_SCALE, "lbf")
LBF_S = Unit(NS.dim, _LBF_SCALE, "lbf*s")

KM = Unit(M.dim, 1000, "km")
# km/h is the everyday unit that is not an SI one. Its scale is exactly
# 1000/3600, which is not a float, which is the whole reason scales are
# Fractions here.
KMH = Unit((M / S).dim, Fraction(1000, 3600), "km/h")

# Time, at the scales the physical world actually uses. A contact bounce is
# milliseconds, a flip-flop setup is picoseconds, and nothing in KEEL lets the
# two meet without passing through this table.
MS = Unit(S.dim, Fraction(1, 1000), "ms")
US = Unit(S.dim, Fraction(1, 10 ** 6), "us")
NSEC = Unit(S.dim, Fraction(1, 10 ** 9), "ns")
PSEC = Unit(S.dim, Fraction(1, 10 ** 12), "ps")
HZ = Unit((S ** -1).dim, 1, "Hz")

MM = Unit(M.dim, Fraction(1, 1000), "mm")
M2 = Unit((M ** 2).dim, 1, "m^2")

# Electrical. These are the same seven base dimensions; no new machinery was
# needed to reach a circuit board, because SI already goes there.
A = Unit((0, 0, 0, 1, 0, 0, 0, 0), 1, "A")
MA = Unit(A.dim, Fraction(1, 1000), "mA")
V = Unit((KG * M ** 2 / (S ** 3 * A)).dim, 1, "V")
OHM = Unit((V / A).dim, 1, "ohm")
FARAD = Unit((A * S / V).dim, 1, "F")
WATT = Unit((NEWTON * M / S).dim, 1, "W")

# Mechanical.
PA = Unit((NEWTON / M ** 2).dim, 1, "Pa")
MPA = Unit(PA.dim, 10 ** 6, "MPa")
N_M = Unit((NEWTON * M).dim, 1, "N*m")

# Photometric -- `cd` was in the dimension vector from the first line of this
# file and went unused. A UI luminance is cd/m^2, so a contrast ratio is a
# dimensionless quotient of two photometric quantities and is checkable by
# exactly the same arithmetic as a thrust margin.
CD = Unit((0, 0, 0, 0, 0, 0, 1, 0), 1, "cd")
CD_M2 = Unit((CD / M ** 2).dim, 1, "cd/m^2")
NIT = Unit(CD_M2.dim, 1, "nit")


# --------------------------------------------------------------------- levels
# Ordered cheapest -> most expensive. Arithmetic is only legal on REG.
MEM_LEVELS = ("REG", "L1", "L2", "DRAM", "NET")

# Design assurance level. A is the most critical (catastrophic failure).
DAL_RANK = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

# Unit roundoff. The relative error introduced by one rounded FP operation.
EPS = {"f32": 2.0 ** -24, "f64": 2.0 ** -53}
FP_FORMATS = tuple(EPS)


# ---------------------------------------------------------------------- types
class Ty:
    """Base. All types are value objects: structural equality, hashable."""

    def canon(self):
        raise NotImplementedError

    def __eq__(self, o):
        return isinstance(o, Ty) and self.canon() == o.canon()

    def __hash__(self):
        return hash(self.canon())


@dataclass(frozen=True, eq=False)
class Bool(Ty):
    def canon(self):
        return ("bool",)

    def __repr__(self):
        return "bool"


@dataclass(frozen=True, eq=False)
class Int(Ty):
    width: int = 64
    signed: bool = True

    def canon(self):
        return ("int", self.width, self.signed)

    def __repr__(self):
        return "%s%d" % ("i" if self.signed else "u", self.width)


@dataclass(frozen=True, eq=False)
class Real(Ty):
    """A floating point scalar.

    unit   -- exact unit, scale included
    frame  -- reference frame tag, or None for a frame-free scalar
    repro  -- is this value bitwise reproducible across decompositions?
    err    -- PROVEN bound on accumulated relative error (not an estimate)
    """
    fmt: str = "f64"
    unit: Unit = ONE
    frame: Optional[str] = None
    repro: bool = True
    err: float = 0.0

    def canon(self):
        return ("real", self.fmt, self.unit.canon(), self.frame, self.repro)

    def with_err(self, e):
        return Real(self.fmt, self.unit, self.frame, self.repro, e)

    def __repr__(self):
        s = "%s<%s" % (self.fmt, self.unit)
        if self.frame:
            s += "@" + self.frame
        if not self.repro:
            s += ",nonrepro"
        return s + ">"


@dataclass(frozen=True, eq=False)
class Vec(Ty):
    """Fixed-length vector of reals, all sharing one unit and one frame."""
    elem: Real = Real()
    n: int = 3
    frame: Optional[str] = None

    def canon(self):
        return ("vec", self.elem.canon(), self.n, self.frame)

    def __repr__(self):
        return "vec%d<%s@%s>" % (self.n, self.elem.unit, self.frame or "-")


@dataclass(frozen=True, eq=False)
class Arr(Ty):
    """Statically shaped array at an explicit memory level.

    There is no dynamic allocation in KEEL, so shape is always static.
    """
    elem: Ty = Real()
    shape: Tuple[int, ...] = (1,)
    level: str = "DRAM"

    def canon(self):
        return ("arr", self.elem.canon(), tuple(self.shape), self.level)

    @property
    def count(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def __repr__(self):
        return "arr%s<%s>@%s" % (list(self.shape), self.elem, self.level)


@dataclass(frozen=True, eq=False)
class Rot(Ty):
    """A rotation that maps vectors expressed in `frm` to `to`."""
    frm: str = "ECI"
    to: str = "BODY"

    def canon(self):
        return ("rot", self.frm, self.to)

    def __repr__(self):
        return "rot<%s->%s>" % (self.frm, self.to)


def fp_fmt(ty):
    """The FP format a type rounds in, or None if it is not floating point."""
    if isinstance(ty, Real):
        return ty.fmt
    if isinstance(ty, Vec):
        return ty.elem.fmt
    if isinstance(ty, Arr):
        return fp_fmt(ty.elem)
    return None


def elem_unit(ty):
    if isinstance(ty, Real):
        return ty.unit
    if isinstance(ty, Vec):
        return ty.elem.unit
    if isinstance(ty, Arr):
        return elem_unit(ty.elem)
    return None


def width_bytes(ty):
    if isinstance(ty, Real):
        return 4 if ty.fmt == "f32" else 8
    if isinstance(ty, Int):
        return max(1, ty.width // 8)
    if isinstance(ty, Bool):
        return 1
    if isinstance(ty, Vec):
        return width_bytes(ty.elem) * ty.n
    if isinstance(ty, Arr):
        return width_bytes(ty.elem) * ty.count
    if isinstance(ty, Rot):
        return 32
    return 8


# ============================================================  physical sites
# `MEM_LEVELS` above was a human taxonomy: the memory hierarchy of a computer,
# which is only one family of places a value can live. A GPIO pin, a copper
# trace, a relay contact, a display panel and a DRAM bank differ in their
# NUMBERS, not in their kind -- each is a place where a value is committed or
# sensed, each has a settling time, a transit delay, an energy price, a time
# domain, and a current it can source or draws.
#
# So there are no UI / shell / embedded / HPC "planes". There is one lattice of
# sites, and a domain is a REGION of it rather than a category above it.

@dataclass(frozen=True)
class Site:
    """A place a value can be committed to or sensed from.

    settle   -- seconds after a transition before the value is valid.
                Contact bounce, flip-flop setup, LCD response: all this field.
    prop     -- seconds of transit delay to reach or read it.
    pj_byte  -- energy per byte moved.
    clock    -- time domain. None means asynchronous / continuous. Two sites
                with different clocks are as incompatible as two reference
                frames, and for the same reason.
    readback -- can what was driven be sensed again?
    arith    -- is arithmetic legal here? Only registers.
    drive_a  -- amperes this site can source (0 = it is not a driver).
    load_a   -- amperes one sink draws from it.

    A site that COMPUTES declares four more numbers. These are not a new
    category of thing: an FMA port, a tensor core, a DSP48 slice and an ALU
    are places a value is committed to, exactly like a pin or a DRAM bank,
    and they are told apart by their numbers rather than by their kind.

    lanes    -- values processed per issue (0 = not an execution unit).
    thru     -- issues per cycle on this unit's port.
    lat      -- result latency in cycles.
    port     -- the issue resource contended for. Two units naming the same
                port compete; this is what makes a throughput bound provable.
    hz       -- this unit's OWN clock rate (0 = the machine's). An AVX-512
                unit really does run slower than the core it sits in, and a
                200 MHz fabric is not a 3 GHz core. A rate, not a domain.
    fmts     -- accepted float formats (() = any). An f64 multiply cannot
                issue on a unit that only carries f16, and that is a fact
                about the silicon, not about the program.
    banks    -- independent parallel sub-arrays (0 = not banked). DRAM banks,
                GPU shared-memory banks, true-dual-port BRAM and register
                file read ports are the same number.
    ooo      -- is issue scheduled by hardware out of order? If so, naming a
                unit is a CONSTRAINT that can be proven, not a COMMAND that
                is obeyed, and the checker reports which one you got. Saying
                otherwise would be the one lie this language must not tell.

    kind     -- reporting only; the checker never branches on it.
    """
    name: str
    settle: float = 0.0
    prop: float = 0.0
    pj_byte: float = 0.0
    clock: Optional[str] = None
    readback: bool = True
    arith: bool = False
    drive_a: float = 0.0
    load_a: float = 0.0
    lanes: int = 0
    thru: float = 0.0
    lat: float = 0.0
    port: Optional[str] = None
    hz: float = 0.0
    fmts: Tuple[str, ...] = ()
    banks: int = 0
    ooo: bool = False
    kind: str = "generic"


SITES = {}


def site(s):
    """Register a site. Re-registering the same name with the same physics is
    idempotent; changing the physics under a name is refused, because a site
    name is a claim about the world and silently redefining it is exactly the
    substitution attack content addressing exists to prevent."""
    old = SITES.get(s.name)
    if old is not None and old != s:
        raise ValueError("site %r is already registered with different "
                         "physics: %r vs %r" % (s.name, old, s))
    SITES[s.name] = s
    return s.name


# The compute hierarchy is just five sites. Delays are seconds at ~3 GHz, so
# they compose with a switch's 5 ms bounce and a display's 8 ms frame in one
# arithmetic rather than through a conversion between "domains".
site(Site("REG", prop=3.3e-10, pj_byte=0.025, arith=True, kind="register"))
site(Site("L1", prop=1.3e-9, pj_byte=0.5, kind="cache"))
site(Site("L2", prop=4.7e-9, pj_byte=2.5, kind="cache"))
site(Site("DRAM", settle=1.5e-8, prop=6.7e-8, pj_byte=40.0, kind="dram"))
site(Site("NET", prop=3.3e-6, pj_byte=500.0, kind="interconnect"))

# Execution units. The point of putting them in the SAME registry as a relay
# contact is that nothing new was needed: an FMA port is a place a value lands
# after a stated delay, which is the definition the switch already satisfied.
#
# `ooo=True` is the honest part. On an out-of-order core you do not get to say
# which port retires your multiply; the renamer and the scheduler are silicon
# and they are not addressable. What you DO get is a provable bound on port
# pressure, so `bind` there means "prove this fits", and the report says so.
# On a tensor core or an FPGA slice the same operation really is a command.
site(Site("ALU64", arith=True, lanes=1, thru=2.0, lat=4.0, port="p01",
          ooo=True, fmts=("f64", "f32"), prop=3.3e-10, pj_byte=0.025,
          kind="x86-64-sse"))
site(Site("FMA512", arith=True, lanes=8, thru=2.0, lat=4.0, port="p05",
          ooo=True, fmts=("f64",), hz=2.1e9, prop=3.3e-10, pj_byte=0.025,
          kind="x86-64-avx512"))
site(Site("TENSOR", arith=True, lanes=128, thru=1.0, lat=16.0, port="mma",
          ooo=False, fmts=("f16",), hz=1.4e9, prop=7.1e-10, pj_byte=0.004,
          kind="ptx-mma-m16n8k16"))
site(Site("SHMEM", banks=32, prop=2.0e-8, pj_byte=0.08, kind="gpu-shared"))
site(Site("DSP48", arith=True, lanes=1, thru=1.0, lat=1.0, port="dsp",
          ooo=False, fmts=("f32",), clock="FPGA_200M", hz=2.0e8,
          settle=1.2e-9, prop=2.5e-9, pj_byte=0.3, kind="fpga-dsp48"))
site(Site("BRAM", banks=2, clock="FPGA_200M", prop=5.0e-9, pj_byte=0.6,
          kind="fpga-bram"))


# =====================================================================
# General-purpose types.
#
# Everything above exists so that a number cannot lie about what it is. What
# follows exists so that a PROGRAM can be written -- records, variants,
# strings, buffers, capabilities and functions -- without giving up any of it.
#
# The constraint that shaped these: nothing may introduce a way for a value to
# be absent, aliased or unbounded. So there is no null (a missing value is a
# variant you must handle), no shared mutable reference (a buffer is linear),
# no unbounded string (a byte string carries its length in its type) and no
# unbounded recursion (a function's cost is part of its contract).
# =====================================================================


@dataclass(frozen=True, eq=False)
class Rec(Ty):
    """A record. Fields are ordered, named and fully typed.

    No field may be absent. KEEL has no null, so "this field is sometimes not
    there" must be written as a `Sum`, where the checker can force you to
    handle the case you would otherwise have forgotten.
    """
    name: str = "rec"
    fields: Tuple[Tuple[str, Ty], ...] = ()

    def canon(self):
        return ("rec", self.name,
                tuple((f, t.canon()) for f, t in self.fields))

    @property
    def names(self):
        return tuple(f for f, _ in self.fields)

    def index(self, fname):
        for i, (f, _) in enumerate(self.fields):
            if f == fname:
                return i
        return None

    def ty_of(self, fname):
        i = self.index(fname)
        return None if i is None else self.fields[i][1]

    def __repr__(self):
        return "%s{%s}" % (self.name, ", ".join(
            "%s:%r" % (f, t) for f, t in self.fields))


@dataclass(frozen=True, eq=False)
class Sum(Ty):
    """A tagged union. Exactly one variant holds, and which one is not known
    statically -- so every read of it goes through `match`, which the checker
    requires to be exhaustive.

    This is where a missing value, an error result and a state machine's state
    all live. There is no other way to express "or" in KEEL, which is the
    point: an unhandled case is a type error rather than a surprise.
    """
    name: str = "sum"
    variants: Tuple[Tuple[str, Ty], ...] = ()

    def canon(self):
        return ("sum", self.name,
                tuple((t, ty.canon()) for t, ty in self.variants))

    @property
    def tags(self):
        return tuple(t for t, _ in self.variants)

    def ty_of(self, tag):
        for t, ty in self.variants:
            if t == tag:
                return ty
        return None

    def __repr__(self):
        return "%s[%s]" % (self.name, " | ".join(
            "%s %r" % (t, ty) for t, ty in self.variants))


@dataclass(frozen=True, eq=False)
class Bytes(Ty):
    """A byte string of STATICALLY KNOWN length, with a declared encoding.

    Length in the type is not pedantry. Every buffer overflow is a program
    that knew a length at one place and not at another; here there is only one
    place. `enc` is carried for the same reason a unit is: utf8 and latin1 are
    the same bytes and different text, and mixing them is the Mars Climate
    Orbiter of string handling.
    """
    n: int = 0
    enc: str = "bytes"

    def canon(self):
        return ("bytes", self.n, self.enc)

    def __repr__(self):
        return "bytes%d<%s>" % (self.n, self.enc)


@dataclass(frozen=True, eq=False)
class Cap(Ty):
    """A capability: permission to perform one class of effect, as a value.

    LINEAR. It is consumed by the effect it authorises and returned fresh, so
    a function that was handed one network capability cannot make two calls
    out of it, and a function that was handed none cannot reach the network at
    all no matter what it declares. Authority travels in the arguments, which
    means it is visible in the type of every function on the path.
    """
    kind: str = "NET"

    def canon(self):
        return ("cap", self.kind)

    @property
    def linear(self):
        return True

    def __repr__(self):
        return "cap<%s>" % self.kind


@dataclass(frozen=True, eq=False)
class Buf(Ty):
    """A mutable buffer at an explicit memory level. LINEAR.

    `store` consumes a buffer and produces the updated one, so there is never
    more than one name for it. Aliasing is therefore not prevented by an
    analysis; it is unrepresentable, because a second use of the same value is
    a second EDGE in the graph and the checker counts edges.
    """
    elem: Ty = Real()
    shape: Tuple[int, ...] = (1,)
    level: str = "DRAM"

    def canon(self):
        return ("buf", self.elem.canon(), tuple(self.shape), self.level)

    @property
    def linear(self):
        return True

    @property
    def count(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def frozen_ty(self):
        return Arr(self.elem, self.shape, self.level)

    def __repr__(self):
        return "buf%s<%r>@%s" % (list(self.shape), self.elem, self.level)


def is_linear(t):
    """Linearity is a property of the TYPE, so the check is one lookup and
    cannot be forgotten at a use site."""
    return bool(getattr(t, "linear", False))


def leaf_types(t):
    """The scalar leaves of a type, in layout order. Records and vectors are
    trees; the machine sees a flat row of doubles, and something has to say
    which is which exactly once."""
    if isinstance(t, Rec):
        out = []
        for _, ft in t.fields:
            out.extend(leaf_types(ft))
        return out
    if isinstance(t, Vec):
        return [t.elem] * t.n
    return [t]
