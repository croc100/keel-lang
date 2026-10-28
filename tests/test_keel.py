"""
KEEL test suite. Plain asserts, no framework:  python tests/test_keel.py

Every test is of the form "the checker must reject X" or "the checker must
accept Y". There is nothing in between -- no warning level, no degraded mode,
no bottom value. That is the property under test.

Note the recurring pattern around tolerances. A value whose proven range
includes zero has NO finite relative error bound, because relative error is
undefined there. Such a definition must declare an ABSOLUTE tolerance (atol).
A value bounded away from zero can use a relative one (tol). The checker says
which, and says so precisely, because its reader is a generator that has to
act on the answer.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keel import wire                                                # noqa: E402
from keel.check import check                                         # noqa: E402
from keel.build import B
from keel.ir import Def, Node                                     # noqa: E402
from keel.types import (Arr, Int, KM, LBF_S, M, MPA, MPS, NS, ONE,  # noqa: E402
                        PA, Real, Site, site)
import keel.backend as emitmod                                       # noqa: E402

PASS, FAIL = [], []


def expect(name, rep, accept, kind=None):
    ok = rep.ok if accept else (not rep.ok)
    if ok and kind is not None:
        ok = any(v.kind == kind for v in rep.violations)
    (PASS if ok else FAIL).append(name)
    detail = "" if ok else "   got: %s" % (
        [v.kind for v in rep.violations] or "accepted")
    print("  %s %-46s%s" % ("ok  " if ok else "FAIL", name, detail))


def note(name, cond, msg):
    (PASS if cond else FAIL).append(name)
    print("  %s %s" % ("ok  " if cond else "FAIL", msg))


def D(name, root, g, **kw):
    kw.setdefault("dal", "C")
    return Def(name, root, g, **kw)


def ARR(n=16, lo=-1.0, hi=1.0, level="DRAM"):
    return Arr(Real("f64", M), (n,), level), lo, hi


print("\n--- units: dimension is not enough, scale must match --------------")

b = B()
acc = b.param("acc", Real("f64", NS), 0, 1e6)
imp = b.param("imp", Real("f64", LBF_S), 0, 1e3)
expect("lbf*s + N*s rejected (Mars Climate Orbiter)",
       check(D("u1", b.add(acc, imp), b.g, atol=1e-9)), False, "unit")

b = B()
acc = b.param("acc", Real("f64", NS), 0, 1e6)
imp = b.param("imp", Real("f64", LBF_S), 0, 1e3)
expect("explicit convert accepted",
       check(D("u2", b.add(acc, b.convert(imp, NS)), b.g, atol=1e-9)), True)

b = B()
x = b.param("x", Real("f64", M), 1, 2)
expect("sqrt of m rejected (odd dimension)",
       check(D("u3", b.sqrt(x), b.g, tol=1e-9)), False, "unit")


print("\n--- reference frames ----------------------------------------------")

b = B()
p = lambda n: b.param(n, Real("f64", MPS), -1e3, 1e3)
v_eci = b.vec3(p("a"), p("b"), p("c"), "ECI")
w_body = b.vec3(p("d"), p("e"), p("f"), "BODY")
expect("ECI + BODY rejected",
       check(D("f1", b.add(v_eci, w_body), b.g, atol=1e-9)), False, "frame")

b = B()
R = b.rot("R", "ECI", "BODY")
p = lambda n: b.param(n, Real("f64", MPS), -1e3, 1e3)
v_eci = b.vec3(p("a"), p("b"), p("c"), "ECI")
w_body = b.vec3(p("d"), p("e"), p("f"), "BODY")
expect("rotation into BODY, then add, accepted",
       check(D("f2", b.add(b.rot_apply(R, v_eci), w_body), b.g, atol=1e-9)),
       True)

b = B()
R = b.rot("R", "ECI", "BODY")
p = lambda n: b.param(n, Real("f64", MPS), -1e3, 1e3)
expect("rot<ECI->BODY> applied to a BODY vector rejected",
       check(D("f3", b.rot_apply(R, b.vec3(p("a"), p("b"), p("c"), "BODY")),
               b.g, atol=1e-9)), False, "frame")


print("\n--- obligations replace the bottom value --------------------------")

b = B()
one = b.const(1.0, Real("f64", ONE))
d = b.param("d", Real("f64", M), -5, 5)
expect("divide by a range that straddles 0 rejected",
       check(D("o1", b.div(one, d), b.g, tol=1e-9)), False, "obligation")

b = B()
one = b.const(1.0, Real("f64", ONE))
d = b.param("d", Real("f64", M), -5, 5)
expect("same divide accepted once a clamp refines the range",
       check(D("o2", b.div(one, b.clamp(d, 1e-3, 5)), b.g, tol=1e-9)), True)

b = B()
x = b.param("x", Real("f64", ONE), -4, 9)
expect("sqrt of a possibly-negative value rejected",
       check(D("o3", b.sqrt(x), b.g, tol=1e-9)), False, "obligation")

b = B()
t, lo, hi = ARR()
arr = b.param("u", t, lo, hi)
i = b.param("i", Int(), 0, 99)
expect("array index [0,99] into 16 elements rejected",
       check(D("o4", b.load(arr, i), b.g, atol=1e-9)), False, "obligation")

b = B()
t, lo, hi = ARR()
arr = b.param("u", t, lo, hi)
expect("array index from a bounded loop variable accepted",
       check(D("o5", b.load(arr, b.loopvar(16)), b.g, atol=1e-9)), True)

b = B()
t, lo, hi = ARR()
expect("array parameter without element bounds rejected",
       check(D("o6", b.load(b.param("u", t), b.loopvar(16)), b.g, atol=1e-9)),
       False, "bounds")

b = B()
x = b.param("x", Real("f64", ONE), 1, 2)
expect("repeat with a non-positive bound rejected",
       check(D("o7", b.repeat(x, 0), b.g, tol=1e-9)), False, "bounds")


print("\n--- the builder is untrusted --------------------------------------")

b = B()
x = b.param("x", Real("f64", M), 0, 1)
y = b.param("y", Real("f64", M), 0, 1)
lie = Node("add", (x, y), {}, Real("f64", NS))     # claims newton-seconds
b.g.add(lie)
expect("a node that lies about its own type rejected",
       check(D("b1", lie.addr, b.g, atol=1e-9)), False, "declared")


print("\n--- floating point error is proven, not hoped ---------------------")

b = B()
a = b.param("a", Real("f64", M), 1, 2)
c = b.param("c", Real("f64", M), -2, -1)
expect("catastrophic cancellation rejected against a relative tolerance",
       check(D("e1", b.add(a, c), b.g, tol=1e-9)), False, "tolerance")

b = B()
a = b.param("a", Real("f64", M), 1, 2)
c = b.param("c", Real("f64", M), -2, -1)
expect("the same cancelling sum accepted with an absolute tolerance",
       check(D("e2", b.add(a, c), b.g, atol=1e-15)), True)

b = B()
a = b.param("a", Real("f64", M), 1, 2)
c = b.param("c", Real("f64", M), 1, 2)
expect("same-sign addition accepted relatively (cannot cancel)",
       check(D("e3", b.add(a, c), b.g, tol=1e-15)), True)

b = B()
a = b.param("a", Real("f64", M), 1, 2)
c = b.param("c", Real("f64", M), -2, -1)
r = check(D("e4", b.add(a, c), b.g))
msg = " ".join(v.msg for v in check(D("e4", b.add(a, c), b.g, tol=1e-9)).violations)
note("advice", "declare an absolute tolerance" in msg,
     "rejection tells the generator exactly what to declare instead")

N = 1 << 20
errs = {}
for assoc in ("ordered", "tree"):
    b = B()
    t, lo, hi = ARR(N)
    u = b.param("u", t, lo, hi)
    rr = b.reduce(u, assoc)
    errs[assoc] = check(D("r." + assoc, rr, b.g)).aerr[rr]
ratio = errs["ordered"] / errs["tree"]
note("tree beats ordered", ratio > 1e4,
     "tree reduction error bound is %.0fx tighter than sequential "
     "(%.2e vs %.2e)" % (ratio, errs["tree"], errs["ordered"]))


print("\n--- reproducibility is part of the type ---------------------------")

for assoc, accept in (("nondet", False), ("tree", True)):
    b = B()
    t, lo, hi = ARR(1024)
    u = b.param("u", t, lo, hi)
    expect("require_repro with assoc=%s" % assoc,
           check(D("p." + assoc, b.reduce(u, assoc), b.g, require_repro=True)),
           accept, None if accept else "repro")


print("\n--- mixed criticality ---------------------------------------------")

b = B()
s = b.param("sensor", Real("f64", M), 0, 1, dal="D")
expect("DAL D input reaching a DAL A output rejected",
       check(Def("c1", b.add(s, s), b.g, dal="A", atol=1e-9)), False, "dal")

b = B()
s = b.param("sensor", Real("f64", M), 0, 1, dal="D")
mon = b.monitor(s, "A", check="range + staleness + cross-string vote")
expect("same input accepted through a declared monitor",
       check(Def("c2", b.add(mon, mon), b.g, dal="A", atol=1e-9)), True)

b = B()
x = b.param("x", Real("f64", M), 0, 1)
expect("DAL A without any declared tolerance rejected",
       check(Def("c3", b.add(x, x), b.g, dal="A")), False, "tolerance")

b = B()
s = b.param("sensor", Real("f64", M), 0, 1, dal="D")
mon = b.monitor(s, "A", check=None)
expect("monitor without a declared check rejected",
       check(Def("c4", b.add(mon, mon), b.g, dal="A", atol=1e-9)), False, "dal")


print("\n--- memory placement and capabilities -----------------------------")

b = B()
t, lo, hi = ARR(8)
a1 = b.param("a", t, lo, hi)
a2 = b.param("b", t, lo, hi)
expect("arithmetic directly on a DRAM array rejected",
       check(D("m1", b.add(a1, a2), b.g, atol=1e-9)), False, "memory")

b = B()
s = b.sys("downlink", [], Real("f64", ONE), "NET")
expect("effect outside the declared ceiling rejected",
       check(D("k1", s, b.g, ceiling=frozenset())), False, "capability")

b = B()
s = b.sys("downlink", [], Real("f64", ONE), "NET")
expect("same effect accepted inside the ceiling",
       check(D("k2", s, b.g, ceiling=frozenset({"NET"}))), True)


print("\n--- WCET is computed from proven bounds ---------------------------")

b = B()
R = b.rot("R", "ECI", "BODY")
p = lambda n: b.param(n, Real("f64", MPS), -8e3, 8e3)
v = b.rot_apply(R, b.vec3(p("x"), p("y"), p("z"), "ECI"))
root = b.div(b.const(1.0, Real("f64", ONE)), b.clamp(b.norm(v), 1e-3, 1e5))
g, rt = b.g, root
expect("WCET over budget rejected",
       check(D("w1", rt, g, tol=1e-12, wcet_cycles=10)), False, "wcet")
expect("WCET within budget accepted",
       check(D("w2", rt, g, tol=1e-12, wcet_cycles=500)), True)


print("\n--- content addressing -------------------------------------------")

b1 = B()
x1 = b1.param("x", Real("f64", M), 0, 1)
e1 = b1.mul(x1, x1)
b2 = B()
x2 = b2.param("x", Real("f64", M), 0, 1)
e2 = b2.mul(x2, x2)
note("addressing stable", e1 == e2,
     "identical expressions get identical addresses (%s)" % e1[:8])

b = B()
x = b.param("x", Real("f64", M), 0, 1)
for _ in range(50):
    b.mul(x, x)
note("dedup", len(b.g.nodes) == 2,
     "50 rebuilds of one subexpression stored once (%d nodes)" % len(b.g.nodes))

b = B()
x = b.param("x", Real("f64", M), 0, 1)
y = b.param("x", Real("f64", NS), 0, 1)     # same name, different type
note("type in address", x != y,
     "address depends on type, so a name cannot be reused to smuggle a body")


print("\n--- substrates: one lattice, not one plane per discipline ---------")
# These sites exist only for the tests. Note what they have in common with
# DRAM: five numbers and a time domain. Nothing else distinguishes a relay
# coil from a cache line as far as the checker is concerned.
site(Site("T_BOUNCE", settle=5.0e-3, readback=True, load_a=1.0e-5))
site(Site("T_PIN", prop=2.0e-8, drive_a=20.0e-3, readback=True, clock="T_FAST"))
site(Site("T_BUF", prop=1.0e-7, drive_a=500.0e-3, readback=False))
site(Site("T_SLOW", prop=1.0e-9, readback=True, clock="T_SLOWCLK"))
site(Site("T_PANEL", settle=1.6e-2, prop=8.3e-3, drive_a=1.0, readback=False))


def sensed(after, s="T_BOUNCE", ty=None, **kw):
    b = B()
    r = b.sense(s, ty or Real("f64", ONE), after=after, lo=0.0, hi=1.0, **kw)
    return Def("t.sense", r, b.g, dal="C"), b


expect("sense inside the settling window rejected",
       check(sensed(1.0e-3)[0]), False, "obligation")
expect("sense after the settling window accepted",
       check(sensed(2.0e-2)[0]), True)
expect("sense with no declared elapsed time rejected",
       check(sensed(None)[0]), False, "obligation")

b = B()
r = b.sense("T_NOWHERE", Real("f64", ONE), after=1.0, lo=0, hi=1)
expect("sense of an unregistered site rejected",
       check(Def("t.unknown_site", r, b.g, dal="C")), False, "substrate")

b = B()
r = b.sense("T_PANEL", Real("f64", ONE), after=1.0, lo=0, hi=1)
expect("sense of a write-only site rejected",
       check(Def("t.writeonly", r, b.g, dal="C")), False, "substrate")

# The time domain is a property of the SITE. A builder that claims otherwise
# is caught by the same declared-vs-derived check that catches a lied unit.
b = B()
r = b.sense("T_PIN", Real("f64", ONE, None), after=1.0, lo=0, hi=1)
expect("builder cannot lie about a site's time domain",
       check(Def("t.clock_lie", r, b.g, dal="C")), False, "declared")


def driven(s, load_a, loads=1):
    b = B()
    v = b.const(1.0, Real("f64", ONE))
    r = b.drive(v, s, effect="e", loads=loads, load_a=load_a)
    return Def("t.drive", r, b.g, dal="C", ceiling=frozenset(["e"]))


expect("driving 70mA from a 20mA pin rejected",
       check(driven("T_PIN", 70.0e-3)), False, "obligation")
expect("the same load behind a buffer accepted",
       check(driven("T_BUF", 70.0e-3)), True)
expect("fanout of 30 x 1mA on a 20mA pin rejected",
       check(driven("T_PIN", 1.0e-3, loads=30)), False, "obligation")
# A site where current is not the limiting quantity owes no current proof.
# Demanding amperes everywhere would be one discipline colonising the rest.
expect("a site that draws no current owes no current proof",
       check(driven("T_PANEL", 0.0)), True)


def crossing(depth=None):
    b = B()
    slow = b.sense("T_SLOW", Real("f64", ONE, "T_SLOWCLK"), after=1.0,
                   lo=0.0, hi=1.0)
    fast = b.sense("T_PIN", Real("f64", ONE, "T_FAST"), after=1.0,
                   lo=0.0, hi=1.0)
    v = slow if depth is None else b.sync(slow, "T_FAST", depth=depth)
    return Def("t.cdc", b.add(v, fast), b.g, dal="C")


# A clock domain is a reference frame for time, so this is caught by the rule
# written for ECI + BODY, with no clock-specific code anywhere in the checker.
expect("unsynchronised clock-domain crossing rejected",
       check(crossing()), False, "frame")
expect("depth-1 synchroniser rejected (metastability)",
       check(crossing(1)), False, "frame")
expect("depth-2 synchroniser accepted", check(crossing(2)), True)

b = B()
v = b.sense("T_PIN", Real("f64", ONE, "T_FAST"), after=1.0, lo=0, hi=1)
expect("sync to the domain a value is already in rejected",
       check(Def("t.nop_sync", b.sync(v, "T_FAST"), b.g, dal="C")),
       False, "frame")


def stress(bound, unit, n=1):
    b = B()
    from keel.types import M2, NEWTON
    area = b.const(n * 2.01e-5, Real("f64", M2))
    f = b.param("load", Real("f64", NEWTON), 0.0, 8000.0)
    r = b.require(b.div(f, area), 0.0, bound, unit, "proof stress")
    return Def("t.stress", r, b.g, dal="C", atol=1.0)


expect("stress over the allowable rejected", check(stress(3.2e8, PA)),
       False, "obligation")
expect("stress inside the allowable accepted",
       check(stress(3.2e8, PA, n=2)), True)
# The Mars Climate Orbiter failure mode, applied to an invariant: a bound in
# MPa against a value in Pa is dimensionally identical and 1e6 wrong.
expect("invariant bounded in MPa against a value in Pa rejected",
       check(stress(320.0, MPA, n=2)), False, "unit")

b = B()
v = b.param("x", Real("f64", ONE), -5.0, 5.0)
expect("require with no declared unit rejected",
       check(Def("t.nounit", b.require(v, 0.0, 1.0, None, "w"), b.g,
                 dal="C", atol=1.0)), False, "unit")

# require PROVES; clamp NARROWS. Mixing the two up is how an unprovable
# program gets quietly made provable, so the distinction is tested.
b = B()
v = b.param("x", Real("f64", ONE), -5.0, 5.0)
req = b.require(v, -10.0, 10.0, ONE, "w")
one = b.const(1.0, Real("f64", ONE))
expect("require does not narrow an interval the way clamp does",
       check(Def("t.req_no_narrow", b.div(one, req), b.g, dal="C", atol=1.0)),
       False, "obligation")
b = B()
v = b.param("x", Real("f64", ONE), -5.0, 5.0)
one = b.const(1.0, Real("f64", ONE))
expect("clamp narrows, discharging the same divide",
       check(Def("t.clamp_narrows", b.div(one, b.clamp(v, 1.0, 5.0)), b.g,
                 dal="C", atol=1.0)), True)


def panel_chain(hops, deadline):
    b = B()
    v = b.sense("T_BOUNCE", Real("f64", ONE), after=2.0e-2, lo=0.0, hi=1.0)
    for i in range(hops):
        v = b.drive(v, "T_PANEL", effect="d%d" % i)
    return Def("t.deadline", v, b.g, dal="C", deadline_s=deadline,
               ceiling=frozenset("d%d" % i for i in range(hops)))


# A deadline in seconds is not a cycle count: 20 ms of contact bounce plus two
# 24 ms panel commits is physical time that no clock speed makes smaller.
expect("physical deadline met accepted", check(panel_chain(1, 0.1)), True)
expect("physical deadline missed rejected",
       check(panel_chain(5, 0.1)), False, "deadline")
note("seconds are physical, not cycles",
     abs(check(panel_chain(1, 0.1)).cost.seconds - (0.02 + 0.0243)) < 1e-6,
     "20ms debounce + 24.3ms panel commit = %.4fs end to end"
     % check(panel_chain(1, 0.1)).cost.seconds)

b = B()
u = b.param("u", Arr(Real("f64", M), (16,), "T_NOWHERE"), -1.0, 1.0)
expect("array at an unregistered site rejected",
       check(Def("t.arr_site", b.reduce(u, "tree"), b.g, dal="C", atol=1.0)),
       False, "substrate")
b = B()
u = b.param("u", Arr(Real("f64", M), (16,), "T_PANEL"), -1.0, 1.0)
expect("reduction over a write-only site rejected",
       check(Def("t.arr_wo", b.reduce(u, "tree"), b.g, dal="C", atol=1.0)),
       False, "substrate")

site(Site("T_DUP", settle=1.0))
try:
    site(Site("T_DUP", settle=2.0))
    _redef = True
except ValueError:
    _redef = False
note("a site name cannot be silently redefined", not _redef,
     "re-registering T_DUP with different physics is refused, for the same "
     "reason content addressing exists")


print("\n--- silicon: an execution unit is a site, not a new idea -----------")


def bound(unit, fmt="f64", lanes=1, extra=None):
    b = B()
    x = b.param("x", Real(fmt, ONE), 1.0, 2.0)
    v = b.bind(b.mul(x, x), unit, lanes=lanes)
    if extra:
        v = extra(b, v, x)
    return Def("t.bind." + unit, v, b.g, dal="C", atol=1e3)


expect("bind to an unregistered unit rejected",
       check(bound("T_NOSUCH")), False, "substrate")
expect("bind to a site that does no arithmetic rejected",
       check(bound("SHMEM")), False, "substrate")
# A datapath that carries f16 has no f64 to issue. This is not a convention.
expect("f64 bound to an f16 unit rejected",
       check(bound("TENSOR")), False, "substrate")
expect("f64 bound to an f32 unit rejected",
       check(bound("DSP48")), False, "substrate")
expect("f32 bound to the f32 fabric unit accepted",
       check(bound("DSP48", fmt="f32")), True)
expect("more lanes than the unit has rejected",
       check(bound("FMA512", lanes=16)), False, "substrate")
expect("lanes within the unit accepted", check(bound("FMA512", lanes=8)), True)

r = check(bound("ALU64"))
note("an out-of-order port is a constraint, not a command",
     r.binds and r.binds[-1][3] is False,
     "ALU64 (ooo) reports pressure-bound; naming a port on an OoO core is a "
     "claim to prove, not an order the silicon obeys")
r = check(bound("DSP48", fmt="f32"))
note("a fabric slice really is commanded",
     r.binds and r.binds[-1][3] is True,
     "DSP48 (not ooo) reports commanded, because on a fabric the placement "
     "is the program")

# An AVX-512 unit is clocked below the core it sits in. Cycles cannot say so;
# seconds can, which is the whole reason they are counted separately.
sa = check(bound("ALU64")).cost.seconds
sf = check(bound("FMA512")).cost.seconds
note("a unit's own clock rate shows up in physical time", sf > sa,
     "same multiply: ALU64 %.3fns vs FMA512 %.3fns (AVX-512 downclock)"
     % (sa * 1e9, sf * 1e9))

# Computing on the fabric puts the result in the fabric's time domain, so the
# rule written for Vec3<ECI> + Vec3<BODY> does the CPU/FPGA boundary unchanged.
expect("fabric result added to a core value rejected",
       check(bound("DSP48", fmt="f32",
                   extra=lambda b, v, x: b.add(v, x))), False, "frame")
expect("the same crossing through a depth-2 synchroniser accepted",
       check(bound("DSP48", fmt="f32",
                   extra=lambda b, v, x: b.add(b.sync(v, None, depth=2), x))),
       True)


def banked(stride, lanes=32, where="SHMEM"):
    b = B()
    a = b.param("a", Arr(Real("f64", M), (4096,), where), -1.0, 1.0)
    return check(Def("t.bank", b.load(a, b.const(0, Int()), lanes=lanes,
                                      stride=stride), b.g, dal="C", atol=1.0))


# With B banks and stride s the residues {i*s mod B} form a subgroup of order
# B/gcd(s,B), so serialization is forced by gcd. Nothing is measured.
note("stride 1 over 32 banks has no conflict",
     banked(1).conflicts[0][4] == 1, "stride 1 -> serialize x1")
note("stride 2 serializes in half", banked(2).conflicts[0][4] == 2,
     "stride 2 -> serialize x2")
note("stride 32 collapses onto one bank",
     banked(32).conflicts[0][4] == 32, "stride 32 -> serialize x32")
note("padding by one element restores full parallelism",
     banked(33).conflicts[0][4] == 1,
     "stride 33 -> serialize x1; this is why GPU programmers pad by one, and "
     "the factor is gcd(33,32)=1 rather than folklore")
note("a conflict costs real cycles",
     banked(32).cost.cycles == 32 * banked(1).cost.cycles,
     "stride 32 costs %.0f cycles against stride 1's %.0f"
     % (banked(32).cost.cycles, banked(1).cost.cycles))
note("an unbanked site records no conflict",
     banked(32, where="DRAM").conflicts == [],
     "DRAM declares no banks here, so it owes no bank proof -- obligations "
     "come from declared physics, not from a checklist")


def ports(units, budget=None):
    b = B()
    v = b.param("v", Real("f64", ONE), 1.0, 2.0)
    acc = v
    for u in units:
        acc = b.bind(b.mul(acc, v), u)
    return check(Def("t.port", acc, b.g, dal="C", atol=1e9,
                     wcet_cycles=budget))


note("operations on one port contend",
     ports(["ALU64"] * 8).cost.throughput_cycles() == 4.0,
     "8 issues at 2/cycle on p01 is a 4-cycle floor")
note("spreading over two ports halves the floor",
     ports(["ALU64", "FMA512"] * 4).cost.throughput_cycles() == 2.0,
     "the same 8 multiplies across p01 and p05 is a 2-cycle floor")
expect("a WCET budget under the port floor is rejected",
       ports(["ALU64"] * 64, budget=8), False, "wcet")


print("\n--- machine code: the claim meets the silicon ----------------------")


def scalar(build, lo=2.0, hi=8.0, lo2=1.0, hi2=4.0, tol=1e-6):
    b = B()
    x = b.param("x", Real("f64", ONE), lo, hi)
    y = b.param("y", Real("f64", ONE), lo2, hi2)
    return Def("t.emit", build(b, x, y), b.g, dal="C", atol=tol), b


try:
    _d, _ = scalar(lambda b, x, y: b.mul(x, y), tol=1e9)
    _rep = check(_d)
    _code = emitmod.emit(_d, _rep)
    _CAN_RUN = _code.run(3.0, 4.0) == 12.0
except Exception as _e:                                    # noqa: BLE001
    _CAN_RUN = False
    print("  note executable memory unavailable here (%s); emission is still "
          "tested" % _e)

# Every emitted operation must agree with IEEE 754 double arithmetic EXACTLY.
# Not approximately: these are the same instructions Python's own floats use,
# so any difference is a bug in the emitter and not a rounding question.
CASES = [
    ("add", lambda b, x, y: b.add(x, y), lambda a, c: a + c),
    ("sub", lambda b, x, y: b.sub(x, y), lambda a, c: a - c),
    ("mul", lambda b, x, y: b.mul(x, y), lambda a, c: a * c),
    ("div", lambda b, x, y: b.div(x, y), lambda a, c: a / c),
    ("sqrt", lambda b, x, y: b.sqrt(x), lambda a, c: a ** 0.5),
    ("neg", lambda b, x, y: b.neg(x), lambda a, c: -a),
    ("clamp", lambda b, x, y: b.clamp(x, 3.0, 5.0),
     lambda a, c: min(max(a, 3.0), 5.0)),
    ("nested", lambda b, x, y: b.div(b.add(x, y), b.sqrt(x)),
     lambda a, c: (a + c) / a ** 0.5),
]
if _CAN_RUN:
    bad = []
    for nm, build, ref in CASES:
        d, _ = scalar(build, tol=1e9)
        code = emitmod.emit(d, check(d))
        for a in (2.0, 3.5, 7.0, 8.0):
            for c in (1.0, 2.25, 4.0):
                got = code.run(*[a, c][:len(code.params)])
                want = ref(a, c)
                if got != want:
                    bad.append((nm, a, c, got, want))
    note("every emitted operation is bit-exact against IEEE 754",
         not bad, "%d operations x 12 input pairs, all exact%s"
         % (len(CASES), "" if not bad else " -- MISMATCH %r" % (bad[:3],)))

    # convert carries the exact rational scale into one multiply
    b = B()
    km = b.param("d", Real("f64", KM), 1.0, 5.0)
    d = Def("t.conv", b.convert(km, M), b.g, dal="C", atol=1e-6)
    code = emitmod.emit(d, check(d))
    note("an exact unit conversion lowers to one multiply",
         code.run(3.0) == 3000.0,
         "3 km -> %g m, and the scale came from a Fraction, not a float "
         "literal in the backend" % code.run(3.0))

    # the loop that matters: proven interval vs what the CPU actually does
    for nm, build, _ref in CASES:
        d, _ = scalar(build, tol=1e9)
        res = emitmod.verify(d, check(d), samples=5)
        note("proven interval holds on real hardware: %s" % nm,
             res["inside"],
             "%-7s %3d points executed, all inside [%g, %g]  (%d bytes)"
             % (nm, res["points"], res["proven"][0], res["proven"][1],
                res["bytes"]))

    # A declaration is not an instruction. bind/require/monitor emitted a copy
    # each; nothing in them computes, because the proof was paid at check time.
    b = B()
    x = b.param("x", Real("f64", ONE), 2.0, 8.0)
    v = b.require(b.bind(b.mul(x, x), "ALU64"), 0.0, 64.0, ONE, "square")
    d = Def("t.decl", v, b.g, dal="C", atol=1e9)
    code = emitmod.emit(d, check(d))
    ops = [o for _, o, _, _ in code.nodes]
    note("a discharged obligation emits no arithmetic",
         code.run(5.0) == 25.0 and ops.count("require") == 1,
         "bind and require lower to a move each: %s" % ops)

_d, _ = scalar(lambda b, x, y: b.div(x, b.sub(y, y)), tol=1e9)
_rej = check(_d)
try:
    emitmod.emit(_d, _rej)
    _refused = False
except emitmod.Unsupported:
    _refused = True
note("a rejected definition is refused by the backend",
     _refused and not _rej.ok,
     "the code generator will not lower an unproven program; the proof is "
     "the precondition for emission, not a parallel activity")

b = B()
_args = [b.param("p%d" % i, Real("f64", ONE), 1.0, 2.0) for i in range(24)]
_prods = [b.mul(_args[i - 1], _args[i]) for i in range(1, 24)]
_acc = _prods[0]
for _a in _args[1:]:
    pass
for _pp in _prods[1:]:
    _acc = b.add(_acc, _pp)
_d5 = Def("t.wide", _acc, b.g, dal="C", atol=1e9)
_c5 = emitmod.emit(_d5, check(_d5))
_vals = [1.0 + i / 24.0 for i in range(24)]
_want = _vals[0] * _vals[1]
for i in range(2, 24):
    _want = _want + _vals[i - 1] * _vals[i]
note("the parameter count is bounded by the proof, not by the register file",
     _c5.run(*_vals) == _want and _c5.nargs == 24,
     "24 parameters and 45 operations, exact; the ABI is a pointer pair, so "
     "xmm0-3 stopped being the limit")
note("register allocation keeps 45 operations out of memory",
     _c5.stats["spills"] == 0,
     "spills=%d reloads=%d -- values live in xmm registers and touch the "
     "stack only when they must" % (_c5.stats["spills"], _c5.stats["reloads"]))

b = B()
_u = b.param("u", Arr(Real("f64", M), (8,), "DRAM"), -1.0, 1.0)
_dr = Def("t.noloop", b.reduce(_u, "tree"), b.g, dal="C", atol=1.0)
try:
    emitmod.emit(_dr, check(_dr))
    _unsup = False
except emitmod.Unsupported:
    _unsup = True
note("an operation with no lowering is an error, not silence", _unsup,
     "reduce has no x86-64 path in this backend and emission fails loudly")


print("\n--- TACET wire: single-glyph corruption is survivable -------------")

b = B()
R = b.rot("R", "ECI", "BODY")
p = lambda n: b.param(n, Real("f64", MPS), -8e3, 8e3)
v = b.rot_apply(R, b.vec3(p("x"), p("y"), p("z"), "ECI"))
root = b.div(b.const(1.0, Real("f64", ONE)), b.clamp(b.norm(v), 1e-3, 1e5))
stream, nb = wire.encode(b.g, root)
truth = wire._ser(b.g, root)

clean, f, br = wire.decode(stream, nb)
note("roundtrip", clean == truth,
     "clean roundtrip (%d glyphs, %d bytes)" % (len(stream), nb))

rng = random.Random(7)
bad_pos, corrected = [], 0
for pos in range(len(stream)):
    for _ in range(3):
        got, fx, bk = wire.decode(wire.corrupt(stream, pos, rng), nb)
        if got != truth:
            bad_pos.append(pos)
        corrected += fx
note("single-error survivable", not bad_pos,
     "every single-glyph substitution at all %d positions still decodes "
     "exactly (%d corrections)" % (len(stream), corrected))

two = wire.corrupt(wire.corrupt(stream, 3, rng), 7, rng)
got2, f2, b2 = wire.decode(two, nb)
note("double-error detected", b2 > 0,
     "two errors in one frame are detected, not silently mis-corrected")


print("\n" + "=" * 70)
print("%d passed, %d failed" % (len(PASS), len(FAIL)))
if FAIL:
    print("failed: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
