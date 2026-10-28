"""
Three cases that decide whether this language is serious.

Each one is a real failure mode from the domains in question, and in each one
the checker must reject the broken version at load time -- not log, not warn,
not substitute a default, and above all not absorb it into a bottom value and
keep flying.

  1. Mars Climate Orbiter       -- lbf*s mixed with N*s. Dimensionally
                                   identical, scale off by 4.448. The orbiter
                                   entered the atmosphere and was lost.
  2. attitude control step      -- cross-frame arithmetic, and a divide by a
                                   norm that nobody proved is non-zero.
  3. exascale stencil+reduction -- f64 addition is not associative, so the
                                   answer depends on the reduction tree, and
                                   data movement, not arithmetic, is the cost.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keel import wire                                            # noqa: E402
from keel.check import check                                     # noqa: E402
from keel.explain import explain                                 # noqa: E402
from keel.build import B                                         # noqa: E402
from keel.ir import Def                                           # noqa: E402
from keel.types import Arr, LBF_S, M, MPS, NS, ONE, Real  # noqa: E402

BAR = "=" * 78


def banner(t):
    print("\n" + BAR + "\n" + t + "\n" + BAR)


# ============================================================== case 1: MCO
def mco(stage):
    """Three stages, because the interesting part is not that the checker says
    no -- it is that it says no in a form the GENERATOR can act on without a
    human in the loop."""
    b = B()
    acc = b.param("impulse_total", Real("f64", NS), 0.0, 1.0e6)
    imp = b.param("thruster_impulse", Real("f64", LBF_S), 0.0, 1.0e3)
    x = imp if stage == "bug" else b.convert(imp, NS)
    root = b.add(acc, x)
    return Def("mco.accumulate." + stage, root, b.g,
               intent="Accumulate angular momentum desaturation impulse for "
                      "trajectory correction. Ground software reports lbf*s; "
                      "flight software integrates in N*s.",
               dal="A", wcet_cycles=1000, ceiling=frozenset(),
               tol=None if stage == "final" else 1e-12,
               atol=1e-9 if stage == "final" else None)


# =================================================== case 2: attitude control
def attitude(variant):
    b = B()
    R = b.rot("R_eci_to_body", "ECI", "BODY")
    vx = b.param("v_x", Real("f64", MPS), -8000.0, 8000.0)
    vy = b.param("v_y", Real("f64", MPS), -8000.0, 8000.0)
    vz = b.param("v_z", Real("f64", MPS), -8000.0, 8000.0)
    v_eci = b.vec3(vx, vy, vz, "ECI")

    if variant == "frame":
        # Reaction-wheel rate vector, already expressed in the body frame.
        wx = b.param("w_x", Real("f64", MPS), -10.0, 10.0)
        w_body = b.vec3(wx, wx, wx, "BODY")
        root = b.add(v_eci, w_body)          # <-- ECI + BODY
        tol = 1e-12
    else:
        v_body = b.rot_apply(R, v_eci)
        spd = b.norm(v_body)                 # m/s, provably only in [0, inf)
        one = b.const(1.0, Real("f64", ONE))
        if variant == "unguarded":
            root = b.div(one, spd)           # <-- nobody proved spd != 0
        else:
            safe = b.clamp(spd, 1.0e-3, 1.0e5)
            root = b.div(one, safe)
        tol = 1e-12

    return Def("attitude.step." + variant, root, b.g,
               intent="Body-frame velocity direction for the attitude "
                      "control law, one 100 Hz minor frame.",
               dal="A", tol=tol, wcet_cycles=500)


# ======================================================= case 3: exascale HPC
N = 1 << 20


def stencil(assoc):
    b = B()
    # Element bounds are mandatory: without a magnitude bound there is no
    # provable error bound for any reduction over this array.
    u = b.param("u", Arr(Real("f64", M), (N,), "DRAM"), -1.0, 1.0)
    lap = b.stencil(u, [-1, 0, 1], [1.0, -2.0, 1.0],
                    out_level="DRAM", boundary="clamp")
    root = b.reduce(lap, assoc)
    # A sum that can straddle zero has no finite RELATIVE bound, so the
    # tolerance that actually means something here is an absolute one.
    return Def("hpc.laplacian_norm." + assoc, root, b.g,
               intent="Residual norm of a 1-D Laplacian over %d points, "
                      "reduced across ranks." % N,
               dal="C", atol=1e-6, require_repro=True)


# =============================================================== run them all
def show(d, verbose=False):
    rep = check(d)
    print(rep.summary())
    if verbose:
        print()
        print(explain(d, rep))
    return rep


banner("CASE 1  Mars Climate Orbiter -- same dimension, wrong scale")
print("stage 1: the bug as flown --")
show(mco("bug"))
print("\nstage 2: unit fixed, but the tolerance claim is the wrong KIND --")
show(mco("advice"))
print("\nstage 3: the generator took the advice --")
r1 = show(mco("final"), verbose=True)

banner("CASE 2  attitude control -- frames and an unproven divisor")
show(attitude("frame"))
print()
show(attitude("unguarded"))
print()
r2 = show(attitude("clamped"), verbose=True)

banner("CASE 3  exascale stencil -- reproducibility and the error bound")
for assoc in ("nondet", "ordered", "tree"):
    dd = stencil(assoc)
    r = check(dd)
    print(r.summary())
    print("    proven absolute error bound: %.3e   (budget 1e-06)"
          % r.aerr[dd.root])
    print()

_tree = stencil("tree")
r3 = check(_tree)
c = r3.cost.report()
print("cost of the accepted exascale version (%d points):" % N)
print("    flops                 %.3e" % c["flops"])
print("    DRAM traffic          %.3e bytes" % c["bytes"].get("DRAM", 0))
print("    L1 traffic            %.3e bytes" % c["bytes"].get("L1", 0))
print("    arithmetic intensity  %.3f flop/byte" % c["arithmetic_intensity"])
print("    energy                %.3e pJ" % c["energy_pj"])
dram_pj = c["bytes"].get("DRAM", 0) * 40.0
print("    -> %.1f%% of the energy is DRAM traffic, not arithmetic"
      % (100.0 * dram_pj / c["energy_pj"]))

banner("WIRE  TACET glyph transport with single-glyph correction")
d = attitude("clamped")
stream, nbytes = wire.encode(d.graph, d.root)
print("DAG nodes            %d" % len(d.graph.topo(d.root)))
print("serialized payload   %d bytes" % nbytes)
print("TACET stream         %d glyphs" % len(stream))
print("first 72 glyphs      %s" % stream[:72])

clean, f0, b0 = wire.decode(stream, nbytes)
print("\nclean decode         corrected=%d uncorrectable=%d match=%s"
      % (f0, b0, clean == wire._ser(d.graph, d.root)))

one = wire.corrupt(stream, 5)
got, f1, b1 = wire.decode(one, nbytes)
print("1 glyph flipped      corrected=%d uncorrectable=%d match=%s"
      % (f1, b1, got == clean))

two = wire.corrupt(wire.corrupt(stream, 3), 7)
got2, f2, b2 = wire.decode(two, nbytes)
print("2 glyphs, one frame  corrected=%d uncorrectable=%d match=%s  (detected)"
      % (f2, b2, got2 == clean))

banner("TRUST ROOT SIZE")
# Being honest about this is the whole argument. A human never reads a KEEL
# program, so control rests entirely on these files being small enough to
# actually read to the end. Anything that can make a false claim about a
# program belongs on this list, not just the checker.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
trusted = [
    ("keel/check.py", "fourteen checks + the obligation discharger"),
    ("keel/types.py", "what units, frames and sites MEAN"),
    ("keel/ir.py", "content addressing -- identity of every node"),
    ("keel/cost.py", "the WCET and energy claims"),
    ("keel/wire.py", "decoding: reconstructs the program from a stream"),
]
untrusted = [
    ("keel/explain.py", "non-authoritative view; cannot approve anything"),
    ("examples/demo.py", "not part of the system"),
    ("examples/substrate.py", "not part of the system"),
    ("examples/silicon.py", "not part of the system"),
]


def sloc(f):
    return sum(1 for ln in open(os.path.join(ROOT, f), encoding="utf-8")
               if ln.strip() and not ln.strip().startswith("#"))


tot = 0
for f, why in trusted:
    n = sloc(f)
    tot += n
    print("  TRUSTED    %-18s %4d  %s" % (f, n, why))
print("  %-29s %4d  <- everything a human must read" % ("TOTAL TRUST ROOT", tot))
print()
for f, why in untrusted:
    print("  untrusted  %-18s %4d  %s" % (f, sloc(f), why))
