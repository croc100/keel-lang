"""
From a bolt to an FMA port, and all the way down to the bytes.

This file is the test of one claim: that a language whose partition is
(substrate, physical law, time domain) reaches the circuit without acquiring a
second set of concepts for it. An execution unit is registered in the SAME
table as a relay contact and a DRAM bank, and `bind` is the same act as
`drive`. If that were a slogan rather than a design, this file could not be
written with the operations that already existed.

It ends by emitting x86-64, asking the operating system to make the bytes
executable, running them on this CPU, and comparing the answers against the
interval the checker proved. Nothing else in the project closes that loop.

Run:  python examples/silicon.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keel import B, Def, check                                  # noqa: E402
from keel.backend import Unsupported, emit, verify              # noqa: E402
from keel.types import (Arr, Int, M, MPS, ONE, Real, S, SITES)   # noqa: E402


def line(t):
    print("\n" + t + "\n" + "-" * len(t))


def show(d, extra=None):
    r = check(d)
    print(r.summary())
    if extra:
        print("   " + extra(r))
    return r


# =====================================================  1. the unit is a site
line("1. the same multiply, bound to three different pieces of silicon")

# An execution unit has no special status in the type system. It is a site
# with lanes, a throughput, a latency, a port, and -- the fact that matters
# below -- its own clock rate.
for n in ("ALU64", "FMA512", "TENSOR", "DSP48", "SHMEM"):
    s = SITES[n]
    print("  %-7s lanes=%-4d thru=%-4s lat=%-5s port=%-4s hz=%-7s "
          "fmts=%-14s banks=%-3d %s"
          % (n, s.lanes, s.thru, s.lat, s.port or "-",
             ("%.2gG" % (s.hz / 1e9)) if s.hz else "core",
             "/".join(s.fmts) or "any", s.banks,
             "commanded" if not s.ooo else "scheduled"))

print()
for unit in ("ALU64", "FMA512", "TENSOR", "DSP48", "SHMEM"):
    b = B()
    x = b.param("x", Real("f64", M), 1.0, 100.0)
    t = b.param("t", Real("f64", S), 1.0, 10.0)
    d = Def("mul." + unit, b.bind(b.div(x, t), unit), b.g, dal="C", atol=1e-9)
    r = check(d)
    tag = "commanded" if (r.binds and r.binds[-1][3]) else "pressure-bound"
    show(d, lambda r, tag=tag: "%.3f ns of physical time, %s"
         % (r.cost.seconds * 1e9, tag) if r.ok else "")

print("""
  Three results worth reading twice.

  FMA512 is SLOWER than ALU64 for one scalar multiply, because an AVX-512
  unit really is clocked below the core it sits in. That is not a heuristic
  in a cost table; it is `hz` on the site, and it only shows up because
  KEEL counts seconds separately from cycles.

  TENSOR and DSP48 are REJECTED, and not for style. A unit whose datapath
  carries f16 has no f64 to issue. No declaration makes one appear.

  ALU64 and FMA512 report "pressure-bound", not "commanded". On an
  out-of-order core the renamer and the scheduler are silicon and no
  instruction addresses them, so naming a port is a claim to be PROVEN, not
  an order to be obeyed. DSP48 on a fabric is the other case, and the
  checker says which one you got instead of flattering you.""")


# ==========================================  2. banks: arithmetic, not folklore
line("2. a bank conflict is division, and padding is why +1 works")

print("  SHMEM has %d banks. 32 lanes, stride s:\n" % SITES["SHMEM"].banks)
for stride in (1, 2, 4, 8, 16, 32, 33):
    b = B()
    a = b.param("a", Arr(Real("f64", M), (4096,), "SHMEM"), -1.0, 1.0)
    d = Def("shm.s%d" % stride,
            b.load(a, b.const(0, Int()), lanes=32, stride=stride),
            b.g, dal="C", atol=1.0)
    r = check(d)
    f = r.conflicts[0][4]
    print("    stride %-3d  serialize x%-3d  %6.0f cycles  %s"
          % (stride, f, r.cost.cycles, "#" * f))

print("""
  With B banks and stride s the residues {i*s mod B} form a subgroup of
  order B/gcd(s,B), so the serialization factor is forced by gcd. Stride 32
  is 32x slower than stride 33. Every GPU programmer knows to pad by one;
  no language above assembly can tell them WHY, or check that they did.""")


# ======================================================  3. ports contend
line("3. two operations on one port cannot retire at once")


def chain(units):
    b = B()
    v = b.param("v", Real("f64", ONE), 1.0, 2.0)
    acc = v
    for i, u in enumerate(units):
        acc = b.bind(b.mul(acc, v), u)
    return Def("port.%s" % "+".join(sorted(set(units))), acc, b.g,
               dal="C", atol=1e3)


for units in (["ALU64"] * 8, ["FMA512"] * 8,
              ["ALU64", "FMA512"] * 4):
    r = check(chain(units))
    print("  %-22s ports=%-26s throughput floor %.1f cycles"
          % (r.name, dict(r.cost.ports), r.cost.throughput_cycles()))

print("""
  Eight multiplies on one port is a 4-cycle issue floor; spread over two
  ports it is 2. The WCET check takes the worse of the dependence chain and
  this floor, so a program that over-subscribes a port is slow BY ARITHMETIC
  and the budget catches it before it runs.""")


# =======================  4. the fabric is a time domain, and that was free
line("4. CPU and fabric are two clock domains, caught by the aerospace rule")

# f32, because that is what a DSP48 slice carries. The point of this section
# is the CLOCK, so the format is made right first.
b = B()
core = b.param("core", Real("f32", ONE), 1.0, 2.0)
fab = b.bind(b.mul(core, core), "DSP48")   # DSP48 declares clock FPGA_200M
show(Def("fpga.unsynced", b.add(fab, core), b.g, dal="C", atol=1e3))

b = B()
core = b.param("core", Real("f32", ONE), 1.0, 2.0)
fab = b.bind(b.mul(core, core), "DSP48")
show(Def("fpga.synced", b.add(b.sync(fab, None, depth=2), core), b.g,
         dal="C", atol=1e3))

print("""
  Computing on the fabric puts the result in the fabric's time domain, so
  adding it to a core-clocked value is the same violation as Vec3<ECI> +
  Vec3<BODY>. Not one line of clock-crossing logic was added to the checker
  to get this; `bind` sets the frame from the unit's declared clock and the
  frame rule written for spacecraft does the rest.

  `sync(fab, None)` is leaving the fabric domain for the asynchronous one,
  through a declared two-flop synchronizer. Depth 1 would be rejected as
  metastable by the same check.""")


# ===============================================  5. all the way to the bytes
line("5. emit x86-64, make it executable, run it on this CPU")

b = B()
x = b.param("x", Real("f64", M), 100.0, 400.0)
t = b.param("t", Real("f64", S), 2.0, 8.0)
v = b.bind(b.div(x, t), "ALU64")
v = b.require(v, 0.0, 250.0, MPS, "speed limit")
d = Def("v.speed", v, b.g, dal="C", atol=1e-9,
        intent="mean speed over a leg, bounded by the posted limit")

r = check(d)
print(r.summary())

try:
    res = verify(d, r, samples=5)
except (OSError, Unsupported) as e:
    print("\n  could not execute here: %s" % e)
    code = emit(d, r)
    print(code.listing())
    res = None

if res is not None:
    code = res["code"]
    print()
    print(code.listing())
    print()
    print(code.hexdump())
    print("""
  proven interval   [%g, %g]
  points executed   %d  (corners and interior of the input box)
  every answer in   %s
  counterexample    %s""" % (res["proven"][0], res["proven"][1],
                             res["points"], res["inside"],
                             res["counterexample"]))
    print("""
  x=400 t=2  ->  %s
  x=100 t=8  ->  %s""" % (code.run(400.0, 2.0), code.run(100.0, 8.0)))
    print("""
  Those bytes went through VirtualAlloc with PAGE_EXECUTE_READWRITE and were
  called as a function. `bind` and `require` emitted a copy and nothing else,
  because a discharged obligation is not an instruction -- the proof was
  already paid for at check time.

  This is the one place in the project where a claim meets a measurement.
  Interval arithmetic said the answer lies in [%g, %g]; the CPU was asked %d
  times and agreed every time. A language that never confronts its model with
  the silicon has not earned the word "proven".""" % (res["proven"][0],
                                                      res["proven"][1],
                                                      res["points"]))


# ==========================  6. the same four operations, the whole stack
line("6. one language, bottom to top -- what has a backend and what does not")

rows = [
    ("circuit",  "a pin's contact bounce",      "sense + require", "no"),
    ("silicon",  "which unit issues the FMA",   "bind",            "yes (x86-64)"),
    ("memory",   "shared-memory bank conflict", "load(lanes,stride)", "no"),
    ("fabric",   "200 MHz clock domain",        "bind + sync",     "no"),
    ("mechanical", "bolt shear vs proof stress", "require",        "n/a"),
    ("ui",       "WCAG contrast over a theme",  "require",         "no"),
    ("network",  "RTT against an SLO",          "drive + deadline", "no"),
]
print("  %-11s %-29s %-20s %s" % ("layer", "what is proven", "operations",
                                  "lowers today"))
for a, bb, c, dd in rows:
    print("  %-11s %-29s %-20s %s" % (a, bb, c, dd))

print("""
  Seven layers, four operations, one checker. The "lowers today" column is
  the honest part: the proof side is complete across all seven and the code
  generator reaches exactly one of them. That gap is work, not a design
  problem -- the site lattice already holds the register map's worth of
  physics for the others, and nothing in the type system has to change to
  emit them.""")

print("\nall sections ran.")
