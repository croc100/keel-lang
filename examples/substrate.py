"""
One language across every substrate, and no partition by engineering discipline.

The question this file answers: a mechanical bolt, a PCB net, a UI button, a
backend socket and a DRAM bank get sliced into separate languages, separate
toolchains and separate job titles -- but is that division IN THE PHYSICS, or
is it an artifact of how humans divide labour?

It is an artifact. Nothing below is a plugin, a plane, a dialect or a mode.
Five cases in five disciplines, built from the same four operations:

    sense(site, after)   read a value off a substrate
    drive(value, site)   commit a value to a substrate
    sync(value, clock)   cross a time domain
    require(v, lo, hi)   prove a quantity stays inside a physical bound

and checked by the same checker, which contains no vocabulary for any of the
five disciplines. A switch's contact bounce, a flip-flop's setup window and an
LCD panel's response time are one obligation with three numbers in it.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keel.check import check                                      # noqa: E402
from keel.build import B                                         # noqa: E402
from keel.ir import Def                                           # noqa: E402
from keel.types import (CD_M2, M2, MM, MPA, NEWTON, ONE, PA, PSEC,  # noqa: E402
                        Real, SITES, Site, Unit, site)

BAR = "=" * 78


def banner(t):
    print("\n" + BAR + "\n" + t + "\n" + BAR)


def show(d, note=None):
    rep = check(d)
    print(rep.summary())
    if note:
        print("    " + note)
    return rep


# =========================================================== the substrates
# Every site is a claim about the physical world, stated once, in SI, with the
# same five fields whether it is a relay coil or a cache line.

site(Site("SW1", settle=5.0e-3, readback=True, clock=None,
          load_a=1.0e-5, kind="spdt toggle switch, 5 ms contact bounce"))
site(Site("GPIO_D7", prop=2.0e-8, drive_a=20.0e-3, readback=True,
          clock="CLK_1G", kind="MCU push-pull pin, 20 mA absolute max"))
site(Site("Q1_C", prop=1.0e-7, drive_a=500.0e-3, readback=False,
          clock=None, kind="NPN collector, 500 mA"))
site(Site("K1_COIL", settle=1.0e-2, drive_a=0.0, load_a=70.0e-3,
          readback=False, clock=None, kind="relay coil, 70 mA, 10 ms armature"))
site(Site("XTAL_25M", prop=1.0e-9, readback=True, clock="CLK_25M",
          kind="25 MHz oscillator domain"))
site(Site("BTN", settle=1.2e-2, readback=True, clock=None,
          load_a=1.0e-5, kind="touch button, 12 ms debounce+controller"))
site(Site("PANEL", settle=1.6e-2, prop=8.3e-3, pj_byte=0.05, drive_a=1.0,
          readback=False, clock=None, kind="60 Hz LCD, 8 ms response"))
site(Site("SOCK", prop=3.0e-2, pj_byte=500.0, readback=True, clock=None,
          kind="TCP socket, 30 ms p99 round trip"))
site(Site("STRUT", settle=0.0, prop=0.0, readback=False, clock=None,
          kind="load-bearing member; a value here is a stress, not a signal"))

PS_PER_MM = Unit((PSEC / MM).dim, PSEC.scale / MM.scale, "ps/mm")


# ============================================= 1. a switch. the simplest thing
def switch(after_s):
    """A toggle switch. There is no software here at all, and the language
    still has something to prove: contacts bounce for 5 ms, so a read taken
    inside that window is not a value, it is noise."""
    b = B()
    s = b.sense("SW1", Real("f64", ONE), after=after_s, lo=0.0, hi=1.0)
    return Def("switch.read@%gms" % (after_s * 1e3), s, b.g,
               intent="Latch the operator's arm/safe toggle once the contacts "
                      "have stopped bouncing.",
               dal="C", deadline_s=0.05)


# ============================================ 2. a circuit board. two failures
def board(variant):
    """Setup/hold on a 1 GHz net, and a GPIO asked to pull a relay coil.

    Both are the same shape as a thrust margin: a physical quantity with a
    unit, an interval proven from declared ranges, and a bound it must respect.
    """
    b = B()
    period = b.const(1000.0, Real("f64", PSEC))          # 1 GHz
    tco = b.const(400.0, Real("f64", PSEC))              # driver output delay
    tsu = b.const(200.0, Real("f64", PSEC))              # receiver setup window
    vel = b.const(6.7, Real("f64", PS_PER_MM))           # FR-4 stripline

    # The router has not placed this net yet, so its length is a RANGE.
    hi_mm = 50.0 if variant == "routed" else 120.0
    length = b.param("net_len", Real("f64", MM), 30.0, hi_mm)
    tprop = b.mul(length, vel)

    slack = b.sub(b.sub(b.sub(period, tco), tprop), tsu)
    slack = b.require(slack, 0.0, None, PSEC, "setup/hold slack")

    # The same definition also drives a relay. One of these two sites can do
    # it and the other cannot, and the difference is 50 mA.
    coil_site = "GPIO_D7" if variant == "direct" else "Q1_C"
    cmd = b.const(1.0, Real("f64", ONE))
    out = b.drive(cmd, coil_site, effect="pin:" + coil_site,
                  loads=1, load_a=70.0e-3)

    root = b.add(slack, b.mul(out, b.const(0.0, Real("f64", PSEC))))
    return Def("board.timing." + variant, root, b.g,
               intent="Capture the ADC data net at 1 GHz and energise the "
                      "safety interlock relay.",
               dal="C", ceiling=frozenset(["pin:" + coil_site]),
               deadline_s=0.05)


def board_clock_bug():
    """A 25 MHz signal added to a 1 GHz one. This is metastability, and it is
    caught by the check written for ECI-plus-BODY vectors, unchanged."""
    b = B()
    slow = b.sense("XTAL_25M", Real("f64", ONE, "CLK_25M"), after=1e-6,
                   lo=0.0, hi=1.0)
    fast = b.sense("GPIO_D7", Real("f64", ONE, "CLK_1G"), after=1e-6,
                   lo=0.0, hi=1.0)
    return Def("board.cdc.unsynchronised", b.add(slow, fast), b.g,
               intent="Sample the external strobe in the core clock domain.",
               dal="C", deadline_s=0.05)


def board_clock_fixed():
    b = B()
    slow = b.sense("XTAL_25M", Real("f64", ONE, "CLK_25M"), after=1e-6,
                   lo=0.0, hi=1.0)
    fast = b.sense("GPIO_D7", Real("f64", ONE, "CLK_1G"), after=1e-6,
                   lo=0.0, hi=1.0)
    return Def("board.cdc.synchronised",
               b.add(b.sync(slow, "CLK_1G", depth=2), fast), b.g,
               intent="Sample the external strobe through a two-flop "
                      "synchroniser.",
               dal="C", deadline_s=0.05)


# ================================================ 3. UI/UX. two human physics
def ui(variant):
    """A button, a repaint, and a contrast ratio.

    Both UI rules here are usually enforced by a linter that looks at one pair
    of colours and a stopwatch that measures one run. Here the contrast is
    proven across the whole declared RANGE of a themeable background, and the
    latency is proven from the panel's own response time.
    """
    b = B()
    # Luminances in cd/m^2, so the ratio is dimensionless by construction.
    fg = b.param("fg_luminance", Real("f64", CD_M2), 90.0, 110.0)
    bg_hi = 22.0 if variant == "theme_wide" else 20.0
    bg = b.param("bg_luminance", Real("f64", CD_M2), 18.0, bg_hi)
    ratio = b.div(fg, bg)
    ratio = b.require(ratio, 4.5, None, ONE, "WCAG 1.4.3 contrast ratio")

    press = b.sense("BTN", Real("f64", ONE), after=1.5e-2, lo=0.0, hi=1.0)
    frame = b.mul(ratio, press)
    hops = 4 if variant == "remote_render" else 0
    for i in range(hops):
        frame = b.drive(frame, "SOCK", effect="net:render%d" % i)
    root = b.drive(frame, "PANEL", effect="display:main")

    eff = ["display:main"] + ["net:render%d" % i for i in range(hops)]
    return Def("ui.button_feedback." + variant, root, b.g,
               intent="Acknowledge a button press on screen within one "
                      "perceptual interval, legibly.",
               dal="C", ceiling=frozenset(eff), deadline_s=0.1)


# ======================================= 4. mechanical. the same require again
def bolt(variant):
    """An M6 bolt in shear. No clock, no signal, no software in the loop --
    and the obligation is character for character the one the PCB net used."""
    b = B()
    n = 1 if variant in ("single", "unit_bug") else 2
    area = b.const(n * 2.01e-5, Real("f64", M2))
    force = b.param("shear_load", Real("f64", NEWTON), 0.0, 8000.0)
    stress = b.div(force, area)
    # Grade 8.8: 640 MPa proof stress, safety factor 2 -> 320 MPa allowable.
    if variant == "unit_bug":
        # The bound is written in MPa and the value is in Pa. Dimensionally
        # identical. This is the Mars Climate Orbiter, in a bolt.
        stress = b.require(stress, 0.0, 320.0, MPA, "shear stress vs 8.8 proof")
    else:
        stress = b.require(stress, 0.0, 3.2e8, PA, "shear stress vs 8.8 proof")
    # A load-bearing member is a site like any other. It draws no current, so
    # it owes no current proof -- the obligations a substrate carries come from
    # the physics it declares, not from a fixed checklist.
    stress = b.drive(stress, "STRUT", effect="struct:bracket")
    return Def("strut.bolt." + variant, stress, b.g,
               intent="Keep the actuator bracket bolts below half the 8.8 "
                      "proof stress at limit load.",
               dal="B", atol=1e-3, deadline_s=1.0,
               ceiling=frozenset(["struct:bracket"]))


# ============================================== 5. a backend request, same law
def server(hops):
    b = B()
    v = b.sense("SOCK", Real("f64", ONE), after=0.0, lo=0.0, hi=1.0)
    for i in range(hops):
        v = b.drive(v, "SOCK", effect="net:hop%d" % i)
    return Def("api.request.%dhop" % hops, v, b.g,
               intent="Serve one API request inside the 200 ms SLO.",
               dal="D",
               ceiling=frozenset("net:hop%d" % i for i in range(hops)),
               deadline_s=0.2)


# ================================================================== run it all
banner("1  A SWITCH -- 5 ms of contact bounce is a proof obligation")
show(switch(1.0e-3))
show(switch(2.0e-2), "the rejection above named the wait; this one took it")

banner("2  A CIRCUIT BOARD -- setup/hold slack, and 70 mA off a 20 mA pin")
r = check(board("direct"))
print(r.summary())
print("    both failures are check 4 and check 11 over amperes and picoseconds")
print()
show(board("routed"), "net length capped at 50 mm, relay moved behind Q1")

banner("2b CLOCK DOMAINS ARE REFERENCE FRAMES")
show(board_clock_bug())
print()
show(board_clock_fixed(), "same rule that rejects ECI + BODY, same fix shape")

banner("3  UI/UX -- contrast over a RANGE, latency from the panel's physics")
show(ui("theme_wide"))
print()
show(ui("remote_render"), "contrast fine; four network hops miss the "
                          "perceptual budget")
print()
show(ui("ok"), "background range narrowed to [18,20] cd/m^2, rendered locally")

banner("4  MECHANICAL -- one bolt, two bolts, and a unit written in MPa")
show(bolt("single"))
print()
show(bolt("unit_bug"))
print()
show(bolt("double"), "320 MPa allowable held across the whole load range")

banner("5  A BACKEND REQUEST -- the same deadline arithmetic")
show(server(3))
print()
show(server(8))

# ============================================================ the actual claim
banner("DOES THE CHECKER KNOW ABOUT ANY OF THESE DISCIPLINES?")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS = ("ui", "ux", "gui", "widget", "pixel", "css", "shell", "gpio", "pcb",
         "trace", "relay", "bolt", "server", "socket", "http", "firmware",
         "embedded", "mechanical", "circuit", "frontend", "backend", "kernel",
         "driver", "button", "display", "panel", "switch", "spacecraft")
checker = io.open(os.path.join(ROOT, "keel/check.py"), encoding="utf-8").read()
code = "\n".join(ln for ln in checker.splitlines()
                 if not ln.strip().startswith("#"))
code = code.split('"""')[2]          # drop the module docstring prose
hits = sorted(w for w in WORDS
              if any(w == t.strip(" .,()[]{}:;%\"'<>-_=")
                     for t in code.lower().replace("\n", " ").split()))
print("discipline words in the executable part of the trust root: %s"
      % (", ".join(hits) if hits else "none"))

ops = {}
for d in (switch(2e-2), board("routed"), board_clock_fixed(), ui("ok"),
          bolt("double"), server(3)):
    for a in d.graph.topo(d.root):
        ops[d.graph[a].op] = ops.get(d.graph[a].op, 0) + 1
print("operations used across all five disciplines: %d distinct" % len(ops))
print("  " + "  ".join("%s=%d" % kv for kv in sorted(ops.items())))
print()
print("sites registered, each one just five numbers and a time domain:")
for k in sorted(SITES):
    s = SITES[k]
    print("  %-9s settle=%-9.4g prop=%-9.4g clock=%-8s %s"
          % (s.name, s.settle, s.prop, s.clock or "async", s.kind))
