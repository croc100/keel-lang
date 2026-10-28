"""
Confronting the site registry with the silicon.

KEEL's strongest claim is that obligations come from declared physics. Its
weakest point is the word DECLARED: `lat`, `thru`, `hz` and `lanes` are typed
in by a human reading a datasheet, and a misread datasheet means proving the
wrong thing perfectly. Nothing in the checker can notice, because the checker
is reasoning correctly from what it was told.

This module is the answer to that, and it is deliberately not in the trust
root. It builds programs whose runtime is dominated by ONE declared number,
runs them on this machine, and reports where the measurement disagrees with
the declaration. A disagreement is not an error in the checker. It is a bug
report against the registry, which is the only place such a bug can live.

Three numbers are testable this way:

  lat    a chain of N operations where each depends on the last cannot run
         faster than N x lat cycles. Measure it and lat falls out.
  thru   the same N operations spread over independent chains are limited by
         issue rate instead, so the ratio of the two measurements is the
         ratio the site declares between its latency and its throughput.
  lanes  CPUID says what the machine has. If a site claims more, a program
         bound to it is proven against a unit this machine does not contain.

What is NOT testable here: `settle`, `prop`, `pj_byte` and the mechanical and
optical sites. Those need an instrument, not a program, and this file says so
rather than inventing a number.
"""

import time

from .backend import Unsupported, cpu, emit
from .check import check
from .cost import M0
from .build import B
from .ir import Def
from .types import ONE, SITES, Real


def _chain(unit, n, ways=1, op="mul"):
    """`ways` independent chains of `n` dependent operations, summed.

    ways=1 is latency-bound: every operation waits for the one before it.
    ways>1 is issue-bound once there is more parallelism than the unit's
    latency can absorb. The same DAG measures both numbers.
    """
    b = B()
    one = b.const(1.0, Real("f64", ONE))
    ends = []
    for w in range(ways):
        v = b.param("x%d" % w, Real("f64", ONE), 1.0, 1.0)
        for _ in range(n):
            if op == "mul":
                step = b.mul(v, one)
            elif op == "add":
                step = b.add(v, one)
            elif op == "div":
                step = b.div(v, one)
            elif op == "sqrt":
                step = b.sqrt(v)
            elif op == "fma":
                step = b.fma(v, one, one)
            else:
                raise ValueError(op)
            v = b.bind(step, unit) if unit else step
        ends.append(v)
    acc = ends[0]
    for e in ends[1:]:
        acc = b.add(acc, e)
    return Def("cal.%s.%dx%d" % (unit or "none", ways, n), acc, b.g,
               dal="E", atol=1e12)


def _seconds(code, args, reps, rounds=9):
    """Minimum over rounds: the least-disturbed sample, not the average. An
    average measures the operating system; a minimum measures the machine."""
    code.run(*args)
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            code.run(*args)
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


def _per_op(unit, n, ways, reps, op="mul"):
    """Time two chain lengths and divide the difference by the difference.

    No counted loop: the reorder buffer spans a backward jump, and because
    each iteration reloads its input from memory the iterations are
    independent, so a short chain in a loop measures roughly half the real
    latency. One long chain has nothing to overlap with.

    The difference between n and n//2 cancels the call overhead exactly,
    which is better than measuring the overhead separately and hoping it was
    the same overhead.
    """
    out = []
    for length in (n, n // 2):
        d = _chain(unit, length, ways, op=op)
        r = check(d)
        if not r.ok:
            raise Unsupported("calibration program rejected: %s"
                              % r.violations[0].msg)
        out.append((emit(d, r), ways * length, r))
    (c1, ops1, rep), (c2, ops2, _) = out
    args = [1.0] * ways
    t1 = _seconds(c1, args, reps)
    t2 = _seconds(c2, args, reps)
    return max(t1 - t2, 0.0) / (ops1 - ops2), c1, rep


# What the machine model declares each operation's latency to be. These are
# the numbers under test.
DECLARED = {"mul": "cycles_per_flop", "add": "cycles_per_flop",
            "fma": "cycles_per_flop", "div": "cycles_div",
            "sqrt": "cycles_sqrt"}


def declared_cycles(op, site=None):
    if site is not None and site.lat and op in ("mul", "add", "fma"):
        return site.lat
    return getattr(M0, DECLARED[op])


def calibrate(unit, n=4000, ways=1, reps=40, op="mul"):
    """Measure one site and compare with what it declares."""
    s = SITES.get(unit) if unit else None
    per, code, rep = _per_op(unit, n, ways, reps, op=op)
    hz = (s.hz if s and s.hz else M0.hz)
    lat = declared_cycles(op, s)
    out = {
        "unit": unit or "(unbound)",
        "op": op,
        "ways": ways,
        "s_per_op": per,
        "declared_cycles": lat,
        "declared_hz": hz,
        "predicted_s": lat / hz if hz else None,
        "implied_hz": (lat / per) if per > 0 else None,
        "bytes": len(code),
    }
    if out["predicted_s"]:
        out["ratio"] = per / out["predicted_s"]
    return out


def lanes_disagreement():
    """CPUID against the registry. A site that claims more lanes than the
    machine has is a site this machine cannot host, and a program bound to it
    is proven against a unit that is not there."""
    f = cpu()
    rows = []
    for name, s in SITES.items():
        if not s.arith or not s.lanes or not s.kind.startswith("x86"):
            continue
        have = f["lanes"]
        rows.append((name, s.lanes, have, s.lanes <= have))
    return f, rows


def report(out=None):
    """Print the whole confrontation. Returns the list of disagreements, so a
    test can fail on them rather than a human having to read a table."""
    import sys
    w = out or sys.stdout
    f, rows = lanes_disagreement()
    bad = []

    w.write("machine: %s\n" % f["brand"])
    w.write("  avx=%s fma=%s avx512f=%s -> %d f64 lanes per instruction\n\n"
            % (f["avx"], f["fma"], f["avx512f"], f["lanes"]))

    w.write("declared lanes against CPUID\n")
    for name, want, have, ok in rows:
        w.write("  %-8s declares %d lane(s), machine offers %d   %s\n"
                % (name, want, have, "ok" if ok else "OVER-DECLARED"))
        if not ok:
            bad.append("%s declares %d lanes; this machine has %d"
                       % (name, want, have))
    w.write("\n")

    w.write("%-10s %-5s %10s %10s %8s %10s\n"
            % ("unit", "chains", "measured", "predicted", "ratio", "implied"))
    for unit in ("ALU64", "FMA512"):
        if unit not in SITES:
            continue
        for ways in (1, 6):
            try:
                r = calibrate(unit, ways=ways)
            except (Unsupported, OSError) as e:
                w.write("  %-8s %-5s  could not run: %s\n" % (unit, ways, e))
                continue
            w.write("  %-8s %-5d %8.3f ns %8.3f ns %7.2fx %7.2f GHz\n"
                    % (r["unit"], ways, r["s_per_op"] * 1e9,
                       r["predicted_s"] * 1e9, r["ratio"],
                       (r["implied_hz"] or 0) / 1e9))
            if ways == 1 and (r["ratio"] > 3.0 or r["ratio"] < 0.34):
                bad.append("%s: a dependent chain measured %.2fx its declared "
                           "latency" % (unit, r["ratio"]))

    # The machine model's clock is itself a declaration, and a dependent chain
    # of known latency measures it. This is the program weighing the model.
    try:
        r = calibrate("ALU64", ways=1)
        implied = r["implied_hz"]
        w.write("\nmachine model declares %.2f GHz; a dependent chain of "
                "%d-cycle\noperations implies %.2f GHz (%.0f%% off)\n"
                % (M0.hz / 1e9, int(r["declared_cycles"]), implied / 1e9,
                   abs(implied - M0.hz) / M0.hz * 100.0))
        if abs(implied - M0.hz) / M0.hz > 0.25:
            bad.append("machine model declares %.2f GHz, measurement implies "
                       "%.2f GHz" % (M0.hz / 1e9, implied / 1e9))
    except (Unsupported, OSError):
        pass

    # ---- the clock-free part. A ratio of two latencies does not contain the
    #      clock, so the model's PROPORTIONS can be tested on their own.
    w.write("\nlatency ratios, which contain no clock at all\n")
    w.write("  %-6s %10s %10s %9s %9s\n"
            % ("op", "measured", "declared", "meas/mul", "decl/mul"))
    base_m = base_d = None
    rows2 = []
    for op in ("mul", "add", "fma", "div", "sqrt"):
        try:
            r = calibrate(None, n=2000, op=op)
        except (Unsupported, OSError) as e:
            w.write("  %-6s could not run: %s\n" % (op, e))
            continue
        if base_m is None:
            base_m, base_d = r["s_per_op"], r["declared_cycles"]
        rows2.append((op, r["s_per_op"], r["declared_cycles"],
                      r["s_per_op"] / base_m, r["declared_cycles"] / base_d))
    for op, ms, dc, mr, dr in rows2:
        flag = "" if abs(mr - dr) <= 0.35 * max(dr, 1.0) else "   <-- wrong"
        w.write("  %-6s %7.3f ns %7.1f cyc %8.2fx %8.2fx%s\n"
                % (op, ms * 1e9, dc, mr, dr, flag))
        if flag:
            bad.append("%s: declared %.2fx a multiply, measured %.2fx"
                       % (op, dr, mr))
    w.write("\n  A ratio that is right while the absolute numbers are wrong\n"
            "  means the clock is misdeclared. A ratio that is wrong means\n"
            "  the relative latencies are. They are different bugs.\n")

    w.write("\nnot measurable by a program: settle, prop, pj_byte, and every\n"
            "mechanical and optical site. Those need an instrument.\n")
    if bad:
        w.write("\n%d disagreement(s):\n" % len(bad))
        for s in bad:
            w.write("  - %s\n" % s)
    else:
        w.write("\nthe registry and the silicon agree within tolerance.\n")
    return bad


if __name__ == "__main__":
    report()
