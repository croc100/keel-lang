"""
The KEEL command line.

    keel check   FILE          check every definition in a module
    keel explain FILE [NAME]   print the non-authoritative human view
    keel addr    FILE          addresses and seals
    keel fmt     FILE          rewrite in canonical glyph form
    keel emit    FILE [NAME]   x86-64 listing and hexdump
    keel run     FILE [NAME] ARG...   execute on this CPU
    keel verify  FILE [NAME]   run it and confront the proof with the silicon
    keel cpu                   what this machine actually is
    keel calibrate             the site registry against the silicon
    keel selftest              the whole test suite

A module is a .kl file: a KEEL-GLYPH stream. Programs are authored with the
builder API and written out with `keel.surface.emit`; see examples/.
"""

import sys

from . import surface
from .backend import Unsupported, cpu, emit as x86_emit, isa_of, verify
from .explain import explain


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _pick(defs, reports, name):
    if name is None:
        return len(defs) - 1
    for i, d in enumerate(defs):
        if d.name == name:
            return i
    sys.stderr.write("no definition named %r; have %s\n"
                     % (name, ", ".join(d.name for d in defs)))
    raise SystemExit(2)


def cmd_check(args):
    defs, funcs, reports = surface.load(_read(args[0]))
    worst = 0
    for d, r in zip(defs, reports):
        print(r.summary())
        if not r.ok:
            worst = 1
            for v in r.violations:
                print("    %-11s %s" % (v.kind, v.msg))
        else:
            c = r.cost
            print("    %r  %g cycles  %.3g s  %g pJ  effects %s"
                  % (r.iv[d.root], max(c.cycles, c.throughput_cycles()),
                     c.seconds, c.energy_pj(), sorted(r.effects) or ["none"]))
    return worst


def cmd_explain(args):
    defs, funcs, reports = surface.load(_read(args[0]))
    i = _pick(defs, reports, args[1] if len(args) > 1 else None)
    print(explain(defs[i], reports[i]))
    return 0


def cmd_addr(args):
    defs, funcs, reports = surface.load(_read(args[0]))
    for d in defs:
        print("%s  %-16s  %d node(s)"
              % (d.graph[d.root].addr, d.name, len(d.graph.topo(d.root))))
    return 0


def cmd_fmt(args):
    defs, funcs, reports = surface.load(_read(args[0]))
    out = surface.emit(defs)
    if len(args) > 1:
        with open(args[1], "w", encoding="utf-8") as fh:
            fh.write(out + "\n")
    else:
        print(out)
    return 0


def cmd_emit(args):
    defs, funcs, reports = surface.load(_read(args[0]))
    i = _pick(defs, reports, args[1] if len(args) > 1 else None)
    if not reports[i].ok:
        sys.stderr.write("refusing to emit a rejected definition\n")
        return 1
    code = x86_emit(defs[i], reports[i])
    print(code.listing())
    print()
    print(code.hexdump())
    return 0


def cmd_run(args):
    defs, funcs, reports = surface.load(_read(args[0]))
    rest = args[1:]
    name = None
    if rest and not _isnum(rest[0]):
        name, rest = rest[0], rest[1:]
    i = _pick(defs, reports, name)
    if not reports[i].ok:
        sys.stderr.write("refusing to run a rejected definition\n")
        return 1
    code = x86_emit(defs[i], reports[i])
    vals = [float(x) for x in rest]
    if len(vals) != code.nargs:
        sys.stderr.write("%s takes %d double(s), %d given\n"
                         % (code.name, code.nargs, len(vals)))
        return 2
    out = code.run(*vals)
    print(out)
    iv = reports[i].iv[defs[i].root]
    vs = out if isinstance(out, tuple) else (out,)
    if not all(iv.lo <= v <= iv.hi for v in vs):
        sys.stderr.write("THE ANSWER IS OUTSIDE THE PROVEN INTERVAL %r\n" % iv)
        return 3
    return 0


def _isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def cmd_verify(args):
    defs, funcs, reports = surface.load(_read(args[0]))
    i = _pick(defs, reports, args[1] if len(args) > 1 else None)
    if not reports[i].ok:
        sys.stderr.write("refusing to verify a rejected definition\n")
        return 1
    try:
        res = verify(defs[i], reports[i], samples=5, time_it=True)
    except (Unsupported, OSError) as e:
        sys.stderr.write("cannot execute here: %s\n" % e)
        return 1
    print("%s: %d bytes, isa=%s" % (res["name"], res["bytes"], res["isa"]))
    print("  proven interval   %r" % (res["proven"],))
    print("  points executed   %d" % res["points"])
    print("  every answer in   %s" % res["inside"])
    print("  counterexample    %r" % (res["counterexample"],))
    print("  throughput        %.2f ns/execution in a stream"
          % (res["throughput_s"] * 1e9))
    print("  WCET claim        %.2f ns for one execution"
          % (res["claimed_s"] * 1e9))
    print("  consistent        %s   (%s)"
          % (res["wcet_consistent"], res["kind"]))
    return 0 if res["inside"] else 3


def cmd_cpu(args):
    f = cpu()
    print("brand    %s" % f["brand"])
    print("avx      %s" % f["avx"])
    print("fma      %s" % f["fma"])
    print("avx512f  %s" % f["avx512f"])
    print("f64 lanes per instruction  %d" % f["lanes"])
    print("isa in use               %s"
          % ("+".join(sorted(isa_of())) or "sse2"))
    return 0


def cmd_calibrate(args):
    from .calibrate import report
    return 1 if report() else 0


def cmd_selftest(args):
    import os
    import subprocess
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rc = 0
    for name in ("test_keel.py", "test_lang.py"):
        rc |= subprocess.call(
            [sys.executable, os.path.join(here, "tests", name)], cwd=here)
    return rc


CMDS = {
    "check": cmd_check, "explain": cmd_explain, "addr": cmd_addr,
    "fmt": cmd_fmt, "emit": cmd_emit, "run": cmd_run, "verify": cmd_verify,
    "cpu": cmd_cpu, "calibrate": cmd_calibrate, "selftest": cmd_selftest,
}


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__.strip())
        return 0
    fn = CMDS.get(argv[0])
    if fn is None:
        sys.stderr.write("unknown command %r; try --help\n" % argv[0])
        return 2
    need = argv[0] not in ("cpu", "calibrate", "selftest")
    if need and len(argv) < 2:
        sys.stderr.write("%s needs a FILE\n" % argv[0])
        return 2
    try:
        return fn(argv[1:])
    except surface.SurfaceError as e:
        sys.stderr.write("surface: %s\n" % e)
        return 2
    except FileNotFoundError as e:
        sys.stderr.write("%s\n" % e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
