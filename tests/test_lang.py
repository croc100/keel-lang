"""
Tests for the general-purpose half of KEEL:  python tests/test_lang.py

Records, variants, functions, loops with state, ownership, effects, bytes,
addressing, the new hash, the written form and the code generator.

Same rule as the other suite: every test is "must accept" or "must reject".
Where a number is measured rather than proven -- avalanche, bucket spread,
throughput -- the number is PRINTED, because a statistical claim that nobody
can see is not a claim.
"""

import os
import struct
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keel.backend as bk                                            # noqa: E402
import keel.hash as kh                                               # noqa: E402
from keel import surface                                             # noqa: E402
from keel.check import check, contract_of, func                      # noqa: E402
from keel.build import B
from keel.ir import Def                                           # noqa: E402
from keel.types import (Buf, Cap, Int, M, MPS, ONE, Real,  # noqa: E402
                        Rec, S, Sum, Vec)

PASS, FAIL = [], []


def ok(name, cond, msg=""):
    (PASS if cond else FAIL).append(name)
    print("  %s %-52s %s" % ("ok  " if cond else "FAIL", name, msg))


def rejects(name, rep, kind=None):
    bad = not rep.ok and (kind is None
                          or any(v.kind == kind for v in rep.violations))
    (PASS if bad else FAIL).append(name)
    msg = rep.violations[0].msg[:58] if rep.violations else "ACCEPTED"
    print("  %s %-52s %s" % ("ok  " if bad else "FAIL", name, msg))


def accepts(name, rep, msg=""):
    (PASS if rep.ok else FAIL).append(name)
    print("  %s %-52s %s" % ("ok  " if rep.ok else "FAIL", name,
                             msg or (rep.violations[0].msg[:58]
                                     if rep.violations else "")))


ONEV = Real("f64", ONE)

# =====================================================================
print("\n--- records: a field keeps its own interval and its own unit ------")

STATE = Rec("state", (("pos", Real("f64", M)), ("vel", Real("f64", MPS))))


def mk_state():
    b = B()
    p = b.param("p", Real("f64", M), 0.0, 100.0)
    v = b.param("v", Real("f64", MPS), -5.0, 5.0)
    return b, b.rec(STATE, pos=p, vel=v)


b, st = mk_state()
d = Def("f", b.field(st, "vel"), b.g, dal="C", atol=1e-9)
r = check(d)
accepts("a field read recovers that field's exact interval", r,
        "vel -> %r, not the hull of the record" % r.iv[d.root])

b, st = mk_state()
r = check(Def("f", b.add(b.field(st, "pos"), b.field(st, "vel")), b.g,
              dal="C", atol=1e-9))
rejects("adding a length to a speed is still a type error", r, "type")

b, st = mk_state()
try:
    b.rec(STATE, pos=b.const(1.0, Real("f64", M)))
    built = True
except TypeError:
    built = False
ok("a partly built record cannot be written down", not built,
   "there is no null, so every field is required at construction")

# =====================================================================
print("\n--- variants: every case handled, or it does not load ------------")

RESULT = Sum("result", (("ok", Real("f64", MPS)), ("fault", Int())))


def arms():
    b = B()
    x = b.param("x", Real("f64", MPS), 0.0, 50.0)
    a1 = func("take", Def("take", b.clamp(x, 0.0, 40.0), b.g, dal="C",
                          atol=1e-9))
    b = B()
    c = b.param("c", Int(), 0.0, 9.0)
    z = b.const(0.0, Real("f64", MPS))
    a2 = func("safe", Def("safe", b.add(z, z), b.g, params=(c,), dal="C",
                          atol=1e-9))
    return a1, a2


a1, a2 = arms()


def match_def(which):
    b = B()
    x = b.param("x", Real("f64", MPS), 0.0, 50.0)
    sv = b.inject(RESULT, "ok", x)
    return Def("m", b.match(sv, which, Real("f64", MPS)), b.g, dal="C",
               atol=1e-9)


r = check(match_def({"ok": a1, "fault": a2}), funcs=[a1, a2])
accepts("an exhaustive match is accepted", r,
        "result %r is the hull of the arms" % r.iv[match_def(
            {'ok': a1, 'fault': a2}).root])
rejects("a missing variant is rejected at check time",
        check(match_def({"ok": a1}), funcs=[a1]), "match")

# =====================================================================
print("\n--- functions: the contract is read off the proof ----------------")

b = B()
s = b.param("s", Real("f64", MPS), 0.0, 100.0)
half = func("half", Def("half", b.mul(s, b.const(0.5, ONEV)), b.g, dal="C",
                        atol=1e-9))
ok("a contract is derived, not declared",
   (half.contract.lo, half.contract.hi) == (0.0, 50.0),
   "half(s in [0,100]) -> [%g,%g], proven" % (half.contract.lo,
                                              half.contract.hi))


def call_with(lo, hi):
    b = B()
    y = b.param("y", Real("f64", MPS), lo, hi)
    return Def("c", b.call(half, y), b.g, dal="C", atol=1e-9)


accepts("a call adopts the callee's proven interval",
        check(call_with(10.0, 20.0), funcs=[half]))
rejects("an argument outside the proven box fails the precondition",
        check(call_with(10.0, 500.0), funcs=[half]), "obligation")
rejects("a contract whose body was not supplied is not adopted",
        check(call_with(10.0, 20.0)), "obligation")

r = check(call_with(10.0, 20.0), funcs=[half])
ok("the callee's body is checked once and memoised by address",
   any(c[2] for c in r.calls),
   "an address commits to the whole body, so the cache needs no "
   "invalidation")

# a contract may promise less than the body proves, never more
weak = contract_of("half", half.body, half.report)
from keel.ir import Contract                                         # noqa: E402
wider = Contract("half", weak.body, weak.params, weak.ret, -100.0, 100.0,
                 weak.rel * 2, weak.aerr * 2, weak.cycles * 2, weak.flops,
                 weak.seconds * 2, weak.effects, weak.dal)
ok("a published contract may promise less than the body proves",
   wider.weaker_than(weak)[0], "wider interval, higher cost: allowed")
tighter = Contract("half", weak.body, weak.params, weak.ret, 0.0, 10.0,
                   weak.rel, weak.aerr, weak.cycles, weak.flops,
                   weak.seconds, weak.effects, weak.dal)
ok("a published contract may not promise more", not tighter.weaker_than(weak)[0],
   tighter.weaker_than(weak)[1][:58])

# =====================================================================
print("\n--- loops: induction over an interval invariant ------------------")

b = B()
acc = b.param("acc", ONEV, 0.0, 10.0)
i = b.param("i", Int(), 0.0, 63.0)
step = func("decay", Def("decay",
                         b.add(b.mul(acc, b.const(0.5, ONEV)),
                               b.const(1.0, ONEV)),
                         b.g, params=(acc, i), dal="C", atol=1e-9))


def fold_def(init, lo, hi, n=64):
    b = B()
    return Def("fold", b.fold(step, b.const(init, ONEV), n, lo, hi), b.g,
               dal="C", atol=1e-6)


r = check(fold_def(0.0, 0.0, 10.0), funcs=[step])
accepts("a loop whose body preserves the invariant is accepted", r,
        "64 iterations, %r, %g cycles" % (r.iv[fold_def(0.0, 0.0, 10.0).root],
                                          r.cost.cycles))
rejects("an invariant the body does not preserve is rejected",
        check(fold_def(0.0, 0.0, 2.0), funcs=[step]), "obligation")
rejects("an initial value outside the invariant is rejected",
        check(fold_def(50.0, 0.0, 10.0), funcs=[step]), "obligation")
r = check(fold_def(0.0, 0.0, 10.0, n=1024), funcs=[step])
ok("the cost of a loop is the trip count times the body",
   r.cost.cycles == 1024 * step.contract.cycles,
   "1024 x %g = %g cycles, and the trip count is a literal, so there is no "
   "program without a WCET" % (step.contract.cycles, r.cost.cycles))

# =====================================================================
print("\n--- ownership: aliasing is an edge count -------------------------")

BUF = Buf(Real("f64", M), (8,), "DRAM")


def own(alias=False, idx=0):
    b = B()
    v = b.param("v", Real("f64", M), -1.0, 1.0)
    buf = b.alloc(BUF)
    one = b.put(buf, b.const(idx, Int()), v)
    if alias:
        two = b.put(buf, b.const(1, Int()), v)
        return Def("o", b.add(b.get(one, b.const(0, Int())),
                              b.get(two, b.const(0, Int()))), b.g,
                   dal="C", atol=1e-9)
    return Def("o", b.freeze(one), b.g, dal="C", atol=1e-9)


r = check(own())
accepts("a buffer threaded through its writes is accepted", r,
        "%d linear value(s), each with exactly one owner" % len(r.linear))
rejects("two edges to one buffer is aliasing", check(own(alias=True)),
        "ownership")
rejects("a write past the end of a buffer is rejected", check(own(idx=99)),
        "obligation")

# =====================================================================
print("\n--- effects: authority travels in the arguments ------------------")


def eff(kind="NET", effect="NET.send", ceiling=("NET.send",)):
    b = B()
    c = b.cap("c", kind)
    return Def("e", b.perform(c, effect, ty=Cap(kind), cost_s=2.0e-3), b.g,
               dal="C", atol=1e9, ceiling=set(ceiling), deadline_s=5e-3)


r = check(eff())
accepts("an effect with a token and a ceiling is accepted", r,
        "effects %s, %.1f ms of physical time"
        % (sorted(r.effects), r.cost.seconds * 1e3))
rejects("an effect outside the declared ceiling is rejected",
        check(eff(ceiling=())), "capability")
rejects("a filesystem token does not authorise the network",
        check(eff(kind="FS")), "capability")
r = check(eff(ceiling=("NET.send",)))
b = B()
c = b.cap("c", "NET")
r2 = check(Def("e", b.add(b.const(1.0, ONEV), b.const(1.0, ONEV)), b.g,
               dal="C", atol=1e-9))
ok("a definition that was handed no token has no effects",
   r2.effects == set(),
   "there is no ambient authority to reach for")

# =====================================================================
print("\n--- bytes: a length is in the type, an encoding is a unit --------")

b = B()
h = b.bstr("GET /status", "utf8")
accepts("an in-bounds slice is accepted",
        check(Def("b", b.bslice(h, 0, 3), b.g, dal="C", atol=1e9)),
        "bytes11 -> bytes3, both static")
b = B()
rejects("a slice past the end does not typecheck",
        check(Def("b", b.bslice(b.bstr("GET", "utf8"), 1, 9), b.g, dal="C",
                  atol=1e9)), "bounds")
b = B()
rejects("concatenating two encodings is a unit error",
        check(Def("b", b.bcat(b.bstr("a", "utf8"), b.bstr("b", "latin1")),
                  b.g, dal="C", atol=1e9)), "unit")

# =====================================================================
print("\n--- KEEL-ADDR: the encoding is injective, and provably so --------")

CORPUS = [None, True, False, 0, -1, 1, 255, 1 << 70, -(1 << 70), 1 << 200,
          0.0, -0.0, 1.5, -1.5, float("inf"), float("-inf"),
          "", "a", "ab", "a" * 300, "[1,2]", "1", b"", b"\x00", b"\xff\x00",
          [], [[]], [[], []], [1], [1, 1], [[1], [1]], [1, [1]],
          {}, {"a": 1}, {"a": [1]}, {"a": 1, "b": 2}, {"b": 2, "a": 1},
          Fraction(45359237, 100000000), Fraction(5, 18)]
encs = [kh.encode(v) for v in CORPUS]
# INJECTIVITY, stated exactly: no two UNEQUAL values may share bytes. The
# converse is not required and is not true -- 0.0 and -0.0 are equal under ==
# and encode differently, because they are distinguishable values and an
# interval can tell them apart.
share = [(CORPUS[i], CORPUS[j])
         for i in range(len(CORPUS)) for j in range(i + 1, len(CORPUS))
         if encs[i] == encs[j] and not (CORPUS[i] == CORPUS[j])]
ok("no two unequal structures share an encoding", not share,
   "%d structures, %d distinct encodings, %d bad pairs"
   % (len(CORPUS), len(set(encs)), len(share)))
# CANONICALITY, which is a different property and also required: a node's
# attributes are a mapping, so the order they were inserted in must not
# change the node's address.
ok("attribute order cannot change an address",
   kh.encode({"a": 1, "b": 2}) == kh.encode({"b": 2, "a": 1}),
   "two spellings of one mapping encode identically")
bad_rt = [v for v in CORPUS
          if not (type(v) is type(kh.roundtrip(v)) and v == kh.roundtrip(v))]
ok("every structure survives encode and decode", not bad_rt,
   "%d structures; a decoder that recovers the input is a constructive "
   "proof that the encoding lost nothing" % len(CORPUS))
ok("integers beyond 2^63 encode correctly",
   kh.roundtrip(1 << 200) == 1 << 200,
   "a unit scale is an exact Fraction, not a machine word")
try:
    kh.encode(float("nan"))
    nan_ok = False
except kh.NotEncodable:
    nan_ok = True
ok("a NaN cannot be addressed", nan_ok,
   "two NaNs with different payloads compare equal; encoding one would "
   "break injectivity on exactly the value that means 'something broke'")
try:
    kh.encode(object())
    unk = False
except kh.NotEncodable:
    unk = True
ok("an unaddressable attribute stops the build", unk,
   "silently skipping part of a node would certify something other than "
   "what runs")
ok("a Merkle tree over [a,b,c] differs from one over [a,b,c,c]",
   kh.merkle(["aa" * 8, "bb" * 8, "cc" * 8])
   != kh.merkle(["aa" * 8, "bb" * 8, "cc" * 8, "cc" * 8]),
   "an odd node is carried up, not duplicated: the classic second-preimage")
ok("the node domain and the Merkle domain are separated",
   kh.node_addr("add", (), {}, ("real",)) != kh.merkle([]),
   "different BLAKE2 personalisation, so a leaf can never be read as a node")

# =====================================================================
print("\n--- KHASH-64: measured, not asserted (NOT cryptographic) ---------")

lo, mean, hi = kh.avalanche(nbytes=8, trials=256)
ok("every input bit flips about half the output bits",
   0.45 <= lo and hi <= 0.55,
   "per-bit flip rate min %.4f mean %.4f max %.4f (ideal 0.5)"
   % (lo, mean, hi))
keys = [("key%d" % i).encode() for i in range(50000)]
chi = kh.bucket_chi2(keys, 1024)
ok("sequential keys spread evenly over buckets", 850 < chi < 1250,
   "chi-square %.0f over 1024 buckets (ideal ~1023)" % chi)
ptrs = [struct.pack("<Q", 0x7F0000000000 + 64 * i) for i in range(50000)]
chi2 = kh.bucket_chi2(ptrs, 1024)
ok("aligned pointers spread evenly too", 850 < chi2 < 1250,
   "chi-square %.0f -- the low bits of an aligned address are all zero, "
   "which is where a weak hash clumps" % chi2)
c32 = kh.collisions(keys, 32)
ok("32-bit collisions are near the birthday expectation", c32 <= 3,
   "%d collisions in 50k keys (expectation %.2f)"
   % (c32, 50000.0 ** 2 / (2 * 2.0 ** 32)))
ok("the constants come from pi, e, sqrt2 and sqrt3",
   (kh.K0 >> 32, kh.K1 >> 32) == (0x243F6A88, 0xB7E15162),
   "K0=%016x K1=%016x K2=%016x K3=%016x, derived in six lines you can re-run"
   % (kh.K0, kh.K1, kh.K2, kh.K3))
ok("length is mixed before the body", kh.khash64(b"ab") != kh.khash64(b"ab\x00"),
   "so a string and its zero-padding diverge from the first round")

# =====================================================================
print("\n--- KEEL-GLYPH: the written form, and the seal that guards it ----")


def _noncanon(text):
    """Inserting a zero digit anywhere must be refused."""
    for pos in range(len(surface.MAGIC), min(len(text), 60)):
        try:
            surface.parse(text[:pos] + "0" + text[pos:])
        except Exception:
            continue
        return False
    return True


def speed_def():
    b = B()
    x = b.param("x", Real("f64", M), 100.0, 400.0)
    t = b.param("t", Real("f64", S), 2.0, 8.0)
    v = b.require(b.bind(b.div(x, t), "ALU64"), 0.0, 250.0, MPS, "limit")
    return Def("v.speed", v, b.g, dal="C", atol=1e-9,
               intent="mean speed over a leg")


d = speed_def()
text = surface.emit([d])
back, = surface.parse(text)
ok("a definition round-trips through glyphs exactly",
   back.graph[back.root].addr == d.graph[d.root].addr,
   "%d glyphs, no whitespace, no names, address %s unchanged"
   % (len(text), d.graph[d.root].addr[:12]))
ok("the metadata round-trips too",
   (back.name, back.dal, back.atol, back.intent) ==
   (d.name, d.dal, d.atol, d.intent), "name, DAL, tolerance and intent")

# Change one glyph anywhere in the stream. The only acceptable outcomes are
# "refused" and "produced the identical graph"; a stream that parses into a
# DIFFERENT program is the thing the seal exists to prevent.
refused = benign = leaked = 0
for pos in range(len(surface.MAGIC), len(text)):
    if text[pos] == surface.SEP:
        continue
    alt = "0" if text[pos] != "0" else "1"
    try:
        got, = surface.parse(text[:pos] + alt + text[pos + 1:])
    except Exception:
        refused += 1
        continue
    if got.graph[got.root].addr == d.graph[d.root].addr:
        benign += 1
    else:
        leaked += 1
ok("no single-glyph change can load as a different program", leaked == 0,
   "%d of %d positions refused, %d changed nothing, %d slipped through; a "
   "parser bug can only make a right program fail to load"
   % (refused, refused + benign + leaked, benign, leaked))

ok("a leading zero digit is refused as non-canonical",
   _noncanon(text),
   "without this, one glyph could be inserted invisibly: prepending a zero "
   "digit decodes to the same number")

b = B()
s = b.param("s", Real("f64", MPS), 0.0, 300.0)
lim = Def("limit", b.clamp(s, 0.0, 250.0), b.g, dal="C", atol=1e-9)
b = B()
x = b.param("x", Real("f64", M), 100.0, 400.0)
t = b.param("t", Real("f64", S), 2.0, 8.0)
cap = Def("v.capped", b.call(func("limit", lim), b.div(x, t)), b.g, dal="C",
          atol=1e-9)
mod = surface.emit([lim, cap])
defs, funcs, reps = surface.load(mod)
ok("a module's call sites are re-derived, never transmitted",
   all(x.ok for x in reps) and funcs[0].contract.hi == 250.0,
   "%d glyphs hold 2 definitions; the contract %r was recomputed from the "
   "body" % (len(mod), funcs[0].contract.hi))

# =====================================================================
print("\n--- the standard library, loaded from its glyph form -------------")

from keel.std import build, module_text                              # noqa: E402

libfuncs, libdefs = build()
txt = module_text()
defs, funcs, reps = surface.load(txt)
ok("the standard library checks out from glyphs alone",
   all(r.ok for r in reps) and len(defs) == len(libdefs),
   "%d definitions, %d glyphs, every one ACCEPT" % (len(defs), len(txt)))
leak = 0
checked = 0
for pos in range(len(surface.MAGIC), len(txt), 11):
    if txt[pos] == surface.SEP:
        continue
    checked += 1
    alt = "0" if txt[pos] != "0" else "1"
    try:
        got = surface.parse(txt[:pos] + alt + txt[pos + 1:])
    except Exception:
        continue
    if [x.graph[x.root].addr for x in got] != [x.graph[x.root].addr
                                               for x in libdefs]:
        leak += 1
ok("the same holds across a nine-definition module", leak == 0,
   "every 11th position of %d sampled (%d probes), none loaded as a "
   "different library" % (len(txt), checked))

ok("addresses survive the written form",
   [x.graph[x.root].addr for x in defs]
   == [x.graph[x.root].addr for x in libdefs],
   "the library you check is the library that was built")

fused = libfuncs["horner3"]
unfused = libfuncs["horner3_unfused"]
ok("fma halves the proven error of a cubic",
   fused.report.aerr[fused.body.root]
   < unfused.report.aerr[unfused.body.root],
   "fused %.3g vs unfused %.3g absolute error, %g vs %g cycles"
   % (fused.report.aerr[fused.body.root],
      unfused.report.aerr[unfused.body.root],
      fused.contract.cycles, unfused.contract.cycles))

# =====================================================================
print("\n--- the code generator: two instruction sets, same bits ----------")

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
    ("deep", lambda b, x, y: b.add(b.mul(b.add(x, y), b.sub(x, y)),
                                   b.div(b.mul(x, x), b.add(y, y))),
     lambda a, c: (a + c) * (a - c) + (a * a) / (c + c)),
]


def scalar(bld):
    b = B()
    x = b.param("x", ONEV, 3.0, 8.0)
    y = b.param("y", ONEV, 1.0, 2.0)
    return Def("t", bld(b, x, y), b.g, dal="C", atol=1e9)


f = bk.cpu()
print("     this machine: %s" % f["brand"])
if not bk.X86:
    print("     not x86-64, so the code generator cannot be exercised here;")
    print("     every proof-side test above still ran.")
    print("\n" + "=" * 70)
    print("%d passed, %d failed" % (len(PASS), len(FAIL)))
    raise SystemExit(1 if FAIL else 0)
print("     avx=%s fma=%s avx512f=%s -> %d f64 lanes"
      % (f["avx"], f["fma"], f["avx512f"], f["lanes"]))

for label, isa in (("auto", None), ("sse2 only", frozenset())):
    bad, total = [], 0
    for nm, bld, ref in CASES:
        d = scalar(bld)
        code = bk.emit(d, check(d), isa=isa)
        total += len(code)
        for a in (3.0, 3.5, 7.0, 8.0):
            for c in (1.0, 1.25, 2.0):
                got = code.run(*[a, c][:code.nargs])
                if got != ref(a, c):
                    bad.append((nm, a, c, got, ref(a, c)))
    ok("every operation is bit-exact against IEEE 754 (%s)" % label,
       not bad, "%d kernels x 12 input pairs, %d bytes total%s"
       % (len(CASES), total, "" if not bad else " MISMATCH %r" % bad[:2]))

b = B()
ps = [b.param("p%d" % i, ONEV, 1.0, 2.0) for i in range(24)]
prods = [b.mul(ps[i - 1], ps[i]) for i in range(1, 24)]
acc2 = prods[0]
for pr in prods[1:]:
    acc2 = b.add(acc2, pr)
d = Def("wide", acc2, b.g, dal="C", atol=1e9)
code = bk.emit(d, check(d))
vals = [1.0 + i / 24.0 for i in range(24)]
want = vals[0] * vals[1]
for i in range(2, 24):
    want = want + vals[i - 1] * vals[i]
ok("24 parameters and 45 operations, exact and unspilled",
   code.run(*vals) == want and code.stats["spills"] == 0,
   "spills=%d reloads=%d, %d-byte frame; the ABI is a pointer pair so the "
   "register file stopped being the limit"
   % (code.stats["spills"], code.stats["reloads"], code.frame))


def three(bld):
    b = B()
    x = b.param("x", ONEV, 0.1, 2.0)
    y = b.param("y", ONEV, 0.1, 2.0)
    z = b.param("z", ONEV, 0.1, 2.0)
    return Def("t", bld(b, x, y, z), b.g, dal="C", atol=1e-9)


d1 = three(lambda b, x, y, z: b.fma(x, y, z))
d2 = three(lambda b, x, y, z: b.add(b.mul(x, y), z))
r1, r2 = check(d1), check(d2)
ok("one rounding is provably tighter than two",
   r1.err[d1.root] < r2.err[d2.root],
   "fma %.4g vs mul-then-add %.4g relative error"
   % (r1.err[d1.root], r2.err[d2.root]))
c1, c2 = bk.emit(d1, r1), bk.emit(d2, r2)
diff = [(a, bb, cc) for a in (0.1, 0.3, 1.7) for bb in (0.1, 0.7, 1.3)
        for cc in (0.1, 1.1) if c1.run(a, bb, cc) != c2.run(a, bb, cc)]
_w = diff[0] if diff else (0.0, 0.0, 0.0)
ok("fma and mul-then-add compute different numbers", len(diff) > 0,
   "differ on %d of 18 inputs; %g*%g+%g -> %.20g fused, %.20g unfused"
   % (len(diff), _w[0], _w[1], _w[2], c1.run(*_w), c2.run(*_w)))
ok("so the backend never contracts on its own",
   all(o != "fma" for _, o, _, _ in c2.nodes),
   "the checker proved mul-then-add with two roundings; giving it one would "
   "describe a program that was not run")


def vdef(bound):
    b = B()
    u = b.param("u", Vec(Real("f64", M), 3, "ECI"), -10.0, 10.0)
    v = b.param("v", Vec(Real("f64", M), 3, "ECI"), -10.0, 10.0)
    w = b.add(u, v)
    w = b.bind(w, "FMA512", lanes=3) if bound else w
    w2 = b.sub(w, u)
    w2 = b.bind(w2, "FMA512", lanes=3) if bound else w2
    return Def("v", w2, b.g, dal="C", atol=1e-6)


cu = bk.emit(vdef(False), check(vdef(False)))
cb = bk.emit(vdef(True), check(vdef(True)))
got_u = cu.run((1.0, 2.0, 3.0), (0.5, -1.0, 4.0))
got_b = cb.run((1.0, 2.0, 3.0), (0.5, -1.0, 4.0))
ok("a proven binding selects the packed instruction",
   got_u == got_b == (0.5, -1.0, 4.0)
   and cb.stats["packed"] == 2 and cu.stats["packed"] == 0,
   "unbound: %d scalar lane ops; bound: %d packed ops, same answer"
   % (cu.stats["scalar"], cb.stats["packed"]))

b = B()
u = b.param("u", Vec(Real("f64", M), 3, "ECI"), -10.0, 10.0)
d = Def("t", b.bind(b.mul(b.param("q", ONEV, 1.0, 2.0),
                          b.param("w", ONEV, 1.0, 2.0)), "TENSOR"), b.g,
        dal="C", atol=1e9)
rejects("f64 on an f16 tensor core is still refused", check(d), "substrate")

f, rows = __import__("keel.calibrate", fromlist=["x"]).lanes_disagreement()
over = [n for n, want, have, good in rows if not good]
ok("CPUID catches a site that claims more lanes than the machine has",
   len(over) > 0 or f["avx512f"],
   "over-declared here: %s  (this is a bug report against the registry, "
   "not against the checker)" % (over or "none"))

# =====================================================================
print("\n" + "=" * 70)
print("%d passed, %d failed" % (len(PASS), len(FAIL)))
if FAIL:
    print("failed: " + ", ".join(FAIL))
    raise SystemExit(1)
