"""
A tour of the general-purpose half of KEEL.

Everything up to this point in the project was about one expression: a number
with a unit, a frame, an interval and a proven error bound. One expression is
not a language. This file is the answer to "can you write a program in it" --
functions, loops with state, records, variants, mutation, effects and strings,
none of which gives up any of the proof.

Run:  python examples/language.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keel import B, Def, check                                 # noqa: E402
from keel.check import func                                    # noqa: E402
from keel.surface import emit, load                            # noqa: E402
from keel.types import (Buf, Cap, Int, KMH, M, MPS, MPS2,  # noqa: E402
                        ONE, Real, Rec, Sum)

ONEV = Real("f64", ONE)


def line(t):
    print("\n" + t + "\n" + "-" * len(t))


def show(d, funcs=None, extra=None):
    r = check(d, funcs=funcs)
    print(" ", r.summary())
    if r.ok and extra:
        print("    " + extra(r))
    return r


# =============================================  1. a function is a proof
line("1. a function's contract is read off its proof, not written beside it")

b = B()
v = b.param("v", Real("f64", MPS), 0.0, 100.0)
a = b.param("a", Real("f64", MPS2), 1.0, 12.0)
dist = b.div(b.mul(v, v), b.mul(b.const(2.0, ONEV), a))
brake = func("brake", Def("brake", dist, b.g, params=(v, a), dal="B",
                          atol=1e-9,
                          intent="stopping distance at constant deceleration"))
print("  %r" % brake.contract)
print("""
  Nothing in that line was typed by an author. The parameter boxes are the
  ones the parameters declared, the result interval and the error bound and
  the cycle count came out of the checker, and the word "pure" is the
  complete effect set. A library function cannot document a range it does
  not have, because the documentation is generated from the proof.""")

b = B()
s = b.param("s", Real("f64", MPS), 0.0, 80.0)
g = b.param("g", Real("f64", MPS2), 2.0, 9.0)
show(Def("ok", b.call(brake, s, g), b.g, dal="B", atol=1e-9), [brake],
     lambda r: "the caller adopts [%g, %g] without re-deriving the body"
     % (r.iv[r.ty and list(r.iv)[-1]].lo, r.iv[list(r.iv)[-1]].hi))

b = B()
s = b.param("s", Real("f64", MPS), 0.0, 400.0)
g = b.param("g", Real("f64", MPS2), 2.0, 9.0)
show(Def("too-fast", b.call(brake, s, g), b.g, dal="B", atol=1e-9), [brake])

b = B()
s = b.param("s", Real("f64", MPS), 0.0, 80.0)
g = b.param("g", Real("f64", MPS2), 2.0, 9.0)
show(Def("unverified", b.call(brake, s, g), b.g, dal="B", atol=1e-9))
print("""
  The third one is the important one. A contract whose body was not supplied
  is an UNDISCHARGED OBLIGATION, not a convenience. Adopting a promise that
  nobody checked would turn modular verification from a way of dividing work
  into a way of hiding it.""")


# ================================================  2. a loop is induction
line("2. a loop with state, proven by induction over an interval")

b = B()
acc = b.param("acc", ONEV, 0.0, 2.0)
i = b.param("i", Int(), 0.0, 1023.0)
step = func("lowpass", Def("lowpass",
                           b.add(b.mul(acc, b.const(0.5, ONEV)),
                                 b.const(0.5, ONEV)),
                           b.g, params=(acc, i), dal="C", atol=1e-12,
                           intent="one first-order step toward 1"))
print("  %r" % step.contract)

b = B()
show(Def("settle", b.fold(step, b.const(0.0, ONEV), 256, 0.0, 2.0), b.g,
         dal="C", atol=1e-9), [step],
     lambda r: "256 iterations, %d cycles, and the proof is two obligations"
     % r.cost.cycles)

b = B()
show(Def("bad-invariant", b.fold(step, b.const(0.0, ONEV), 256, 0.0, 0.4),
         b.g, dal="C", atol=1e-9), [step])
print("""
  The accepted one was not simulated 256 times. Two facts were checked: the
  initial value is inside [0,2], and the body maps [0,2] into [0,2]. That is
  induction, and it is why a loop costs the same to verify at 256 iterations
  as at 256 million. The trip count is a literal, so the cost is a number --
  there is no KEEL program without a WCET.""")


# ===========================================  3. records and variants
line("3. no null: a value that may be absent is a variant you must handle")

STATE = Rec("state", (("pos", Real("f64", M)), ("vel", Real("f64", MPS))))
READING = Sum("reading", (("good", Real("f64", MPS)),
                          ("stale", Int())))

b = B()
x = b.param("x", Real("f64", MPS), 0.0, 50.0)
good = func("trust", Def("trust", b.clamp(x, 0.0, 40.0), b.g, dal="C",
                         atol=1e-9))
b = B()
age = b.param("age", Int(), 0.0, 9.0)
z = b.const(0.0, Real("f64", MPS))
fallback = func("hold", Def("hold", b.add(z, z), b.g, params=(age,),
                            dal="C", atol=1e-9))

b = B()
x = b.param("x", Real("f64", MPS), 0.0, 50.0)
rd = b.inject(READING, "good", x)
show(Def("handled", b.match(rd, {"good": good, "stale": fallback},
                            Real("f64", MPS)), b.g, dal="C", atol=1e-9),
     [good, fallback],
     lambda r: "result is the hull of the arms; cost is the WORST arm")

b = B()
x = b.param("x", Real("f64", MPS), 0.0, 50.0)
rd = b.inject(READING, "good", x)
show(Def("forgotten", b.match(rd, {"good": good}, Real("f64", MPS)), b.g,
         dal="C", atol=1e-9), [good])
print("""
  A sensor reading that might be stale is not a float and a flag, and it is
  certainly not a float that is sometimes NaN. It is a variant, and the only
  way to read one is a match the checker counts. Forgetting the stale case
  is a rejected program rather than a surprise in flight.""")


# =====================================  4. mutation without aliasing
line("4. aliasing is not analysed away -- it cannot be written down")

BUF = Buf(Real("f64", M), (8,), "DRAM")

b = B()
v = b.param("v", Real("f64", M), -1.0, 1.0)
buf = b.alloc(BUF)
buf = b.put(buf, b.const(0, Int()), v)
buf = b.put(buf, b.const(7, Int()), v)
show(Def("threaded", b.freeze(buf), b.g, dal="C", atol=1e-9), None,
     lambda r: "%d linear values, each with exactly one owner" % len(r.linear))

b = B()
v = b.param("v", Real("f64", M), -1.0, 1.0)
buf = b.alloc(BUF)
one = b.put(buf, b.const(0, Int()), v)
two = b.put(buf, b.const(1, Int()), v)
show(Def("aliased", b.add(b.get(one, b.const(0, Int())),
                          b.get(two, b.const(0, Int()))), b.g,
         dal="C", atol=1e-9))

b = B()
v = b.param("v", Real("f64", M), -1.0, 1.0)
buf = b.put(b.alloc(BUF), b.const(99, Int()), v)
show(Def("overflow", b.freeze(buf), b.g, dal="C", atol=1e-9))
print("""
  A buffer is linear, so it must have exactly ONE edge pointing at it, and
  the graph is content-addressed -- it already counts edges. Two names for
  one mutable thing IS two edges. Rust's borrow checker is a fan-out count
  here; nothing was invented for it, and the index check is the same
  obligation machinery that proves a divisor is non-zero.""")


# ==========================================  5. effects are arguments
line("5. no ambient authority: an effect is a token you were handed")

b = B()
net = b.cap("net", "NET")
out = b.perform(net, "NET.send", ty=Cap("NET"), cost_s=2.0e-3)
show(Def("send", out, b.g, dal="C", atol=1e9, ceiling={"NET.send"},
         deadline_s=5e-3),
     None, lambda r: "effects %s, %.1f ms of physical time against a 5 ms "
     "deadline" % (sorted(r.effects), r.cost.seconds * 1e3))

b = B()
net = b.cap("net", "NET")
show(Def("no-ceiling", b.perform(net, "NET.send", ty=Cap("NET")), b.g,
         dal="C", atol=1e9, ceiling=set()))

b = B()
fs = b.cap("fs", "FS")
show(Def("wrong-token", b.perform(fs, "NET.send", ty=Cap("FS")), b.g,
         dal="C", atol=1e9, ceiling={"NET.send"}))
print("""
  A capability is linear too, so a function handed one network token cannot
  make two calls out of it, and a function handed none cannot reach the
  network however it is written. Authority travels in the arguments, which
  means it is visible in the type of every function on the path -- and the
  cost of an effect is seconds, so a network round trip and a display frame
  are compared in the same arithmetic as a multiply.""")


# ===============================================  6. the written form
line("6. what a KEEL program actually looks like")

b = B()
v = b.param("v", Real("f64", KMH), 0.0, 400.0)
d = Def("mps", b.convert(v, MPS), b.g, params=(v,), dal="B", atol=1e-12,
        intent="road speed in SI units")
text = emit([d])
print("  %s" % text)
print("""
  No keywords, no names, no delimiters, no whitespace. Every value is
  referenced by distance back in the stream, so two identical programs are
  identical text -- which is what content addressing needs. The last field is
  the SEAL: the root address, written down.""")
defs, funcs, reps = load(text)
print("  loaded back: %s" % reps[0].summary())
print("  address declared in the stream == address of the graph built from it")
# change one glyph somewhere in the body and report what happens; the
# interesting case to print is the one the seal itself catches
seal_caught = other = leaked = 0
first_seal_msg = ""
for pos in range(len(text)):
    alt = "0" if text[pos] != "0" else "1"
    try:
        load(text[:pos] + alt + text[pos + 1:])
        leaked += 1
    except Exception as e:
        if "seal mismatch" in str(e):
            seal_caught += 1
            first_seal_msg = first_seal_msg or str(e).split(". ")[0]
        else:
            other += 1
print("  one glyph changed, at each of %d positions:" % len(text))
print("    %3d refused by the seal, %3d refused while parsing, %d loaded as "
      "a different program" % (seal_caught, other, leaked))
print("    %s" % first_seal_msg[:118])
print("""
  That is why the parser is outside the trust root. A parser bug produces a
  different graph, which produces a different address, which does not match
  the seal. A bug here can make a correct program fail to load; it cannot
  make an incorrect one load.""")

print("\nall sections ran.")
