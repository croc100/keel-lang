"""
The KEEL standard library, written in KEEL.

Every entry here is a definition that went through the checker and came back
accepted. That is why the contracts printed by `python -m keel check` beside
each one are not documentation: they are what was proven, read off the proof.
A library function whose stated range is wrong cannot exist, because the
statement is generated from the body rather than written next to it.

Read this file as the answer to "can you actually write anything in it".

Build the module with:

    python -m keel.std > std.kl          # canonical glyph form
    python -m keel check std.kl          # and check it back

Note what is NOT here: no string formatting, no collections, no date
arithmetic, no sockets. Those need the bytes and effect operations to reach
the backend first, and claiming a standard library that cannot run would be
the kind of thing this project exists to avoid.
"""

from .check import func
from .build import B
from .ir import Def
from .types import KMH, M, M2, MPS, MPS2, ONE, Real, Vec

ONEV = Real("f64", ONE)


# ============================================================ conversions

def mps_from_kmh():
    """km/h -> m/s, exactly.

    The scale is 1000/3600 = 5/18, which is not representable as a float. It
    is carried as a Fraction and lowered to a single multiply by the nearest
    double to 5/18, and the error analysis knows that is what happened.
    """
    b = B()
    v = b.param("v", Real("f64", KMH), 0.0, 400.0)
    return func("mps_from_kmh",
                Def("mps_from_kmh", b.convert(v, MPS), b.g,
                    dal="B", atol=1e-12, params=(v,),
                    intent="road speed in SI units"))


# ============================================================== kinematics

def brake_distance(deps):
    """v^2 / (2a). The divisor's interval excludes zero, which is the entire
    reason this is allowed to contain a division."""
    b = B()
    v = b.param("v", Real("f64", MPS), 0.0, 100.0)
    a = b.param("a", Real("f64", MPS2), 1.0, 12.0)
    two = b.const(2.0, ONEV)
    d = b.div(b.mul(v, v), b.mul(two, a))
    d = b.require(d, 0.0, 5000.0, M, "stopping distance stays on the map")
    return func("brake_distance",
                Def("brake_distance", d, b.g, dal="B", atol=1e-9,
                    params=(v, a),
                    intent="distance to stop from v at constant deceleration"))


def lerp(deps):
    """a + (b - a) * t, for t in [0,1].

    The subtraction can straddle zero, so there is no relative error bound
    here and an absolute one is declared instead. The checker says so; it
    does not quietly report a relative bound of infinity as if it were fine.
    """
    b = B()
    p = b.param("p", Real("f64", M), -1000.0, 1000.0)
    q = b.param("q", Real("f64", M), -1000.0, 1000.0)
    t = b.param("t", ONEV, 0.0, 1.0)
    out = b.add(p, b.mul(b.sub(q, p), t))
    return func("lerp", Def("lerp", out, b.g, dal="C", atol=1e-9,
                            params=(p, q, t),
                            intent="interpolate between two positions"))


# =================================================================== fma

def horner3(deps):
    """A cubic by Horner's rule, using fma.

    Three fused multiply-adds instead of three multiplies and three adds. The
    point is not that it is shorter: it is that each fma rounds ONCE, so the
    proven error bound is about half of what the unfused form can promise,
    and the checker computes both rather than being told.
    """
    b = B()
    x = b.param("x", ONEV, -2.0, 2.0)
    c3 = b.const(0.5, ONEV)
    c2 = b.const(-1.25, ONEV)
    c1 = b.const(2.0, ONEV)
    c0 = b.const(-0.75, ONEV)
    h = b.fma(c3, x, c2)
    h = b.fma(h, x, c1)
    h = b.fma(h, x, c0)
    return func("horner3", Def("horner3", h, b.g, dal="C", atol=1e-9,
                               params=(x,),
                               intent="0.5x^3 - 1.25x^2 + 2x - 0.75"))


def horner3_unfused(deps):
    """The same polynomial without fma, so the two error bounds can be put
    side by side. Nothing in the library needs this; the test suite does."""
    b = B()
    x = b.param("x", ONEV, -2.0, 2.0)
    c3 = b.const(0.5, ONEV)
    c2 = b.const(-1.25, ONEV)
    c1 = b.const(2.0, ONEV)
    c0 = b.const(-0.75, ONEV)
    h = b.add(b.mul(c3, x), c2)
    h = b.add(b.mul(h, x), c1)
    h = b.add(b.mul(h, x), c0)
    return func("horner3_unfused",
                Def("horner3_unfused", h, b.g, dal="C", atol=1e-9,
                    params=(x,), intent="the same cubic, unfused"))


# =============================================================== vectors

def vmag(deps):
    """|u|, with the packed path selected by a proven binding."""
    b = B()
    u = b.param("u", Vec(Real("f64", M), 3, "BODY"), -1.0e4, 1.0e4)
    # bind the OPERATION, not the parameter: a binding names the unit that
    # executes something, and a parameter does not execute
    n = b.bind(b.norm(u), "FMA512", lanes=3)
    return func("vmag", Def("vmag", n, b.g, dal="C", atol=1e-6, params=(u,),
                            intent="magnitude of a body-frame displacement"))


def vdot(deps):
    b = B()
    u = b.param("u", Vec(Real("f64", M), 3, "BODY"), -1.0e3, 1.0e3)
    v = b.param("v", Vec(Real("f64", M), 3, "BODY"), -1.0e3, 1.0e3)
    out = b.require(b.bind(b.dot(u, v), "FMA512", lanes=3),
                    -3.0e6, 3.0e6, M2, "dot stays in range")
    return func("vdot", Def("vdot", out, b.g, dal="C", atol=1.0,
                            params=(u, v), intent="body-frame inner product"))


# ================================================================= loops

def decay_body(deps):
    """One step of a first-order low-pass: acc <- acc/2 + x/2 with x == 1.

    This is a fold BODY, so it takes (acc, i). It ignores the index, which is
    exactly why a definition is allowed to declare its signature: a body that
    does not use its index still has two parameters, and its callers have to
    agree about that.
    """
    b = B()
    acc = b.param("acc", ONEV, 0.0, 2.0)
    i = b.param("i", __import__("keel.types", fromlist=["Int"]).Int(),
                0.0, 1023.0)
    half = b.const(0.5, ONEV)
    out = b.add(b.mul(acc, half), half)
    return func("decay_body", Def("decay_body", out, b.g, dal="C",
                                  atol=1e-12, params=(acc, i),
                                  intent="one low-pass step toward 1"))


def settle(deps):
    """Run the low-pass 64 times from zero.

    The invariant [0,2] is what makes this provable: the initial value is
    inside it, and the body maps it into itself, so the result is inside it
    after any number of iterations. That is induction, and it is the whole
    proof -- there is no symbolic execution of 64 steps.
    """
    body = deps["decay_body"]
    b = B()
    z = b.const(0.0, ONEV)
    out = b.fold(body, z, 64, 0.0, 2.0)
    return func("settle", Def("settle", out, b.g, dal="C", atol=1e-9,
                              intent="low-pass settling from rest"),
                funcs=[body])


# ========================================================= the whole module

ORDER = (
    ("mps_from_kmh", mps_from_kmh),
    ("brake_distance", brake_distance),
    ("lerp", lerp),
    ("horner3", horner3),
    ("horner3_unfused", horner3_unfused),
    ("vmag", vmag),
    ("vdot", vdot),
    ("decay_body", decay_body),
    ("settle", settle),
)


def build():
    """Check the library in dependency order and return (funcs, defs).

    A name becomes available to the entries after it only once it has been
    accepted, which is the same rule a module file follows when it is loaded.
    """
    out, defs = {}, []
    for name, mk in ORDER:
        f = mk(out) if mk.__code__.co_argcount else mk()
        out[name] = f
        defs.append(f.body)
    return out, defs


def module_text():
    from .surface import emit
    return emit(build()[1])


if __name__ == "__main__":
    import sys
    sys.stdout.write(module_text() + "\n")
