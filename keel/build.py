"""
The builder -- UNTRUSTED, and that is the point.

Everything in here is convenience. It picks types, it fills in attributes, it
threads addresses together. None of it is believed: `check.py` re-derives
every type from scratch and rejects any node whose declared type does not
match what it derives, so a builder that is buggy, out of date or hostile
cannot produce an accepted program that is wrong.

That separation is what makes it safe for this file to be large and clever
while the trust root stays small. It is also what makes the builder
REPLACEABLE: this one is written in Python because that is what was to hand,
and a KEEL program contains none of it -- a program is a content-addressed
graph and a glyph stream. Rewriting this file in another language changes
nothing about any existing program, which is the test of whether a language
really is independent of its host.
"""

from .ir import Func, Graph, Node
from .types import (Arr, Bool, Buf, Bytes, Cap, Int, Real, Rec, Rot,
                    Sum, Unit, Vec)


# --------------------------------------------------------------------- builder
class B:
    """Untrusted convenience builder. Declares types; the checker verifies."""

    def __init__(self, graph=None):
        self.g = graph or Graph()

    def _n(self, op, args, attrs, ty):
        return self.g.add(Node(op, args, attrs, ty))

    def ty(self, a):
        return self.g[a].ty

    # ---- leaves
    def const(self, value, ty):
        return self._n("const", (), {"value": value}, ty)

    def param(self, name, ty, lo=None, hi=None, dal=None, err=0.0):
        """Parameters MUST declare a numeric range.

        A human would never bother. Without it no interval analysis is
        possible, so no obligation can be discharged, so nothing loads.

        `dal` declares the assurance level of the SOURCE. An input from a
        DAL D sensor cannot reach a DAL A output except through a monitor.
        """
        return self._n("param", (), {"name": name, "lo": lo, "hi": hi,
                                     "dal": dal, "err": err}, ty)

    def loopvar(self, bound, name="i"):
        return self._n("loopvar", (), {"bound": bound, "name": name}, Int())

    # ---- arithmetic
    def add(self, a, b):
        return self._n("add", (a, b), {}, self.ty(a))

    def sub(self, a, b):
        return self._n("sub", (a, b), {}, self.ty(a))

    def mul(self, a, b):
        ta, tb = self.ty(a), self.ty(b)
        if isinstance(ta, Real) and isinstance(tb, Real):
            ty = Real(ta.fmt, ta.unit * tb.unit, ta.frame or tb.frame,
                      ta.repro and tb.repro)
        else:
            ty = ta
        return self._n("mul", (a, b), {}, ty)

    def div(self, a, b):
        ta, tb = self.ty(a), self.ty(b)
        if isinstance(ta, Real) and isinstance(tb, Real):
            ty = Real(ta.fmt, ta.unit / tb.unit, ta.frame or tb.frame,
                      ta.repro and tb.repro)
        else:
            ty = ta
        return self._n("div", (a, b), {}, ty)

    def fma(self, a, b, c):
        """a*b + c with ONE rounding, the way the silicon does it.

        Separate from mul-then-add because it computes a DIFFERENT NUMBER.
        Asking for it explicitly is the only way the error analysis and the
        instruction stream can be talking about the same program.
        """
        ta, tb, tc = self.ty(a), self.ty(b), self.ty(c)
        ty = tc
        if isinstance(ta, Real) and isinstance(tb, Real):
            ty = Real(ta.fmt, ta.unit * tb.unit,
                      ta.frame or tb.frame or getattr(tc, "frame", None),
                      ta.repro and tb.repro and getattr(tc, "repro", True))
        return self._n("fma", (a, b, c), {}, ty)

    def neg(self, a):
        return self._n("neg", (a,), {}, self.ty(a))

    def sqrt(self, a):
        ta = self.ty(a)
        ty = Real(ta.fmt, Unit(tuple(d // 2 for d in ta.unit.dim),
                               ta.unit.scale), ta.frame, ta.repro)
        return self._n("sqrt", (a,), {}, ty)

    def convert(self, a, to_unit):
        """The ONLY way to change scale. Explicit, typed, and costed."""
        ta = self.ty(a)
        return self._n("convert", (a,), {"to": to_unit},
                       Real(ta.fmt, to_unit, ta.frame, ta.repro))

    # ---- vectors and frames
    def vec3(self, x, y, z, frame):
        tx = self.ty(x)
        return self._n("vec3", (x, y, z), {"frame": frame},
                       Vec(Real(tx.fmt, tx.unit, frame, tx.repro), 3, frame))

    def rot(self, name, frm, to):
        return self._n("param", (), {"name": name, "lo": None, "hi": None},
                       Rot(frm, to))

    def rot_apply(self, r, v):
        tr, tv = self.ty(r), self.ty(v)
        return self._n("rot_apply", (r, v), {},
                       Vec(Real(tv.elem.fmt, tv.elem.unit, tr.to, tv.elem.repro),
                           tv.n, tr.to))

    def dot(self, a, b):
        ta, tb = self.ty(a), self.ty(b)
        return self._n("dot", (a, b), {},
                       Real(ta.elem.fmt, ta.elem.unit * tb.elem.unit, None,
                            ta.elem.repro and tb.elem.repro))

    def norm(self, v):
        tv = self.ty(v)
        return self._n("norm", (v,), {},
                       Real(tv.elem.fmt, tv.elem.unit, None, tv.elem.repro))

    def vscale(self, v, s):
        tv, ts = self.ty(v), self.ty(s)
        return self._n("vscale", (v, s), {},
                       Vec(Real(tv.elem.fmt, tv.elem.unit * ts.unit, tv.frame,
                                tv.elem.repro and ts.repro), tv.n, tv.frame))

    # ---- memory
    def load(self, arr, idx, lanes=1, stride=1):
        """`lanes` concurrent accesses at `stride` apart.

        On a banked site these two numbers decide, by arithmetic alone,
        whether the banks answer in parallel or serialize. That is a circuit
        fact the program controls and that nothing above assembly exposes.
        """
        ta = self.ty(arr)
        e = ta.elem
        ty = Real(e.fmt, e.unit, e.frame, e.repro) if isinstance(e, Real) else e
        return self._n("load", (arr, idx), {"lanes": lanes, "stride": stride},
                       ty)

    def stencil(self, arr, offsets, coeffs, out_level="DRAM", boundary=None):
        ta = self.ty(arr)
        return self._n("stencil", (arr,),
                       {"offsets": list(offsets), "coeffs": list(coeffs),
                        "out_level": out_level, "boundary": boundary},
                       Arr(ta.elem, ta.shape, out_level))

    def reduce(self, arr, assoc):
        """assoc in {'ordered','tree','nondet'}.

        'nondet' means the parallel decomposition is free to reassociate, so
        the result is NOT bitwise reproducible, and the type says so.
        """
        ta = self.ty(arr)
        e = ta.elem
        return self._n("reduce", (arr,), {"assoc": assoc},
                       Real(e.fmt, e.unit, e.frame, assoc != "nondet"))

    # ---- control / assurance
    def clamp(self, a, lo, hi):
        """Saturating refinement. Narrows the proven interval, which is how a
        downstream obligation (a divide, an index) gets discharged."""
        return self._n("clamp", (a,), {"lo": lo, "hi": hi}, self.ty(a))

    def repeat(self, inner, bound):
        return self._n("repeat", (inner,), {"bound": bound}, self.ty(inner))

    def monitor(self, a, dal, check):
        """Raise a value's criticality, backed by a declared runtime check.

        This is the ONLY way low-DAL data may flow into high-DAL computation.
        """
        return self._n("monitor", (a,), {"dal": dal, "check": check}, self.ty(a))

    def cmp(self, op, a, b):
        return self._n("cmp", (a, b), {"op": op}, Bool())

    def select(self, c, a, b):
        return self._n("select", (c, a, b), {}, self.ty(a))

    def sys(self, name, args, ty, effect):
        """A capability-gated foreign call. `effect` is checked against the
        definition's ceiling at load time."""
        return self._n("sys", tuple(args), {"name": name, "effect": effect}, ty)

    # ---- the physical world
    # Four operations cover every substrate: sense it, drive it, cross a time
    # domain, and prove an invariant about a quantity. A switch, a trace, a
    # panel and a DRAM bank use the same four.

    def sense(self, site, ty, after, lo=None, hi=None, err=0.0):
        """Read a value off a physical site.

        `after` is the seconds elapsed since the site last transitioned. The
        checker requires it to exceed that site's settling time, which is
        contact debounce, flip-flop setup and display response all at once.
        """
        return self._n("sense", (), {"site": site, "after": after,
                                     "lo": lo, "hi": hi, "err": err}, ty)

    def drive(self, a, site, effect, loads=1, load_a=None):
        """Commit a value to a physical site.

        `loads` is how many sinks this site feeds; the checker proves the site
        can source their current. `effect` is capability-gated like any other.
        """
        return self._n("drive", (a,), {"site": site, "effect": effect,
                                       "loads": loads, "load_a": load_a},
                       self.ty(a))

    def sync(self, a, to, depth=2):
        """Cross a clock domain through a declared synchronizer.

        A clock domain is a reference frame for time, so this is `rot_apply`
        for clocks and it is checked by the same frame rule. Depth < 2 is
        metastability.
        """
        ta = self.ty(a)
        return self._n("sync", (a,), {"to": to, "depth": depth},
                       Real(ta.fmt, ta.unit, to, ta.repro))

    def bind(self, a, unit, lanes=1):
        """Name the physical unit that executes this operation.

        An execution unit is a site. Binding to one is the same act as driving
        a pin: you state where the value lands and the checker proves the
        landing is possible -- the format the unit carries, the lanes it has,
        and the time domain it runs in.

        On an out-of-order core this is a CONSTRAINT, not a command: the
        renamer and the scheduler are silicon and no instruction addresses
        them. The checker says which one you got rather than pretending.
        """
        from .types import SITES
        ta = self.ty(a)
        u = SITES.get(unit)
        if u is not None and u.clock and isinstance(ta, Real):
            ta = Real(ta.fmt, ta.unit, u.clock, ta.repro)
        return self._n("bind", (a,), {"unit": unit, "lanes": lanes}, ta)

    def require(self, a, lo, hi, unit, why):
        """Prove a quantity stays inside a physical bound.

        This is the one operation every domain-specific rule reduces to:
        contact settling, setup/hold slack, fanout current, contrast ratio,
        bolt stress, torque limit, thermal margin, latency budget. `unit` must
        match the operand exactly, because a bound in lbf and a value in N are
        dimensionally identical and that is how missions are lost.
        """
        return self._n("require", (a,), {"lo": lo, "hi": hi, "unit": unit,
                                         "why": why}, self.ty(a))

    # ---------------------------------------------------------- records
    def rec(self, ty, **vals):
        """Build a record. Every field must be given; KEEL has no null, so
        there is no such thing as a partly built value."""
        if not isinstance(ty, Rec):
            raise TypeError("rec needs a Rec type, got %r" % (ty,))
        missing = [f for f in ty.names if f not in vals]
        if missing:
            raise TypeError("record %s is missing %s" % (ty.name, missing))
        extra = [k for k in vals if ty.index(k) is None]
        if extra:
            raise TypeError("record %s has no field %s" % (ty.name, extra))
        args = tuple(vals[f] for f in ty.names)
        return self._n("rec", args, {"rty": ty}, ty)

    def field(self, r, fname):
        """Read one field. Total: the field exists or this does not build."""
        t = self.ty(r)
        if not isinstance(t, Rec) or t.index(fname) is None:
            raise TypeError("%r has no field %r" % (t, fname))
        return self._n("field", (r,), {"f": fname}, t.ty_of(fname))

    # ------------------------------------------------------- sum types
    def inject(self, ty, tag, value):
        """Make a tagged value. The tag is part of the address, so two
        variants carrying the same payload are different values."""
        if not isinstance(ty, Sum) or ty.ty_of(tag) is None:
            raise TypeError("%r has no variant %r" % (ty, tag))
        return self._n("inject", (value,), {"sty": ty, "tag": tag}, ty)

    def match(self, scrutinee, arms, ret):
        """Branch on which variant holds. `arms` maps tag -> Func.

        Exhaustive by construction here and re-checked by the checker: a
        missing arm is a rejected program, not a runtime surprise. The result
        interval is the hull of the arms and the cost is the WORST arm, which
        is the only thing a WCET claim can mean about a branch.
        """
        t = self.ty(scrutinee)
        if not isinstance(t, Sum):
            raise TypeError("match needs a Sum, got %r" % (t,))
        items = tuple(sorted(arms.items()))
        return self._n("match", (scrutinee,),
                       {"arms": tuple((tag, f.contract) for tag, f in items),
                        "rty": ret}, ret)

    # --------------------------------------------------------- calling
    def call(self, f, *args):
        """Call a proven function.

        The callee's body is NOT re-derived here. Its contract is adopted,
        and the only thing checked at this site is that the arguments are
        inside the boxes the callee was proven over.
        """
        c = f.contract if isinstance(f, Func) else f
        if len(args) != len(c.params):
            raise TypeError("%s takes %d argument(s), got %d"
                            % (c.name, len(c.params), len(args)))
        return self._n("call", tuple(args), {"fn": c}, c.ret)

    # ------------------------------------------------- bounded iteration
    def fold(self, body, init, n, inv_lo, inv_hi):
        """Iterate `body` n times over a carried accumulator.

        body : (acc, i) -> acc, as a Func.
        inv  : the INVARIANT box for the accumulator.

        The checker proves two things and nothing else: the initial value is
        inside the box, and the body maps the box into the box. That is
        induction, and it is what makes a loop with state provable by the
        same interval arithmetic that proves a single multiply.

        `n` is a literal, so the loop terminates and its cost is n times the
        body's. There is no unbounded loop in KEEL and therefore no program
        without a WCET.
        """
        c = body.contract if isinstance(body, Func) else body
        if not (isinstance(n, int) and n > 0):
            raise TypeError("fold needs a positive literal trip count")
        return self._n("fold", (init,),
                       {"fn": c, "n": n, "lo": inv_lo, "hi": inv_hi}, c.ret)

    # --------------------------------------------- linear buffers (mutation)
    def alloc(self, ty, fill=0.0):
        """A fresh buffer. LINEAR: it has exactly one owner from here on."""
        if not isinstance(ty, Buf):
            raise TypeError("alloc needs a Buf type")
        return self._n("alloc", (), {"bty": ty, "fill": fill}, ty)

    def put(self, buf, idx, value):
        """Write one element, CONSUMING the buffer and producing the new one.

        There is no in-place write that leaves the old name usable, so there
        is never a second name for a mutated buffer. Aliasing is not analysed
        away; it cannot be written down.
        """
        t = self.ty(buf)
        if not isinstance(t, Buf):
            raise TypeError("put needs a Buf, got %r" % (t,))
        return self._n("put", (buf, idx, value), {}, t)

    def get(self, buf, idx):
        """Read one element. Reading does not consume: the buffer flows out
        of this node as its second result, which in a DAG means `get` must be
        threaded like `put`."""
        t = self.ty(buf)
        if not isinstance(t, Buf):
            raise TypeError("get needs a Buf, got %r" % (t,))
        return self._n("get", (buf, idx), {}, t.elem)

    def freeze(self, buf):
        """Spend the buffer and get an immutable array. The one-owner rule is
        what makes this sound without a copy."""
        t = self.ty(buf)
        if not isinstance(t, Buf):
            raise TypeError("freeze needs a Buf, got %r" % (t,))
        return self._n("freeze", (buf,), {}, t.frozen_ty())

    # ------------------------------------------------- capabilities (effects)
    def cap(self, name, kind):
        """A capability parameter. LINEAR, so it cannot be copied, and a
        function that was not handed one cannot reach that effect at all."""
        return self._n("param", (), {"name": name, "lo": None, "hi": None},
                       Cap(kind))

    def perform(self, cap, effect, args=(), ty=None, cost_s=0.0):
        """Do something to the outside world, spending the capability.

        The capability comes back out of this node, so a sequence of effects
        is a chain and not a set -- which is what gives an effectful program
        a defined order without introducing statements.
        """
        t = self.ty(cap)
        if not isinstance(t, Cap):
            raise TypeError("perform needs a Cap, got %r" % (t,))
        return self._n("perform", (cap,) + tuple(args),
                       {"effect": effect, "cost_s": cost_s},
                       ty if ty is not None else t)

    # ------------------------------------------------------------ bytes
    def bstr(self, data, enc="utf8"):
        """A byte literal. Its length is in its type."""
        b = data.encode(enc) if isinstance(data, str) else bytes(data)
        return self._n("bstr", (), {"data": list(b), "enc": enc},
                       Bytes(len(b), enc))

    def bslice(self, b, off, n):
        """A substring with STATIC bounds. Out of range does not build."""
        t = self.ty(b)
        if not isinstance(t, Bytes):
            raise TypeError("bslice needs Bytes, got %r" % (t,))
        return self._n("bslice", (b,), {"off": off, "n": n},
                       Bytes(n, t.enc))

    def bcat(self, x, y):
        tx, ty_ = self.ty(x), self.ty(y)
        if not (isinstance(tx, Bytes) and isinstance(ty_, Bytes)):
            raise TypeError("bcat needs Bytes")
        return self._n("bcat", (x, y), {}, Bytes(tx.n + ty_.n, tx.enc))

    def blen(self, b):
        return self._n("blen", (b,), {}, Int())

    def beq(self, x, y):
        return self._n("beq", (x, y), {}, Bool())

    def bhash(self, b, bits=64):
        """A content hash of a byte string, as a value in the language."""
        return self._n("bhash", (b,), {"bits": bits}, Int())
