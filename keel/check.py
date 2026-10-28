"""
KEEL checker -- THE TRUST ROOT.

This is the only file a human is required to read. A KEEL program may be a
million nodes of proof-carrying DAG that no person will ever look at; control
over it comes from the fact that THIS file is small enough to be read to the
end, and nothing loads unless this file says yes.

    unreadable programs, readable checker.

Deliberately included here, even though it would be tidier elsewhere: the
obligation discharger and the error analysis. Both are things you are
trusting, so hiding them in another module would be dishonest about the size
of the trust root.

There are no warnings. There is no partial success. A definition is accepted
or it is rejected, and a rejected definition does not run. In particular there
is NO bottom value: an error is not something that flows, it is a proof
obligation that was not discharged.

Seventeen checks, one pass over the DAG in dependency order:

  1. type re-derivation        -- the builder is untrusted; derive and compare
  2. units                     -- dimension AND exact scale, no implicit coercion
  3. frames                    -- spatial frames AND clock domains, one rule
  4. obligations               -- div/=0, sqrt>=0, index in range, invariants
  5. floating point error      -- proven bounds vs declared tolerance
  6. mixed criticality         -- low-DAL data may not reach high-DAL output
  7. reproducibility           -- bitwise determinism is part of the type
  8. placement                 -- arithmetic only where a site permits it
  9. capabilities and WCET     -- effects within ceiling, cycles within budget
 10. settling                  -- no sensing a site before its value is valid
 11. drive                     -- no sourcing more current than a site can
 12. deadline                  -- end-to-end PHYSICAL time, not just cycles
 13. unit binding              -- the execution unit can actually issue this
 14. bank conflict             -- concurrent accesses that serialize in silicon
 15. ownership                 -- a linear value has exactly one owner
 16. contracts                 -- a published contract is implied by its body
 17. exhaustiveness            -- every variant of a match is handled

Checks 3, 8, 10, 11 and 12 are not separate subsystems for separate domains.
Contact bounce on a switch, setup time on a trace, response time on a display
panel and row activation in a DRAM bank are all check 10 with different
numbers. A fanout margin and a thrust margin are both check 4 over a typed
physical quantity. The partition is by SUBSTRATE and PHYSICAL LAW; never by
engineering discipline, because discipline boundaries are an artifact of how
humans divide labour and they do not appear anywhere in the physics.
"""

import math

from .cost import total_cost
from .ir import Contract, Func, declared_params
from .types import (Arr, Bool, Buf, Bytes, Cap, DAL_RANK, EPS, Int, Real,
                    Rec, Rot, SITES, Sum, Unit, Vec, elem_unit, fp_fmt,
                    is_linear)
from math import gcd

INF = float("inf")


# ============================================================  1/4 intervals
# Small, conservative, and the basis for discharging obligations. "Unbounded"
# is a legitimate answer; it simply fails to discharge anything.

class IV:
    """An interval, and -- for a record or a variant -- a TREE of intervals.

    A record's abstract value is one interval per field, and a variant's is
    one per possible tag. `lo`/`hi` on a composite are the hull of its
    leaves, so every rule written for scalars stays sound on composites by
    widening instead of by crashing. Reading a field recovers the exact
    field interval; doing arithmetic on a whole record cannot typecheck, so
    the widening is never the thing a proof rests on.
    """

    __slots__ = ("lo", "hi", "parts", "arms")

    def __init__(self, lo=-INF, hi=INF, parts=None, arms=None):
        self.lo, self.hi = float(lo), float(hi)
        self.parts = parts          # tuple of IV, for a record
        self.arms = arms            # {tag: IV}, for a variant

    @staticmethod
    def tup(parts):
        parts = tuple(parts)
        if not parts:
            return IV(0.0, 0.0, parts=())
        return IV(min(p.lo for p in parts), max(p.hi for p in parts),
                  parts=parts)

    @staticmethod
    def tagged(tag, v):
        return IV(v.lo, v.hi, arms={tag: v})

    def part(self, i):
        if self.parts is None or i >= len(self.parts):
            return IV()
        return self.parts[i]

    def arm(self, tag):
        if self.arms is None:
            return IV()
        return self.arms.get(tag, IV())

    def inside(self, lo, hi):
        return self.lo >= lo and self.hi <= hi

    @staticmethod
    def at(v):
        return IV(v, v)

    @staticmethod
    def sym(b):
        return IV(-abs(b), abs(b))

    def __repr__(self):
        return "[%g,%g]" % (self.lo, self.hi)

    @property
    def amax(self):
        return max(abs(self.lo), abs(self.hi))

    @property
    def amin(self):
        return 0.0 if (self.lo <= 0.0 <= self.hi) else min(abs(self.lo), abs(self.hi))

    def excludes_zero(self):
        return self.lo > 0.0 or self.hi < 0.0

    def nonneg(self):
        return self.lo >= 0.0

    def within(self, n):
        return self.lo >= 0.0 and self.hi <= n - 1

    def add(self, o):
        return IV(self.lo + o.lo, self.hi + o.hi)

    def sub(self, o):
        return IV(self.lo - o.hi, self.hi - o.lo)

    def mul(self, o):
        c = [self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi]
        c = [x for x in c if not math.isnan(x)]
        return IV(min(c), max(c)) if c else IV()

    def div(self, o):
        if not o.excludes_zero():
            return IV()
        c = [self.lo / o.lo, self.lo / o.hi, self.hi / o.lo, self.hi / o.hi]
        return IV(min(c), max(c))

    def neg(self):
        return IV(-self.hi, -self.lo)

    def sqrt(self):
        return IV(math.sqrt(max(self.lo, 0.0)), math.sqrt(max(self.hi, 0.0)))

    def hull(self, o):
        return IV(min(self.lo, o.lo), max(self.hi, o.hi))


# ===============================================================  2/4 report

class V:
    """A violation. Carries the node so the generator can be told precisely
    what to regenerate."""

    def __init__(self, kind, addr, msg):
        self.kind, self.addr, self.msg = kind, addr, msg

    def __repr__(self):
        return "%-12s %s  %s" % (self.kind, (self.addr or "")[:6].ljust(6), self.msg)


class Report:
    def __init__(self, name):
        self.name = name
        self.violations = []
        self.obligations = []      # (addr, kind, detail, discharged, why)
        self.ty = {}
        self.iv = {}
        self.err = {}              # proven RELATIVE error bound
        self.aerr = {}             # proven ABSOLUTE error bound
        self.dal = {}
        self.effects = set()
        self.binds = []            # (addr, unit, lanes_used, commanded)
        self.calls = []            # (addr, contract, verified)
        self.linear = []           # (addr, ty, uses)
        self.conflicts = []        # (addr, site, lanes, stride, factor)
        self.cost = None

    def bad(self, kind, addr, msg):
        self.violations.append(V(kind, addr, msg))

    def oblige(self, addr, kind, detail, discharged, why):
        self.obligations.append((addr, kind, detail, discharged, why))
        if not discharged:
            self.bad("obligation", addr, "%s: %s (%s)" % (kind, detail, why))

    @property
    def ok(self):
        return not self.violations

    def summary(self):
        n_ob = len(self.obligations)
        n_ok = sum(1 for o in self.obligations if o[3])
        head = "%s  %s   obligations %d/%d discharged" % (
            "ACCEPT" if self.ok else "REJECT", self.name, n_ok, n_ob)
        if self.violations:
            head += "\n" + "\n".join("  " + repr(v) for v in self.violations)
        return head


# =====================================================  3/4 type derivation
# Leaf types (const/param/loopvar) are AXIOMS -- declarations the generator
# makes and the signer vouches for. Every interior node's type is DERIVED here
# and compared against what the untrusted builder claimed.

def _shape(t):
    """Type identity for add/sub compatibility: format, unit and frame must
    match exactly. Reproducibility and error are computed, not required."""
    if isinstance(t, Real):
        return ("real", t.fmt, t.unit, t.frame)
    if isinstance(t, Vec):
        return ("vec", t.n, t.elem.fmt, t.elem.unit, t.frame)
    if isinstance(t, Arr):
        return ("arr", t.shape, _shape(t.elem), t.level)
    return t.canon()


def _eps(t):
    return EPS.get(fp_fmt(t), 0.0)


def _derive(rep, node, addr, argtys, argivs):
    """Returns (declared-or-derived type, result interval).
    Appends violations and obligations to `rep`."""
    op, at = node.op, node.attrs
    bad = lambda k, m: rep.bad(k, addr, m)

    # ---------------------------------------------------------- leaf axioms
    if op in ("const", "param", "loopvar"):
        t = node.ty
        if op == "const":
            v = at.get("value")
            return t, (IV.at(v) if isinstance(v, (int, float)) else IV())
        if op == "loopvar":
            b = at.get("bound")
            if not (isinstance(b, int) and b > 0):
                bad("bounds", "loopvar bound must be a positive literal")
                return t, IV()
            return t, IV(0, b - 1)
        lo, hi = at.get("lo"), at.get("hi")
        if isinstance(t, Arr) and (lo is None or hi is None):
            # No element bounds means no magnitude bound, which means no
            # provable error bound for any reduction over this array.
            bad("bounds", "array parameter %r must declare element bounds "
                          "(lo/hi)" % at.get("name"))
        return t, IV(-INF if lo is None else lo, INF if hi is None else hi)

    a0 = argtys[0] if argtys else None
    a1 = argtys[1] if len(argtys) > 1 else None

    # ------------------------------------------------------ add / sub (2)(3)
    if op in ("add", "sub"):
        if _shape(a0) != _shape(a1):
            fa, fb = getattr(a0, "frame", None), getattr(a1, "frame", None)
            ua, ub = elem_unit(a0), elem_unit(a1)
            if fa != fb:
                bad("frame", "%s mixes frames %s and %s; route one operand "
                             "through a declared change of frame (rot_apply "
                             "for a spatial frame, sync for a time domain)"
                    % (op, fa, fb))
            elif ua and ub and ua.same_dim(ub) and ua != ub:
                # THE Mars Climate Orbiter check. Dimensions agree; only the
                # scale differs, which is exactly what dimension-only analysis
                # misses and what cost the mission.
                bad("unit", "%s mixes %r and %r: same dimension, scale differs "
                            "by %.10gx. An explicit convert is required."
                    % (op, ua, ub, float(ub.scale / ua.scale)))
            else:
                bad("type", "%s requires identical operands: %r vs %r"
                    % (op, a0, a1))
            return a0, IV()
        ia = argivs[0]
        ib = argivs[1] if op == "add" else argivs[1].neg()
        if isinstance(a0, Real):
            t = Real(a0.fmt, a0.unit, a0.frame, a0.repro and a1.repro)
        elif isinstance(a0, Vec):
            t = Vec(Real(a0.elem.fmt, a0.elem.unit, a0.frame,
                         a0.elem.repro and a1.elem.repro), a0.n, a0.frame)
        else:
            bad("type", "%s is not defined on %r" % (op, a0))
            t = a0
        return t, ia.add(ib)

    # --------------------------------------------------------- mul / div (2)
    if op in ("mul", "div"):
        if not (isinstance(a0, Real) and isinstance(a1, Real)):
            bad("type", "%s expects two scalars, got %r, %r" % (op, a0, a1))
            return a0, IV()
        if a0.frame and a1.frame and a0.frame != a1.frame:
            bad("frame", "%s mixes frames %s and %s" % (op, a0.frame, a1.frame))
        u = a0.unit * a1.unit if op == "mul" else a0.unit / a1.unit
        if op == "div":
            # (4) the obligation that replaces v1's bottom value
            ok = argivs[1].excludes_zero()
            rep.oblige(addr, "nonzero", "divisor %r" % (argivs[1],), ok,
                       "interval excludes 0" if ok else
                       "cannot prove divisor != 0 from declared ranges")
        iv = argivs[0].mul(argivs[1]) if op == "mul" else argivs[0].div(argivs[1])
        return Real(a0.fmt, u, a0.frame or a1.frame, a0.repro and a1.repro), iv

    # ----------------------------------------------------------- fma (2)(3)
    if op == "fma":
        a2 = argtys[2] if len(argtys) > 2 else None
        if not all(isinstance(t, Real) for t in (a0, a1, a2)):
            bad("type", "fma expects three scalars, got %r, %r, %r"
                % (a0, a1, a2))
            return a0, IV()
        u = a0.unit * a1.unit
        if u != a2.unit:
            if u.same_dim(a2.unit):
                # The Mars Climate Orbiter check again, in the one place a
                # fused operation could have hidden it.
                bad("unit", "fma adds %r to a product of %r: same dimension, "
                            "scale differs by %.10gx"
                    % (a2.unit, u, float(a2.unit.scale / u.scale)))
            else:
                bad("unit", "fma adds %r to a product of %r" % (a2.unit, u))
        frames = sorted({t.frame for t in (a0, a1, a2) if t.frame})
        if len(frames) > 1:
            bad("frame", "fma mixes frames %s" % ", ".join(frames))
        iv = argivs[0].mul(argivs[1]).add(argivs[2])
        return Real(a0.fmt, u, frames[0] if frames else None,
                    a0.repro and a1.repro and a2.repro), iv

    if op == "neg":
        return a0, argivs[0].neg()

    # ================================================== records (1)(2)
    if op == "rec":
        t = at.get("rty")
        if not isinstance(t, Rec):
            bad("type", "rec needs a record type")
            return node.ty, IV()
        if len(argtys) != len(t.fields):
            bad("type", "record %s has %d field(s), %d given"
                % (t.name, len(t.fields), len(argtys)))
            return t, IV()
        for (fname, fty), got in zip(t.fields, argtys):
            if _shape(fty) != _shape(got):
                bad("type", "field %s.%s is %r, got %r"
                    % (t.name, fname, fty, got))
        return t, IV.tup(argivs)

    if op == "field":
        if not isinstance(a0, Rec):
            bad("type", "field needs a record, got %r" % (a0,))
            return node.ty, IV()
        i = a0.index(at.get("f"))
        if i is None:
            bad("type", "record %s has no field %r" % (a0.name, at.get("f")))
            return node.ty, IV()
        return a0.fields[i][1], argivs[0].part(i)

    # =================================================== variants (1)(4)
    if op == "inject":
        t = at.get("sty")
        tag = at.get("tag")
        if not isinstance(t, Sum) or t.ty_of(tag) is None:
            bad("type", "no variant %r in %r" % (tag, t))
            return node.ty, IV()
        if _shape(t.ty_of(tag)) != _shape(a0):
            bad("type", "variant %s.%s carries %r, got %r"
                % (t.name, tag, t.ty_of(tag), a0))
        return t, IV.tagged(tag, argivs[0])

    if op == "match":
        if not isinstance(a0, Sum):
            bad("type", "match needs a variant, got %r" % (a0,))
            return node.ty, IV()
        arms = dict(at.get("arms") or ())
        want, have = set(a0.tags), set(arms)
        if want != have:
            # (4) EXHAUSTIVENESS. An unhandled variant is the single most
            # common way a program meets a state its author never considered,
            # and it is a counting problem, so a machine should do the
            # counting.
            miss, extra = sorted(want - have), sorted(have - want)
            bad("match", "match on %s is not exhaustive%s%s" % (
                a0.name,
                "; unhandled: %s" % miss if miss else "",
                "; no such variant: %s" % extra if extra else ""))
            return at.get("rty"), IV()
        ret = at.get("rty")
        acc = None
        for tag in a0.tags:
            c = arms[tag]
            if len(c.params) != 1:
                bad("match", "arm %r must take exactly the payload" % tag)
                continue
            pname, pty, plo, phi = c.params[0]
            if _shape(pty) != _shape(a0.ty_of(tag)):
                bad("type", "arm %r takes %r but %s.%s carries %r"
                    % (tag, pty, a0.name, tag, a0.ty_of(tag)))
            armiv = argivs[0].arm(tag)
            if isinstance(a0.ty_of(tag), Real) and not armiv.inside(plo, phi):
                rep.oblige(addr, "arm-domain",
                           "arm %r over %r" % (tag, armiv), False,
                           "payload may be outside the arm's proven box "
                           "[%g,%g]" % (plo, phi))
            if _shape(c.ret) != _shape(ret):
                bad("type", "arm %r returns %r, match returns %r"
                    % (tag, c.ret, ret))
            civ = IV(c.lo, c.hi)
            acc = civ if acc is None else acc.hull(civ)
        return ret, (acc if acc is not None else IV())

    # ===================================================== calling (1)(4)
    if op == "call":
        c = at.get("fn")
        if not isinstance(c, Contract):
            bad("type", "call needs a contract")
            return node.ty, IV()
        if len(argtys) != len(c.params):
            bad("type", "%s takes %d argument(s), %d given"
                % (c.name, len(c.params), len(argtys)))
            return c.ret, IV(c.lo, c.hi)
        for (pname, pty, plo, phi), got, giv in zip(c.params, argtys, argivs):
            if _shape(pty) != _shape(got):
                bad("type", "%s expects %s:%r, got %r" % (c.name, pname,
                                                          pty, got))
                continue
            if isinstance(pty, (Real, Int)):
                # THE PRECONDITION, and the only one. The body was proven
                # over this box; outside it, nothing was proven at all.
                ok = giv.inside(plo, phi)
                rep.oblige(addr, "precondition",
                           "%s(%s=%r)" % (c.name, pname, giv), ok,
                           "inside [%g,%g]" % (plo, phi) if ok else
                           "argument may leave the box [%g,%g] the callee "
                           "was proven over" % (plo, phi))
        for e in c.effects:
            rep.effects.add(e)
        return c.ret, IV(c.lo, c.hi)

    # ============================================ bounded iteration (4)
    if op == "fold":
        c = at.get("fn")
        n = at.get("n")
        lo, hi = at.get("lo"), at.get("hi")
        if not isinstance(c, Contract):
            bad("type", "fold needs a body contract")
            return node.ty, IV()
        if not (isinstance(n, int) and n > 0):
            bad("bounds", "fold needs a positive literal trip count")
            return c.ret, IV()
        if len(c.params) != 2:
            bad("type", "a fold body takes (acc, i), got %d parameter(s)"
                % len(c.params))
            return c.ret, IV()
        (an, aty, alo, ahi), (inm, ity, ilo, ihi) = c.params
        if _shape(aty) != _shape(c.ret):
            bad("type", "a fold body must return what it accumulates: "
                        "%r vs %r" % (aty, c.ret))
        if _shape(a0) != _shape(aty):
            bad("type", "fold starts from %r but the body accumulates %r"
                % (a0, aty))
        if not isinstance(ity, Int):
            bad("type", "a fold body's second parameter is the index and "
                        "must be an integer, got %r" % (ity,))
        elif not IV(0, n - 1).inside(ilo, ihi):
            rep.oblige(addr, "index-domain", "fold index over [0,%d]" % (n - 1),
                       False, "the body was proven only over [%g,%g]"
                       % (ilo, ihi))
        # INDUCTION, in two obligations and nothing else.
        base = argivs[0].inside(lo, hi)
        rep.oblige(addr, "invariant-base", "initial %r" % (argivs[0],), base,
                   "starts inside [%g,%g]" % (lo, hi) if base else
                   "the initial value is outside the declared invariant "
                   "[%g,%g]" % (lo, hi))
        step = IV(c.lo, c.hi).inside(lo, hi) and IV(lo, hi).inside(alo, ahi)
        rep.oblige(addr, "invariant-step",
                   "body maps [%g,%g] to [%g,%g]" % (lo, hi, c.lo, c.hi),
                   step,
                   "the invariant is preserved" if step else
                   "the body was proven over [%g,%g] and returns [%g,%g]; "
                   "that does not map the invariant [%g,%g] into itself"
                   % (alo, ahi, c.lo, c.hi, lo, hi))
        for e in c.effects:
            rep.effects.add(e)
        return c.ret, IV(lo, hi)

    # ================================================ linear buffers (1)
    if op == "alloc":
        t = at.get("bty")
        if not isinstance(t, Buf):
            bad("type", "alloc needs a buffer type")
            return node.ty, IV()
        s = SITES.get(t.level)
        if s is None:
            bad("substrate", "buffer lives at unknown site %r" % t.level)
        return t, IV.at(float(at.get("fill", 0.0)))

    if op == "put":
        if not isinstance(a0, Buf):
            bad("type", "put needs a buffer, got %r" % (a0,))
            return node.ty, IV()
        if not isinstance(a1, Int):
            bad("type", "a buffer index must be an integer, got %r" % (a1,))
        else:
            ok = argivs[1].within(a0.count)
            rep.oblige(addr, "index", "put index %r into %d element(s)"
                       % (argivs[1], a0.count), ok,
                       "index is in range" if ok else
                       "cannot prove 0 <= i < %d" % a0.count)
        vt = argtys[2] if len(argtys) > 2 else None
        if vt is not None and _shape(vt) != _shape(a0.elem):
            bad("type", "buffer holds %r, writing %r" % (a0.elem, vt))
        # the buffer's element range widens to include what was written
        return a0, argivs[0].hull(argivs[2] if len(argivs) > 2 else IV())

    if op == "get":
        if not isinstance(a0, Buf):
            bad("type", "get needs a buffer, got %r" % (a0,))
            return node.ty, IV()
        if not isinstance(a1, Int):
            bad("type", "a buffer index must be an integer, got %r" % (a1,))
        else:
            ok = argivs[1].within(a0.count)
            rep.oblige(addr, "index", "get index %r from %d element(s)"
                       % (argivs[1], a0.count), ok,
                       "index is in range" if ok else
                       "cannot prove 0 <= i < %d" % a0.count)
        return a0.elem, argivs[0]

    if op == "freeze":
        if not isinstance(a0, Buf):
            bad("type", "freeze needs a buffer, got %r" % (a0,))
            return node.ty, IV()
        return a0.frozen_ty(), argivs[0]

    # ================================================== capabilities (9)
    if op == "perform":
        if not isinstance(a0, Cap):
            bad("capability", "perform needs a capability, got %r. There is "
                              "no ambient authority in KEEL: an effect is "
                              "reachable only by spending a token that was "
                              "passed in." % (a0,))
            return node.ty, IV()
        eff = at.get("effect")
        if not eff:
            bad("capability", "perform must name its effect")
        rep.effects.add(eff)
        if eff and not eff.startswith(a0.kind):
            bad("capability", "capability %r does not authorise effect %r"
                % (a0.kind, eff))
        return node.ty, IV()

    # ======================================================== bytes (1)(4)
    if op == "bstr":
        data = at.get("data") or []
        return Bytes(len(data), at.get("enc", "bytes")), IV(0, 255)

    if op == "bslice":
        if not isinstance(a0, Bytes):
            bad("type", "bslice needs bytes, got %r" % (a0,))
            return node.ty, IV()
        off, n = at.get("off"), at.get("n")
        if not (isinstance(off, int) and isinstance(n, int)
                and off >= 0 and n >= 0 and off + n <= a0.n):
            # Every buffer overflow is a length known in one place and not in
            # another. Here there is one place.
            bad("bounds", "bslice [%r,+%r) leaves a %d-byte string"
                % (off, n, a0.n))
            return Bytes(max(n or 0, 0), a0.enc), IV()
        return Bytes(n, a0.enc), argivs[0]

    if op == "bcat":
        if not (isinstance(a0, Bytes) and isinstance(a1, Bytes)):
            bad("type", "bcat needs bytes")
            return node.ty, IV()
        if a0.enc != a1.enc:
            # utf8 and latin1 are the same bytes and different text. The
            # unit check, applied to strings.
            bad("unit", "bcat joins %s and %s; an encoding is a unit and "
                        "there is no implicit conversion" % (a0.enc, a1.enc))
        return Bytes(a0.n + a1.n, a0.enc), argivs[0].hull(argivs[1])

    if op == "blen":
        if not isinstance(a0, Bytes):
            bad("type", "blen needs bytes, got %r" % (a0,))
            return node.ty, IV()
        return Int(), IV.at(a0.n)

    if op == "beq":
        if not (isinstance(a0, Bytes) and isinstance(a1, Bytes)):
            bad("type", "beq needs bytes")
        return Bool(), IV(0, 1)

    if op == "bhash":
        if not isinstance(a0, Bytes):
            bad("type", "bhash needs bytes, got %r" % (a0,))
            return node.ty, IV()
        bits = at.get("bits", 64)
        return Int(), IV(0, float(2 ** bits - 1))

    # -------------------------------------------------------------- sqrt (4)
    if op == "sqrt":
        if any(d % 2 for d in a0.unit.dim):
            bad("unit", "sqrt of %r has non-integral dimensions" % a0.unit)
        if a0.unit.scale != 1:
            bad("unit", "sqrt requires an SI-scaled operand, got %r" % a0.unit)
        ok = argivs[0].nonneg()
        rep.oblige(addr, "nonneg", "sqrt operand %r" % (argivs[0],), ok,
                   "interval is non-negative" if ok else
                   "cannot prove operand >= 0 from declared ranges")
        u = Unit(tuple(d // 2 for d in a0.unit.dim), a0.unit.scale)
        return Real(a0.fmt, u, a0.frame, a0.repro), argivs[0].sqrt()

    # ----------------------------------------------------------- convert (2)
    if op == "convert":
        to = at.get("to")
        if not isinstance(to, Unit):
            bad("unit", "convert needs a target unit")
            return a0, IV()
        if not a0.unit.same_dim(to):
            bad("unit", "convert %r -> %r changes dimension" % (a0.unit, to))
        k = float(a0.unit.scale / to.scale)
        return Real(a0.fmt, to, a0.frame, a0.repro), argivs[0].mul(IV.at(k))

    # ------------------------------------------------------- vectors / frames
    if op == "vec3":
        fr = at.get("frame")
        if len({_shape(t) for t in argtys}) != 1:
            bad("type", "vec3 components disagree: %r" % (argtys,))
        c = argtys[0]
        iv = argivs[0].hull(argivs[1]).hull(argivs[2])
        return Vec(Real(c.fmt, c.unit, fr, all(t.repro for t in argtys)), 3, fr), iv

    if op == "rot_apply":
        if not isinstance(a0, Rot):
            bad("type", "rot_apply expects a rotation, got %r" % a0)
            return a1, IV()
        if a1.frame != a0.frm:
            bad("frame", "rot<%s->%s> cannot be applied to a vector in %s"
                % (a0.frm, a0.to, a1.frame))
        return Vec(Real(a1.elem.fmt, a1.elem.unit, a0.to, a1.elem.repro),
                   a1.n, a0.to), IV.sym(a1.n * argivs[1].amax)

    if op == "dot":
        if a0.frame != a1.frame:
            bad("frame", "dot mixes frames %s and %s" % (a0.frame, a1.frame))
        if a0.n != a1.n:
            bad("type", "dot length %d vs %d" % (a0.n, a1.n))
        bound = a0.n * argivs[0].amax * argivs[1].amax
        return Real(a0.elem.fmt, a0.elem.unit * a1.elem.unit, None,
                    a0.elem.repro and a1.elem.repro), IV.sym(bound)

    if op == "norm":
        return Real(a0.elem.fmt, a0.elem.unit, None, a0.elem.repro), \
            IV(0, math.sqrt(a0.n) * argivs[0].amax)

    if op == "vscale":
        return Vec(Real(a0.elem.fmt, a0.elem.unit * a1.unit, a0.frame,
                        a0.elem.repro and a1.repro), a0.n, a0.frame), \
            IV.sym(argivs[0].amax * argivs[1].amax)

    # ------------------------------------------------------------- memory (8)
    if op == "load":
        if not isinstance(a0, Arr):
            bad("type", "load expects an array, got %r" % a0)
            return a0, IV()
        ok = argivs[1].within(a0.count)
        rep.oblige(addr, "inrange", "index %r into %d" % (argivs[1], a0.count),
                   ok, "index interval within bounds" if ok else
                   "cannot prove 0 <= i < %d" % a0.count)
        return a0.elem, argivs[0]

    if op == "stencil":
        offs, co = at.get("offsets", []), at.get("coeffs", [])
        if len(offs) != len(co):
            bad("type", "stencil has %d offsets but %d coefficients"
                % (len(offs), len(co)))
        if max((abs(o) for o in offs), default=0) > 0 and \
                at.get("boundary") not in ("clamp", "periodic", "shrink"):
            bad("bounds", "stencil reads outside the array and declares no "
                          "boundary policy (clamp|periodic|shrink)")
        lvl = at.get("out_level", a0.level)
        if lvl not in SITES:
            bad("substrate", "unknown site %r" % lvl)
        sc = sum(abs(c) for c in co) or 1.0
        return Arr(a0.elem, a0.shape, lvl), IV.sym(sc * argivs[0].amax)

    if op == "reduce":
        assoc = at.get("assoc")
        if assoc not in ("ordered", "tree", "nondet"):
            bad("repro", "reduce needs assoc in {ordered,tree,nondet}")
            assoc = "nondet"
        el = a0.elem
        return Real(el.fmt, el.unit, el.frame, assoc != "nondet"), \
            IV.sym(a0.count * argivs[0].amax)

    # ------------------------------------------------------- refinement (4)
    if op == "clamp":
        lo, hi = at.get("lo"), at.get("hi")
        if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))
                and lo <= hi):
            bad("bounds", "clamp needs literal lo <= hi")
            return a0, IV()
        # Saturation is how flight software actually narrows a range. Here the
        # type system gets to know about it, so a downstream divide-by-zero
        # obligation becomes dischargeable instead of unprovable.
        return a0, IV(max(argivs[0].lo, lo), min(argivs[0].hi, hi))

    # ------------------------------------------------------------ control (4)
    if op == "repeat":
        b = at.get("bound")
        if not (isinstance(b, int) and b > 0):
            bad("bounds", "repeat bound must be a positive literal (no "
                          "unbounded loops in KEEL)")
        return a0, argivs[0]

    if op == "monitor":
        if at.get("dal") not in DAL_RANK:
            bad("dal", "monitor needs a target DAL")
        if not at.get("check"):
            bad("dal", "monitor must declare the runtime check it performs")
        return a0, argivs[0]

    if op == "cmp":
        if _shape(a0) != _shape(a1):
            bad("type", "cmp on %r vs %r" % (a0, a1))
        return Bool(), IV(0, 1)

    if op == "select":
        if not isinstance(a0, Bool):
            bad("type", "select condition must be bool")
        if _shape(a1) != _shape(argtys[2]):
            bad("type", "select branches disagree: %r vs %r" % (a1, argtys[2]))
        return a1, argivs[1].hull(argivs[2])

    if op == "sys":
        rep.effects.add(at.get("effect"))
        return node.ty, IV()

    # ------------------------------------------------- the physical world (10)
    if op == "sense":
        s = SITES.get(at.get("site"))
        if s is None:
            bad("substrate", "unknown site %r" % at.get("site"))
            return node.ty, IV()
        if not s.readback:
            bad("substrate", "site %r is write-only and cannot be sensed"
                % s.name)
        after = at.get("after")
        if not isinstance(after, (int, float)):
            rep.oblige(addr, "settle", "sense of %s" % s.name, False,
                       "no elapsed time declared; a sense must state how long "
                       "after the last transition it samples")
        else:
            ok = after >= s.settle
            rep.oblige(addr, "settle",
                       "sense %s at t+%.4gs (settles in %.4gs)"
                       % (s.name, after, s.settle), ok,
                       "elapsed time exceeds the settling time" if ok else
                       "samples %.4gs early; wait at least %.4gs"
                       % (s.settle - after, s.settle))
        # The time domain comes from the SITE, never from the builder's claim.
        t = node.ty
        if isinstance(t, Real):
            t = Real(t.fmt, t.unit, s.clock, t.repro)
        lo, hi = at.get("lo"), at.get("hi")
        return t, IV(-INF if lo is None else lo, INF if hi is None else hi)

    # ------------------------------------------------------------- drive (11)
    if op == "drive":
        s = SITES.get(at.get("site"))
        if s is None:
            bad("substrate", "unknown site %r" % at.get("site"))
            return a0, IV()
        rep.effects.add(at.get("effect"))
        n = at.get("loads", 1)
        per = at.get("load_a")
        per = s.load_a if per is None else per
        need = n * per
        # An obligation exists only where some declared physics creates one.
        # A socket has no amperes, and demanding them would be this language
        # importing one discipline's quantities into every substrate -- the
        # exact mistake the site lattice exists to avoid. Current has to be
        # proven when current is actually drawn, and not otherwise.
        if need > 0.0 or s.drive_a > 0.0:
            ok = s.drive_a > 0.0 and need <= s.drive_a
            rep.oblige(addr, "drive",
                       "%s sources %.4gA, %d load(s) draw %.4gA"
                       % (s.name, s.drive_a, n, need), ok,
                       "within the drive capability" if ok else
                       ("site %r sources no current, but %.4gA is drawn from "
                        "it" % (s.name, need) if s.drive_a <= 0.0 else
                        "overloaded by %.4gA; split the net or buffer it"
                        % (need - s.drive_a)))
        return a0, argivs[0]

    # ----------------------------------- clock domains are frames for time (3)
    if op == "sync":
        to, depth = at.get("to"), at.get("depth")
        if not (isinstance(depth, int) and depth >= 2):
            bad("frame", "a clock-domain crossing needs a synchronizer of "
                         "depth >= 2; depth %r leaves the receiving flop "
                         "metastable" % (depth,))
        if getattr(a0, "frame", None) == to:
            bad("frame", "sync to %r is a no-op: the value is already in that "
                         "time domain" % (to,))
        return Real(a0.fmt, a0.unit, to, a0.repro), argivs[0]

    # ------------------------------ every physical invariant, one operation (4)
    if op == "require":
        lo, hi = at.get("lo"), at.get("hi")
        u, why = at.get("unit"), at.get("why")
        ua = elem_unit(a0)
        if not isinstance(u, Unit):
            bad("unit", "require must declare the unit its bound is written in")
        elif ua != u:
            if ua is not None and ua.same_dim(u):
                # The Mars Climate Orbiter failure mode, now on invariants.
                bad("unit", "require %r bounds in %r but the value is %r: same "
                            "dimension, scale differs by %.10gx"
                    % (why, u, ua, float(ua.scale / u.scale)))
            else:
                bad("unit", "require %r bounds in %r but the value is %r"
                    % (why, u, ua))
        iv = argivs[0]
        lo = -INF if lo is None else float(lo)
        hi = INF if hi is None else float(hi)
        ok = iv.lo >= lo and iv.hi <= hi
        rep.oblige(addr, "invariant",
                   "%s: %r must lie in [%g,%g]" % (why, iv, lo, hi), ok,
                   "proven interval is inside the bound" if ok else
                   "proven interval %r leaves [%g,%g] by %.4g"
                   % (iv, lo, hi, max(lo - iv.lo, iv.hi - hi)))
        # require PROVES and does not narrow; clamp narrows by saturating.
        return a0, iv

    # ------------------------------- the unit that performs the operation (13)
    if op == "bind":
        s = SITES.get(at.get("unit"))
        if s is None:
            bad("substrate", "unknown execution unit %r" % at.get("unit"))
            return a0, argivs[0]
        if not s.arith:
            bad("substrate", "site %r performs no arithmetic; it cannot be "
                             "bound as an execution unit" % s.name)
        f = fp_fmt(a0)
        if s.fmts and f not in s.fmts:
            # Not a convention. A unit that carries f16 through its datapath
            # has no f64 to issue, and no amount of declaring makes one.
            bad("substrate", "unit %r carries %s; this operation is %s"
                % (s.name, "/".join(s.fmts), f))
        want = at.get("lanes", 1)
        if not (isinstance(want, int) and want >= 1):
            bad("substrate", "bind must declare how many lanes it uses")
        elif s.lanes and want > s.lanes:
            bad("substrate", "unit %r has %d lane(s); %d requested"
                % (s.name, s.lanes, want))
        else:
            rep.binds.append((addr, s.name, want, not s.ooo))
        # A unit with its own clock puts its result in that time domain, so
        # mixing it with a core-clocked value needs a sync. The rule written
        # for ECI + BODY is doing the CPU/fabric boundary with no new code.
        t = a0
        if s.clock and isinstance(t, Real):
            t = Real(t.fmt, t.unit, s.clock, t.repro)
        return t, argivs[0]

    bad("op", "unknown operation %r" % op)
    return node.ty, IV()


# ====================================================  4/4 error propagation
# Two domains, because neither alone is honest.
#
#   RELATIVE error is preserved by multiplicative operations (mul, div, sqrt,
#   convert, scaling) and is the right currency there.
#
#   ABSOLUTE error is the right currency for additive operations, because a
#   sum that can straddle zero has NO finite relative bound -- that is
#   catastrophic cancellation, and pretending otherwise is how you ship a bug.
#
# Each op reports in whichever domain it can justify, and the two tighten each
# other against the value's proven interval. A reduction reports absolute only,
# because |computed - exact| <= k*eps*sum|x_i| says nothing about the result's
# own magnitude. That is why a reduction needs a declared ABSOLUTE tolerance.

def _m(x, y):
    """Products where one factor is an unbounded magnitude and the other an
    exactly-zero error are zero, not NaN. Interval arithmetic has to say so
    out loud."""
    if x == 0.0 or y == 0.0:
        return 0.0
    return x * y


def _error(node, argtys, argivs, rel, aerr, ty, iv):
    op, at = node.op, node.attrs
    eps = _eps(ty)
    r, a = INF, INF

    if op in ("const", "loopvar"):
        r, a = 0.0, 0.0
    elif op == "param":
        a = float(at.get("err", 0.0) or 0.0)
        r = 0.0 if a == 0.0 else INF
    elif op in ("add", "sub"):
        ia = argivs[0]
        ib = argivs[1] if op == "add" else argivs[1].neg()
        a = aerr[0] + aerr[1] + eps * iv.amax
        if iv.amin > 0.0:
            r = (ia.amax * rel[0] + ib.amax * rel[1]) / iv.amin + eps
    elif op == "mul":
        r = rel[0] + rel[1] + eps
        # An ABSOLUTE bound as well. Without it, a product of anything that
        # can be zero had no error bound at all -- the relative domain
        # reports infinity there, and multiplying by an unbounded relative
        # error gave an unbounded absolute one. The standard first-order
        # bound is |b|.da + |a|.db + eps|ab| and it is finite whenever the
        # operands' absolute errors are.
        a = (_m(argivs[1].amax, aerr[0]) + _m(argivs[0].amax, aerr[1])
             + eps * iv.amax)
    elif op == "div":
        r = rel[0] + rel[1] + eps
        if argivs[1].amin > 0.0:
            a = ((aerr[0] + _m(iv.amax, aerr[1])) / argivs[1].amin
                 + eps * iv.amax)
    elif op == "fma":
        # ONE rounding, and that is the whole point. mul-then-add pays an eps
        # for the product and another for the sum; this pays one. The bound
        # is therefore strictly tighter, provably, not by assertion.
        ab = argivs[0].mul(argivs[1])
        a = (_m(argivs[1].amax, aerr[0]) + _m(argivs[0].amax, aerr[1])
             + aerr[2] + eps * iv.amax)
        if iv.amin > 0.0:
            r = (_m(ab.amax, rel[0] + rel[1]) + _m(argivs[2].amax, rel[2])) \
                / iv.amin + eps
    elif op == "neg":
        r, a = rel[0], aerr[0]
    elif op == "sqrt":
        r = rel[0] / 2.0 + eps
        # d(sqrt x) = dx / (2 sqrt x), which needs x bounded away from zero.
        # Near zero sqrt is infinitely steep and no absolute bound exists, so
        # none is claimed.
        if argivs[0].lo > 0.0:
            a = aerr[0] / (2.0 * math.sqrt(argivs[0].lo)) + eps * iv.amax
    elif op == "convert":
        k = abs(float(argtys[0].unit.scale / at["to"].scale))
        r = rel[0] + eps
        a = aerr[0] * k + eps * iv.amax
    elif op == "vec3":
        r, a = max(rel), max(aerr)
    elif op == "rot_apply":
        n = argtys[1].n
        r = rel[1] + n * eps
        a = n * aerr[1] + (2 * n - 1) * eps * iv.amax
    elif op == "dot":
        n = argtys[0].n
        a = n * (argivs[0].amax * aerr[1] + argivs[1].amax * aerr[0]) \
            + (2 * n - 1) * eps * iv.amax
    elif op == "norm":
        r = rel[0] / 2.0 + 2 * argtys[0].n * eps
    elif op == "vscale":
        r = rel[0] + rel[1] + eps
    elif op in ("load", "clamp", "repeat", "monitor", "drive", "sync",
                "require", "bind"):
        r, a = rel[0], aerr[0]
    elif op == "sense":
        # A sensed value carries the transducer error, which is absolute.
        a = float(at.get("err", 0.0) or 0.0)
        r = 0.0 if a == 0.0 else INF
    elif op == "stencil":
        sc = sum(abs(c) for c in at.get("coeffs", [])) or 1.0
        k = max(1, len(at.get("offsets", [])))
        a = sc * aerr[0] + (2 * k - 1) * eps * iv.amax
    elif op == "reduce":
        n = argtys[0].count
        assoc = at.get("assoc")
        # A tree reduction's error grows with log2(n); a sequential one with n.
        # Nobody computes this by hand, which is why nobody optimizes for it.
        k = max(1, int(math.ceil(math.log2(n)))) if assoc == "tree" \
            else max(1, n - 1)
        a = n * aerr[0] + k * eps * iv.amax
    elif op in ("rec", "inject", "field", "freeze", "get", "put"):
        r, a = (max(rel) if rel else 0.0), (max(aerr) if aerr else 0.0)
    elif op == "alloc":
        r, a = 0.0, 0.0
    elif op == "call":
        # The callee PROVED these. Adopting them is the whole point of a
        # contract; re-deriving the body here would be doing the work twice
        # and would make a thousand-function program a thousand times slower
        # to check.
        c = at.get("fn")
        r, a = c.rel, c.aerr
    elif op == "fold":
        c = at.get("fn")
        n = at.get("n", 1)
        # n applications of the body, each contributing its own absolute
        # error. Conservative, which is the only acceptable direction.
        a = n * c.aerr if c.aerr < INF else INF
        r = n * c.rel if c.rel < INF else INF
    elif op == "match":
        arms = dict(at.get("arms") or ())
        if arms:
            r = max(c.rel for c in arms.values())
            a = max(c.aerr for c in arms.values())
    elif op in ("bstr", "blen", "beq", "bhash", "bslice", "bcat", "perform"):
        r, a = 0.0, 0.0
    elif op == "cmp":
        r, a = 0.0, 0.0
    elif op == "select":
        r, a = max(rel[1], rel[2]), max(aerr[1], aerr[2])

    # the two domains tighten each other through the proven interval
    if iv.amin > 0.0 and a < INF:
        r = min(r, a / iv.amin)
    if r < INF and iv.amax < INF:
        a = min(a, r * iv.amax)
    return r, a


# ============================================================  the entry point

def contract_of(name, d, rep):
    """Mint the contract a checked body is ENTITLED to publish.

    Note what this is not: it is not a declaration the author writes and the
    checker tries to believe. It is read off the proof. A caller that adopts
    it is adopting something that was proven, which is why a call site does
    not have to re-derive the body.
    """
    return Contract(
        name=name, body=d.root, params=declared_params(d),
        ret=rep.ty[d.root], lo=rep.iv[d.root].lo, hi=rep.iv[d.root].hi,
        rel=rep.err[d.root], aerr=rep.aerr[d.root],
        cycles=max(rep.cost.cycles, rep.cost.throughput_cycles()),
        flops=rep.cost.flops, seconds=rep.cost.seconds,
        effects=rep.effects, dal=rep.dal[d.root])


def func(name, d, declare=None, machine=None, funcs=None):
    """Check a definition and turn it into something callable.

    A rejected definition does not become a function. That is the whole
    mechanism by which a large program stays proven: there is no way to get a
    callable value except through the checker.
    """
    rep = check(d, machine=machine, funcs=funcs)
    if not rep.ok:
        raise ValueError("cannot make a function of a rejected definition "
                         "%r: %s" % (name, rep.violations[0].msg))
    actual = contract_of(name, d, rep)
    use = actual
    if declare is not None:
        ok, why = declare.weaker_than(actual)
        if not ok:
            raise ValueError("declared contract for %r is not implied by its "
                             "body: %s" % (name, why))
        use = declare
    f = Func(use, d)
    f.report = rep
    f.proven = actual
    return f


def check(d, machine=None, funcs=None, _memo=None):
    """Check one definition. Returns a Report. Nothing else may load it.

    `funcs` supplies the bodies of everything this definition calls. Without
    them a contract cannot be verified, and an unverified contract is an
    UNDISCHARGED OBLIGATION rather than a convenience -- adopting a promise
    nobody checked would make modular verification a way of hiding work
    instead of a way of dividing it.
    """
    rep = Report(d.name)
    g = d.graph
    fmap = {}
    for f in ((funcs.values() if isinstance(funcs, dict) else funcs) or ()):
        fmap[f.contract.body] = f
    memo = {} if _memo is None else _memo

    for addr in g.topo(d.root):
        node = g[addr]
        argtys = [rep.ty[a] for a in node.args]
        argivs = [rep.iv[a] for a in node.args]

        ty, iv = _derive(rep, node, addr, argtys, argivs)

        # (1) the builder is untrusted: declared must equal derived
        if _shape(ty) != _shape(node.ty):
            rep.bad("declared", addr,
                    "%s declares %r but derives %r" % (node.op, node.ty, ty))

        # (5) proven error bounds, in both domains
        r, a = _error(node, argtys, argivs,
                      [rep.err[x] for x in node.args],
                      [rep.aerr[x] for x in node.args], ty, iv)

        rep.ty[addr], rep.iv[addr] = ty, iv
        rep.err[addr], rep.aerr[addr] = r, a

        # (6) mixed criticality: a value is only as assured as its weakest input
        if node.op == "monitor":
            rep.dal[addr] = node.attrs.get("dal") or d.dal
        elif node.op in ("call", "fold"):
            # A result is only as assured as the weakest of its inputs AND
            # the function that produced it. Criticality crosses a call.
            cands = [rep.dal[x] for x in node.args]
            cands.append(node.attrs["fn"].dal)
            rep.dal[addr] = max(cands, key=lambda k: DAL_RANK[k])
        elif node.args:
            rep.dal[addr] = max((rep.dal[x] for x in node.args),
                                key=lambda k: DAL_RANK[k])
        else:
            rep.dal[addr] = node.attrs.get("dal") or d.dal

        # (8) arithmetic happens in registers; arrays are only touched by the
        # three operations that are allowed to move data
        if node.op not in ("load", "stencil", "reduce", "drive"):
            for t in argtys:
                if isinstance(t, Arr):
                    rep.bad("memory", addr,
                            "%s operates directly on %r; data must be loaded "
                            "to REG first" % (node.op, t))

        # (8) placement is about the SITE physics, not about a hierarchy of
        # caches. An array living somewhere unreadable cannot be loaded.
        for t in argtys:
            if isinstance(t, Arr):
                s = SITES.get(t.level)
                if s is None:
                    rep.bad("substrate", addr,
                            "array lives at unknown site %r" % t.level)
                elif node.op in ("load", "stencil", "reduce") \
                        and not s.readback:
                    rep.bad("substrate", addr,
                            "site %r is write-only; %s cannot read from it"
                            % (s.name, node.op))
                # (14) Bank conflicts. With B banks and stride s, the residues
                # {i*s mod B} form a subgroup of size B/gcd(s,B), so lanes
                # landing on one bank -- and therefore the serialization --
                # follow from arithmetic. This is why padding an array by one
                # element can triple a kernel's speed, and why no language
                # above assembly can tell you so.
                if s is not None and s.banks and node.op == "load":
                    lanes = node.attrs.get("lanes", 1)
                    stride = node.attrs.get("stride", 1)
                    if lanes > 1:
                        gb = gcd(abs(stride) or s.banks, s.banks)
                        distinct = max(1, min(lanes, s.banks // gb))
                        factor = -(-lanes // distinct)
                        rep.conflicts.append((addr, s.name, lanes, stride,
                                              factor))

    root_ty = rep.ty[d.root]
    reach = g.topo(d.root)

    # (15) OWNERSHIP. A linear value must have exactly one edge pointing at
    # it. That is the whole rule, and it is the whole of aliasing: two edges
    # to one buffer IS two names for one mutable thing, and a
    # content-addressed graph already counts edges, so nothing had to be
    # invented. Rust's borrow checker is a fan-out count here.
    uses = {}
    for a in reach:
        for dep in g[a].args:
            uses[dep] = uses.get(dep, 0) + 1
    for a in reach:
        t = rep.ty.get(a)
        if not is_linear(t):
            continue
        u = uses.get(a, 0)
        rep.linear.append((a, t, u))
        if u == 1 or (a == d.root and u == 0):
            continue
        if u == 0:
            rep.bad("ownership", a,
                    "%r is produced and never consumed; a linear value must "
                    "be spent" % t)
        else:
            rep.bad("ownership", a,
                    "%r is used %d times. A linear value has exactly one "
                    "owner, and two edges to one buffer is aliasing -- which "
                    "is why the rule is a count and not an analysis" % (t, u))

    # (16) CONTRACTS. Every adopted contract is verified against the body it
    # claims to describe, and the check is memoised BY ADDRESS -- which is
    # sound without any invalidation logic, because an address commits to the
    # entire body. The same function called a thousand times is checked once.
    for a in reach:
        node = g[a]
        cs = []
        if node.op in ("call", "fold"):
            cs = [node.attrs.get("fn")]
        elif node.op == "match":
            cs = [c for _, c in (node.attrs.get("arms") or ())]
        for c in cs:
            if not isinstance(c, Contract):
                continue
            f = fmap.get(c.body)
            if f is None:
                rep.oblige(a, "contract", c.name, False,
                           "the body of %r was not supplied, so its contract "
                           "could not be verified (pass funcs=[...])" % c.name)
                continue
            if c.body in memo and memo[c.body] is None:
                # No recursion. Not a restriction for its own sake: an
                # unbounded call depth is an unbounded cost, and a program
                # without a cost bound is not a program this language admits.
                rep.bad("contract", a,
                        "%r is recursive; KEEL has no unbounded call depth, "
                        "because a WCET is not optional" % c.name)
                continue
            if c.body not in memo:
                memo[c.body] = None
                memo[c.body] = check(f.body, machine, funcs, memo)
            sub = memo[c.body]
            if not sub.ok:
                rep.bad("contract", a, "the body of %r does not check out: %s"
                        % (c.name, sub.violations[0].msg))
                continue
            ok, why = c.weaker_than(contract_of(c.name, f.body, sub))
            rep.oblige(a, "contract", c.name, ok,
                       "the body proves at least this" if ok else why)
            rep.calls.append((a, c, ok))

    # (6) the definition may not claim more assurance than its inputs provide
    if DAL_RANK[rep.dal[d.root]] > DAL_RANK[d.dal]:
        rep.bad("dal", d.root,
                "declared DAL %s but the value is only DAL %s; route the "
                "low-assurance input through a monitor"
                % (d.dal, rep.dal[d.root]))

    # (5) flight-critical code must claim a bound, which in turn forces the
    # cancellation analysis to be answered rather than ignored
    if d.dal in ("A", "B") and d.tol is None and d.atol is None:
        rep.bad("tolerance", None,
                "DAL %s requires a declared error tolerance (tol or atol)"
                % d.dal)
    if d.tol is not None and rep.err[d.root] > d.tol:
        extra = ""
        if rep.err[d.root] == INF:
            extra = ("; the value can be zero, so no relative bound exists -- "
                     "declare an absolute tolerance (atol=%.3g would hold)"
                     % rep.aerr[d.root]) if rep.aerr[d.root] < INF else ""
        rep.bad("tolerance", d.root,
                "proven relative error %.3g exceeds declared tolerance %.3g%s"
                % (rep.err[d.root], d.tol, extra))
    if d.atol is not None and rep.aerr[d.root] > d.atol:
        rep.bad("tolerance", d.root,
                "proven absolute error %.3g exceeds declared tolerance %.3g"
                % (rep.aerr[d.root], d.atol))

    # (7) bitwise reproducibility is part of the type, not a hope
    if d.require_repro:
        r = getattr(root_ty, "repro", None)
        if r is None and isinstance(root_ty, (Vec, Arr)):
            r = getattr(root_ty.elem, "repro", True)
        if not r:
            rep.bad("repro", d.root,
                    "definition requires bitwise reproducibility but the "
                    "result is not reproducible (check reduce assoc)")

    # (9) capabilities: no ambient authority, ceiling enforced at load
    for eff in sorted(x for x in rep.effects if x):
        if eff not in d.ceiling:
            rep.bad("capability", None,
                    "effect %r is outside the declared ceiling %s"
                    % (eff, sorted(d.ceiling)))

    # (9) WCET: computed from proven bounds, never measured
    rep.cost = total_cost(g, d.root) if machine is None \
        else total_cost(g, d.root, machine)
    # (9) WCET is the worse of the dependence chain and the issue-port bound.
    # A unit's throughput is a property of the silicon, so a program that
    # issues more work to one port than the port can retire is slow by
    # arithmetic, not by luck.
    wc = max(rep.cost.cycles, rep.cost.throughput_cycles())
    if d.wcet_cycles is not None and wc > d.wcet_cycles:
        bound = "port pressure" if rep.cost.throughput_cycles() > \
            rep.cost.cycles else "dependence chain"
        rep.bad("wcet", None, "WCET %.0f cycles exceeds budget %d (bound by "
                              "%s)" % (wc, d.wcet_cycles, bound))

    # (12) the deadline a cycle count cannot express. Settling elapses whether
    # or not a clock runs through it, so a 5 ms contact bounce and a 100 ms
    # perceptual budget are compared in the same arithmetic as a FLOP.
    if d.deadline_s is not None and rep.cost.seconds > d.deadline_s:
        rep.bad("deadline", None,
                "end-to-end %.6gs exceeds the %.6gs deadline (%.2fx over); "
                "the settling times on this path are physics, not scheduling"
                % (rep.cost.seconds, d.deadline_s,
                   rep.cost.seconds / d.deadline_s))

    return rep
