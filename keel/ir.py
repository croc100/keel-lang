"""
KEEL IR -- a typed, content-addressed Merkle DAG.

A KEEL program is NOT a string. Strings exist because humans type and read
linearly; a machine reader has no such constraint. The native form is a DAG
whose nodes are addressed by the hash of their own content:

    addr(n) = blake2s( op | arg addrs | attrs | declared type )

Consequences:
  * identical subexpressions unify automatically and globally
  * identity is intrinsic and STABLE ACROSS VERSIONS -- it is tied to content,
    not to position, so a delta update to a vehicle 300M km away ships only
    the changed Merkle path
  * supply-chain substitution is not expressible: you cannot reuse a name to
    smuggle in a different body, because there are no names

IMPORTANT -- the builder is UNTRUSTED.
Nodes carry a *declared* type. The checker re-derives the true type
independently and rejects any mismatch. So a buggy or hostile generator
cannot lie about what it built. The builder below is a convenience; it is
not part of the trust root.
"""

from .hash import node_addr
from .types import Unit

ADDR_LEN = 16  # hex chars of blake2s


def _canon_attr(v):
    if isinstance(v, Unit):
        return ["unit", list(v.dim), str(v.scale)]
    if hasattr(v, "canon") and callable(v.canon):
        # A type or a contract in an attribute canonicalises through its own
        # canon(), so a node's address commits to it exactly. A call node's
        # address therefore commits to the callee's whole proof.
        return _canon_attr(v.canon())
    if isinstance(v, (list, tuple)):
        return [_canon_attr(x) for x in v]
    if isinstance(v, dict):
        return {k: _canon_attr(v[k]) for k in sorted(v)}
    if isinstance(v, float):
        return ["f", repr(v)]
    return v


def _canon_ty(t):
    return _canon_attr(list(t.canon()) if isinstance(t.canon(), tuple) else t.canon())


class Node:
    __slots__ = ("op", "args", "attrs", "ty", "addr")

    def __init__(self, op, args, attrs, ty):
        self.op = op
        self.args = tuple(args)
        self.attrs = dict(attrs or {})
        self.ty = ty              # DECLARED type; the checker re-derives it
        # KEEL-ADDR. See keel/hash.py for why this is not json.dumps: an
        # address function may not be "probably injective".
        self.addr = node_addr(op, self.args, _canon_attr(self.attrs),
                              _canon_ty(ty), ADDR_LEN // 2)

    def __repr__(self):
        return "%s:%s%s" % (self.addr[:6], self.op, list(self.args and
                                                         [a[:6] for a in self.args]))


class Def:
    """A named, signed definition -- the unit of loading and certification.

    The 4-tuple envelope from the design principles lives here:
      glyphs/graph   -- the content-addressed body (root + nodes)
      intent         -- the human-owned specification (natural language)
      contract       -- machine-checked obligations the checker must discharge
      ceiling        -- effect / capability upper bound, enforced at load
    plus the things the hard bar adds: dal, wcet budget, error tolerance.
    """

    def __init__(self, name, root, graph, *, intent="", dal="C",
                 ceiling=frozenset(), wcet_cycles=None, tol=None, atol=None,
                 require_repro=False, deadline_s=None, params=None):
        self.name = name
        self.root = root
        self.graph = graph
        # A SIGNATURE, when the author wants one. Without it the parameters
        # are whichever ones the body reaches, in declaration order -- fine
        # for an expression, wrong for a function: a body that ignores its
        # index parameter still has two parameters, and its callers have to
        # agree about that.
        self.params = tuple(params) if params is not None else None
        self.intent = intent
        self.dal = dal
        self.ceiling = frozenset(ceiling)
        self.wcet_cycles = wcet_cycles
        self.tol = tol                      # max allowed RELATIVE error
        self.atol = atol                    # max allowed ABSOLUTE error
        self.require_repro = require_repro
        self.deadline_s = deadline_s        # end-to-end PHYSICAL time budget

    @property
    def addr(self):
        return self.graph.nodes[self.root].addr


class Graph:
    def __init__(self):
        self.nodes = {}
        self.order = []

    def add(self, node):
        if node.addr not in self.nodes:
            self.nodes[node.addr] = node
            self.order.append(node.addr)
        return node.addr

    def __getitem__(self, addr):
        return self.nodes[addr]

    def topo(self, root):
        """Addresses reachable from root, in dependency order.

        Iterative on purpose. A recursive walk makes the deepest expression
        the language can hold a property of the HOST's stack rather than of
        KEEL, and a chain of four thousand operations is an ordinary thing to
        want -- it is how you measure a unit's latency. The graph is acyclic
        by construction, because an address is a hash of its arguments and a
        cycle would need a node that contains its own digest.
        """
        seen, out = set(), []
        stack = [(root, False)]
        while stack:
            a, expanded = stack.pop()
            if expanded:
                out.append(a)
                continue
            if a in seen:
                continue
            seen.add(a)
            stack.append((a, True))
            # pushed in reverse so the FIRST argument is expanded first: the
            # order a reader would expect, and the one the cost and register
            # schedules were measured against
            for d in reversed(self.nodes[a].args):
                if d not in seen:
                    stack.append((d, False))
        return out


# =====================================================================
# Functions, and therefore programs.
#
# Up to here a definition was one expression. One expression is not a
# language: there is no abstraction, so there is no reuse, so a large program
# is a large graph with nothing to say about itself.
#
# A Func is a Def plus a CONTRACT, and the contract is not a promise -- it is
# DERIVED FROM THE CALLEE'S OWN PROOF. At a call site the checker does not
# re-derive the body; it verifies the arguments land inside the parameter
# boxes the callee was proven over and then adopts the callee's proven
# interval, error, cost and effects. That is modular verification, and it is
# the difference between checking a program and checking an expression.
#
# None of this weakens the Merkle structure. A contract carries the body's
# root address and canonicalises into the call node's attributes, so the call
# node's address commits to the exact body that was proven. Swap the body and
# every caller's address changes.
# =====================================================================


class Contract:
    """What a caller may rely on, and what it must establish first.

    requires -- a box per parameter. The caller proves its arguments are
                inside; that is the precondition, and it is the only one.
    ensures  -- the proven result interval, relative and absolute error.
    costs    -- cycles, flops, physical seconds. Cost is part of the
                interface, because a caller with a deadline cannot be asked
                to guess.
    effects  -- the complete set. A function that does not declare an effect
                cannot perform one, and cannot call anything that does.
    """

    __slots__ = ("name", "body", "params", "ret", "lo", "hi", "rel", "aerr",
                 "cycles", "flops", "seconds", "effects", "dal")

    def __init__(self, name, body, params, ret, lo, hi, rel, aerr,
                 cycles, flops, seconds, effects, dal):
        self.name = name
        self.body = body            # root address of the proven body
        self.params = tuple(params)  # ((pname, ty, lo, hi), ...)
        self.ret = ret
        self.lo, self.hi = float(lo), float(hi)
        self.rel, self.aerr = float(rel), float(aerr)
        self.cycles, self.flops = float(cycles), float(flops)
        self.seconds = float(seconds)
        self.effects = frozenset(effects)
        self.dal = dal

    def canon(self):
        return ("contract", self.name, self.body,
                tuple((p, t.canon(), repr(lo), repr(hi))
                      for p, t, lo, hi in self.params),
                self.ret.canon(), repr(self.lo), repr(self.hi),
                repr(self.rel), repr(self.aerr), repr(self.cycles),
                repr(self.flops), repr(self.seconds),
                tuple(sorted(self.effects)), self.dal)

    def weaker_than(self, other):
        """Is `self` a contract this body is entitled to publish?

        A published contract may promise LESS than the body proves: a wider
        result interval, a larger error, a higher cost, more effects. It may
        never promise more. It may also require MORE of its caller (narrower
        parameter boxes), never less.
        """
        if len(self.params) != len(other.params):
            return False, "parameter count"
        for (n1, t1, l1, h1), (n2, t2, l2, h2) in zip(self.params,
                                                      other.params):
            if t1.canon() != t2.canon():
                return False, "parameter %s type" % n1
            if l1 < l2 or h1 > h2:
                return False, ("parameter %s box [%g,%g] is wider than the "
                               "body was proven over [%g,%g]"
                               % (n1, l1, h1, l2, h2))
        if self.ret.canon() != other.ret.canon():
            return False, "result type"
        if self.lo > other.lo or self.hi < other.hi:
            return False, ("result interval [%g,%g] is narrower than the "
                           "proven [%g,%g]" % (self.lo, self.hi,
                                               other.lo, other.hi))
        for f, a, b in (("relative error", self.rel, other.rel),
                        ("absolute error", self.aerr, other.aerr),
                        ("cycles", self.cycles, other.cycles),
                        ("seconds", self.seconds, other.seconds)):
            if a < b:
                return False, "%s %g is below the proven %g" % (f, a, b)
        if not other.effects <= self.effects:
            return False, ("effects %s are not declared"
                           % sorted(other.effects - self.effects))
        return True, "ok"

    def __repr__(self):
        return "%s(%s) -> %r in [%g,%g] <=%g cyc %s" % (
            self.name, ", ".join("%s:%r" % (p, t) for p, t, _, _ in
                                 self.params),
            self.ret, self.lo, self.hi, self.cycles,
            "+".join(sorted(self.effects)) or "pure")


class Func:
    """A Def that may be called. Build one with `func(name, def, report)`."""

    def __init__(self, contract, body_def):
        self.contract = contract
        self.body = body_def

    @property
    def name(self):
        return self.contract.name

    @property
    def addr(self):
        return self.contract.body

    def __repr__(self):
        return "func %r" % self.contract


def param_addrs(d):
    """The parameter nodes of a Def, in signature order."""
    g = d.graph
    if d.params is not None:
        for a in d.params:
            if g[a].op != "param":
                raise ValueError("%r is not a parameter" % a[:6])
        return tuple(d.params)
    reach = set(g.topo(d.root))
    return tuple(a for a in g.order
                 if a in reach and g[a].op == "param")


def declared_params(d):
    """Parameters of a Def, in signature order with their declared boxes."""
    g = d.graph
    out = []
    for a in param_addrs(d):
        n = g[a]
        out.append((n.attrs["name"], n.ty,
                    -float("inf") if n.attrs.get("lo") is None
                    else float(n.attrs["lo"]),
                    float("inf") if n.attrs.get("hi") is None
                    else float(n.attrs["hi"])))
    return tuple(out)
