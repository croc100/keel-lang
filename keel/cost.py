"""
KEEL cost semantics.

v1 of this project claimed "the cost model is tokens". At the supercomputer
bar that is a toy. Energy per f64 FMA is ~1 pJ; energy to pull that word from
DRAM is two to three orders of magnitude more. At exascale you do not optimize
instructions, you optimize BYTES MOVED across the memory hierarchy and the
interconnect.

So cost here is a vector, not a scalar, and it is computed symbolically from
the DAG rather than measured:

    flops                    -- arithmetic performed
    bytes[level]             -- traffic at each level of the hierarchy
    msgs                     -- interconnect messages
    cycles                   -- WORST CASE, used as the WCET bound
    energy_pj                -- derived from the above via the machine model

The machine model is explicit and swappable. The absolute numbers below are
illustrative order-of-magnitude figures for a current-generation node; what
matters structurally is the ~100x spread between REG and DRAM, which is what
makes data movement rather than arithmetic the thing to minimize.

Multiplicity: because the DAG is content-addressed, common subexpressions are
already unified, so a shared node is evaluated ONCE. Cost accounting exploits
that exactly. Nodes under a `repeat` are multiplied by its proven bound, and
where a node is reachable at several nesting depths we take the maximum --
conservative, which is the only acceptable direction for a WCET claim.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict

from .types import MEM_LEVELS, SITES, Vec, width_bytes


@dataclass(frozen=True)
class Machine:
    """Illustrative single-node model. Swap freely; the checker only cares
    that a model is fixed and declared."""
    pj_per_byte: Dict[str, float] = field(default_factory=lambda: {
        "REG": 0.025,     # ~0.2 pJ per 8B word
        "L1": 0.5,        # ~4 pJ
        "L2": 2.5,        # ~20 pJ
        "DRAM": 40.0,     # ~320 pJ   <-- the cliff
        "NET": 500.0,     # ~4000 pJ
    })
    pj_per_flop: float = 1.0
    cycles_per_flop: float = 4.0
    latency_cycles: Dict[str, float] = field(default_factory=lambda: {
        "REG": 1.0, "L1": 4.0, "L2": 14.0, "DRAM": 200.0, "NET": 10000.0,
    })
    hz: float = 3.0e9         # so cycles and seconds are one quantity
    cycles_div: float = 20.0
    cycles_sqrt: float = 22.0
    pj_per_msg: float = 4000.0


M0 = Machine()


def _pj(level, m):
    """Energy per byte at a site. The machine model may override the compute
    hierarchy; anything else answers from its own registered physics."""
    if level in m.pj_per_byte:
        return m.pj_per_byte[level]
    s = SITES.get(level)
    return s.pj_byte if s else 0.0


def _latency(level, m):
    if level in m.latency_cycles:
        return m.latency_cycles[level]
    s = SITES.get(level)
    return (s.prop + s.settle) * m.hz if s else 0.0


@dataclass
class Cost:
    flops: float = 0.0
    bytes: Dict[str, float] = field(
        default_factory=lambda: defaultdict(
            float, {lv: 0.0 for lv in MEM_LEVELS}))
    msgs: float = 0.0
    cycles: float = 0.0
    # PHYSICAL time, which a cycle count cannot express: a 5 ms contact
    # bounce is not a number of clocks.
    seconds: float = 0.0
    ports: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    lanes_used: float = 0.0
    lanes_cap: float = 0.0

    def throughput_cycles(self):
        """The slowest issue port. Two operations naming the same port cannot
        retire at once, so this is a floor no scheduling can get under."""
        return max(self.ports.values()) if self.ports else 0.0

    def utilization(self):
        return self.lanes_used / self.lanes_cap if self.lanes_cap else None

    def scaled(self, k):
        return Cost(self.flops * k,
                    defaultdict(float, {l: v * k for l, v in self.bytes.items()}),
                    self.msgs * k, self.cycles * k, self.seconds * k,
                    defaultdict(float, {q: v * k for q, v in self.ports.items()}),
                    self.lanes_used * k, self.lanes_cap * k)

    def __iadd__(self, o):
        self.flops += o.flops
        for l, v in o.bytes.items():
            self.bytes[l] = self.bytes.get(l, 0.0) + v
        self.msgs += o.msgs
        self.cycles += o.cycles
        self.seconds += o.seconds
        for q, v in o.ports.items():
            self.ports[q] += v
        self.lanes_used += o.lanes_used
        self.lanes_cap += o.lanes_cap
        return self

    def energy_pj(self, m=M0):
        e = self.flops * m.pj_per_flop + self.msgs * m.pj_per_msg
        for l, v in self.bytes.items():
            e += v * _pj(l, m)
        return e

    def report(self, m=M0):
        mv = {l: v for l, v in self.bytes.items() if v}
        return {
            "flops": self.flops,
            "bytes": mv,
            "msgs": self.msgs,
            "wcet_cycles": self.cycles,
            "seconds": self.seconds,
            "throughput_cycles": self.throughput_cycles(),
            "ports": {q: v for q, v in self.ports.items() if v},
            "lane_utilization": self.utilization(),
            "energy_pj": self.energy_pj(m),
            "arithmetic_intensity": (self.flops / sum(mv.values())
                                     if sum(mv.values()) else float("inf")),
        }


_FLOP1 = {"add", "sub", "mul", "neg", "convert", "cmp", "select", "clamp"}
_LANG_OPS = frozenset((
    "call", "fold", "match", "alloc", "put", "get", "freeze",
    "perform", "rec", "field", "inject", "bstr", "bslice", "bcat",
    "blen", "beq", "bhash"))


def node_cost(node, argtys, m=M0):
    """Cost of evaluating one node once, given its argument types."""
    c = Cost()
    op = node.op

    if op in _FLOP1:
        n = 1
        t = argtys[0] if argtys else None
        if isinstance(t, Vec):
            n = t.n
        c.flops = n
        c.cycles = n * m.cycles_per_flop

    elif op == "fma":
        # Two flops for one instruction's latency. This is exactly why an FMA
        # unit doubles a machine's quoted peak and why HPC kernels are written
        # to reach it.
        c.flops, c.cycles = 2, m.cycles_per_flop

    elif op == "div":
        c.flops, c.cycles = 1, m.cycles_div

    elif op == "sqrt":
        c.flops, c.cycles = 1, m.cycles_sqrt

    elif op == "dot":
        n = argtys[0].n
        c.flops = 2 * n - 1
        c.cycles = c.flops * m.cycles_per_flop

    elif op == "norm":
        n = argtys[0].n
        c.flops = 2 * n
        c.cycles = (2 * n - 1) * m.cycles_per_flop + m.cycles_sqrt

    elif op == "vscale":
        n = argtys[0].n
        c.flops = n
        c.cycles = n * m.cycles_per_flop

    elif op == "rot_apply":
        n = argtys[1].n
        c.flops = 2 * n * n - n          # 3x3 matvec -> 15 flops
        c.cycles = c.flops * m.cycles_per_flop

    elif op == "load":
        a = argtys[0]
        w = width_bytes(a.elem)
        lanes = node.attrs.get("lanes", 1)
        stride = node.attrs.get("stride", 1)
        s = SITES.get(a.level)
        factor = 1
        if s is not None and s.banks and lanes > 1:
            from math import gcd
            g = gcd(abs(stride) or s.banks, s.banks)
            distinct = max(1, min(lanes, s.banks // g))
            factor = -(-lanes // distinct)
        c.bytes[a.level] += w * lanes
        c.cycles = _latency(a.level, m) * factor

    elif op == "stencil":
        a = argtys[0]
        k = len(node.attrs["offsets"])
        w = width_bytes(a.elem)
        n = a.count
        out = node.attrs.get("out_level", a.level)
        c.flops = n * (2 * k - 1)
        # compulsory traffic: stream in once, stream out once ...
        c.bytes[a.level] += n * w
        c.bytes[out] += n * w
        # ... plus the k-point neighbourhood served from cache
        c.bytes["L1"] += n * k * w
        c.cycles = n * (2 * k - 1) * m.cycles_per_flop + \
            (n * w / 64.0) * _latency(a.level, m)

    elif op == "reduce":
        a = argtys[0]
        w = width_bytes(a.elem)
        n = a.count
        c.flops = n - 1
        c.bytes[a.level] += n * w
        if node.attrs.get("assoc") == "nondet":
            c.msgs = 1
        c.cycles = (n - 1) * m.cycles_per_flop + \
            (n * w / 64.0) * _latency(a.level, m)

    elif op == "sys":
        c.msgs = 1
        c.cycles = _latency("NET", m)

    elif op in ("sense", "drive"):
        s = SITES.get(node.attrs.get("site"))
        if s is not None:
            w = width_bytes(node.ty)
            c.bytes[s.name] += w
            # Settling is real elapsed time whether or not a clock is running
            # through it, and it dominates everything a switch or a panel does.
            wait = node.attrs.get("after") if op == "sense" else None
            c.seconds = s.prop + (s.settle if wait is None else max(wait, 0.0))

    elif op == "bind":
        s = SITES.get(node.attrs.get("unit"))
        if s is not None:
            hz = s.hz or m.hz
            # Latency and port pressure are charged here IN ADDITION to the
            # wrapped operation's generic cost. That double-counts slightly,
            # which is the only acceptable direction for a WCET claim.
            c.cycles = s.lat
            if s.port and s.thru:
                c.ports[s.port] += 1.0 / s.thru
            c.lanes_used = node.attrs.get("lanes", 1)
            c.lanes_cap = s.lanes
            # An AVX-512 unit really is clocked below the core it sits in, and
            # a 200 MHz fabric is not a 3 GHz pipeline. Seconds know this;
            # cycles cannot.
            c.seconds = s.lat / hz + s.prop

    elif op == "sync":
        # Two flops in the receiving domain, by definition.
        c.cycles = float(node.attrs.get("depth", 2))

    elif op in _LANG_OPS:
        return lang_cost(node, argtys, m)

    # `require` costs nothing: it is discharged at check time and compiles to
    # no instruction. A proven invariant is free; a runtime assertion is not.
    # const / param / loopvar / monitor / vec3 / repeat: no intrinsic cost
    if not c.seconds:
        c.seconds = c.cycles / m.hz
    return c


def multiplicity(graph, root):
    """How many times each node is evaluated, worst case."""
    order = graph.topo(root)
    mult = {a: 0.0 for a in order}
    mult[root] = 1.0
    for a in reversed(order):
        n = graph[a]
        factor = n.attrs["bound"] if n.op == "repeat" else 1
        for d in n.args:
            mult[d] = max(mult[d], mult[a] * factor)
    return mult


def total_cost(graph, root, m=M0):
    mult = multiplicity(graph, root)
    total = Cost()
    for a in graph.topo(root):
        n = graph[a]
        argtys = [graph[d].ty for d in n.args]
        total += node_cost(n, argtys, m).scaled(mult[a])
    return total


# =====================================================================
# Cost of the general-purpose operations.
#
# The interesting cases are the three that do not evaluate a subgraph: a
# call, a fold and a match all take their cost from a CONTRACT. That is what
# makes cost compositional -- a caller does not walk the callee to find out
# what it costs, any more than it walks it to find out what it returns.
# =====================================================================

def lang_cost(node, argtys, m=M0):
    op = node.op
    c = Cost()

    if op == "call":
        k = node.attrs["fn"]
        c.cycles, c.flops, c.seconds = k.cycles, k.flops, k.seconds

    elif op == "fold":
        k, n = node.attrs["fn"], node.attrs["n"]
        # n applications, and the trip count is a literal, so this is a
        # number rather than an estimate.
        c.cycles, c.flops = k.cycles * n, k.flops * n
        c.seconds = k.seconds * n

    elif op == "match":
        # The WORST arm. A branch's cost under a WCET claim cannot mean
        # anything else; averaging would be a performance model, not a bound.
        arms = [k for _, k in (node.attrs.get("arms") or ())]
        if arms:
            c.cycles = max(k.cycles for k in arms)
            c.flops = max(k.flops for k in arms)
            c.seconds = max(k.seconds for k in arms)

    elif op == "alloc":
        t = node.attrs["bty"]
        c.bytes[t.level] += t.count * width_bytes(t.elem)
        c.cycles = _latency(t.level, m)

    elif op in ("put", "get"):
        t = argtys[0]
        c.bytes[t.level] += width_bytes(t.elem)
        c.cycles = _latency(t.level, m)

    elif op == "perform":
        # An effect's cost is physical time, not cycles. A network round trip
        # and a display frame are both seconds.
        c.seconds = float(node.attrs.get("cost_s", 0.0) or 0.0)
        c.msgs = 1

    elif op in ("bcat", "bslice"):
        n = getattr(node.ty, "n", 0)
        c.bytes["L1"] += n
        c.cycles = n / 8.0

    elif op in ("beq", "bhash"):
        n = getattr(argtys[0], "n", 0) if argtys else 0
        c.bytes["L1"] += n
        c.cycles = n / 8.0 + 4.0

    # rec / field / inject / freeze / blen / bstr: no instruction of their own.
    # A record is a layout, a field read is an offset, freezing is a change of
    # type, and a byte string's length is in its type.
    if not c.seconds:
        c.seconds = c.cycles / m.hz
    return c
