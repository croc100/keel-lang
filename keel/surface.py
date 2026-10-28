"""
KEEL-GLYPH -- the written form of the language.

A KEEL program is a graph, not a string, so the written form is a
SERIALISATION of that graph and not a grammar that gives it meaning. That
inverts the usual relationship and it is the reason this file can look the
way it does.

    ~0/7cs.U8;0hCdkdkdkdkdkd...

There are no keywords, no variable names, no delimiters and no whitespace.
That is not obfuscation; each of them was removed for a reason:

  no keywords      every operation is one glyph, because the operation table
                   is finite and a machine does not need a mnemonic.
  no names         a value is referenced by DISTANCE BACK in the stream. A
                   name is a human's index into their own memory, and it is
                   the thing that makes two identical programs textually
                   different -- which breaks content addressing for nothing.
  no delimiters    arity is fixed per operation, so the stream parses with no
                   lookahead and no brackets to balance.
  no whitespace    a space is not a separator here; every number is
                   self-delimiting. So whitespace in a stream is corruption,
                   which is what the TACET parity layer is for.

WHY THIS IS SAFE TO LET AN UNTRUSTED PARSER READ

Every definition ends with its own root address -- the SEAL. The loader
parses the glyphs, builds the graph, recomputes the address from the graph
it actually built, and compares. A parser that misreads anything produces a
different graph, which produces a different address, which is refused. So a
bug in this file cannot make a wrong program load; it can only make a right
program fail to load. The parser is therefore outside the trust root, like
the builder and the backend.

HOW A HUMAN READS IT

They do not. They read `explain`, which prints the graph in words, and they
read the checker's report. The point of the language was never that a person
could read the program; it was that a person could read the RULES.

A module is a sequence of definitions separated by `~`. A definition may call
the ones before it by back-index, and the loader checks them in order, so a
contract is never transmitted -- it is re-derived from the body that is
already in hand. There is nothing in the stream to forge.
"""

import struct
from fractions import Fraction

from . import hash as kh
from . import types as T
from .ir import Def, Graph, Node
from .wire import ALPHABET

GA = ALPHABET
IDX = {g: i for i, g in enumerate(GA)}
RADIX = 44                      # 0..43 continue, 44..87 terminate
SEP = GA[88]                    # between definitions; the only structural mark
MAGIC = GA[0] + GA[7]           # version 0, format 7

FMTS = ("f64", "f32", "f16")


class SurfaceError(Exception):
    pass


# =====================================================================
# self-delimiting numbers
# =====================================================================

def put_uint(n):
    if n < 0:
        raise SurfaceError("put_uint of %r" % n)
    digits = []
    while True:
        digits.append(n % RADIX)
        n //= RADIX
        if not n:
            break
    digits.reverse()
    return "".join(GA[d] for d in digits[:-1]) + GA[digits[-1] + RADIX]


def put_int(n):
    return put_uint(2 * n if n >= 0 else -2 * n - 1)


def put_float(x):
    return put_uint(struct.unpack("<Q", struct.pack("<d", float(x)))[0])


def put_str(s):
    b = s.encode("utf-8") if isinstance(s, str) else bytes(s)
    return put_uint(len(b)) + put_uint(int.from_bytes(b, "big") if b else 0)


def put_opt(x, f):
    return put_uint(0) if x is None else put_uint(1) + f(x)


class Cursor:
    def __init__(self, text, i=0):
        self.s = text
        self.i = i

    def eof(self):
        return self.i >= len(self.s)

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else ""

    def uint(self):
        """A self-delimiting number, and a CANONICAL one.

        Leading zero digits are refused. Without that the encoding is not
        injective as a stream: prepending a zero digit decodes to the same
        value, so a glyph could be inserted invisibly -- which is exactly how
        one corrupted stream out of 124 still loaded while this was being
        tested. Zero itself is one terminal digit, so a leading zero digit is
        always a non-canonical spelling of something.
        """
        n = 0
        first = True
        while True:
            if self.i >= len(self.s):
                raise SurfaceError("stream ended inside a number")
            g = self.s[self.i]
            self.i += 1
            d = IDX.get(g)
            if d is None:
                raise SurfaceError("glyph %r is not in the alphabet" % g)
            if d < RADIX:
                if first and d == 0:
                    raise SurfaceError(
                        "non-canonical number at %d: a leading zero digit "
                        "spells a value that has a shorter spelling"
                        % (self.i - 1))
                n = n * RADIX + d
            else:
                return n * RADIX + (d - RADIX)
            first = False

    def int(self):
        n = self.uint()
        return (n >> 1) if not (n & 1) else -((n + 1) >> 1)

    def float(self):
        return struct.unpack("<d", struct.pack("<Q", self.uint()))[0]

    def str(self):
        n = self.uint()
        v = self.uint()
        return v.to_bytes(n, "big").decode("utf-8") if n else ""

    def opt(self, f):
        return f() if self.uint() else None


# =====================================================================
# units, sites and types
# =====================================================================

def _unit_table():
    out = {}
    for name in dir(T):
        v = getattr(T, name)
        if isinstance(v, T.Unit) and v.label:
            out.setdefault(v.label, v)
    return tuple(sorted(out.items()))


UNITS = _unit_table()
UNIT_IX = {lbl: i for i, (lbl, _) in enumerate(UNITS)}


def put_unit(u):
    if u.label and u.label in UNIT_IX and UNITS[UNIT_IX[u.label]][1] == u:
        return put_uint(0) + put_uint(UNIT_IX[u.label])
    # not in the table: write the dimension vector and the exact scale. A unit
    # is never approximated, here least of all.
    f = Fraction(u.scale)
    return (put_uint(1) + "".join(put_int(d) for d in u.dim)
            + put_int(f.numerator) + put_uint(f.denominator))


def get_unit(c):
    if c.uint() == 0:
        return UNITS[c.uint()][1]
    dim = tuple(c.int() for _ in range(T.NDIM))
    num, den = c.int(), c.uint()
    return T.Unit(dim, Fraction(num, den))


TY_INT, TY_BOOL, TY_REAL, TY_VEC, TY_ARR = 0, 1, 2, 3, 4
TY_ROT, TY_BYTES, TY_CAP, TY_BUF, TY_REC, TY_SUM = 5, 6, 7, 8, 9, 10


def put_ty(t):
    if isinstance(t, T.Int):
        return put_uint(TY_INT)
    if isinstance(t, T.Bool):
        return put_uint(TY_BOOL)
    if isinstance(t, T.Real):
        return (put_uint(TY_REAL) + put_uint(FMTS.index(t.fmt))
                + put_unit(t.unit) + put_opt(t.frame, put_str)
                + put_uint(1 if t.repro else 0))
    if isinstance(t, T.Vec):
        return (put_uint(TY_VEC) + put_ty(t.elem) + put_uint(t.n)
                + put_opt(t.frame, put_str))
    if isinstance(t, T.Arr):
        return (put_uint(TY_ARR) + put_ty(t.elem) + put_uint(len(t.shape))
                + "".join(put_uint(d) for d in t.shape) + put_str(t.level))
    if isinstance(t, T.Rot):
        return put_uint(TY_ROT) + put_str(t.frm) + put_str(t.to)
    if isinstance(t, T.Bytes):
        return put_uint(TY_BYTES) + put_uint(t.n) + put_str(t.enc)
    if isinstance(t, T.Cap):
        return put_uint(TY_CAP) + put_str(t.kind)
    if isinstance(t, T.Buf):
        return (put_uint(TY_BUF) + put_ty(t.elem) + put_uint(len(t.shape))
                + "".join(put_uint(d) for d in t.shape) + put_str(t.level))
    if isinstance(t, T.Rec):
        return (put_uint(TY_REC) + put_str(t.name) + put_uint(len(t.fields))
                + "".join(put_str(f) + put_ty(ft) for f, ft in t.fields))
    if isinstance(t, T.Sum):
        return (put_uint(TY_SUM) + put_str(t.name) + put_uint(len(t.variants))
                + "".join(put_str(g) + put_ty(gt) for g, gt in t.variants))
    raise SurfaceError("no glyph form for type %r" % (t,))


def get_ty(c):
    k = c.uint()
    if k == TY_INT:
        return T.Int()
    if k == TY_BOOL:
        return T.Bool()
    if k == TY_REAL:
        fmt = FMTS[c.uint()]
        u = get_unit(c)
        fr = c.opt(c.str)
        return T.Real(fmt, u, fr, bool(c.uint()))
    if k == TY_VEC:
        e = get_ty(c)
        n = c.uint()
        return T.Vec(e, n, c.opt(c.str))
    if k == TY_ARR:
        e = get_ty(c)
        nd = c.uint()
        shape = tuple(c.uint() for _ in range(nd))
        return T.Arr(e, shape, c.str())
    if k == TY_ROT:
        return T.Rot(c.str(), c.str())
    if k == TY_BYTES:
        return T.Bytes(c.uint(), c.str())
    if k == TY_CAP:
        return T.Cap(c.str())
    if k == TY_BUF:
        e = get_ty(c)
        nd = c.uint()
        shape = tuple(c.uint() for _ in range(nd))
        return T.Buf(e, shape, c.str())
    if k == TY_REC:
        name = c.str()
        n = c.uint()
        return T.Rec(name, tuple((c.str(), get_ty(c)) for _ in range(n)))
    if k == TY_SUM:
        name = c.str()
        n = c.uint()
        return T.Sum(name, tuple((c.str(), get_ty(c)) for _ in range(n)))
    raise SurfaceError("unknown type tag %d" % k)


# =====================================================================
# the operation table
#
# APPEND ONLY. An operation's index is part of every program ever written in
# this form, so inserting one in the middle would silently reinterpret old
# streams -- the same reason a wire protocol never renumbers a field.
#
# Each entry is (name, arity, field-codes). Field codes:
#   s string   i int   f float   o optional float   u unit   t type   b bytes
# =====================================================================

OPS = (
    ("param", 0, ("s", "t", "o", "o", "S", "f")),
    ("const", 0, ("f", "t")),
    ("add", 2, ()),
    ("sub", 2, ()),
    ("mul", 2, ()),
    ("div", 2, ()),
    ("neg", 1, ()),
    ("sqrt", 1, ()),
    ("fma", 3, ()),
    ("convert", 1, ("u",)),
    ("clamp", 1, ("f", "f")),
    ("require", 1, ("f", "f", "u", "s")),
    ("bind", 1, ("s", "i")),
    ("monitor", 1, ("s", "s")),
    ("sync", 1, ("s", "i")),
    ("vec3", 3, ("s",)),
    ("dot", 2, ()),
    ("norm", 1, ()),
    ("vscale", 2, ()),
    ("load", 2, ("i", "i")),
    ("select", 3, ()),
    ("cmp", 2, ("s",)),
    ("rec", -1, ("t",)),
    ("field", 1, ("s",)),
    ("inject", 1, ("t", "s")),
    ("bstr", 0, ("b", "s")),
    ("bslice", 1, ("i", "i")),
    ("bcat", 2, ()),
    ("blen", 1, ()),
    ("beq", 2, ()),
    ("bhash", 1, ("i",)),
    ("alloc", 0, ("t", "f")),
    ("put", 3, ()),
    ("get", 2, ()),
    ("freeze", 1, ()),
    ("perform", -1, ("s", "f", "t")),
    ("call", -1, ("i",)),
    ("fold", 1, ("i", "i", "f", "f")),
    ("loopvar", 0, ("i", "s")),
    ("repeat", 1, ("i",)),
    ("reduce", 1, ("s",)),
)
OP_IX = {name: i for i, (name, _, _) in enumerate(OPS)}

# attribute key order per op, matching the field codes above
KEYS = {
    "param": ("name", "ty", "lo", "hi", "dal", "err"),
    "const": ("value", "ty"),
    "convert": ("to",),
    "clamp": ("lo", "hi"),
    "require": ("lo", "hi", "unit", "why"),
    "bind": ("unit", "lanes"),
    "monitor": ("dal", "check"),
    "sync": ("to", "depth"),
    "vec3": ("frame",),
    "load": ("lanes", "stride"),
    "cmp": ("op",),
    "rec": ("rty",),
    "field": ("f",),
    "inject": ("sty", "tag"),
    "bstr": ("data", "enc"),
    "bslice": ("off", "n"),
    "bhash": ("bits",),
    "alloc": ("bty", "fill"),
    "perform": ("effect", "cost_s", "ty"),
    "call": ("fnref",),
    "fold": ("fnref", "n", "lo", "hi"),
    "loopvar": ("bound", "name"),
    "repeat": ("bound",),
    "reduce": ("assoc",),
}


def _put_field(code, v):
    if code == "s":
        return put_str("" if v is None else str(v))
    if code == "i":
        return put_int(int(v if v is not None else 0))
    if code == "f":
        return put_float(0.0 if v is None else v)
    if code == "o":
        return put_opt(v, put_float)
    if code == "S":
        return put_opt(v, put_str)
    if code == "u":
        return put_unit(v)
    if code == "t":
        return put_ty(v)
    if code == "b":
        return put_uint(len(v)) + "".join(put_uint(x) for x in v)
    raise SurfaceError("field code %r" % code)


def _get_field(code, c):
    if code == "s":
        return c.str()
    if code == "i":
        return c.int()
    if code == "f":
        return c.float()
    if code == "o":
        return c.opt(c.float)
    if code == "S":
        return c.opt(c.str)
    if code == "u":
        return get_unit(c)
    if code == "t":
        return get_ty(c)
    if code == "b":
        return [c.uint() for _ in range(c.uint())]
    raise SurfaceError("field code %r" % code)


# =====================================================================
# writing
# =====================================================================

def emit_def(d, fnref=None):
    """One definition as glyphs. `fnref` maps a contract body address to the
    back-index of the definition that proves it."""
    g = d.graph
    from .ir import param_addrs
    order = g.topo(d.root)
    # A declared parameter the body does not use is still part of the
    # signature, so it has to be in the stream. Parameters have no arguments,
    # so putting them first keeps the order topological.
    unused = [a for a in param_addrs(d) if a not in set(order)]
    order = unused + order
    pos = {a: i for i, a in enumerate(order)}
    out = [put_str(d.name), put_str(d.dal),
           put_opt(d.tol, put_float), put_opt(d.atol, put_float),
           put_opt(d.wcet_cycles, lambda x: put_uint(int(x))),
           put_opt(d.deadline_s, put_float),
           put_uint(1 if d.require_repro else 0),
           put_uint(len(d.ceiling)),
           "".join(put_str(e) for e in sorted(d.ceiling)),
           put_str(d.intent)]
    # the signature, when one was declared
    if d.params is None:
        out.append(put_uint(0))
    else:
        out.append(put_uint(1) + put_uint(len(d.params))
                   + "".join(put_uint(len(order) - pos[a])
                             for a in d.params))
    out.append(put_uint(len(order)))
    for i, a in enumerate(order):
        n = g[a]
        ix = OP_IX.get(n.op)
        if ix is None:
            raise SurfaceError("operation %r has no glyph form" % n.op)
        name, arity, codes = OPS[ix]
        out.append(put_uint(ix))
        if arity < 0:
            out.append(put_uint(len(n.args)))
        elif len(n.args) != arity:
            raise SurfaceError("%s takes %d argument(s), node has %d"
                               % (name, arity, len(n.args)))
        # A reference is the DISTANCE BACK to the value, so the same
        # expression written at two places in a module is the same glyphs.
        for dep in n.args:
            out.append(put_uint(i - pos[dep]))
        keys = KEYS.get(n.op, ())
        for code, key in zip(codes, keys):
            if key == "ty":
                # the node's DECLARED type, which lives beside the attributes
                # and not inside them -- putting it in attrs would change the
                # address of every node that carries one
                out.append(put_ty(n.ty))
            elif key == "fnref":
                c = n.attrs["fn"]
                if fnref is None or c.body not in fnref:
                    raise SurfaceError(
                        "%r calls %r but that body is not in this module; a "
                        "contract is never transmitted, so the callee has to "
                        "be here" % (d.name, c.name))
                out.append(put_int(fnref[c.body]))
            else:
                out.append(_put_field(code, n.attrs.get(key)))
    # THE SEAL, over the ENVELOPE and not just the body.
    #
    # Sealing the root address alone was not enough, and the way that was
    # found is worth recording: flipping one glyph at each position of a
    # stream and loading the result, 16 of 124 positions produced a program
    # that LOADED. Every one of them had landed in metadata -- the name, the
    # DAL, a tolerance, the capability ceiling -- which the body's address
    # does not cover. A corrupted ceiling is the worst possible thing to
    # accept silently, since it is the list of effects the program may
    # perform.
    out.append(put_str(envelope(d, g[d.root].addr)))
    return "".join(out)


def envelope(d, root_addr):
    """The address of the whole definition: body AND everything a loader
    decides with. Two programs with the same body and different ceilings are
    different programs to anyone choosing whether to run one."""
    return kh.seal(root_addr, d.intent,
                   ("env", d.name, d.dal, repr(d.tol), repr(d.atol),
                    repr(d.wcet_cycles), repr(d.deadline_s),
                    bool(d.require_repro),
                    len(d.params) if d.params is not None else -1),
                   d.ceiling)


def emit(defs):
    """A module: definitions in dependency order, separated by `~`."""
    parts, fnref = [], {}
    for i, d in enumerate(defs):
        parts.append(emit_def(d, fnref))
        fnref[d.graph[d.root].addr] = i
    return MAGIC + SEP + SEP.join(parts)


# =====================================================================
# reading
# =====================================================================

def parse(text, funcs=None):
    """Glyphs -> a list of Defs. Raises if any seal does not match.

    `funcs` lets a caller supply already-proven functions for call sites; the
    normal path is `load`, which checks each definition as it goes and so
    always has them.
    """
    text = "".join(ch for ch in text if not ch.isspace())
    if not text.startswith(MAGIC):
        raise SurfaceError("not a KEEL-GLYPH stream")
    chunks = text[len(MAGIC):].split(SEP)
    chunks = [x for x in chunks if x]
    out, by_index = [], list(funcs or ())
    for ci, chunk in enumerate(chunks):
        out.append(_parse_def(chunk, by_index))
        if len(by_index) <= ci:
            by_index.append(None)
    return out


def _parse_def(chunk, by_index):
    c = Cursor(chunk)
    name, dal = c.str(), c.str()
    tol, atol = c.opt(c.float), c.opt(c.float)
    wcet = c.opt(c.uint)
    deadline = c.opt(c.float)
    repro = bool(c.uint())
    ceiling = frozenset(c.str() for _ in range(c.uint()))
    intent = c.str()
    has_sig = c.uint()
    sig_back = [c.uint() for _ in range(c.uint())] if has_sig else None
    count = c.uint()

    g = Graph()
    addrs = []
    for i in range(count):
        ix = c.uint()
        if ix >= len(OPS):
            raise SurfaceError("unknown operation index %d" % ix)
        op, arity, codes = OPS[ix]
        nargs = c.uint() if arity < 0 else arity
        args = []
        for _ in range(nargs):
            back = c.uint()
            j = i - back
            if j < 0 or j >= i:
                raise SurfaceError("reference %d back from position %d is "
                                   "outside the stream" % (back, i))
            args.append(addrs[j])
        attrs = {}
        decl = None
        for code, key in zip(codes, KEYS.get(op, ())):
            if key == "ty":
                decl = get_ty(c)
            elif key == "fnref":
                k = c.int()
                if k < 0 or k >= len(by_index) or by_index[k] is None:
                    raise SurfaceError("call refers to definition %d, which "
                                       "is not proven yet" % k)
                attrs["fn"] = by_index[k].contract
            else:
                attrs[key] = _get_field(code, c)
        ty = decl if decl is not None else _derive_decl_ty(op, attrs, args, g)
        node = Node(op, args, attrs, ty)
        g.add(node)
        addrs.append(node.addr)

    root = addrs[-1] if addrs else None
    declared_seal = c.str()
    if root is None:
        raise SurfaceError("definition %r has no nodes" % name)
    params = None
    if sig_back is not None:
        params = tuple(addrs[count - b] for b in sig_back)
    d = Def(name, root, g, intent=intent, dal=dal, ceiling=ceiling,
            wcet_cycles=wcet, tol=tol, atol=atol, require_repro=repro,
            deadline_s=deadline, params=params)
    got = envelope(d, g[root].addr)
    if got != declared_seal:
        raise SurfaceError(
            "seal mismatch in %r: the stream declares %s and the definition "
            "built from it is %s. The parser read something other than what "
            "was written, and content addressing caught it."
            % (name, declared_seal, got))
    return d


def _derive_decl_ty(op, attrs, args, g):
    """The DECLARED type of a parsed node.

    The stream does not carry a type per node, because the checker re-derives
    every type anyway and carrying it would mean carrying something that can
    disagree with itself. What is carried is the handful of types that are
    genuinely declarations: a parameter's, a constant's, a record's, a
    variant's. Everything else is reconstructed by the same rule the builder
    uses, and if that reconstruction is wrong the seal will not match.
    """
    ty = lambda a: g[a].ty
    if op == "alloc":
        return attrs["bty"]
    if op == "rec":
        return attrs["rty"]
    if op == "inject":
        return attrs["sty"]
    if op == "field":
        t = ty(args[0])
        return t.ty_of(attrs["f"])
    if op in ("add", "sub", "neg", "clamp", "require", "monitor", "repeat"):
        return ty(args[0])
    if op == "sync":
        t = ty(args[0])
        return T.Real(t.fmt, t.unit, attrs.get("to") or None, t.repro) \
            if isinstance(t, T.Real) else t
    if op == "bind":
        t = ty(args[0])
        s = T.SITES.get(attrs["unit"])
        if s is not None and s.clock and isinstance(t, T.Real):
            return T.Real(t.fmt, t.unit, s.clock, t.repro)
        return t
    if op == "mul":
        ta, tb = ty(args[0]), ty(args[1])
        return T.Real(ta.fmt, ta.unit * tb.unit, ta.frame or tb.frame,
                      ta.repro and tb.repro)
    if op == "div":
        ta, tb = ty(args[0]), ty(args[1])
        return T.Real(ta.fmt, ta.unit / tb.unit, ta.frame or tb.frame,
                      ta.repro and tb.repro)
    if op == "fma":
        ta, tb, tc = (ty(a) for a in args)
        return T.Real(ta.fmt, ta.unit * tb.unit,
                      ta.frame or tb.frame or tc.frame,
                      ta.repro and tb.repro and tc.repro)
    if op == "sqrt":
        t = ty(args[0])
        return T.Real(t.fmt, T.Unit(tuple(d // 2 for d in t.unit.dim),
                                    t.unit.scale), t.frame, t.repro)
    if op == "convert":
        t = ty(args[0])
        return T.Real(t.fmt, attrs["to"], t.frame, t.repro)
    if op == "vec3":
        t = ty(args[0])
        return T.Vec(T.Real(t.fmt, t.unit, attrs.get("frame") or None,
                            t.repro), 3, attrs.get("frame") or None)
    if op == "dot":
        ta, tb = ty(args[0]), ty(args[1])
        return T.Real(ta.elem.fmt, ta.elem.unit * tb.elem.unit, None,
                      ta.elem.repro and tb.elem.repro)
    if op == "norm":
        t = ty(args[0])
        return T.Real(t.elem.fmt, t.elem.unit, None, t.elem.repro)
    if op == "vscale":
        ta, tb = ty(args[0]), ty(args[1])
        return T.Vec(T.Real(ta.elem.fmt, ta.elem.unit * tb.unit, ta.frame,
                            ta.elem.repro and tb.repro), ta.n, ta.frame)
    if op == "load":
        return ty(args[0]).elem
    if op == "select":
        return ty(args[1])
    if op == "cmp" or op == "beq":
        return T.Bool()
    if op in ("blen", "bhash"):
        return T.Int()
    if op == "loopvar":
        return T.Int()
    if op == "bstr":
        return T.Bytes(len(attrs["data"]), attrs["enc"])
    if op == "bslice":
        return T.Bytes(attrs["n"], ty(args[0]).enc)
    if op == "bcat":
        ta, tb = ty(args[0]), ty(args[1])
        return T.Bytes(ta.n + tb.n, ta.enc)
    if op == "put":
        return ty(args[0])
    if op == "get":
        return ty(args[0]).elem
    if op == "freeze":
        return ty(args[0]).frozen_ty()
    if op in ("call", "fold"):
        return attrs["fn"].ret
    if op == "reduce":
        return ty(args[0]).elem
    raise SurfaceError("cannot reconstruct the declared type of %r" % op)


# =====================================================================
# loading: parse, check, and refuse
# =====================================================================

def load(text, machine=None):
    """Parse a module and check every definition in order.

    Returns (defs, funcs, reports). Raises on a seal mismatch; a definition
    that fails to CHECK is reported rather than raised, because that is the
    normal, expected outcome of writing a program that is wrong, and the
    report is the useful artefact.
    """
    from .check import check, func
    text = "".join(ch for ch in text if not ch.isspace())
    if not text.startswith(MAGIC):
        raise SurfaceError("not a KEEL-GLYPH stream")
    chunks = [x for x in text[len(MAGIC):].split(SEP) if x]
    defs, funcs, reports = [], [], []
    for chunk in chunks:
        d = _parse_def(chunk, funcs)
        defs.append(d)
        rep = check(d, machine=machine, funcs=funcs)
        reports.append(rep)
        if rep.ok:
            funcs.append(func(d.name, d, machine=machine, funcs=funcs))
        else:
            funcs.append(None)
    return defs, funcs, reports


def roundtrip(defs):
    """Write a module and read it back. Used by the tests to assert that the
    written form loses nothing -- the same argument the address encoder
    makes, applied one level up."""
    return parse(emit(defs))
