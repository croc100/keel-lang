"""
TACET wire format -- the glyph transport encoding, demoted from v1's "language"
to v2's "serialization".

A KEEL program is a DAG, not a string. But it still has to cross channels that
only carry tokens, so it needs an encoding, and the encoding is where v1's one
genuinely good idea lives on.

Alphabet: 89 printable ASCII glyphs. 89 is prime, so glyph indices form GF(89)
and we get cheap algebra for error correction. No whitespace at all -- a space
in a TACET stream is a corruption signal, never a separator.

Frame: 16 data glyphs + 2 parity glyphs (12.5% overhead).

    s0 = sum(a_i)                mod 89
    s1 = sum((i+1) * a_i)        mod 89

A single substituted glyph at position j with magnitude e gives
d0 = e and d1 = (j+1)*e, so j = d1/d0 - 1 and e = d0. Position AND magnitude
are recovered, so the error is corrected, not merely detected. Two errors in
one frame are detected.

Why a language would carry this when no human language does: the two failure
modes that motivate it are both machine-specific.

  1. the author is a sampler, so it does not make typos -- it drifts. A single
     mis-sampled glyph is the overwhelmingly common corruption.
  2. the program image sits in memory on a vehicle outside the magnetosphere,
     where single event upsets flip stored bits for real.

A human would never maintain parity by hand, so no human language asks. For a
machine it is free.
"""

import json

DIGITS = "0123456789"
LOWER = "abcdefghijklmnopqrstuvwxyz"
UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYMS = "!#$%&()*+,-./:<=>?@[]^_{|}~"
ALPHABET = DIGITS + LOWER + UPPER + SYMS
assert len(ALPHABET) == 89 == len(set(ALPHABET))

P = 89
IDX = {g: i for i, g in enumerate(ALPHABET)}

DATA = 16          # data glyphs per frame
PAR = 2            # parity glyphs per frame
CHUNK = 8          # bytes per group
GPC = 10           # glyphs per group (89**10 > 2**64)


def _inv(a):
    return pow(a, P - 2, P)


# ------------------------------------------------------------------ base 89

def _bytes_to_glyphs(bs):
    out = []
    for i in range(0, len(bs), CHUNK):
        grp = bs[i:i + CHUNK]
        n = int.from_bytes(grp.ljust(CHUNK, b"\0"), "big")
        g = []
        for _ in range(GPC):
            g.append(ALPHABET[n % P])
            n //= P
        out.append("".join(reversed(g)))
    return "".join(out), len(bs)


def _glyphs_to_bytes(gs, nbytes):
    out = bytearray()
    for i in range(0, len(gs), GPC):
        n = 0
        for c in gs[i:i + GPC]:
            n = n * P + IDX.get(c, 0)
        out += n.to_bytes(CHUNK, "big")
    return bytes(out[:nbytes])


# -------------------------------------------------------------- framing + ECC

def _frame(body):
    out = []
    for i in range(0, len(body), DATA):
        blk = body[i:i + DATA].ljust(DATA, ALPHABET[0])
        v = [IDX[c] for c in blk]
        s0 = sum(v) % P
        s1 = sum((j + 1) * x for j, x in enumerate(v)) % P
        out.append(blk + ALPHABET[(-s0) % P] + ALPHABET[(-s1) % P])
    return "".join(out)


def _deframe(stream):
    """Returns (body, n_corrected, n_uncorrectable)."""
    body, fixed, broken = [], 0, 0
    W = DATA + PAR
    for i in range(0, len(stream), W):
        blk = stream[i:i + W]
        if len(blk) < W:
            broken += 1
            body.append(blk)
            continue
        v = [IDX.get(c, 0) for c in blk[:DATA]]
        p0, p1 = IDX.get(blk[DATA], 0), IDX.get(blk[DATA + 1], 0)
        d0 = (sum(v) + p0) % P
        d1 = (sum((j + 1) * x for j, x in enumerate(v)) + p1) % P
        if d0 == 0 and d1 == 0:
            body.append(blk[:DATA])
            continue
        if d0 == 0:                       # parity glyph itself was hit, or 2+ errors
            broken += 1
            body.append(blk[:DATA])
            continue
        j = (d1 * _inv(d0)) % P - 1
        if 0 <= j < DATA:
            v[j] = (v[j] - d0) % P
            body.append("".join(ALPHABET[x] for x in v))
            fixed += 1
        else:
            broken += 1
            body.append(blk[:DATA])
    return "".join(body), fixed, broken


# --------------------------------------------------------------- public API

def _ser(graph, root):
    recs = []
    for a in graph.topo(root):
        n = graph[a]
        recs.append([n.op, [x[:8] for x in n.args],
                     {k: _plain(v) for k, v in sorted(n.attrs.items())},
                     repr(n.ty), a[:8]])
    return json.dumps([root[:8], recs], separators=(",", ":")).encode()


def _plain(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return repr(v)


def encode(graph, root):
    """DAG -> TACET glyph stream (error-correcting framed)."""
    body, n = _bytes_to_glyphs(_ser(graph, root))
    return "%s%s" % (_frame(body), ""), n


def decode(stream, nbytes):
    """TACET glyph stream -> (payload bytes, corrected, uncorrectable)."""
    body, fixed, broken = _deframe(stream)
    return _glyphs_to_bytes(body, nbytes), fixed, broken


def corrupt(stream, pos, rng=None):
    """Substitute one glyph -- a mis-sample, or a cosmic ray."""
    import random
    rng = rng or random.Random(0)
    cur = stream[pos]
    new = cur
    while new == cur:
        new = ALPHABET[rng.randrange(P)]
    return stream[:pos] + new + stream[pos + 1:]
