"""
KEEL addressing and hashing.

Two things live here and they are not the same kind of thing, so they are
labelled separately and loudly.

1. KEEL-ADDR -- the content address of a node. SECURITY RELEVANT.

   The whole language rests on the claim that an address names exactly one
   structure. If two different nodes can be fed to the compression function
   as the same byte string, then one program's address certifies another
   program's proof, and everything above it is decoration.

   What is invented here is the ENCODING, not the primitive. The compression
   function is BLAKE2s, which has been attacked by people who do that for a
   living. Writing a new compression function and calling it secure would be
   exactly the lie this project refuses to tell everywhere else.

   What the encoding has to be is INJECTIVE: distinct structures must produce
   distinct bytes. The old encoding was `json.dumps(..., sort_keys=True)`,
   which is probably injective and is not provably so -- a string containing a
   bracket, a float whose repr is platform-shaped, a dict key that collides
   with a list rendering. "Probably" is not a word the trust root may use.

   So every value here is written as tag / length / payload, which makes each
   encoding self-delimiting and therefore the concatenation uniquely
   parseable. The proof of that is constructive and it is in this file: there
   is a DECODER, and `tests` round-trips a corpus through it. An encoding you
   can decode cannot have lost anything.

2. KHASH-64 -- a fast 64-bit hash for interning and the lemma cache.
   NOT CRYPTOGRAPHIC. NOT COLLISION RESISTANT AGAINST AN ADVERSARY.

   This one is new, and the reason it is allowed to be new is that nothing
   depends on an attacker being unable to collide it. It picks hash-table
   buckets and memoisation keys. Its claim is statistical and it is measured
   rather than asserted: avalanche, bias and bucket distribution are computed
   in the test suite and the numbers are printed, so a reader can see what
   they are getting instead of being told it is good.

   Its constants are derived in this file from the fractional digits of pi, e,
   sqrt(2) and sqrt(3). Nothing is up the sleeve: the derivation is six lines
   of arithmetic you can re-run.
"""

import hashlib
import struct
from fractions import Fraction

MASK64 = (1 << 64) - 1
ADDR_BYTES = 8                 # 16 hex characters
PERSON_NODE = b"KEEL-NOD"      # domain separation, via BLAKE2's own parameter
PERSON_MERKLE = b"KEEL-MRK"
PERSON_DEF = b"KEEL-DEF"


# =====================================================================
# 1. the canonical, injective encoding
# =====================================================================

T_NULL, T_FALSE, T_TRUE = 0x01, 0x02, 0x03
T_INT, T_FLOAT, T_STR, T_BYTES = 0x04, 0x05, 0x06, 0x07
T_LIST, T_DICT, T_UNIT, T_ADDR = 0x08, 0x09, 0x0A, 0x0B
T_FRAC = 0x0C


class NotEncodable(Exception):
    """Raised rather than guessed. An attribute the address function does not
    understand must stop the build, because an address that silently ignored
    part of a node would certify something other than what runs."""


def _uvarint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _read_uvarint(buf, i):
    n, shift = 0, 0
    while True:
        b = buf[i]
        i += 1
        n |= (b & 0x7F) << shift
        if not (b & 0x80):
            return n, i
        shift += 7


def _zigzag(n):
    """Arbitrary precision, deliberately. The usual (n << 1) ^ (n >> 63) form
    is a 64-bit identity, and KEEL integers are not 64-bit: a unit scale is an
    exact Fraction and an array index is a mathematical integer. Applying the
    64-bit trick to a bignum silently corrupts anything above 2^63, which an
    address function may not do."""
    return 2 * n if n >= 0 else -2 * n - 1


def _unzigzag(n):
    return (n >> 1) if not (n & 1) else -((n + 1) >> 1)


def encode(v):
    """Canonical bytes for one value. Self-delimiting, therefore injective
    under concatenation."""
    if v is None:
        return bytes([T_NULL])
    if v is True:
        return bytes([T_TRUE])
    if v is False:
        return bytes([T_FALSE])
    if isinstance(v, int):
        p = _uvarint(_zigzag(v))
        return bytes([T_INT]) + _uvarint(len(p)) + p
    if isinstance(v, float):
        if v != v:
            # A NaN in an attribute is a bug upstream, and two NaNs with
            # different payloads are the same float to ==. Encoding one would
            # make the address function non-injective on exactly the value
            # that already means "something went wrong".
            raise NotEncodable("NaN cannot be addressed")
        p = struct.pack(">d", v)
        return bytes([T_FLOAT]) + _uvarint(8) + p
    if isinstance(v, str):
        p = v.encode("utf-8")
        return bytes([T_STR]) + _uvarint(len(p)) + p
    if isinstance(v, (bytes, bytearray)):
        return bytes([T_BYTES]) + _uvarint(len(v)) + bytes(v)
    if isinstance(v, Fraction):
        a = encode(v.numerator) + encode(v.denominator)
        return bytes([T_FRAC]) + _uvarint(len(a)) + a
    if isinstance(v, (list, tuple)):
        parts = b"".join(encode(x) for x in v)
        return bytes([T_LIST]) + _uvarint(len(v)) + _uvarint(len(parts)) + parts
    if isinstance(v, dict):
        keys = sorted(v)
        for k in keys:
            if not isinstance(k, str):
                raise NotEncodable("dict keys must be strings, got %r" % (k,))
        parts = b"".join(encode(k) + encode(v[k]) for k in keys)
        return (bytes([T_DICT]) + _uvarint(len(keys))
                + _uvarint(len(parts)) + parts)
    # A Unit is given its own tag rather than being flattened into a list, so
    # that a unit and a list that happens to look like one cannot collide.
    if hasattr(v, "dim") and hasattr(v, "scale"):
        a = encode(list(v.dim)) + encode(Fraction(v.scale))
        return bytes([T_UNIT]) + _uvarint(len(a)) + a
    if hasattr(v, "canon") and callable(v.canon):
        a = encode(v.canon())
        return bytes([T_ADDR]) + _uvarint(len(a)) + a
    raise NotEncodable("cannot address a %s" % type(v).__name__)


def decode(buf, i=0):
    """The decoder exists to make injectivity CHECKABLE rather than claimed.
    If every structure in a corpus survives encode->decode unchanged, then no
    two of them encoded to the same bytes."""
    t = buf[i]
    i += 1
    if t == T_NULL:
        return None, i
    if t == T_TRUE:
        return True, i
    if t == T_FALSE:
        return False, i
    if t == T_INT:
        n, i = _read_uvarint(buf, i)
        raw, j = _read_uvarint(buf, i)
        return _unzigzag(raw), i + n
    if t == T_FLOAT:
        n, i = _read_uvarint(buf, i)
        return struct.unpack(">d", buf[i:i + 8])[0], i + n
    if t == T_STR:
        n, i = _read_uvarint(buf, i)
        return buf[i:i + n].decode("utf-8"), i + n
    if t == T_BYTES:
        n, i = _read_uvarint(buf, i)
        return buf[i:i + n], i + n
    if t == T_FRAC:
        n, i = _read_uvarint(buf, i)
        num, j = decode(buf, i)
        den, j = decode(buf, j)
        return Fraction(num, den), i + n
    if t == T_LIST:
        cnt, i = _read_uvarint(buf, i)
        n, i = _read_uvarint(buf, i)
        end, out = i + n, []
        for _ in range(cnt):
            v, i = decode(buf, i)
            out.append(v)
        return out, end
    if t == T_DICT:
        cnt, i = _read_uvarint(buf, i)
        n, i = _read_uvarint(buf, i)
        end, out = i + n, {}
        for _ in range(cnt):
            k, i = decode(buf, i)
            v, i = decode(buf, i)
            out[k] = v
        return out, end
    if t in (T_UNIT, T_ADDR):
        n, i = _read_uvarint(buf, i)
        v, _ = decode(buf, i)
        return (t, v), i + n
    raise NotEncodable("unknown tag 0x%02x at %d" % (t, i - 1))


def roundtrip(v):
    out, i = decode(encode(v))
    return out


# =====================================================================
# the address functions
# =====================================================================

def node_addr(op, args, attrs, ty_canon, nbytes=ADDR_BYTES):
    """KEEL-ADDR. The address of a node is a hash of what the node IS:
    its operation, the addresses of its arguments, its attributes and its
    declared type. Change any of them and every value downstream is renamed,
    which is what makes a program's root address a certificate for the whole
    program."""
    blob = encode([op, list(args), attrs, ty_canon])
    return hashlib.blake2s(blob, digest_size=nbytes,
                           person=PERSON_NODE).hexdigest()


def merkle(addrs, nbytes=ADDR_BYTES):
    """A binary Merkle root over an ordered list of addresses.

    An odd node is carried up rather than duplicated. Duplicating it is the
    classic second-preimage bug: a tree over [a, b, c] and one over
    [a, b, c, c] would otherwise have the same root.
    """
    if not addrs:
        return hashlib.blake2s(b"", digest_size=nbytes,
                               person=PERSON_MERKLE).hexdigest()
    level = [bytes.fromhex(a) for a in addrs]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(hashlib.blake2s(
                b"\x01" + level[i] + level[i + 1], digest_size=nbytes,
                person=PERSON_MERKLE).digest())
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0].hex()


def seal(root_addr, intent, contract_canon, ceiling, nbytes=ADDR_BYTES):
    """The envelope address: body, intent, contract and ceiling in one digest.

    This is the thing a loader would check a signature over. The body alone
    is not enough, because the same body under a wider capability ceiling is
    a different program to anyone deciding whether to run it.
    """
    blob = encode([root_addr, intent, contract_canon,
                   sorted(ceiling or ())])
    return hashlib.blake2s(blob, digest_size=nbytes,
                           person=PERSON_DEF).hexdigest()


# =====================================================================
# 2. KHASH-64 -- new, fast, and NOT cryptographic
# =====================================================================

def _nothing_up_the_sleeve():
    """Constants from the fractional parts of pi, e, sqrt(2), sqrt(3).

    Re-runnable in six lines, which is the only reason to trust a magic
    number: not that someone says it is fine, but that you can see where it
    came from. Each is forced odd, because an even multiplier throws away a
    bit of state per round.
    """
    from decimal import Decimal, getcontext
    getcontext().prec = 60
    src = [Decimal(0), Decimal(0), Decimal(2).sqrt(), Decimal(3).sqrt()]
    # pi and e to 50 places, as digit strings, so no library is trusted for
    # them either
    src[0] = Decimal("3.14159265358979323846264338327950288419716939937510")
    src[1] = Decimal("2.71828182845904523536028747135266249775724709369995")
    out = []
    for x in src:
        frac = x - int(x)
        out.append((int(frac * (1 << 64)) & MASK64) | 1)
    return tuple(out)


K0, K1, K2, K3 = _nothing_up_the_sleeve()


def _mum(a, b):
    """Folded 128-bit multiply. One multiply diffuses every input bit into
    both halves of the product; xoring the halves keeps that diffusion in 64
    bits. It is not invertible, which for a hash is the point."""
    r = (a * b) & ((1 << 128) - 1)
    return ((r & MASK64) ^ (r >> 64)) & MASK64


def khash64(data, seed=0):
    """A 64-bit hash of a byte string.

    Structure: length is mixed in FIRST, so that "ab" and "abc" diverge from
    the first round rather than at the end; the body is consumed eight bytes
    at a time through a folded multiply; the tail is read with an OVERLAPPING
    eight-byte window instead of byte by byte, which keeps the inner loop
    branch-free at the cost of reading some bytes twice -- harmless, because
    the length is already in the state.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    n = len(data)
    h = (seed ^ K0) & MASK64
    h = _mum(h ^ n, K1)
    i = 0
    while i + 8 <= n:
        v = int.from_bytes(data[i:i + 8], "little")
        h = _mum(h ^ v, K2)
        i += 8
    if i < n:
        if n >= 8:
            v = int.from_bytes(data[n - 8:], "little")   # overlapping window
        else:
            v = int.from_bytes(data[i:] + b"\x00" * (8 - (n - i)), "little")
        h = _mum(h ^ v, K2)
    h = _mum(h ^ (h >> 29), K3)
    return (h ^ (h >> 32)) & MASK64


def khash_bits(data, bits=64, seed=0):
    return khash64(data, seed) >> (64 - bits)


# ----------------------------------------------------- measuring, not claiming

def avalanche(nbytes=8, trials=512, seed=0):
    """For each input bit, the fraction of output bits it flips.

    A good mixer sits at 0.5. Returned as (min, mean, max) over all input
    bit positions, so a single stuck bit cannot hide behind a good average.
    """
    import random
    rng = random.Random(seed)
    nbits = nbytes * 8
    flips = [0] * nbits
    for _ in range(trials):
        base = bytearray(rng.getrandbits(8) for _ in range(nbytes))
        h0 = khash64(bytes(base))
        for b in range(nbits):
            base[b // 8] ^= 1 << (b % 8)
            d = h0 ^ khash64(bytes(base))
            base[b // 8] ^= 1 << (b % 8)
            flips[b] += bin(d).count("1")
    rates = [f / float(trials * 64) for f in flips]
    return min(rates), sum(rates) / len(rates), max(rates)


def bucket_chi2(keys, nbuckets=1024):
    """Chi-square of bucket occupancy. For a uniform hash the statistic is
    about nbuckets-1; far above that means clumping."""
    counts = [0] * nbuckets
    for k in keys:
        counts[khash64(k) % nbuckets] += 1
    exp = len(keys) / float(nbuckets)
    return sum((c - exp) ** 2 / exp for c in counts) if exp else 0.0


def collisions(keys, bits=32):
    """Distinct keys minus distinct hashes: every key that had to share."""
    ks = set(keys)
    return len(ks) - len({khash_bits(k, bits) for k in ks})
