# KEEL

**A programming language whose only reader is a machine.**

KEEL programs are not written to be read. They are written to be *checked*.
The thesis is one sentence:

> Every obligation mainstream languages dropped, they dropped because of
> human cost — not because it had no value. If the author is a machine, the
> human cost is zero, so make all of it mandatory.

Exact units with scale. Reference frames. Proven floating-point error bounds.
Worst-case execution time. Worst-case energy. Mixed-criticality
non-interference. Bitwise reproducibility. Explicit memory placement. Linear
ownership. Capability-scoped effects. Every value in a KEEL program carries
all of it, and nothing loads unless a 2,251-line checker says yes.

A KEEL program is therefore unreadable not because information was removed
but because much more of it was added. The unreadable part is the proof, and
a proof does not need to be read — it needs to be checked.

```
unreadable programs, readable checker.
```

---

## What it looks like

This is a complete program. It converts road speed to SI units.

```
07~L1E8s?J1&IJ3uyzjv3wq1h%IIII&14ga1ij92C7mmnAejvE8yywdkwieymaA?JJKKIJ2/KIIX
IJJIJ3CzDyhHlcDFMIIRJI#Y12y0Azk7sBjvzqgoBash9G4@
```

No keywords, no variable names, no delimiters, no whitespace. Each of those
was removed for a reason, not for effect:

| removed | because |
|---|---|
| keywords | the operation table is finite; a machine needs no mnemonic |
| names | a name is a human's index into their own memory, and it makes two identical programs textually different — which breaks content addressing for nothing |
| delimiters | arity is fixed per operation, so the stream parses with no lookahead |
| whitespace | every number is self-delimiting, so a space in a stream is *corruption*, which the parity layer corrects |

Values are referenced by **distance back** in the stream. The last field is
the **seal**: the address of the whole definition, written down.

Here is what a human reads instead:

```console
$ python -m keel check lib/std.kl
ACCEPT  mps_from_kmh   obligations 0/0 discharged
    [0,111.111]  4 cycles  1.33e-09 s  1 pJ  effects ['none']
ACCEPT  brake_distance   obligations 2/2 discharged
    [0,5000]  28 cycles  9.33e-09 s  3 pJ  effects ['none']
...
```

And here is the program running, as real machine code, on the real CPU:

```console
$ python -m keel run lib/std.kl brake_distance 30 8
56.25

$ python -m keel verify lib/std.kl brake_distance
brake_distance: 78 bytes, isa=avx+fma
  proven interval   (0.0, 5000.0)
  points executed   25
  every answer in   True
  counterexample    None
```

There is no interpreter, no runtime and no virtual machine. `emit` produces
x86-64 bytes, the OS is asked to make them executable, and they run.

---

## Quick start

No dependencies. Python 3.8+.

```bash
git clone https://github.com/croc100/keel-lang && cd keel-lang
python -m keel selftest                  # 156 tests
python examples/language.py              # a tour of the language
python examples/silicon.py               # the same multiply on five kinds of silicon
python -m keel.std > lib/std.kl          # build the standard library
python -m keel check lib/std.kl          # and check it back from glyphs alone
python -m keel calibrate                 # confront the machine model with this machine
```

The build is reproducible, and content addressing is why: `python -m keel.std`
emits a module byte-identical to the committed `lib/std.kl`, because every
address is a function of structure alone and there is nothing else in the
stream.

---

## The inversion

v1 of this project optimised for cheap generation: it made failure impossible
by absorbing every error into a bottom value. That is right for throwaway
code and catastrophic for a vehicle, where a silently absorbed error is data
corruption rather than a failover.

So KEEL has **no bottom value**. An error is not something that flows; it is
a proof obligation that was not discharged. There are no warnings, no
severity levels and no partial success. A definition is accepted or it is
rejected, and a rejected definition does not run.

---

## Seventeen checks, one pass

| # | check | what it refuses |
|---|---|---|
| 1 | type re-derivation | the builder is untrusted — derive independently and compare |
| 2 | units | dimension **and** exact scale; no implicit coercion anywhere |
| 3 | frames | spatial frames and clock domains, by one rule |
| 4 | obligations | division by a possible zero, `sqrt` of a possible negative, an index that may be out of range, an invariant that may not hold |
| 5 | floating-point error | proven relative and absolute bounds against a declared tolerance |
| 6 | mixed criticality | low-assurance data reaching a high-assurance output |
| 7 | reproducibility | a non-deterministic reduction where bitwise determinism was required |
| 8 | placement | arithmetic where no site permits it; a read from a write-only site |
| 9 | capabilities and WCET | an effect outside the declared ceiling; cycles over budget |
| 10 | settling | sensing a site before its value is valid |
| 11 | drive | sourcing more current than a pin can supply |
| 12 | deadline | end-to-end **physical time**, which a cycle count cannot express |
| 13 | unit binding | an execution unit that cannot actually issue this operation |
| 14 | bank conflict | concurrent accesses that serialise in silicon |
| 15 | ownership | a linear value with more than one owner |
| 16 | contracts | a published contract its body does not imply |
| 17 | exhaustiveness | a variant nobody handled |

Checks 3, 8, 10, 11 and 12 are **not** separate subsystems for separate
domains. Contact bounce on a switch, setup time on a trace, response time on
a display panel and row activation in a DRAM bank are all check 10 with
different numbers. A fan-out margin and a thrust margin are both check 4 over
a typed physical quantity.

The partition is by **substrate** and **physical law** — never by engineering
discipline, because discipline boundaries are an artifact of how humans
divide labour and they appear nowhere in the physics.

---

## It is a general-purpose language

Functions, loops with state, records, variants, mutation, effects and strings
— none of which gives up any of the proof.

### A function's contract is read off its proof

```
brake(v:f64<m/s>, a:f64<m/s^2>) -> f64<m> in [0,5000] <=28 cyc pure
```

Nothing in that line was typed by an author. The parameter boxes come from
the parameters, the result interval and error bound and cycle count come out
of the checker, and `pure` is the complete effect set. **A library function
cannot document a range it does not have**, because the documentation is
generated from the proof.

At a call site the body is not re-derived. The checker verifies the arguments
land inside the boxes the callee was proven over — that is the precondition,
and it is the only one — then adopts the proven interval, error, cost and
effects. A contract whose body was not supplied is an *undischarged
obligation*, not a convenience: adopting a promise nobody checked would turn
modular verification from a way of dividing work into a way of hiding it.

### A loop is induction

```python
b.fold(lowpass, b.const(0.0, ONEV), 256, 0.0, 2.0)
```

Two facts are checked: the initial value is inside `[0,2]`, and the body maps
`[0,2]` into `[0,2]`. The loop is not simulated. Verifying 256 iterations
costs exactly what verifying 256 million costs.

The trip count is a literal, so the cost is a number. **There is no KEEL
program without a WCET.**

### No null

A value that may be absent is a variant, and the only way to read one is a
`match` the checker counts. A sensor reading that might be stale is not a
float and a flag, and certainly not a float that is sometimes NaN:

```
REJECT  forgotten
  match   match on reading is not exhaustive; unhandled: ['stale']
```

### Aliasing cannot be written down

A buffer is linear, so it must have exactly **one edge** pointing at it — and
the graph is content-addressed, so it already counts edges. Two names for one
mutable thing *is* two edges:

```
REJECT  aliased
  ownership   buf[8]<f64<m>>@DRAM is used 2 times. A linear value has exactly
              one owner, and two edges to one buffer is aliasing -- which is
              why the rule is a count and not an analysis
```

Rust's borrow checker is a fan-out count here. Nothing was invented for it.

### Effects are arguments

There is no ambient authority. An effect is reachable only by spending a
capability that was passed in, and a capability is linear — so a function
handed one network token cannot make two calls out of it, and a function
handed none cannot reach the network however it is written. Authority travels
in the arguments, which means it is visible in the type of every function on
the path.

The cost of an effect is **seconds**, so a network round trip and a display
frame are compared in the same arithmetic as a multiply.

---

## Down to the circuit, with no new concepts

The question was whether a language partitioned by *(substrate, physical law,
time domain)* could reach the circuit. Rather than assert an answer, a
falsifiable condition was set:

> Register execution units in the **same table** as a relay contact, and
> reach the circuit **without adding a single new concept**. If new concepts
> are needed, the partition was wrong.

An FMA port, a tensor core and a DSP48 slice are *places a value lands after
a stated delay* — the definition a switch contact already satisfied. So
`Site` gained eight numbers and the language gained one operation, `bind`.
That was all.

| field | meaning |
|---|---|
| `lanes` | values per issue (0 = not an execution unit) |
| `thru` / `lat` | issues per cycle / result latency in cycles |
| `port` | the contended issue resource; same port means mutual exclusion |
| `hz` | the unit's **own** clock rate (0 = the core's) |
| `fmts` | the float formats its datapath carries |
| `banks` | independent parallel sub-arrays |
| `ooo` | does hardware schedule issue out of order |

Three things then came for free.

**A unit's own clock shows up in physical time.** The same scalar multiply:
ALU64 2.997 ns, FMA512 **3.568 ns**. AVX-512 is *slower*, because that unit
really is clocked below the core it sits in. Not a heuristic in a cost table
— one field, visible only because KEEL counts seconds separately from cycles.

**A fabric is a time domain.** Binding to DSP48 puts the result in the
`FPGA_200M` domain, so adding it to a core-clocked value is the *same
violation* as `Vec3<ECI> + Vec3<BODY>`. Zero lines of clock-crossing logic
were added to the checker.

**A bank conflict is `gcd`, not folklore.** With B banks and stride s the
residues {i·s mod B} form a subgroup of order B/gcd(s,B), so serialisation is
forced by arithmetic:

| stride | serialise | cycles |
|---|---|---|
| 1 | ×1 | 60 |
| 16 | ×16 | 960 |
| 32 | ×32 | 1,920 |
| **33** | **×1** | **60** |

Every GPU programmer knows to pad by one. No language above assembly can tell
them *why*, or check that they did.

### The one lie this language must not tell

On an out-of-order core, no instruction addresses the renamer or the
scheduler. So `bind` to an `ooo=True` unit is a **constraint to be proven**,
and the report says `pressure-bound`. On a tensor core or an FPGA slice the
same operation really is a **command**, and the report says `commanded`.

Calling a constraint a command is the single thing this language must never
do, and one bit on the site registry keeps it honest.

---

## The proof meets the silicon

Interval arithmetic proves a result lies in `[lo,hi]` *in a model*. Until the
program runs on real hardware, that is a statement about the model. So:

- `check` must accept a definition before `emit` will lower it.
- `verify` re-runs the emitted code over the corners and interior of the
  proven input box and asserts every answer lands inside the proven interval.

Eight operations × twelve input pairs are **bit-exact against IEEE 754**, on
both the AVX and the SSE2-only code paths. A language that never confronts
its model with the silicon has not earned the word *proven*.

### `fma` is an operation, not an optimisation

`a*b+c` and `fma(a,b,c)` compute **different numbers**:

```
0.1 * 0.1 + 0.1
  fused    0.11000000000000000056
  unfused  0.11000000000000001443
```

A compiler that contracts the first into the second has replaced the
expression whose error was analysed with one that was not, so the analysis
describes a program that was never run. That is `-ffp-contract=fast`, and
here it is unrepresentable: `fma` is a separate operation, the author asks
for the single rounding, the checker proves the tighter bound —

```
fma        relative error 1.110e-16    4 cycles
mul + add  relative error 4.148e-15    8 cycles
```

— and **the backend never contracts on its own**.

### `bind` selects the instruction

The gap this project used to admit — `bind("FMA512")` being proven but
lowered as scalar code anyway — is closed. A vector operation lowers to one
256-bit packed instruction when, and only when, it is bound to a unit whose
declared lanes can hold it:

```
unbound:  6 scalar lane operations
bound:    2 packed instructions, same answer
```

The binding the checker proves is the binding that is emitted.

---

## The registry, confronted

KEEL's strongest claim is that obligations come from declared physics. Its
weakest point is the word **declared**: `lat`, `thru`, `hz` and `lanes` are
typed in by a human reading a datasheet, and a misread datasheet means
proving the wrong thing perfectly. The checker cannot notice, because it is
reasoning correctly from what it was told.

`python -m keel calibrate` builds programs whose runtime is dominated by one
declared number, runs them, and reports the disagreements. On the machine
this was developed on it found four real bugs in the registry:

```
machine: AMD Ryzen 5 5600G with Radeon Graphics
  avx=True fma=True avx512f=False -> 4 f64 lanes per instruction

  FMA512   declares 8 lane(s), machine offers 4   OVER-DECLARED

latency ratios, which contain no clock at all
  op       measured   declared  meas/mul  decl/mul
  mul      0.677 ns     4.0 cyc     1.00x     1.00x
  add      0.677 ns     4.0 cyc     1.00x     1.00x
  fma      0.913 ns     4.0 cyc     1.35x     1.00x   <-- wrong
  div      3.015 ns    20.0 cyc     4.45x     5.00x
  sqrt     4.653 ns    22.0 cyc     6.87x     5.50x   <-- wrong

machine model declares 3.00 GHz; a dependent chain of 4-cycle
operations implies 5.96 GHz (99% off)
```

A **ratio contains no clock**, so the model's proportions can be tested
without knowing how fast the machine runs. That separates "the model has the
wrong proportions" from "the model has the wrong clock" — different bugs. The
`fma` row is the good one: the model says a fused multiply-add costs the same
as a multiply; the silicon says 1.35×, and Zen 3's documented 4-versus-3
cycles is 1.33.

A disagreement here is **not** an error in the checker. It is a bug report
against the registry, which is the only place such a bug can live.

Not measurable by a program, and so not claimed: `settle`, `prop`, `pj_byte`,
and every mechanical and optical site. Those need an instrument.

---

## Addressing

A node's address is `BLAKE2s` over a **tag/length/value encoding** that is
injective by construction, with BLAKE2's own personalisation for domain
separation.

The compression function is not new, and that is deliberate: writing one and
calling it secure would be exactly the lie this project refuses everywhere
else. What *is* new is the encoding, because the old one — `json.dumps(...,
sort_keys=True)` — is *probably* injective, and "probably" is not a word the
trust root may use.

Injectivity is **constructively checkable**: there is a decoder, and the test
suite round-trips a corpus through it. An encoding you can decode cannot have
lost anything. Writing it found a real bug: the zigzag integer encoding was
the 64-bit identity `(n << 1) ^ (n >> 63)`, which silently corrupts anything
above 2⁶³ — and a unit scale is an exact `Fraction`, not a machine word.

Two properties, both required and different:

- **injective** — no two unequal values share bytes.
- **canonical** — two spellings of one mapping encode identically, so
  attribute insertion order cannot change an address.

### KHASH-64 — new, fast, and *not* cryptographic

For interning and the lemma cache. Nothing depends on an attacker being
unable to collide it, which is the only reason it is allowed to be new. Its
claim is statistical and it is **measured**, not asserted:

```
per-bit avalanche   min 0.4927  mean 0.5005  max 0.5080   (ideal 0.5)
bucket chi-square   1050 over 1024 buckets                (ideal ~1023)
aligned pointers    945
32-bit collisions   0 in 50k keys                         (expect 0.29)
```

Constants are derived in six re-runnable lines from the fractional parts of
π, e, √2 and √3. Nothing is up the sleeve.

---

## The seal, and why the parser is untrusted

Every definition ends with its own address. The loader parses the glyphs,
builds the graph, recomputes the address **from the graph it actually built**,
and compares.

So a bug in the parser cannot make a wrong program load. It can only make a
right program fail to load. That is what puts the parser outside the trust
root, alongside the builder and the backend.

The property is tested by flipping one glyph at every position of a stream and
loading the result. The only acceptable outcomes are *refused* and *produced
the identical program*. The first run of that test found two real holes:

1. **16 of 124 positions loaded as a different program.** Every one had
   landed in metadata — the name, the DAL, a tolerance, the capability
   ceiling — which the body's address does not cover. A corrupted ceiling is
   the worst possible thing to accept silently, since it is the list of
   effects the program may perform. The seal now covers the whole envelope.

2. **One position still loaded.** Varints were not canonical: prepending a
   zero digit decodes to the same value, so a glyph could be inserted
   invisibly. Leading zero digits are now refused on read.

```
one glyph changed, at each of 124 positions:
   24 refused by the seal, 100 refused while parsing, 0 loaded as a different program
```

---

## The trust root

**2,251 significant lines, five files.** That is the whole of what a human
must read and believe.

| in the trust root | lines | why |
|---|---|---|
| `check.py` | 1,026 | the seventeen checks, the obligation discharger, the error analysis |
| `types.py` | 431 | units with exact scale, frames, sites, linearity |
| `cost.py` | 275 | the cost semantics and the machine model |
| `hash.py` | 298 | the address encoding — if it is not injective, nothing above it means anything |
| `ir.py` | 221 | the graph, the envelope, contract implication |

| outside it | lines | why it is safe |
|---|---|---|
| `build.py` | 346 | the builder declares types; the checker re-derives all of them |
| `backend.py` | 843 | `check` must accept before `emit` lowers, and `verify` re-tests on the CPU |
| `surface.py` | 594 | a misparse produces a different address, which the seal refuses |
| `calibrate.py` | 219 | it reports disagreements; it never decides anything |
| `std.py` | 166 | ordinary programs, checked like any other |
| `explain.py` | 88 | explicitly non-authoritative |
| `wire.py` | 129 | transport only; corruption is corrected or detected |
| `__main__.py` | 180 | a command line |

The obligation discharger and the error analysis are deliberately *inside*
`check.py`, even though it would be tidier to move them out. You are trusting
them, so hiding them in another module would be dishonest about the size of
the trust root.

---

## Honest limits

- **The backend covers scalar `f64` and 3-lane packed `f64`, on x86-64.** The
  proof side spans seven layers; the code generator reaches one and a half.
  There is no scheduling: the port pressure the checker computes is a bound
  the emitter does not yet try to meet. `bind("TENSOR")` is proven and then
  *refused* by the emitter rather than quietly lowered to something else.
- **The site numbers are declared by a human.** `calibrate` makes them
  falsifiable, and it currently reports four disagreements on the development
  machine — which have not yet been fixed in the registry, because a machine
  model that is right for one CPU is wrong for the next one, and picking the
  right abstraction there is unfinished work.
- **Interval arithmetic is conservative.** A program can be rejected for an
  error bound that is loose rather than wrong. Tightening it needs affine or
  Taylor-model arithmetic, which is not here.
- **No recursion, by design.** An unbounded call depth is an unbounded cost.
  This is a real expressiveness limit and it is the price of the WCET
  guarantee.
- **No dynamic allocation, by design.** Every shape is static.
- **Bytes and effects do not reach the backend.** They check, they cost, they
  do not yet emit — so the standard library stops where honesty does.
- **The envelope is sealed but not signed.** There is no cryptographic
  signature and no loader enforcement; addressing is ready, signing is not.
- **You still have to trust 2,251 lines.** That is the one thing left to a
  human, and it is the only part of this design that cannot be automated
  away.

---

## Layout

```
keel/
  check.py      THE TRUST ROOT -- the only file you must read
  types.py      units, frames, sites, linearity
  cost.py       cost semantics: bytes moved, cycles, seconds, picojoules
  hash.py       KEEL-ADDR (injective encoding) and KHASH-64
  ir.py         the graph, the envelope, contracts
  build.py      the builder            (untrusted)
  surface.py    KEEL-GLYPH: the written form and its seal  (untrusted)
  backend.py    x86-64 code generation (untrusted)
  calibrate.py  the registry against the silicon
  explain.py    a non-authoritative human view
  wire.py       TACET: glyph transport with error correction
  std.py        the standard library, written in KEEL
examples/
  language.py   functions, loops, variants, ownership, effects, the glyph form
  silicon.py    the same multiply on five kinds of silicon, down to the bytes
  substrate.py  switches, panels, bolts, DRAM -- one rule, different numbers
  demo.py       the original walkthrough
tests/
  test_keel.py  98 tests -- units, frames, obligations, error, DAL, silicon
  test_lang.py  58 tests -- the language, addressing, the hash, the glyph form
```

---

## On the host language

The checker is implemented in Python. That is the host, not the language.

A KEEL program contains no Python: it is a content-addressed graph, written as
a glyph stream, lowered to x86-64 bytes. The builder is one authoring surface
and it is explicitly untrusted. Reimplementing any of it in another language
changes nothing about any program that already exists — which is the only
real test of whether a language is independent of its host.

The bottom is bytes. That makes the implementation language replaceable.

## Further reading

- [DESIGN.md](DESIGN.md) — what was decided and what was given up.
- [DESIGN-v2.md](DESIGN-v2.md), [PRINCIPLES.md](PRINCIPLES.md) — the original
  design notes, in Korean. They document the earlier state of the language,
  before functions, loops, variants, ownership and effects existed, and they
  are kept as the record rather than rewritten.

## License

MIT. See [LICENSE](LICENSE).
