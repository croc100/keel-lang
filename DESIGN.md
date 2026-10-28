# Design notes

Why KEEL is shaped the way it is. The README says what it does; this says
what was decided and what was given up.

---

## 1. The inversion

Every language is a compromise with its author's patience. Dependent types,
exact units, frame-tagged vectors, WCET annotations, proven error bounds,
effect systems, linear types — all of them exist, all of them work, and all
of them were dropped from mainstream languages for the same reason: a person
will not maintain that much bookkeeping to ship a feature.

If the author is a machine, that reason evaporates. Nothing else about the
tradeoff changes — the obligations are still valuable, still checkable, still
exactly as sound. Only the cost went to zero.

So the design rule is:

> Make mandatory everything a human gave up on, and nothing else.

"Nothing else" matters as much as the first half. KEEL has no feature that
exists only because it would be clever. Every mandatory obligation traces to
a failure mode with a name.

---

## 2. No bottom value

v1 of this project (TACET) absorbed every error into a bottom value that
propagated. A program could not crash. That is the right answer for a
sampler generating throwaway code and the wrong answer for anything
load-bearing: a silently absorbed error is data corruption that arrives
later, somewhere else, wearing a plausible number.

KEEL therefore has no `⊥`, no `NaN`-as-control-flow, no null and no
exceptions. An error is not a value that flows; it is an obligation that was
not discharged, and an undischarged obligation is a rejected program.

Consequences, all of them deliberate:

- a division requires a proof that the divisor is non-zero;
- a `sqrt` requires a proof that the operand is non-negative;
- an index requires a proof that it is in range;
- a value that may be absent must be a variant, and reading one must be an
  exhaustive match.

There is no way to write "and if that fails, carry on".

---

## 3. The partition: substrate, physical law, time domain

The most important decision in the language, and the one with the most
consequences.

A conventional system would have an embedded dialect, a numerics dialect, a
UI framework and an RTOS, each with its own vocabulary for its own kind of
deadline. KEEL has one rule set, partitioned by *what the physics is*:

- a **site** is a place a value is committed to, with declared settling time,
  propagation delay, drive capability, energy per byte and clock domain;
- a **physical law** is an obligation over a typed quantity;
- a **time domain** is a clock a value is valid in.

Contact bounce on a relay, setup time on a trace, response time on an LCD and
row activation in a DRAM bank are the *same check* with different numbers.
Fan-out current and bolt shear stress are the *same check* over different
units. A spatial frame mismatch (`Vec3<ECI> + Vec3<BODY>`) and a clock domain
crossing (core + FPGA fabric) are the *same check*.

The test of this partition was set up to be falsifiable: **register execution
units in the same table as a relay contact, and reach the circuit without
adding a single new concept.** An FMA port is a place a value lands after a
stated delay — the definition a switch contact already satisfied. Eight
numbers on `Site` and one operation (`bind`) were enough, and three results
fell out that were not designed for:

1. an AVX-512 unit's own clock makes it *slower* than the core for a scalar
   multiply, visible only because seconds are counted apart from cycles;
2. computing on an FPGA fabric puts the result in the fabric's time domain,
   so the CPU/FPGA boundary is caught by the rule written for spacecraft
   frames, with zero new checker code;
3. a shared-memory bank conflict is `gcd(stride, banks)` — which is why
   padding an array by one element works, derived rather than measured.

Had any of those needed new vocabulary, the partition would have been wrong.

### Why not partition by engineering discipline

Because discipline boundaries are an artifact of how humans divide labour.
"Embedded", "numerics", "frontend" and "control" are org charts. The physics
does not have them, and a language that encodes them inherits a seam at every
boundary where two sets of rules meet and neither owns the interface.

---

## 4. Unreadable programs, readable checker

A KEEL program is unreadable. This is a design output, not a goal: the
program carries 10–50× more information than its equivalent elsewhere, and
most of that information is proof.

Control therefore cannot come from reading programs. It comes from the fact
that the *rules* are small enough to read to the end:

```
2,248 significant lines, five files.
```

Three rules keep it that way.

**Everything outside is re-derived or re-checked.** The builder declares
types; the checker derives them independently and rejects any mismatch. The
parser produces a graph; the seal rejects any graph that is not the one that
was written. The backend produces bytes; `verify` re-tests them against the
proven interval on the actual CPU. Each of those is a component that may be
buggy or hostile without producing an accepted program that is wrong.

**The uncomfortable parts stay inside.** The obligation discharger and the
floating-point error analysis would be tidier in their own modules. They are
in `check.py` because you are trusting them, and moving them would understate
the trust root.

**The explainer is explicitly non-authoritative.** `explain.py` invents names
and drops proofs. Approving a definition because its explainer output looked
fine is the single most dangerous thing a person can do with this language,
and the module's docstring says so.

---

## 5. Contracts are derived, not declared

A function's contract is read off its own proof: parameter boxes from the
parameters, result interval and error bound and cost from the checker, effect
set from the graph.

This is the opposite of the usual arrangement, where an author writes a
specification and the checker tries to believe it. The consequence is that a
library function *cannot* document a range it does not have.

An author may still publish a contract, and then it is checked for
implication: a published contract may promise **less** than the body proves —
a wider interval, a larger error, a higher cost, more effects — and may
require **more** of its caller. It may never promise more. That is what makes
information hiding possible without making it a loophole.

At a call site the body is not re-derived. The arguments must land inside the
boxes the callee was proven over; that is the precondition, and it is the
only one. Everything else is adopted.

### Why an unverified contract is a rejection

Because the alternative is a hole large enough to drive the whole design
through. If adopting a contract required no evidence, then modular
verification would be a way of *hiding* work rather than dividing it. So a
contract whose body was not supplied is an undischarged obligation.

The verification is memoised by address — which is sound with no invalidation
logic whatsoever, because an address commits to the entire body. A function
called a thousand times is checked once.

---

## 6. A loop is induction, and that is why there is always a WCET

`fold(body, init, n, lo, hi)` carries an accumulator and declares an
invariant box. Two obligations are discharged:

- the initial value is inside the box (base case);
- the body maps the box into the box (inductive step).

The loop is not unrolled or simulated. Verifying 256 iterations costs exactly
what verifying 256 million costs.

The trip count is a literal and recursion is rejected, so every program's
cost is a finite number computed from the graph. This is a real
expressiveness limit — you cannot write a `while` loop whose bound is
unknown — and it is the price of the guarantee. It was paid deliberately.

---

## 7. Linearity is an edge count

A mutable buffer and a capability are linear types. The rule is:

> A linear value must have exactly one edge pointing at it.

That is the whole of aliasing. Two names for one mutable thing *is* two
edges, and the graph is content-addressed, so it already counts edges —
reusing a buffer does not create a copy, it creates fan-out.

Nothing was invented for this. There is no borrow checker, no lifetime
inference and no region analysis; there is a dictionary of in-degrees that
was already being computed for the cost model. The same observation gives
data-race freedom for free: two subgraphs that share no linear value cannot
interfere.

The bounds check on a buffer write is the same obligation machinery that
proves a divisor is non-zero. There is no second mechanism.

---

## 8. Effects are arguments

There is no ambient authority. An effect is performed by spending a
capability token, capabilities are linear, and a definition declares a
ceiling the checker enforces.

Three properties follow:

- a function handed one network token cannot make two calls out of it;
- a function handed none cannot reach the network however it is written;
- authority is visible in the type of every function on the path, because it
  travels in the arguments.

An effect's cost is **seconds**, not cycles. A 2 ms network round trip, an
8 ms display frame and a 5 ms contact bounce are compared in the same
arithmetic as a floating-point multiply, which is the only way an end-to-end
deadline can be checked across a stack that spans a pin and a socket.

---

## 9. The written form

A KEEL program is a graph. The written form is a *serialisation* of that
graph, not a grammar that gives it meaning — which inverts the usual
relationship and is why it can look the way it does.

```
07~L1E8s?J1&IJ3uyzjv3wq1h%IIII&14ga1ij92C7mmnAejvE8yywdkwieymaA?JJKKIJ2/KIIX
IJJIJ3CzDyhHlcDFMIIRJI#Y12y0Azk7sBjvzqgoBash9G4@
```

Each removal has a reason:

- **no keywords** — the operation table is finite, so one glyph per
  operation;
- **no names** — a value is referenced by distance back in the stream. A name
  is a human's index into their own memory, and it makes two identical
  programs textually different, which breaks content addressing for nothing;
- **no delimiters** — arity is fixed per operation, so the stream parses with
  no lookahead and nothing to balance;
- **no whitespace** — every number is self-delimiting, so whitespace in a
  stream is corruption, which the TACET parity layer corrects.

A module is a sequence of definitions. A definition may call the ones before
it by back-index, and the loader checks them in order — so **a contract is
never transmitted**. It is re-derived from the body that is already in hand,
which means there is nothing in the stream to forge.

### The seal

Every definition ends with the address of its whole envelope: body, intent,
DAL, tolerances, WCET budget, deadline, signature and capability ceiling. The
loader parses, builds, recomputes the address *from what it built*, and
compares.

So a parser bug can make a correct program fail to load. It cannot make an
incorrect program load. That is what puts the parser outside the trust root.

Two holes were found by testing exactly that property — flipping one glyph at
every position and loading the result:

1. **Sealing the body alone was not enough.** 16 of 124 positions produced a
   program that loaded, and every one had landed in metadata the body's
   address does not cover. A corrupted capability ceiling is the worst thing
   on that list to accept silently, since it is the list of effects the
   program may perform.
2. **Varints were not canonical.** Prepending a zero digit decoded to the
   same value, so one glyph could be inserted invisibly. Leading zero digits
   are now refused on read.

Both were found by a test rather than by reasoning, which is the argument for
having the test.

---

## 10. Addressing: what may be invented and what may not

`hash.py` holds two things and labels them separately on purpose.

**KEEL-ADDR is security relevant.** The whole language rests on an address
naming exactly one structure. If two different nodes can be fed to the
compression function as the same byte string, one program's address certifies
another program's proof and everything above it is decoration.

So: the compression function is BLAKE2s, which has been attacked by people
who do that for a living. The *encoding* is new, because the old one —
`json.dumps(..., sort_keys=True)` — is only *probably* injective, and
"probably" is not a word the trust root may use. Every value is written as
tag/length/value, making each encoding self-delimiting and the concatenation
uniquely parseable.

The proof of injectivity is constructive and it is in the file: there is a
**decoder**, and the test suite round-trips a corpus through it. An encoding
you can decode cannot have lost anything.

Writing it found a real bug. The zigzag integer encoding was the 64-bit
identity `(n << 1) ^ (n >> 63)`, which silently corrupts anything above 2⁶³ —
and a unit scale is an exact `Fraction`, not a machine word.

**KHASH-64 is not security relevant**, which is the only reason it is allowed
to be new. It picks hash-table buckets and memoisation keys; nothing depends
on an adversary being unable to collide it. Its claim is statistical and it
is measured in the test suite rather than asserted, and its constants are
derived in six re-runnable lines from π, e, √2 and √3.

Writing a new compression function and calling it secure would be exactly the
lie this project refuses everywhere else.

---

## 11. The backend, and what it is allowed to be

`backend.py` is 844 lines of hand-written x86-64 encoding with a register
allocator. It is outside the trust root, and soundly so, for two reasons:

- `check` must accept the definition before `emit` will lower it;
- `verify` re-tests the emitted code against the proven interval on the real
  CPU.

So it may be arbitrarily clever or arbitrarily broken without producing an
accepted-but-wrong program. Three things it does with that licence:

**Register allocation.** Linear scan over the topological order with last-use
liveness and furthest-next-use spilling. A 24-parameter, 45-operation kernel
emits with zero spills and a 32-byte frame.

**Instruction selection driven by `bind`.** A vector operation lowers to one
256-bit packed instruction when, and only when, it is bound to a unit whose
declared lanes can hold it. The binding the checker proves is the binding
that is emitted.

**`fma` as one instruction with one rounding** — and *only* for an explicit
`fma` node. The backend never contracts `a*b+c` on its own, because the
checker proved that expression with two roundings and silently giving it one
would make the proof describe a program that was not run. That is
`-ffp-contract=fast`; here it is unrepresentable.

### What it refuses rather than approximates

`bind("TENSOR")` is proven against the site registry and then *refused* by
the emitter, because this backend emits x86-64 and lowering a tensor-core
binding to scalar SSE2 would be a different program than the one that was
proven. Refusing is the honest failure.

### Why the parameter order is declared

The ABI laid out its input buffer in topological order. Changing the graph
walk from recursive to iterative — a change that cannot affect any result —
silently permuted every function's arguments. A language whose ABI moves when
its traversal changes does not have an ABI, so a definition may now declare
its signature, and parameters appear in declaration order.

This also fixed a quieter problem: a function body that ignores a parameter
still *has* that parameter, and its callers have to agree about that.

---

## 12. Confronting the model with the machine

The strongest claim in the design is that obligations come from declared
physics. The weakest point is the word *declared*. `lat`, `thru`, `hz` and
`lanes` are typed in by a human reading a datasheet, and a misread datasheet
means proving the wrong thing perfectly — the checker cannot notice, because
it is reasoning correctly from what it was told.

`calibrate.py` is the answer, and it is deliberately not in the trust root.
It builds programs whose runtime is dominated by one declared number, runs
them, and reports disagreements. It decides nothing.

The method matters. A **ratio of two latencies contains no clock**, so the
model's proportions can be tested without knowing how fast the machine runs.
That separates "the model has the wrong proportions" from "the model has the
wrong clock", which are different bugs in the registry. It found four real
ones, including that the model prices a fused multiply-add the same as a
multiply while the silicon charges 1.35× — against Zen 3's documented
4-versus-3 cycles, which is 1.33.

Three measurement traps were hit on the way and are worth recording:

1. **A short dependent chain in a counted loop measures about half the real
   latency.** The reorder buffer spans the backward jump, and because each
   iteration reloads its inputs the iterations are independent. One long
   straight chain has nothing to overlap with.
2. **Timing a 9 ns kernel through 2 µs of `ctypes` measures `ctypes`.** The
   fix is to emit the body inside a counted loop and take the difference
   between two loop counts, so everything fixed cancels exactly rather than
   being estimated.
3. **The number that then comes back is throughput, not latency.** So
   `measured ≤ claimed` is *consistent with* the WCET bound and does not
   confirm it. Calling it confirmation would be measuring one thing and
   claiming another, which is the failure this whole project is arranged
   against.

A disagreement reported here is not an error in the checker. It is a bug
report against the registry, which is the only place such a bug can live.

Not measurable by a program, and therefore not claimed: `settle`, `prop`,
`pj_byte`, and every mechanical and optical site. Those need an instrument.

---

## 13. What was given up

- **Recursion.** An unbounded call depth is an unbounded cost.
- **Dynamic allocation.** Every shape is static.
- **Unbounded loops.** Every trip count is a literal.
- **Readability.** Programs are machine artifacts. The checker and the
  explainer are the human-facing surfaces.
- **Tightness of the error analysis.** Interval arithmetic is conservative,
  so a program can be rejected for a bound that is loose rather than wrong.
  Affine arithmetic or Taylor models would fix it and are not here.
- **Breadth of the backend.** The proof side spans seven layers; the code
  generator reaches one and a half.

---

## 14. The host language

The checker is implemented in Python. That is the host, not the language.

A KEEL program contains no Python. It is a content-addressed graph, written
as a glyph stream, lowered to x86-64 bytes. The builder is one authoring
surface and it is explicitly untrusted, which is why it lives in its own file
and why reimplementing it changes nothing about any program that already
exists.

That is the only real test of whether a language is independent of its host,
and it is passable now for one reason: the bottom is bytes.
