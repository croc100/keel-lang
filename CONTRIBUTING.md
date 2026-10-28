# Contributing

The thing that makes KEEL worth anything is the size of its trust root, so
the one rule that matters is about that.

## Adding to `keel/check.py`, `types.py`, `cost.py`, `hash.py`, `ir.py`

These five files are what a human has to read and believe. A change here
needs to answer, in the commit message:

1. **What failure mode does this refuse?** Not "what does it enable" — what
   does it stop. If the answer is a style preference or a convenience, it
   does not belong in the trust root.
2. **Why can it not live outside?** The builder, the parser and the backend
   are all untrusted because something re-derives or re-checks their output.
   If your change can be arranged that way, arrange it that way.
3. **How many lines does the trust root grow by?** Say the number.

New obligations must come from declared physics, not from a checklist. The
test for that: if a site declares no banks, it owes no bank proof. A rule
that fires on every program regardless of what it touches is a lint, and
lints go elsewhere.

## Adding outside the trust root

Much freer. The constraint is only that a bug there must not be able to
produce an accepted-but-wrong program:

- `build.py` — the checker re-derives every type.
- `surface.py` — the seal refuses any graph that is not the one written.
- `backend.py` — `check` gates `emit`, and `verify` re-tests on the CPU.

If you add a code path that is not covered by one of those three arguments,
say so in the pull request.

## Tests

```bash
python -m keel selftest
python -m flake8 keel tests examples
```

Every test is "must accept X" or "must reject Y". There is deliberately
nothing in between — no warning level, no degraded mode — because that is
the property under test.

Where a claim is **statistical** rather than proven (hash avalanche, bucket
spread, measured throughput), print the number. A statistical claim nobody
can see is not a claim.

## Measurements

If you add a number to the site registry, add the way to falsify it to
`calibrate.py`, or say in the commit message why a program cannot measure it.
`settle`, `prop` and `pj_byte` need an instrument, and the registry says so
rather than inventing a figure.

## What will not be merged

- A bottom value, a null, an exception, or any other way for an error to
  flow instead of being an obligation.
- Unbounded recursion or an unbounded loop. Those remove the WCET guarantee,
  which is most of the point.
- A new cryptographic primitive. Using a reviewed one is fine; writing one
  and calling it secure is the exact lie this project is arranged against.
- A check whose failure message does not say what to do about it. The reader
  is a generator that has to act on the answer.
