"""KEEL -- a language whose only reader is a machine.

v1 of this project (TACET) optimized for cheap generation and made failure
impossible by absorbing every error into a bottom value. That is right for
throwaway code and catastrophic for a spacecraft, where a silently absorbed
error is silent data corruption rather than a failover.

KEEL inverts the thesis:

    the author is a machine with unlimited patience for formal obligation,
    so make mandatory everything a human gave up on.

Dependent obligations, exact units with scale, reference frames, static cost
signatures, WCET bounds, proven floating point error bounds, mixed-criticality
noninterference, bitwise reproducibility, explicit memory placement. Every one
of those was dropped by mainstream languages because of HUMAN cost, not
because it lacked value.

So a KEEL program is unreadable not because information was removed but
because 10-50x more of it was added. The unreadable part is the proof, and a
proof does not need to be read -- it needs to be checked. Hence `check.py`:
small, readable, and the only thing a human must trust.

TACET survives as `wire.py`: the glyph transport encoding for token channels,
with error correction that now covers both sampling drift and radiation-
induced bit flips in stored program images.
"""

from .types import (ONE, Arr, Bool, Buf, Bytes, CD_M2, Cap, HZ, Int, KM,
                    KMH, LBF, LBF_S, M, M2, MA, MM, MPA, MPS, MPS2, MS,
                    NEWTON, NS, NSEC, OHM, PA, PSEC, RAD, RADPS, Real, Rec,
                    Rot, S, SITES, Site, Sum, US, Unit, V, Vec, is_linear,
                    site)
from .build import B
from .ir import Contract, Def, Func, Graph, Node, declared_params
from .check import check, contract_of, func
from .cost import Cost, Machine, total_cost
from .explain import explain
from .backend import Code, Unsupported, cpu, emit, isa_of, verify
from .hash import khash64, merkle, node_addr, seal
from .wire import decode, encode, corrupt

__version__ = "0.2.0"

__all__ = [
    # types
    "ONE", "Arr", "Bool", "Buf", "Bytes", "CD_M2", "Cap", "HZ", "Int", "KM",
    "KMH", "LBF", "LBF_S", "M", "M2", "MA", "MM", "MPA", "MPS", "MPS2",
    "MS", "NEWTON", "NS", "NSEC", "OHM", "PA", "PSEC", "RAD", "RADPS",
    "Real", "Rec", "Rot", "S", "SITES", "Site", "Sum", "US", "Unit", "V",
    "Vec", "is_linear", "site",
    # building
    "B", "Contract", "Def", "Func", "Graph", "Node", "declared_params",
    # checking -- the trust root
    "check", "contract_of", "func",
    # cost, explanation, machine code
    "Cost", "Machine", "total_cost", "explain",
    "Code", "Unsupported", "cpu", "emit", "isa_of", "verify",
    # addressing and transport
    "khash64", "merkle", "node_addr", "seal", "encode", "decode", "corrupt",
    "__version__",
]
