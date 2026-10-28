"""
Explainer -- a NON-AUTHORITATIVE view of a KEEL definition.

This exists for debugging and for nothing else. It is deliberately NOT part of
the approval path:

  * it is lossy -- proofs, intervals, error bounds and cost are flattened or
    dropped, and the names are invented here, not in the program
  * it can be wrong, and when it is wrong a human has no way to notice,
    because the thing it is paraphrasing is unreadable by construction

Approving a definition because its explainer output looked fine is the single
most dangerous thing a person can do with this language. Authority lives in
`check.py` and in the signed envelope, never here.
"""

from .types import Vec  # noqa: F401


def _name(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return s


def explain(d, rep=None, show_cost=True):
    g = d.graph
    order = g.topo(d.root)
    nm = {a: _name(i) for i, a in enumerate(order)}
    L = []

    L.append("# NON-AUTHORITATIVE VIEW -- not an approval artifact")
    L.append("# definition %r  dal=%s  root=%s" % (d.name, d.dal, d.root[:8]))
    if d.intent:
        L.append("# intent: " + d.intent)
    L.append("")

    for a in order:
        n = g[a]
        args = ", ".join(nm[x] for x in n.args)
        extra = ""
        if n.op == "const":
            extra = repr(n.attrs.get("value"))
        elif n.op == "param":
            extra = "%r range=[%s,%s]" % (n.attrs.get("name"),
                                          n.attrs.get("lo"), n.attrs.get("hi"))
        elif n.op == "loopvar":
            extra = "0..%d" % (n.attrs.get("bound", 1) - 1)
        elif n.op == "convert":
            extra = "to %r" % n.attrs.get("to")
        elif n.op == "reduce":
            extra = "assoc=%s" % n.attrs.get("assoc")
        elif n.op == "stencil":
            extra = "offsets=%s out=%s" % (n.attrs.get("offsets"),
                                           n.attrs.get("out_level"))
        elif n.op == "clamp":
            extra = "to [%s,%s]" % (n.attrs.get("lo"), n.attrs.get("hi"))
        elif n.op == "repeat":
            extra = "x%d" % n.attrs.get("bound", 1)
        elif n.op == "monitor":
            extra = "-> dal %s via %r" % (n.attrs.get("dal"), n.attrs.get("check"))
        elif n.op == "vec3":
            extra = "@%s" % n.attrs.get("frame")
        elif n.op == "sys":
            extra = "%r effect=%s" % (n.attrs.get("name"), n.attrs.get("effect"))

        ty = rep.ty[a] if rep else n.ty
        note = ""
        if rep:
            e = rep.err.get(a, 0.0)
            ae = rep.aerr.get(a, 0.0)
            iv = rep.iv.get(a)
            bits = []
            if e:
                bits.append("rel<=%.2e" % e)
            if ae:
                bits.append("abs<=%.2e" % ae)
            if iv is not None and (iv.lo != float("-inf") or iv.hi != float("inf")):
                bits.append("in %r" % iv)
            if bits:
                note = "   # " + " ".join(bits)

        L.append("%-4s = %-10s %-22s : %s%s"
                 % (nm[a], n.op, (args + (" " + extra if extra else "")).strip(),
                    ty, note))

    L.append("")
    L.append("return %s" % nm[d.root])

    if rep and show_cost and rep.cost:
        r = rep.cost.report()
        L.append("")
        L.append("# cost  flops=%g  wcet=%g cyc  energy=%.3g pJ  AI=%.3g flop/byte"
                 % (r["flops"], r["wcet_cycles"], r["energy_pj"],
                    r["arithmetic_intensity"]))
        if r["bytes"]:
            L.append("# traffic " + "  ".join("%s=%g B" % kv
                                              for kv in sorted(r["bytes"].items())))
    return "\n".join(L)
