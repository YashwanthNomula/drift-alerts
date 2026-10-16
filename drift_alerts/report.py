"""Terminal and JSON rendering of drift check results."""

from __future__ import annotations

from .detect import DriftResult, SIGNIFICANT, WATCH, fmt


def _marker(verdict: str) -> str:
    if verdict == SIGNIFICANT:
        return "***"
    if verdict == WATCH:
        return " * "
    return "   "


def render_terminal(result: DriftResult) -> str:
    lines = [
        "drift-alerts — model drift check",
        f"source:    {result.source}",
        f"reference: {result.n_ref} rows  {result.ref_start} to {result.ref_end}",
        f"current:   {result.n_cur} rows  {result.cur_start} to {result.cur_end}",
        "",
        f"{'signal':<22}{'kind':<12}{'PSI':>7}{'KS':>7}  verdict",
        f"{'-' * 22}{'-' * 12}{'-' * 7}{'-' * 7}  {'-' * 7}",
    ]
    for s in result.signals + [result.score]:
        ks = f"{s.ks:>7.2f}" if s.ks is not None else f"{'—':>7}"
        lines.append(f"{_marker(s.verdict)} {s.name:<18}{s.kind:<12}"
                     f"{s.psi:>7.2f}{ks}  {s.verdict}")
    lines.append("")
    if result.auc_ref is not None and result.auc_cur is not None:
        delta = result.auc_delta or 0.0
        lines.append(f"ranking quality (y_true vs y_score): "
                     f"AUC {result.auc_ref:.2f} -> {result.auc_cur:.2f} "
                     f"(change {delta:+.2f})")
    else:
        lines.append("ranking quality: AUC unavailable "
                     "(labels missing or not binary)")
    return "\n".join(lines)


def result_to_dict(result: DriftResult, email_path: str | None = None,
                   adapter: str | None = None) -> dict:
    def sig(s):
        return {
            "name": s.name, "kind": s.kind,
            "psi": round(s.psi, 4),
            "ks": round(s.ks, 4) if s.ks is not None else None,
            "verdict": s.verdict,
            "n_ref": s.n_ref, "n_cur": s.n_cur,
            "missing_ref": s.missing_ref, "missing_cur": s.missing_cur,
            "summary": s.summary,
        }
    return {
        "source": result.source,
        "overall": result.overall,
        "reference": {"rows": result.n_ref, "start": result.ref_start,
                      "end": result.ref_end},
        "current": {"rows": result.n_cur, "start": result.cur_start,
                    "end": result.cur_end},
        "features": [sig(s) for s in result.signals],
        "score": sig(result.score),
        "auc_ref": result.auc_ref,
        "auc_cur": result.auc_cur,
        "auc_drop": result.auc_drop,
        "n_drifted": len(result.drifted),
        "n_watch": len(result.watch),
        "alert": ({"email_path": email_path, "adapter": adapter}
                  if email_path else None),
    }
