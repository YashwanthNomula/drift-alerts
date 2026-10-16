"""Drift detection orchestration: per-feature PSI/KS, score drift, AUC check."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import stats
from .csv_io import infer_kind, is_float, present_values
from .split import split_windows, window_range

SIGNIFICANT = "significant drift"
WATCH = "moderate shift — watch"
STABLE = "no significant change"


@dataclass
class SignalDrift:
    """Drift result for one signal (a feature or the model score)."""
    name: str
    kind: str  # "numeric" | "categorical"
    psi: float
    ks: float | None
    verdict: str
    n_ref: int
    n_cur: int
    missing_ref: int
    missing_cur: int
    summary: str = ""  # one-line human description of the shift


@dataclass
class DriftResult:
    source: str
    signals: list[SignalDrift]
    score: SignalDrift  # drift of y_score itself
    n_ref: int
    n_cur: int
    ref_start: str
    ref_end: str
    cur_start: str
    cur_end: str
    auc_ref: float | None = None
    auc_cur: float | None = None

    @property
    def drifted(self) -> list[SignalDrift]:
        return [s for s in self.signals + [self.score]
                if s.verdict == SIGNIFICANT]

    @property
    def watch(self) -> list[SignalDrift]:
        return [s for s in self.signals + [self.score]
                if s.verdict == WATCH]

    @property
    def overall(self) -> str:
        if self.drifted:
            return "drift"
        if self.watch:
            return "watch"
        return "stable"

    @property
    def auc_drop(self) -> float | None:
        """AUC lost between windows (positive means ranking got worse)."""
        if self.auc_ref is None or self.auc_cur is None:
            return None
        return self.auc_ref - self.auc_cur

    @property
    def auc_delta(self) -> float | None:
        """Signed change in AUC (negative means ranking got worse)."""
        if self.auc_ref is None or self.auc_cur is None:
            return None
        return self.auc_cur - self.auc_ref


def auc_score(y_true: list[float], y_score: list[float]) -> float | None:
    """Rank-based AUC (Mann-Whitney). None when one class is absent."""
    pairs = sorted(zip(y_score, y_true), key=lambda p: p[0])
    n = len(pairs)
    if n == 0:
        return None
    # Average ranks for ties.
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    n_pos = sum(1 for _, y in pairs if y == 1.0)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1.0)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _numeric_summary(name: str, ref: list[float], cur: list[float]) -> str:
    rmean = sum(ref) / len(ref)
    cmean = sum(cur) / len(cur)
    if rmean != 0:
        pct = (cmean - rmean) / abs(rmean) * 100
        return (f"mean moved from {fmt(rmean)} to {fmt(cmean)} "
                f"({pct:+.0f}% change)")
    return f"mean moved from {fmt(rmean)} to {fmt(cmean)}"


def _categorical_summary(ref: list[str], cur: list[str], cats: list[str],
                         ref_counts: list[int], cur_counts: list[int]) -> str:
    best, best_gap = None, -1.0
    for cat, r, c in zip(cats, ref_counts, cur_counts):
        rp = r / len(ref) if ref else 0.0
        cp = c / len(cur) if cur else 0.0
        if abs(cp - rp) > best_gap:
            best, best_gap = (cat, rp, cp), abs(cp - rp)
    if best is None:
        return "distribution changed"
    cat, rp, cp = best
    direction = "rose" if cp > rp else "fell"
    return f"'{cat}' share {direction} from {rp:.0%} to {cp:.0%}"


def fmt(x: float) -> str:
    """Compact human number formatting."""
    ax = abs(x)
    if ax >= 10000:
        s = f"{x:,.0f}"
    elif ax >= 100:
        s = f"{x:,.1f}"
    elif ax >= 1:
        s = f"{x:,.2f}"
    else:
        s = f"{x:.3f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def compare_signal(name: str, ref_vals: list[str], cur_vals: list[str],
                   psi_warning: float = stats.PSI_NO_CHANGE,
                   psi_drift: float = stats.PSI_MODERATE,
                   ks_warning: float = stats.KS_NO_CHANGE,
                   ks_drift: float = stats.KS_MODERATE) -> SignalDrift:
    ref_present, ref_missing = present_values(ref_vals)
    cur_present, cur_missing = present_values(cur_vals)
    kind = infer_kind(ref_present + cur_present)

    if kind == "numeric":
        ref_nums = [float(v) for v in ref_present]
        cur_nums = [float(v) for v in cur_present]
        psi_value, _, ref_counts, cur_counts = stats.psi_numeric(ref_nums, cur_nums)
        ks_value: float | None = stats.ks_2samp(ref_nums, cur_nums)
        verdict = stats.worst_verdict(
            stats.psi_verdict(psi_value, psi_warning, psi_drift),
            stats.ks_verdict(ks_value, ks_warning, ks_drift),
        )
        summary = _numeric_summary(name, ref_nums, cur_nums) if ref_nums and cur_nums else ""
    else:
        psi_value, cats, ref_counts, cur_counts = stats.psi_categorical(
            ref_present, cur_present)
        ks_value = None
        verdict = stats.psi_verdict(psi_value, psi_warning, psi_drift)
        summary = _categorical_summary(ref_present, cur_present, cats,
                                       ref_counts, cur_counts)

    return SignalDrift(
        name=name, kind=kind, psi=psi_value, ks=ks_value, verdict=verdict,
        n_ref=len(ref_present), n_cur=len(cur_present),
        missing_ref=ref_missing, missing_cur=cur_missing, summary=summary,
    )


def _auc_for(rows: list[dict[str, str]], label_col: str, score_col: str) -> float | None:
    yt, ys = [], []
    for r in rows:
        y, s = r.get(label_col, ""), r.get(score_col, "")
        if is_float(y) and is_float(s):
            yt.append(float(y))
            ys.append(float(s))
    # Labels must be binary for AUC; anything else means "no AUC available".
    if any(y not in (0.0, 1.0) for y in yt):
        return None
    return auc_score(yt, ys)


def detect_drift(rows: list[dict[str, str]],
                 source: str = "<rows>",
                 timestamp_col: str = "timestamp",
                 label_col: str = "y_true",
                 score_col: str = "y_score",
                 exclude: tuple[str, ...] = (),
                 ref_rows: int | None = None,
                 cur_rows: int | None = None,
                 split_at: str | None = None,
                 psi_warning: float = stats.PSI_NO_CHANGE,
                 psi_drift: float = stats.PSI_MODERATE,
                 ks_warning: float = stats.KS_NO_CHANGE,
                 ks_drift: float = stats.KS_MODERATE,
                 ) -> DriftResult:
    """Run the full drift check: per-feature PSI/KS, score drift, AUC change."""
    reserved = {timestamp_col, label_col, score_col}
    for col in (label_col, score_col):
        if col not in rows[0]:
            raise ValueError(f"required column {col!r} not found in CSV")

    ref, cur = split_windows(rows, ts_col=timestamp_col, ref_rows=ref_rows,
                             cur_rows=cur_rows, split_at=split_at)
    ref_start, ref_end = window_range(ref, timestamp_col)
    cur_start, cur_end = window_range(cur, timestamp_col)

    feature_cols = [c for c in rows[0]
                    if c not in reserved and c not in set(exclude)]
    if not feature_cols:
        raise ValueError("no feature columns left after excluding "
                         f"{sorted(reserved | set(exclude))}")

    signals = []
    for col in feature_cols:
        ref_vals = [r.get(col, "") for r in ref]
        cur_vals = [r.get(col, "") for r in cur]
        signals.append(compare_signal(col, ref_vals, cur_vals,
                                      psi_warning, psi_drift,
                                      ks_warning, ks_drift))

    score = compare_signal(score_col,
                           [r.get(score_col, "") for r in ref],
                           [r.get(score_col, "") for r in cur],
                           psi_warning, psi_drift, ks_warning, ks_drift)

    return DriftResult(
        source=source, signals=signals, score=score,
        n_ref=len(ref), n_cur=len(cur),
        ref_start=ref_start, ref_end=ref_end,
        cur_start=cur_start, cur_end=cur_end,
        auc_ref=_auc_for(ref, label_col, score_col),
        auc_cur=_auc_for(cur, label_col, score_col),
    )
