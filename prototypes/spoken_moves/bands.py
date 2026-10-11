"""Band selection, pass rule, threshold fit and metrics for the FEAT-36 harness. Pure python (no numpy, no onnx).

A row is a dict: id, text, lang, label ('talk' or a move), accept (tuple of moves), scores ({legal move: score}),
talk (score), word (None | ('sure' | 'unsure', [moves]) from the word list).
"""
import itertools
import random

KEYS = ("t_sure", "margin", "g_sure", "t_floor", "g_floor")


def features(scores, talk):
    """(m1, m2, s1, margin, talk_gap) for per-move scores; m2 is None with a single legal move."""
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    m1, s1 = ranked[0]
    m2, s2 = ranked[1] if len(ranked) > 1 else (None, 0.0)
    return m1, m2, s1, s1 - s2, s1 - talk


def band_of(feat, th):
    """(band, buttons) from embedding features and thresholds."""
    m1, m2, s1, margin, gap = feat
    if s1 >= th["t_sure"] and margin >= th["margin"] and gap >= th["g_sure"]:
        return "sure", [m1]
    if s1 >= th["t_floor"] and gap >= th["g_floor"]:
        return "unsure", [m1, m2] if m2 else [m1]
    return "silent", []


def mask_renorm(probs, legal):
    """({legal move: p}, p_talk) from class probabilities {move or 'talk': p}: drop illegal moves, renormalize."""
    keep = {m: p for m, p in probs.items() if m == "talk" or m in legal}
    total = sum(keep.values()) or 1.0
    return {m: p / total for m, p in keep.items() if m != "talk"}, keep.get("talk", 0.0) / total


def word_of(row, cascade):
    """The row's word-list reading when cascade allows it. cascade: False/None = no word list; True = N 1;
    an int N = only when at most N tokens besides the move word and fillers remain (row 'extra', default 0)."""
    if cascade is False or cascade is None:
        return None
    limit = 1 if cascade is True else cascade
    return row.get("word") if row.get("extra", 0) <= limit else None


def decide(row, th, cascade):
    """(band, buttons) for a row; with cascade the word list is consulted first."""
    feat = features(row["scores"], row["talk"])
    word = word_of(row, cascade)
    if word:
        kind, moves = word
        buttons = list(moves)
        if kind == "unsure":
            for m in (feat[0], feat[1]):
                if len(buttons) >= 2:
                    break
                if m and m not in buttons:
                    buttons.append(m)
            return "unsure", buttons[:2]
        return "sure", buttons[:1]
    return band_of(feat, th)


def passes(row, buttons):
    """Move row: label or an accepted move is a button. Talk row: silent, or a button is an accepted move."""
    if row["label"] == "talk":
        return not buttons or any(b in row["accept"] for b in buttons)
    return row["label"] in buttons or any(a in buttons for a in row["accept"])


def evaluate(rows, outcomes):
    """Metrics from rows and their (band, buttons) outcomes."""
    out = {"move_n": 0, "move_pass": 0, "talk_n": 0, "talk_silent": 0, "sure_right": 0, "sure_wrong": 0,
           "by_lang": {}, "by_label": {}, "failures": []}
    for row, (band, buttons) in zip(rows, outcomes):
        ok = passes(row, buttons)
        talk = row["label"] == "talk"
        if talk:
            out["talk_n"] += 1
            out["talk_silent"] += band == "silent"
        else:
            out["move_n"] += 1
            out["move_pass"] += ok
            if band == "sure":
                out["sure_right" if ok else "sure_wrong"] += 1
        for key, name in (("by_lang", row["lang"]), ("by_label", row["label"])):
            n, p = out[key].get(name, (0, 0))
            out[key][name] = (n + 1, p + ok)
        if not ok:
            out["failures"].append((row, band, buttons))
    out["move_rate"] = out["move_pass"] / out["move_n"] if out["move_n"] else 1.0
    out["talk_rate"] = out["talk_silent"] / out["talk_n"] if out["talk_n"] else 1.0
    return out


def run(rows, th, cascade):
    return evaluate(rows, [decide(r, th, cascade) for r in rows])


def quantile_values(values, n):
    vals = sorted(values)
    if not vals:
        return [0.0]
    picks = {vals[round(i * (len(vals) - 1) / (n - 1))] for i in range(n)} if n > 1 else {vals[0]}
    return sorted(round(v, 4) for v in picks)


def default_grid(rows, n=12):
    """Candidate thresholds from quantiles of the rows' own features (plus 'off' values: -inf-like floors, never-sure highs)."""
    feats = [features(r["scores"], r["talk"]) for r in rows]
    s1, mg, gp = ([f[i] for f in feats] for i in (2, 3, 4))
    lo = min(s1 + gp) - 1.0
    hi = max(s1 + mg + gp) + 1.0
    return {"t_sure": quantile_values(s1, n) + [hi], "margin": [0.0] + quantile_values(mg, n) + [hi],
            "g_sure": quantile_values(gp, n) + [hi], "t_floor": [lo] + quantile_values(s1, n),
            "g_floor": [lo] + quantile_values(gp, n)}


def _compile(rows, cascade):
    """Split rows into fixed outcomes (word list) and embedding rows reduced to what the thresholds decide."""
    base = {"move_pass": 0, "talk_silent": 0, "sure_right": 0, "sure_wrong": 0}
    emb = []
    for r in rows:
        talk = r["label"] == "talk"
        if word_of(r, cascade):
            band, buttons = decide(r, {}, cascade)
            ok = passes(r, buttons)
            base["talk_silent"] += talk and band == "silent"
            base["move_pass"] += ok and not talk
            if not talk and band == "sure":
                base["sure_right" if ok else "sure_wrong"] += 1
            continue
        m1, m2, s1, margin, gap = features(r["scores"], r["talk"])
        emb.append((s1, margin, gap, talk, passes(r, [m1]), passes(r, [m1, m2] if m2 else [m1])))
    return base, emb


def fit(rows, cascade, min_move=0.95, grid=None):
    """Grid search: maximize talk-silent rate with move pass >= min_move; tie-break on more right 'sure' bands,
    then fewer wrong ones. With no feasible point, the best move pass wins. Returns (thresholds, metrics)."""
    grid = grid or default_grid(rows)
    base, emb = _compile(rows, cascade)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and emb:
        th = _fit_np(np, rows, base, emb, grid, min_move)
        return th, run(rows, th, cascade)
    move_n = sum(r["label"] != "talk" for r in rows) or 1
    talk_n = sum(r["label"] == "talk" for r in rows) or 1
    best, best_key = None, None
    for combo in itertools.product(*(grid[k] for k in KEYS)):
        ts, mg, gs, tf, gf = combo
        if tf > ts or gf > gs:
            continue
        mp, tk, sr, sw = (base[k] for k in ("move_pass", "talk_silent", "sure_right", "sure_wrong"))
        for s1, margin, gap, talk, ps, pu in emb:
            if s1 >= ts and margin >= mg and gap >= gs:
                if talk:
                    continue
                if ps:
                    mp += 1
                    sr += 1
                else:
                    sw += 1
            elif s1 >= tf and gap >= gf:
                if not talk and pu:
                    mp += 1
            elif talk:
                tk += 1
        mr, tr = mp / move_n, tk / talk_n
        feasible = mr >= min_move
        key = (feasible, tr if feasible else mr, sr, -sw, mr, tr)
        if best_key is None or key > best_key:
            best, best_key = combo, key
    th = dict(zip(KEYS, best))
    return th, run(rows, th, cascade)


def _fit_np(np, rows, base, emb, grid, min_move):
    """Same search as the loop in fit, vectorized over the whole grid (first best in product order wins)."""
    move_n = sum(r["label"] != "talk" for r in rows) or 1
    s1, mg, gp = (np.array([e[i] for e in emb]) for i in (0, 1, 2))
    talk = np.array([e[3] for e in emb])
    ps, pu = np.array([e[4] for e in emb]), np.array([e[5] for e in emb])
    mv = ~talk
    ts, mgs, gss, tfs, gfs = (np.array(grid[k]) for k in KEYS)
    A = np.stack(np.meshgrid(ts, mgs, gss, indexing="ij"), -1).reshape(-1, 3)
    B = np.stack(np.meshgrid(tfs, gfs, indexing="ij"), -1).reshape(-1, 2)
    S = (s1[None] >= A[:, :1]) & (mg[None] >= A[:, 1:2]) & (gp[None] >= A[:, 2:3])
    U = (s1[None] >= B[:, :1]) & (gp[None] >= B[:, 1:2])
    Sf, nS = S.astype(np.float64), (~S).astype(np.float64)
    sr = Sf @ (ps & mv).astype(np.float64)
    sw = Sf @ ((~ps) & mv).astype(np.float64)
    unsure = nS @ (U & (pu & mv)[None]).astype(np.float64).T
    silent = (nS * talk[None]) @ (~U).astype(np.float64).T
    mp = np.rint(sr[:, None] + unsure).astype(np.int64) + base["move_pass"]
    tk = np.rint(silent).astype(np.int64) + base["talk_silent"]
    srI = np.rint(sr).astype(np.int64)[:, None] + base["sure_right"]
    swI = np.rint(sw).astype(np.int64)[:, None] + base["sure_wrong"]
    valid = (B[None, :, 0] <= A[:, None, 0]) & (B[None, :, 1] <= A[:, None, 2])
    feasible = (mp / move_n) >= min_move
    K = len(rows) + 2
    primary = np.where(feasible, tk, mp)
    key = ((((feasible.astype(np.int64) * K + primary) * K + srI) * K + (K - 1 - swI)) * K + mp) * K + tk
    key = np.where(valid, key, -1)
    i, j = divmod(int(np.argmax(key)), key.shape[1])
    return dict(zip(KEYS, (*A[i].tolist(), *B[j].tolist())))


def stratified_folds(rows, k=5, seed=0):
    """k lists of row indices, each label dealt round-robin over the folds after a seeded shuffle."""
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    by_label = {}
    for i, r in enumerate(rows):
        by_label.setdefault(r["label"], []).append(i)
    start = 0
    for label in sorted(by_label):
        idx = by_label[label]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[(start + j) % k].append(i)
        start = (start + len(idx)) % k
    return [sorted(f) for f in folds]


def cross_validate(rows, cascade, k=5, seed=0, min_move=0.95, grid_n=12):
    """Fit on k-1 folds, decide the held-out fold; returns (pooled metrics, per-fold thresholds)."""
    folds = stratified_folds(rows, k, seed)
    outcomes, ths, feasible = [None] * len(rows), [], 0
    for held in folds:
        held_set = set(held)
        train = [r for i, r in enumerate(rows) if i not in held_set]
        th, train_m = fit(train, cascade, min_move, default_grid(train, grid_n))
        feasible += train_m["move_rate"] >= min_move
        ths.append(th)
        for i in held:
            outcomes[i] = decide(rows[i], th, cascade)
    metrics = evaluate(rows, outcomes)
    metrics["folds_feasible"] = feasible
    return metrics, ths
