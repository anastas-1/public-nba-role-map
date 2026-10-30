"""
nba_role_mapper.py - data-derived NBA player positions and an interactive role map,
from public play-by-play data (https://github.com/shufinskiy/nba_data).

Per-75 style stats + a matchup-based size index -> split-half stability picks the
size weight and number of positions -> Gaussian mixture and Bayesian Gaussian
mixture fits -> one map and CSV per model in ./output.

  pip install numpy pandas scikit-learn
  python nba_role_mapper.py --season 2023              # 2023-24, both models
  python nba_role_mapper.py --season 2023 --model bgmm --k 12

No tracking data (potential assists, on-ball time, play types). Matchups start 2017-18.
"""

import argparse
import json
import tarfile
import unicodedata
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

# ---------------------------------------------------------------------------
# Settings (the knobs worth knowing about)
# ---------------------------------------------------------------------------
PER_POSS = 75                  # rates per 75 possessions (set 100 for per-100)
SIZE_WEIGHTS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)   # size axis weight vs 1.0 per style feature
K_RANGE = range(6, 17)         # numbers of positions to try
SEEDS = (0, 1, 2)              # stability is averaged over these seeds
STABILITY_TOLERANCE = 0.02     # models within this of the best count as tied;
                               # ties go to the larger K (finer positions)
MAX_SIZE_REDUNDANCY = 0.50     # reject models whose positions are mostly size bins
VAR_KEEP = 0.90                # PCA variance kept before clustering
BOUNDARY_RATIO = 1.30          # a player links two positions if his 2nd-nearest
                               # position is within this factor of his nearest

STYLE_FEATURES = [
    "rimFGA", "paintFGA", "midFGA", "c3FGA", "ab3FGA",       # shot diet (per 75)
    "floater_share",                                          # floaters / FGA
    "rim_ast_share", "three_ast_share", "self_created",       # creation vs finishing
    "AST", "ASTEFG", "sTOVpct",                               # playmaking + security
    "OREB", "DREB", "BLK", "STL",                             # glass + defense activity
    "FTr", "putback", "transition_share",                     # pressure, putbacks, pace
]
FEATURE_WEIGHTS = {"ASTEFG": 0.5, "sTOVpct": 0.5}   # quality measures count half

PHRASE = {  # what a strongly POSITIVE value of each feature means, for naming
    "rimFGA": "rim attacker", "paintFGA": "paint scorer", "midFGA": "midrange",
    "c3FGA": "corner 3", "ab3FGA": "volume 3", "floater_share": "floaters",
    "rim_ast_share": "cutter/roller", "three_ast_share": "catch-and-shoot",
    "self_created": "self-creator", "AST": "playmaker", "ASTEFG": "high-value passer",
    "sTOVpct": "loose handle", "OREB": "offensive glass", "DREB": "defensive glass",
    "BLK": "rim protector", "STL": "ball hawk", "FTr": "foul drawer",
    "putback": "putbacks", "transition_share": "transition",
}

DATA_URL = "https://github.com/shufinskiy/nba_data/raw/main/datasets/{key}.tar.xz"
POSNUM = {"G": 0.0, "G-F": 0.5, "F-G": 0.5, "F": 1.0, "F-C": 1.5, "C-F": 1.5, "C": 2.0}
ZONES = {"Restricted Area": "rim", "In The Paint (Non-RA)": "paint", "Mid-Range": "mid",
         "Left Corner 3": "c3", "Right Corner 3": "c3", "Above the Break 3": "ab3",
         "Backcourt": "ab3"}


def log(msg):
    # plain ASCII so Windows consoles never choke on accented names
    print(unicodedata.normalize("NFKD", str(msg)).encode("ascii", "ignore").decode(), flush=True)


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
def archive(name, season, data_dir):
    """Path to <name>_<season>.tar.xz, downloaded straight from shufinskiy/nba_data once."""
    key = f"{name}_{season}"
    path = data_dir / f"{key}.tar.xz"
    if not path.exists():
        log(f"downloading {key} ...")
        data_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urlretrieve(DATA_URL.format(key=key), tmp)
        tmp.replace(path)            # no half-written archive if a download is cut off
    return path


def read_archive(path, columns):
    """Read the CSV inside a .tar.xz archive, keeping only the listed columns."""
    with tarfile.open(path) as t:
        member = next(m for m in t.getmembers() if m.name.endswith(".csv"))
        df = pd.read_csv(t.extractfile(member), usecols=lambda c: c in columns,
                         low_memory=False)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}")
    return df


def as_id(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).astype("int64")


def load_pbp(path):
    cols = ["GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "PERIOD", "PCTIMESTRING",
            "HOMEDESCRIPTION", "VISITORDESCRIPTION"]
    for k in (1, 2, 3):
        cols += [f"PLAYER{k}_ID", f"PLAYER{k}_NAME", f"PLAYER{k}_TEAM_ABBREVIATION"]
    df = read_archive(path, cols)
    df = df[df["PERIOD"] <= 4].copy()
    df["GAME_ID"] = as_id(df["GAME_ID"])
    for k in (1, 2, 3):
        pid = as_id(df[f"PLAYER{k}_ID"])
        # team-level events carry a team id and no player name: not a player
        df[f"P{k}"] = pid.where(df[f"PLAYER{k}_NAME"].notna(), 0)
        df[f"T{k}"] = df[f"PLAYER{k}_TEAM_ABBREVIATION"]
    clock = df["PCTIMESTRING"].astype(str).str.split(":", expand=True)
    remaining = pd.to_numeric(clock[0], errors="coerce") * 60 + pd.to_numeric(clock[1], errors="coerce")
    df["t"] = ((df["PERIOD"] - 1) * 720 + (720 - remaining)).astype(float)   # seconds elapsed
    df["desc"] = df["HOMEDESCRIPTION"].fillna("") + " " + df["VISITORDESCRIPTION"].fillna("")
    df = df.sort_values(["GAME_ID", "PERIOD", "EVENTNUM"]).reset_index(drop=True)
    names = pd.concat([df.loc[df[f"P{k}"] > 0, [f"P{k}", f"PLAYER{k}_NAME"]]
                       .set_axis(["pid", "name"], axis=1) for k in (1, 2, 3)])
    name_of = names.groupby("pid")["name"].agg(lambda s: s.mode().iat[0])
    keep = ["GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "PERIOD", "t", "desc",
            "P1", "T1", "P2", "T2", "P3", "T3"]
    return df[keep], name_of


def load_shots(path):
    cols = ["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "PERIOD", "ACTION_TYPE",
            "SHOT_ZONE_BASIC", "SHOT_MADE_FLAG"]
    sh = read_archive(path, cols)
    sh = sh[sh["PERIOD"] <= 4].copy()
    sh["GAME_ID"] = as_id(sh["GAME_ID"])
    sh["pid"] = as_id(sh["PLAYER_ID"])
    sh["zone"] = sh["SHOT_ZONE_BASIC"].map(ZONES).fillna("mid")
    sh["made"] = sh["SHOT_MADE_FLAG"].fillna(0).astype(int)
    sh["floater"] = sh["ACTION_TYPE"].astype(str).str.contains("Floating", na=False)
    return sh[["GAME_ID", "GAME_EVENT_ID", "pid", "zone", "made", "floater"]]


def load_matchups(path):
    cols = ["game_id", "person_id", "position", "matchups_person_id", "partial_possessions"]
    mu = read_archive(path, cols)
    mu["gid"] = as_id(mu["game_id"])
    mu["a"] = as_id(mu["person_id"])
    mu["b"] = as_id(mu["matchups_person_id"])
    mu["w"] = pd.to_numeric(mu["partial_possessions"], errors="coerce").fillna(0)
    listed = (mu.dropna(subset=["position"]).groupby("a")["position"]
                .agg(lambda s: s.mode().iat[0]).map(POSNUM).dropna())
    return mu[mu["w"] > 0][["gid", "a", "b", "w"]], listed


# ---------------------------------------------------------------------------
# 2. Per-player, per-game counts (so any subset of games can be aggregated)
# ---------------------------------------------------------------------------
def minutes_by_game(pbp):
    """Seconds on court per (player, game) from substitutions, plus team."""
    rows = []
    cols = ["EVENTMSGTYPE", "t", "P1", "T1", "P2", "T2", "P3", "T3"]
    for (gid, period), g in pbp.groupby(["GAME_ID", "PERIOD"], sort=False):
        t0, t1 = (period - 1) * 720.0, period * 720.0
        ev = g[cols].to_numpy()
        first, team = {}, {}
        for row in ev:
            for slot, (p, tm) in enumerate(((row[2], row[3]), (row[4], row[5]), (row[6], row[7]))):
                if p and p not in first:
                    # first seen entering the game as a sub -> was on the bench
                    first[p] = "off" if (row[0] == 8 and slot == 1) else "on"
                    team[p] = tm
        on = {p for p, s in first.items() if s == "on"}
        start = {p: t0 for p in on}
        for row in ev:
            if row[0] != 8 or not np.isfinite(row[1]):
                continue
            t, p_out, p_in = row[1], row[2], row[4]
            if p_out in on:
                rows.append((p_out, gid, team[p_out], t - start.pop(p_out)))
                on.discard(p_out)
            if p_in and p_in not in on:
                on.add(p_in)
                start[p_in] = t
        for p in on:
            rows.append((p, gid, team[p], t1 - start[p]))
    m = pd.DataFrame(rows, columns=["pid", "gid", "team", "sec"])
    m = m[m["sec"] > 0]
    return m.groupby(["pid", "gid"]).agg(sec=("sec", "sum"), team=("team", "first"))


def count(df, pid_col, name, gid_col="GAME_ID"):
    s = df[df[pid_col] > 0].groupby([pid_col, gid_col]).size()
    s.index = s.index.set_names(["pid", "gid"])
    return s.rename(name)


def total(df, pid_col, value_col, name, gid_col="GAME_ID"):
    s = df[df[pid_col] > 0].groupby([pid_col, gid_col])[value_col].sum()
    s.index = s.index.set_names(["pid", "gid"])
    return s.rename(name)


def game_table(pbp, shots, matchups, listed):
    """One row per (player, game): seconds, possessions and every count we need."""
    log("rebuilding minutes from substitutions ...")
    mins = minutes_by_game(pbp)

    ev = pbp["EVENTMSGTYPE"]
    fga_ev = pbp[ev.isin([1, 2])]
    is_miss = (ev == 2) | ((ev == 3) & pbp["desc"].str.contains("MISS", na=False))
    pbp = pbp.assign(miss_team=pbp["T1"].where(is_miss))
    pbp["last_miss_team"] = pbp.groupby("GAME_ID")["miss_team"].ffill()
    reb = pbp[(ev == 4) & (pbp["P1"] > 0)]
    oreb = reb[reb["T1"] == reb["last_miss_team"]]
    dreb = reb[reb["T1"] != reb["last_miss_team"]]
    tov = pbp[ev == 5]
    badpass = tov[tov["desc"].str.contains("Bad Pass", na=False)]

    # team possessions per game: FGA + 0.44 FTA + TOV - OREB
    def team_count(df, name):
        return df.dropna(subset=["T1"]).groupby(["T1", "GAME_ID"]).size().rename(name)
    tp = pd.concat([team_count(fga_ev, "fga"), team_count(pbp[ev == 3], "fta"),
                    team_count(tov, "tov"), team_count(oreb, "oreb")], axis=1).fillna(0)
    tp = (tp["fga"] + 0.44 * tp["fta"] + tp["tov"] - tp["oreb"]).rename("team_poss")
    tp.index = tp.index.set_names(["team", "gid"])

    # transition: FGA within 6 s of the same team's defensive rebound or steal
    trig = pd.concat([dreb[["GAME_ID", "t", "T1"]].rename(columns={"T1": "team"}),
                      tov[tov["P2"] > 0][["GAME_ID", "t", "T2"]].rename(columns={"T2": "team"})])
    shots_ev = fga_ev[(fga_ev["P1"] > 0)][["GAME_ID", "t", "T1", "P1"]].rename(columns={"T1": "team"})
    shots_ev = shots_ev.dropna(subset=["t", "team"]).sort_values("t")
    trig = trig.dropna(subset=["t", "team"]).sort_values("t").assign(trig_t=lambda d: d["t"])
    tr = pd.merge_asof(shots_ev, trig[["GAME_ID", "team", "t", "trig_t"]], on="t",
                       by=["GAME_ID", "team"], direction="backward", tolerance=6.0)
    tr = tr[tr["trig_t"].notna()]
    # putback: FGA within 4 s of the shooter's own offensive rebound
    ob = oreb[["GAME_ID", "t", "P1"]].dropna(subset=["t"]).sort_values("t").assign(ob_t=lambda d: d["t"])
    pb = pd.merge_asof(shots_ev, ob[["GAME_ID", "P1", "t", "ob_t"]], on="t",
                       by=["GAME_ID", "P1"], direction="backward", tolerance=4.0)
    pb = pb[pb["ob_t"].notna()]

    # shots joined to the play-by-play to learn who (if anyone) assisted the make
    made_ev = pbp[ev == 1][["GAME_ID", "EVENTNUM", "P2"]].rename(
        columns={"EVENTNUM": "GAME_EVENT_ID", "P2": "assister"})
    sh = shots.merge(made_ev, on=["GAME_ID", "GAME_EVENT_ID"], how="left")
    sh["assister"] = sh["assister"].fillna(0).astype("int64")
    sh["assisted"] = (sh["made"] == 1) & (sh["assister"] > 0)
    sh["pts"] = np.where(sh["zone"].isin(["c3", "ab3"]), 3.0, 2.0)
    made = sh[sh["made"] == 1]
    threes = made[made["zone"].isin(["c3", "ab3"])]
    rim = made[made["zone"] == "rim"]
    ast_made = made[made["assisted"]]

    parts = [count(sh, "pid", "FGA")]
    parts += [count(sh[sh["zone"] == z], "pid", f"{z}FGA") for z in ("rim", "paint", "mid", "c3", "ab3")]
    parts += [count(sh[sh["floater"]], "pid", "floaters"),
              count(rim, "pid", "rim_made"), count(rim[rim["assisted"]], "pid", "rim_ast_made"),
              count(threes, "pid", "three_made"), count(threes[threes["assisted"]], "pid", "three_ast_made"),
              count(made[~made["assisted"]], "pid", "unast_made"),
              count(ast_made, "assister", "ast_given"), total(ast_made, "assister", "pts", "ast_pts"),
              count(pbp[ev == 3], "P1", "FTA"), count(tov, "P1", "TOV"), count(badpass, "P1", "badpass"),
              count(tov, "P2", "STL"), count(pbp[ev == 2], "P3", "BLK"),
              count(oreb, "P1", "OREB"), count(dreb, "P1", "DREB"),
              count(tr, "P1", "trans_fga"), count(pb, "P1", "putback")]

    if matchups is not None:
        # size: listed position of whoever you were matched with, both directions pooled
        both = pd.concat([matchups.rename(columns={"a": "pid", "b": "opp"}),
                          matchups.rename(columns={"b": "pid", "a": "opp"})])
        both["opp_pos"] = both["opp"].map(listed)
        both = both.dropna(subset=["opp_pos"])
        both["size_num"] = both["opp_pos"] * both["w"]
        both = both.rename(columns={"w": "size_den"})
        parts += [total(both, "pid", "size_num", "size_num", gid_col="gid"),
                  total(both, "pid", "size_den", "size_den", gid_col="gid")]

    G = mins.join(pd.concat(parts, axis=1), how="left").fillna(0)
    G = G.join(tp, on=["team", "gid"])
    G["team_poss"] = G["team_poss"].fillna(G["team_poss"].median())
    G["poss"] = G["sec"] / 2880.0 * G["team_poss"]
    return G


# ---------------------------------------------------------------------------
# 3. Season features for any subset of games
# ---------------------------------------------------------------------------
def season_features(G, games=None):
    g = G if games is None else G[G.index.get_level_values("gid").isin(games)]
    S = g.drop(columns=["team"]).groupby(level="pid").sum()
    poss = S["poss"].replace(0, np.nan)
    rate = lambda c: S[c] / poss * PER_POSS
    share = lambda a, b: S[a] / S[b].replace(0, np.nan)
    nonpass = S["TOV"] - S["badpass"]
    F = pd.DataFrame({
        "rimFGA": rate("rimFGA"), "paintFGA": rate("paintFGA"), "midFGA": rate("midFGA"),
        "c3FGA": rate("c3FGA"), "ab3FGA": rate("ab3FGA"),
        "floater_share": share("floaters", "FGA"),
        "rim_ast_share": share("rim_ast_made", "rim_made"),
        "three_ast_share": share("three_ast_made", "three_made"),
        "self_created": rate("unast_made"),
        "AST": rate("ast_given"),
        "ASTEFG": S["ast_pts"] / (2.0 * S["ast_given"].replace(0, np.nan)),
        "sTOVpct": nonpass / (S["FGA"] + 0.44 * S["FTA"] + nonpass).replace(0, np.nan),
        "OREB": rate("OREB"), "DREB": rate("DREB"), "BLK": rate("BLK"), "STL": rate("STL"),
        "FTr": share("FTA", "FGA"), "putback": rate("putback"),
        "transition_share": share("trans_fga", "FGA"),
    })
    size = (S["size_num"] / S["size_den"].replace(0, np.nan)) if "size_num" in S else pd.Series(np.nan, index=S.index)
    return F, size, S["sec"] / 60.0


def finalize(F, size, listed, players):
    """Restrict to players, fill missing shares with medians, fill size from listing."""
    F = F.loc[players].copy()
    F = F.fillna(F.median())
    size = size.reindex(players)
    size = size.fillna(pd.Series(listed).reindex(players)).fillna(1.0)
    return F, size


# ---------------------------------------------------------------------------
# 4. Model: weighted style features + size axis -> PCA -> Gaussian mixture
# ---------------------------------------------------------------------------
def design(F, size, size_weight):
    Z = (F - F.mean()) / F.std(ddof=0).replace(0, 1)
    w = np.array([FEATURE_WEIGHTS.get(c, 1.0) for c in STYLE_FEATURES])
    s = (size - size.mean()) / (size.std(ddof=0) or 1.0)
    return np.column_stack([Z[STYLE_FEATURES].to_numpy() * w, size_weight * s.to_numpy()])


MODELS = {
    "gmm": "Gaussian mixture",            # exactly K positions
    "bgmm": "Bayesian Gaussian mixture",  # K is a ceiling; a Dirichlet-process prior
}                                         # switches off positions the data doesn't support


def fit(X, k, n_comp, seed, model="gmm", n_init=1):
    """PCA then a mixture model. Returns labels, soft memberships, PCA scores, means.
    Unused components (common with the Bayesian model) are dropped and relabeled."""
    P = PCA(n_components=n_comp, random_state=seed).fit_transform(X)
    if model == "bgmm":
        gm = BayesianGaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-3,
                                     weight_concentration_prior_type="dirichlet_process",
                                     weight_concentration_prior=1.0 / k, max_iter=1000,
                                     n_init=n_init, random_state=seed).fit(P)
    else:
        gm = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-3,
                             n_init=n_init, random_state=seed).fit(P)
    proba = gm.predict_proba(P)
    used = np.unique(proba.argmax(axis=1))
    proba = proba[:, used] / proba[:, used].sum(axis=1, keepdims=True)
    return proba.argmax(axis=1), proba, P, gm.means_[used]


def sweep(F_full, size_full, halves, weights, ks, model):
    """Stability (odd vs even games) and size-redundancy for every (weight, K)."""
    (F_odd, s_odd), (F_even, s_even) = halves
    rows = []
    for w in weights:
        X_full = design(F_full, size_full, w)
        n_comp = PCA(n_components=VAR_KEEP).fit(X_full).n_components_
        X_odd, X_even = design(F_odd, s_odd, w), design(F_even, s_even, w)
        for k in ks:
            stab = np.mean([adjusted_rand_score(fit(X_odd, k, n_comp, s, model)[0],
                                                fit(X_even, k, n_comp, s, model)[0]) for s in SEEDS])
            full_lab = fit(X_full, k, n_comp, 0, model)[0]
            k_used = len(np.unique(full_lab))
            size_only = KMeans(n_clusters=k_used, n_init=10, random_state=0).fit_predict(
                size_full.to_numpy().reshape(-1, 1))
            red = adjusted_rand_score(full_lab, size_only)
            rows.append({"size_weight": w, "K": k, "positions_used": k_used,
                         "stability": round(stab, 4), "size_redundancy": round(red, 4),
                         "n_components": n_comp})
        log(f"  size weight {w:>3}: best stability "
            f"{max(r['stability'] for r in rows if r['size_weight'] == w):.3f}")
    table = pd.DataFrame(rows)
    ok = table[table["size_redundancy"] <= MAX_SIZE_REDUNDANCY]
    if ok.empty:
        ok = table
    best = ok["stability"].max()
    tied = ok[ok["stability"] >= best - STABILITY_TOLERANCE]
    choice = tied.sort_values(["positions_used", "stability", "K"], ascending=False).iloc[0]
    return table, float(choice["size_weight"]), int(choice["K"])


# ---------------------------------------------------------------------------
# 5. Naming and the map
# ---------------------------------------------------------------------------
NAME_EXCLUDE = {"sTOVpct", "ASTEFG"}   # quality measures describe how well, not what role


def size_class(p):
    """Individual size class from the player's size percentile (about 40/35/25 split)."""
    return "Guard" if p < 0.40 else ("Wing" if p < 0.75 else "Big")


def size_word(pcts):
    """A position's size word = majority of its members; say so when it is split."""
    classes = pd.Series([size_class(p) for p in pcts]).value_counts(normalize=True)
    if classes.iloc[0] >= 0.60:
        return classes.index[0]
    pair = set(classes.index[:2])
    return {frozenset({"Guard", "Wing"}): "Guard/Wing", frozenset({"Wing", "Big"}): "Wing/Big"}.get(
        frozenset(pair), "Mixed")


def name_positions(F, size_pct, labels, k):
    Z = (F - F.mean()) / F.std(ddof=0).replace(0, 1)
    cents, prefix, words, sigs = {}, {}, {}, {}
    for c in range(k):
        mem = labels == c
        if not mem.any():
            continue
        cz = Z[mem].mean()
        cents[c] = cz
        prefix[c] = size_word(size_pct[mem])
        words[c] = [PHRASE[f] for f, v in cz.sort_values(ascending=False).items()
                    if v > 0.45 and f not in NAME_EXCLUDE][:2]
        top = cz.reindex(cz.abs().sort_values(ascending=False).index)[:5]
        sigs[c] = [f"{f} {v:+.1f}" for f, v in top.items()]
    # positions that would get the same name are told apart by what separates
    # each from its siblings (largest centroid difference vs the sibling mean)
    groups = {}
    for c in cents:
        groups.setdefault((prefix[c], frozenset(words[c])), []).append(c)
    for members in groups.values():
        if len(members) < 2:
            continue
        sib_mean = pd.concat([cents[c] for c in members], axis=1).mean(axis=1)
        for c in members:
            diff = (cents[c] - sib_mean).sort_values(ascending=False)
            extra = next((PHRASE[f] for f, v in diff.items()
                          if f not in NAME_EXCLUDE and PHRASE[f] not in words[c]), None)
            if extra:
                words[c] = words[c] + [f"more {extra}"]
    names, out_sigs = [], []
    for c in range(k):
        if c not in cents:
            names.append(f"(empty position {c})")
            out_sigs.append([])
            continue
        name = f"{prefix[c]}: " + (" / ".join(words[c]) if words[c] else "low-usage role player")
        while name in names:
            name += " II"
        names.append(name)
        out_sigs.append(sigs[c])
    return names, out_sigs


def build_map(players, name_of, team_of, minutes, size_pct, labels, proba, P, means, names, sigs,
              meta):
    k = len(names)
    D = np.linalg.norm(P[:, None, :] - means[None, :, :], axis=2)
    order = np.argsort(D, axis=1)
    edges = {}
    for i in range(len(players)):
        a, b = order[i, 0], order[i, 1]
        if D[i, b] / max(D[i, a], 1e-9) <= BOUNDARY_RATIO:
            edges.setdefault((min(a, b), max(a, b)), []).append(i)
    edge_list = [{"a": int(a), "b": int(b), "type": "boundary", "k": len(v),
                  "players": [name_of.get(players[i], str(players[i]))
                              for i in sorted(v, key=lambda i: -minutes.iloc[i])][:8]}
                 for (a, b), v in edges.items()]
    # guarantee one connected graph: join leftover components by nearest means
    while True:
        adj = {c: set() for c in range(k)}
        for e in edge_list:
            adj[e["a"]].add(e["b"]); adj[e["b"]].add(e["a"])
        seen, comps = set(), []
        for c in range(k):
            if c in seen:
                continue
            stack, comp = [c], set()
            while stack:
                u = stack.pop()
                if u not in comp:
                    comp.add(u); stack.extend(adj[u] - comp)
            seen |= comp; comps.append(comp)
        if len(comps) == 1:
            break
        main = max(comps, key=len)
        pair = min(((a, b) for comp in comps if comp is not main for a in comp for b in main),
                   key=lambda ab: np.linalg.norm(means[ab[0]] - means[ab[1]]))
        edge_list.append({"a": int(pair[0]), "b": int(pair[1]), "type": "proximity", "k": 0, "players": []})

    y_axis = PCA(n_components=1, random_state=0).fit_transform(means).ravel()
    y_axis = (y_axis - y_axis.min()) / (np.ptp(y_axis) or 1.0)
    nodes = []
    for c in range(k):
        mem = np.where(labels == c)[0]
        med = mem[int(np.argmin(D[mem, c]))] if len(mem) else None
        ranked = sorted(mem, key=lambda i: -proba[i, c])
        classes = pd.Series([size_class(p) for p in size_pct[mem]]).value_counts(normalize=True)
        tag, _, traits = names[c].partition(": ")
        nodes.append({
            "id": c, "name": names[c], "tag": tag, "traits": traits or names[c], "n": int(len(mem)),
            "cls2": classes.index[1] if len(classes) > 1 and classes.iloc[1] >= 0.30 else "",
            "x": float(size_pct[labels == c].mean()) if len(mem) else 0.5,
            "y": float(y_axis[c]),
            "cls": (pd.Series([size_class(p) for p in size_pct[mem]]).mode().iat[0]
                    if len(mem) else "Wing"),          # node color = majority size class
            "medoid": name_of.get(players[med], "") if med is not None else "",
            "sig": sigs[c],
            "members": [[name_of.get(players[i], str(players[i])), team_of.get(players[i], ""),
                         round(float(proba[i, c]), 2)] for i in ranked],
        })
    seen = pd.Series([n["traits"] for n in nodes]).value_counts()
    for n in nodes:                      # same traits on two nodes -> add the size word back
        if seen[n["traits"]] > 1:
            n["traits"] = f'{n["traits"]} ({n["tag"]})'
    html = MAP_TEMPLATE.replace("__NODES__", json.dumps(nodes)) \
                       .replace("__EDGES__", json.dumps(edge_list)) \
                       .replace("__META__", json.dumps(meta))
    return html.encode("ascii", "xmlcharrefreplace").decode("ascii")


MAP_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NBA Role Map</title>
<style>
:root{--bg:#121211;--panel:#1c1b19;--line:#3a3935;--text:#e9e7e0;--muted:#9a978d;
--guard:#6f7bf7;--wing:#2bb39a;--big:#e0894a}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--text);
font:13px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
#app{display:flex;height:100vh}#stage{position:relative;flex:1;min-width:0}
canvas{display:block;width:100%;height:100%}
#head{position:absolute;top:12px;left:14px;right:14px;max-width:620px;pointer-events:none}
#head h1{margin:0 0 2px;font-size:17px;font-weight:600}#head p{margin:0;color:var(--muted)}
#legend{position:absolute;bottom:12px;left:14px;right:14px;color:var(--muted);display:flex;gap:6px 14px;flex-wrap:wrap;font-size:12px}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
#tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--line);
border-radius:8px;padding:8px 10px;max-width:320px;display:none;box-shadow:0 6px 20px rgba(0,0,0,.4)}
#tip b{display:block;margin-bottom:3px}#tip .m{color:var(--muted)}
#side{width:300px;border-left:1px solid var(--line);background:var(--panel);overflow:auto;padding:14px;display:none}
#side h2{font-size:15px;margin:0 0 4px}#side .m{color:var(--muted);margin-bottom:10px}
#side table{width:100%;border-collapse:collapse}#side td{padding:3px 0;border-bottom:1px solid #2a2926}
#side td.p{text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
#close{float:right;background:none;border:0;color:var(--muted);font-size:18px;cursor:pointer}
@media (max-width:700px){html,body{height:auto}#app{display:block;height:auto}
#side{width:auto;border-left:0;border-top:1px solid var(--line)}}
</style></head><body>
<div id="app"><div id="stage"><canvas id="c"></canvas>
<div id="head"><h1>NBA Role Map</h1><p id="sub"></p></div>
<div id="legend"><span><i class="sw" style="background:var(--guard)"></i>Guard</span>
<span><i class="sw" style="background:var(--wing)"></i>Wing</span>
<span><i class="sw" style="background:var(--big)"></i>Big</span>
<span>solid line = players between two positions</span><span>dashed = nearest position</span><span>ring = split group</span><span>hover for details, click for roster</span></div>
<div id="tip"></div></div>
<aside id="side"><button id="close" aria-label="Close">&times;</button><h2 id="sname"></h2><div class="m" id="smeta"></div>
<table id="stab"></table></aside></div>
<script>
const N=__NODES__,E=__EDGES__,M=__META__;
const css=getComputedStyle(document.documentElement);
const COL={Guard:css.getPropertyValue('--guard'),Wing:css.getPropertyValue('--wing'),Big:css.getPropertyValue('--big')};
document.getElementById('sub').textContent=[M.season,N.length+' positions',M.players+' players',
M.model,'stability '+M.stability.toFixed(2)].join('  \u00b7  ');
const cv=document.getElementById('c'),tip=document.getElementById('tip'),stage=document.getElementById('stage');
let W=0,H=0,small=false;const rad=n=>(small?6:9)+Math.sqrt(n.n)*(small?1.7:2.6);
const F1=()=>'600 '+(small?11:12.5)+'px system-ui,sans-serif',F2=()=>(small?10:11.5)+'px system-ui,sans-serif';
function layout(){small=innerWidth<700;
stage.style.height=small?Math.max(innerHeight*0.85,N.length*78+170)+'px':'';
W=stage.clientWidth;H=stage.clientHeight;
const m=document.createElement('canvas').getContext('2d');
N.forEach(n=>{m.font=F1();n.hw=Math.max(m.measureText(n.traits).width,2*rad(n))/2+6;n.lh=small?16:19;});
const top=document.getElementById('head').offsetHeight+18,bot=H-document.getElementById('legend').offsetHeight-22;
const L=Math.max(...N.map(n=>n.hw))+8,R=W-L;
N.forEach(n=>{n.px=L+n.x*Math.max(R-L,1);n.py=top+n.lh+rad(n)+(1-n.y)*Math.max(bot-top-n.lh-2*rad(n),1);});
const box=n=>[n.px-n.hw,n.px+n.hw,n.py-rad(n)-n.lh,n.py+rad(n)+4];
for(let it=0;it<400;it++){let moved=false;
for(let i=0;i<N.length;i++)for(let j=i+1;j<N.length;j++){const a=N[i],b=N[j],A=box(a),B=box(b);
const ox=Math.min(A[1],B[1])-Math.max(A[0],B[0]),oy=Math.min(A[3],B[3])-Math.max(A[2],B[2]);
if(ox>0&&oy>0){moved=true;if(ox<oy){const s=(ox/2+1)*(a.px<b.px?-1:1);a.px+=s;b.px-=s;}
else{const s=(oy/2+1)*(a.py<b.py?-1:1);a.py+=s;b.py-=s;}}}
N.forEach(n=>{n.px=Math.max(n.hw+4,Math.min(W-n.hw-4,n.px));
n.py=Math.max(top+n.lh+rad(n),Math.min(bot-rad(n),n.py));});if(!moved)break;}}
function draw(){const dpr=devicePixelRatio||1;cv.width=W*dpr;cv.height=H*dpr;const x=cv.getContext('2d');
x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
E.forEach(e=>{const a=N[e.a],b=N[e.b];x.beginPath();x.moveTo(a.px,a.py);x.lineTo(b.px,b.py);
x.setLineDash(e.type==='proximity'?[5,4]:[]);x.lineWidth=e.type==='proximity'?1.6:Math.min(1.8+e.k/3,6);
x.strokeStyle=e.hot?'#ffffff':(e.type==='proximity'?'rgba(190,187,178,.6)':'rgba(214,211,202,.8)');
x.stroke();});x.setLineDash([]);
N.forEach(n=>{const r=rad(n);x.beginPath();x.arc(n.px,n.py,r,0,7);x.fillStyle=COL[n.cls];
x.globalAlpha=n.hot?1:.88;x.fill();x.globalAlpha=1;
if(n.cls2){x.beginPath();x.arc(n.px,n.py,r+3,0,7);x.lineWidth=3;x.strokeStyle=COL[n.cls2];x.stroke();}
if(n.sel){x.beginPath();x.arc(n.px,n.py,r+(n.cls2?7:3),0,7);x.lineWidth=2;x.strokeStyle='#fff';x.stroke();}
if(r>=15){x.font='600 11px system-ui,sans-serif';x.fillStyle='rgba(18,18,17,.85)';x.textAlign='center';
x.textBaseline='middle';x.fillText(n.n,n.px,n.py+.5);x.textBaseline='alphabetic';}});
x.textAlign='center';x.font=F1();x.fillStyle='#e9e7e0';
N.forEach(n=>x.fillText(n.traits,n.px,n.py-rad(n)-(n.cls2?10:6)));}
function nodeAt(mx,my){let h=null;N.forEach(n=>{if(Math.hypot(mx-n.px,my-n.py)<=rad(n)+3)h=n;});return h;}
function edgeAt(mx,my){let best=null,bd=7;E.forEach(e=>{const a=N[e.a],b=N[e.b];
const L=(b.px-a.px)**2+(b.py-a.py)**2||1;let t=((mx-a.px)*(b.px-a.px)+(my-a.py)*(b.py-a.py))/L;
t=Math.max(.08,Math.min(.92,t));const d=Math.hypot(mx-(a.px+t*(b.px-a.px)),my-(a.py+t*(b.py-a.py)));
if(d<bd){bd=d;best=e;}});return best;}
function show(html,mx,my){tip.innerHTML=html;tip.style.display='block';
const r=stage.getBoundingClientRect();let l=mx+14,t=my+10;
if(l+330>r.width)l=mx-334;if(t+200>r.height)t=my-200;tip.style.left=l+'px';tip.style.top=t+'px';}
let drag=null,moved=false;
cv.addEventListener('mousedown',e=>{const r=cv.getBoundingClientRect();drag=nodeAt(e.clientX-r.left,e.clientY-r.top);moved=false;});
addEventListener('mouseup',e=>{if(drag&&!moved)openSide(drag);drag=null;});
cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
if(drag){drag.px=mx;drag.py=my;moved=true;draw();return;}
N.forEach(n=>n.hot=false);E.forEach(x=>x.hot=false);const n=nodeAt(mx,my),ed=n?null:edgeAt(mx,my);
if(n){n.hot=true;show('<b>'+n.name+'</b><span class="m">'+n.n+' players \u00b7 most typical: '+n.medoid+
'</span><br>'+n.members.slice(0,6).map(m=>m[0]).join(', ')+'<br><span class="m">signature: '+n.sig.join(', ')+'</span>',mx,my);}
else if(ed){ed.hot=true;show(ed.type==='proximity'?'<b>Nearest positions</b><span class="m">No players sit on this boundary; the line only keeps the map connected.</span>':
'<b>'+ed.k+' boundary player'+(ed.k>1?'s':'')+'</b><span class="m">'+N[ed.a].name+' &harr; '+N[ed.b].name+'</span><br>'+ed.players.join(', ')+(ed.k>8?', ...':''),mx,my);}
else tip.style.display='none';cv.style.cursor=n?'grab':'default';draw();});
cv.addEventListener('mouseleave',()=>{tip.style.display='none';N.forEach(n=>n.hot=false);E.forEach(x=>x.hot=false);draw();});
function openSide(n){N.forEach(m=>m.sel=false);n.sel=true;document.getElementById('side').style.display='block';
document.getElementById('sname').textContent=n.name;
document.getElementById('smeta').textContent=n.n+' players. Number = probability of belonging here.';
document.getElementById('stab').innerHTML=n.members.map(m=>'<tr><td>'+m[0]+' <span style="color:#9a978d">'+(m[1]||'')+
'</span></td><td class="p">'+m[2].toFixed(2)+'</td></tr>').join('');
requestAnimationFrame(()=>{resize();});}
document.getElementById('close').onclick=()=>{document.getElementById('side').style.display='none';N.forEach(m=>m.sel=false);resize();};
function resize(){layout();draw();}
layout();draw();addEventListener('resize',resize);
</script></body></html>"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--season", type=int, default=2023, help="start year, e.g. 2023 for 2023-24")
    ap.add_argument("--data-dir", default="data", help="where archives are cached")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--min-minutes", type=float, default=800.0)
    ap.add_argument("--size-weight", type=float, default=None, help="fix the size weight")
    ap.add_argument("--k", type=int, default=None,
                    help="fix the number of positions (a ceiling for bgmm)")
    ap.add_argument("--model", choices=["both", "gmm", "bgmm"], default="both")
    args = ap.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    season = args.season

    pbp, name_of = load_pbp(archive("nbastats", season, data_dir))
    shots = load_shots(archive("shotdetail", season, data_dir))
    try:
        matchups, listed = load_matchups(archive("matchups", season, data_dir))
    except Exception as exc:
        log(f"matchups unavailable ({exc}); running without the size axis")
        matchups, listed = None, pd.Series(dtype=float)
    has_size = matchups is not None

    G = game_table(pbp, shots, matchups, listed)
    team_of = (G.reset_index().groupby(["pid", "team"])["sec"].sum().reset_index()
                .sort_values("sec").groupby("pid")["team"].last().to_dict())

    F_all, size_all, mins_all = season_features(G)
    players = mins_all[mins_all >= args.min_minutes].index
    players = players[F_all.loc[players].notna().sum(axis=1) >= len(STYLE_FEATURES) - 4]
    F_full, size_full = finalize(F_all, size_all, listed, players)
    minutes = mins_all.loc[players]
    log(f"{len(players)} players with at least {args.min_minutes:.0f} minutes")

    games = np.sort(G.index.get_level_values("gid").unique())
    odd, even = games[1::2], games[0::2]
    Fo, so, mo = season_features(G, odd)
    Fe, se, me = season_features(G, even)
    half_floor = args.min_minutes / 2
    shared = players[(mo.reindex(players) >= half_floor).to_numpy() & (me.reindex(players) >= half_floor).to_numpy()]
    halves = (finalize(Fo, so, listed, shared), finalize(Fe, se, listed, shared))

    weights = (args.size_weight,) if args.size_weight is not None else SIZE_WEIGHTS
    if not has_size:
        weights = (0.0,)
    ks = (args.k,) if args.k is not None else K_RANGE
    size_pct = size_full.rank(pct=True).to_numpy()
    models = list(MODELS) if args.model == "both" else [args.model]
    final_labels = {}

    for model in models:
        log(f"\n=== {MODELS[model]} ===")
        log(f"stability sweep on {len(shared)} players who qualify in both halves ...")
        table, w, k = sweep(F_full, size_full, halves, weights, ks, model)
        table.to_csv(out_dir / f"stability_{season}_{model}.csv", index=False)
        stab = float(table[(table["size_weight"] == w) & (table["K"] == k)]["stability"].iloc[0])

        X = design(F_full, size_full, w)
        n_comp = PCA(n_components=VAR_KEEP).fit(X).n_components_
        labels, proba, P, means = fit(X, k, n_comp, seed=0, model=model, n_init=10)
        n_pos = proba.shape[1]
        final_labels[model] = labels
        log(f"chosen: size weight {w}, K = {k} -> {n_pos} positions used, "
            f"split-half stability {stab:.3f}")
        names, sigs = name_positions(F_full, size_pct, labels, n_pos)

        out = pd.DataFrame({"player": [name_of.get(p, str(p)) for p in players],
                            "team": [team_of.get(p, "") for p in players],
                            "minutes": minutes.round(0).to_numpy(),
                            "size_index": size_full.round(3).to_numpy(),
                            "position": [names[c] for c in labels],
                            "membership": proba.max(axis=1).round(3)}, index=players)
        out = out.join(pd.DataFrame(proba.round(3), index=players, columns=names))
        out.index.name = "player_id"
        csv_path = out_dir / f"player_positions_{season}_{model}.csv"
        out.sort_values(["position", "membership"], ascending=[True, False]).to_csv(csv_path)

        for c in np.argsort([-np.sum(labels == c) for c in range(n_pos)]):
            mem = np.where(labels == c)[0]
            top = [name_of.get(players[i], "") for i in sorted(mem, key=lambda i: -minutes.iloc[i])[:6]]
            log(f"\n{names[c]}  ({len(mem)} players)\n   {', '.join(top)}"
                f"\n   signature: {', '.join(sigs[c])}")

        meta = {"season": f"{season}-{str(season + 1)[-2:]}", "players": int(len(players)),
                "stability": float(stab), "size_weight": w,
                "model": {"gmm": "GMM", "bgmm": "Bayesian GMM"}[model]}
        html = build_map(np.array(players), name_of, team_of, minutes, size_pct, labels, proba, P,
                         means, names, sigs, meta)
        map_path = out_dir / f"role_map_{season}_{model}.html"
        map_path.write_text(html)
        log(f"\nwrote {map_path}\nwrote {csv_path}")

    if len(final_labels) == 2:
        agree = adjusted_rand_score(final_labels["gmm"], final_labels["bgmm"])
        log(f"\nagreement between the two models' positions (adjusted Rand): {agree:.3f}")

if __name__ == "__main__":
    main()
