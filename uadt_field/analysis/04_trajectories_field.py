"""
UADT Field Cancerization — Script 04: Directional Trajectories

Tests whether field-site transitions are bidirectional (true field effect)
or unidirectional (spread). Also quantifies synchronous vs metachronous split.

Adapted from coexist_cancers/analysis/07_sir_trajectories.py trajectories().

Outputs:
  results/04_trajectories/field_trajectories.csv
  results/04_trajectories/field_triples.csv
  results/04_trajectories/fig4a_trajectory_graph.png
  results/04_trajectories/fig4b_directionality_bars.png
  results/04_trajectories/fig4c_sync_heatmap.png
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

# ── LOCKED CONSTANTS ──────────────────────────────────────────────────────────
FIELD_SITES = ['C02','C03','C04','C05','C06','C09','C10','C12','C13','C15']
FIELD_LABELS = {
    'C02':'Tongue',      'C03':'Gum',        'C04':'Floor of mouth',
    'C05':'Palate',      'C06':'Oral NOS',   'C09':'Tonsil',
    'C10':'Oropharynx',  'C12':'Pyriform',   'C13':'Hypopharynx',
    'C15':'Esophagus'
}
SYNC_MO     = 6
LANDMARK_MO = 6
STUDY_END   = pd.Timestamp('2020-12-31')
MIN_OBS     = 5

# ── PRE-REGISTERED SECONDARY HYPOTHESIS ──────────────────────────────────────
# SECONDARY HYPOTHESIS: C13↔C15 transition is BIDIRECTIONAL (field effect),
# not unidirectional. Test: binomial test on n_forward vs n_reverse.
# Expected: symmetry ≈ 0.80–0.90, p_binomial > 0.05 (not significant direction).

assert len(FIELD_SITES) == 10

BASE = Path(__file__).parent.parent
OUT  = BASE / "results/04_trajectories"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="white", font_scale=0.95)
lab = FIELD_LABELS.__getitem__


def trajectories(first, sync_mo=SYNC_MO):
    """
    Adapted verbatim from 07_sir_trajectories.py trajectories().
    Restricted to FIELD_SITES only.
    Returns (traj_df, triples_df, symmetry_df).
    """
    first = first[first["site"].isin(FIELD_SITES)].copy()
    g = first.groupby("pid")
    pair_dir = {}
    sync_cnt = {}
    triples  = {}
    for pid, grp in g:
        recs = grp.sort_values("dx")[["site","dx"]].values
        seen = {}
        for s, d in recs:
            if s not in seen: seen[s] = d
        items = sorted(seen.items(), key=lambda kv: kv[1])
        sl = [s for s,_ in items]; dl = [d for _,d in items]
        for i in range(len(sl)):
            for j in range(i+1, len(sl)):
                a, b = sl[i], sl[j]
                if a == b: continue
                gap = abs((dl[j]-dl[i]).days)
                key = frozenset((a,b))
                sync_cnt.setdefault(key,[0,0])
                if gap <= sync_mo*30.44:
                    sync_cnt[key][0] += 1
                else:
                    sync_cnt[key][1] += 1
                    pair_dir[(a,b)] = pair_dir.get((a,b),0) + 1
        for i in range(len(sl)-2):
            t = (sl[i], sl[i+1], sl[i+2])
            if len(set(t)) == 3:
                triples[t] = triples.get(t,0) + 1

    rows = []
    done = set()
    for (a,b), nab in pair_dir.items():
        if frozenset((a,b)) in done: continue
        nba = pair_dir.get((b,a), 0)
        tot = nab + nba
        if tot < MIN_OBS: continue
        done.add(frozenset((a,b)))
        if nab >= nba:
            major, mn = (a,b), nab; minor = nba
        else:
            major, mn = (b,a), nba; minor = nab
        p = binomtest(minor, tot, 0.5).pvalue
        sc = sync_cnt.get(frozenset((a,b)), [0,0])
        symmetry = round(min(nab,nba)/max(nab,nba), 3) if max(nab,nba)>0 else np.nan
        rows.append({
            "from": major[0], "from_label": lab(major[0]),
            "to":   major[1], "to_label":   lab(major[1]),
            "n_forward": mn, "n_reverse": minor, "n_total_metach": tot,
            "symmetry": symmetry,
            "frac_forward": round(mn/tot, 2),
            "n_synchronous": sc[0], "n_metachronous": sc[1],
            "pct_synchronous": round(100*sc[0]/(sc[0]+sc[1]),1) if (sc[0]+sc[1])>0 else np.nan,
            "dir_p": p
        })
    traj = pd.DataFrame(rows)
    if len(traj):
        traj["dir_FDR"] = multipletests(traj["dir_p"], method="fdr_bh")[1]
        traj = traj.sort_values("n_total_metach", ascending=False).reset_index(drop=True)

    trip = pd.DataFrame([
        {"step1": lab(t[0]), "step2": lab(t[1]), "step3": lab(t[2]), "n": n}
        for t, n in triples.items() if n >= 3
    ]).sort_values("n", ascending=False) if triples else pd.DataFrame()
    return traj, trip


def fig_trajectory_graph(traj, min_n=10):
    """Adapted verbatim from 07_sir_trajectories.py fig_trajectory_graph()."""
    t = traj[traj["n_total_metach"] >= min_n].copy()
    if len(t) == 0: t = traj.nlargest(15, "n_total_metach")
    G = nx.DiGraph()
    for _, r in t.iterrows():
        G.add_edge(r["from_label"], r["to_label"],
                   w=r["n_forward"], sync=r["pct_synchronous"])
    if G.number_of_edges() == 0: return
    fig, ax = plt.subplots(figsize=(11, 8))
    pos = nx.spring_layout(G, k=1.5, seed=42, iterations=200)
    deg = dict(G.degree())
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=[300+deg[n]*250 for n in G.nodes()],
                           node_color="#aed6f1", edgecolors="#2471a3", linewidths=1.2)
    ws = [G[u][v]["w"] for u,v in G.edges()]
    wmax = max(ws)
    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=[0.8+3.5*w/wmax for w in ws],
                           edge_color=[G[u][v]["sync"] for u,v in G.edges()],
                           edge_cmap=plt.cm.RdYlBu_r, edge_vmin=0, edge_vmax=100,
                           arrowsize=16, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.08", alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(0,100))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("% synchronous (red=together, blue=sequential)", fontsize=8)
    ax.set_title("UADT Field Cancer Trajectories\n(directed, metachronous; edge width = n patients)",
                 fontsize=11, fontweight="bold")
    ax.axis("off"); fig.tight_layout()
    fig.savefig(OUT / "fig4a_trajectory_graph.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_directionality_bars(traj):
    sub = traj[traj["n_total_metach"] >= 10].sort_values("n_total_metach", ascending=False)
    if len(sub) == 0: return
    fig, ax = plt.subplots(figsize=(9, len(sub)*0.45+1.5))
    y = np.arange(len(sub))
    colors_fwd = ["#dc2626" if r["dir_FDR"]<0.10 else "#93c5fd" for _,r in sub.iterrows()]
    ax.barh(y, sub["n_forward"].values, color=colors_fwd, height=0.4,
            label="n forward (dominant direction)")
    ax.barh(y-0.4, sub["n_reverse"].values, color="#d1d5db", height=0.4,
            label="n reverse")
    pairs = [f"{r['from_label']} → {r['to_label']}" for _,r in sub.iterrows()]
    ax.set_yticks(y-0.2); ax.set_yticklabels(pairs, fontsize=8)
    ax.set_xlabel("Number of patients (metachronous)")
    ax.set_title("UADT Transition Directionality\n(red = FDR<0.10 significant direction)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4b_directionality_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_sync_heatmap(traj):
    labels = [FIELD_LABELS[s] for s in FIELD_SITES]
    M = pd.DataFrame(np.nan, index=labels, columns=labels)
    for _, r in traj.iterrows():
        if r["n_synchronous"] + r["n_metachronous"] > 0:
            M.loc[r["from_label"], r["to_label"]] = r["pct_synchronous"]
            M.loc[r["to_label"],   r["from_label"]] = r["pct_synchronous"]
    mask = np.triu(np.ones_like(M, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(M, mask=mask | M.isna(), cmap="RdYlBu_r", vmin=0, vmax=100,
                annot=True, fmt=".0f", linewidths=0.5, ax=ax,
                cbar_kws={"label":"% synchronous","shrink":0.8},
                annot_kws={"size": 7})
    ax.set_title("% Synchronous Diagnoses (≤6 months apart)\nUADT Field Pairs",
                 fontsize=11, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig4c_sync_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=== UADT Trajectory Analysis ===")
    print("SECONDARY HYPOTHESIS: C13↔C15 transition is bidirectional (field effect)")
    print("Expected: symmetry ≈ 0.80–0.90, dir_p > 0.05\n")

    first = pd.read_csv(BASE / "data/field_patients.csv", parse_dates=["dx","end_fu"])
    traj, trip = trajectories(first, sync_mo=SYNC_MO)

    print(f"Metachronous pairs (n≥{MIN_OBS}): {len(traj)}")
    print(f"Significant direction (dir_FDR<0.10): {(traj['dir_FDR']<0.10).sum()}")

    # SECONDARY HYPOTHESIS RESULT
    for a, b in [("C13","C15"), ("C15","C13"), ("C12","C15")]:
        row = traj[((traj["from"]==a) & (traj["to"]==b)) |
                   ((traj["from"]==b) & (traj["to"]==a))]
        if len(row):
            r = row.iloc[0]
            print(f"\nSECONDARY HYPOTHESIS: {FIELD_LABELS[a]}↔{FIELD_LABELS[b]}: "
                  f"n_fwd={r.n_forward} n_rev={r.n_reverse} "
                  f"symmetry={r.symmetry:.3f} dir_p={r.dir_p:.4f} dir_FDR={r.dir_FDR:.4f}")

    print("\nTop 12 trajectories:")
    print(traj[["from_label","to_label","n_forward","n_reverse","symmetry",
                "pct_synchronous","dir_FDR"]].head(12).to_string(index=False))

    # ── Sensitivity: sync threshold at 3 and 9 months ────────────────────────
    print("\nSync threshold sensitivity (C13↔C15 pair):")
    for mo in [3, 6, 9]:
        t2, _ = trajectories(first, sync_mo=mo)
        row = t2[((t2["from"]=="C13") & (t2["to"]=="C15")) |
                 ((t2["from"]=="C15") & (t2["to"]=="C13"))]
        if len(row):
            r = row.iloc[0]
            print(f"  SYNC_MO={mo}: %sync={r.pct_synchronous:.1f}%, "
                  f"symmetry={r.symmetry:.3f}")

    traj.to_csv(OUT / "field_trajectories.csv", index=False, encoding="utf-8-sig")
    if len(trip): trip.to_csv(OUT / "field_triples.csv", index=False, encoding="utf-8-sig")

    fig_trajectory_graph(traj)
    fig_directionality_bars(traj)
    fig_sync_heatmap(traj)
    print(f"\n✓ Outputs written to {OUT}")


if __name__ == "__main__":
    main()
