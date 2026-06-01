"""
Second-Primary-Cancer epidemiology: Standardized Incidence Ratios (SIR) +
directional disease-trajectory mining (Jensen et al., Nat Commun 2014 style).

SIR: for each index cancer, observed vs expected later primaries, with person-years
     at risk and age(at index)+sex-stratified internal-reference background rates;
     Poisson exact 95% CI; FDR across all ordered pairs.
Trajectories: time-ordered A->B directional tests + 3-step A->B->C paths;
     synchronous (<=2 mo) vs metachronous split.

Outputs: results/05_sir_trajectories/
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from itertools import permutations
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import chi2, poisson
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).parent.parent
RAW  = BASE.parent / "data/processed/all_cancers.csv"
OUT  = BASE / "results/05_sir_trajectories"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)

STUDY_END   = pd.Timestamp("2020-12-31")
WASHOUT_MO  = 2          # months: 2nd primaries within this window = synchronous (excluded from SIR)
SYNC_MO     = 6          # months: synchronous vs metachronous cutpoint for the timing split
MIN_OBS     = 5          # minimum observed co-occurrences to report an SIR pair

SITE = {'C00':'Lip','C01':'Base of tongue','C02':'Tongue','C03':'Gum','C04':'Floor mouth',
'C05':'Palate','C06':'Mouth NOS','C07':'Parotid','C09':'Tonsil','C10':'Oropharynx',
'C11':'Nasopharynx','C12':'Pyriform sinus','C13':'Hypopharynx','C15':'Esophagus','C16':'Stomach',
'C18':'Colon','C19':'Rectosigmoid','C20':'Rectum','C22':'Liver','C23':'Gallbladder','C25':'Pancreas',
'C32':'Larynx','C34':'Lung','C42':'Hematopoietic','C50':'Breast','C53':'Cervix','C54':'Corpus uteri',
'C56':'Ovary','C61':'Prostate','C64':'Kidney','C67':'Bladder','C71':'Brain','C73':'Thyroid',
'C77':'Lymph nodes','C82':'Follicular lym','C83':'DLBCL','C85':'NHL','C90':'Myeloma',
'C91':'Lymphoid leuk','C92':'Myeloid leuk'}
def lab(c): return SITE.get(c, c)


def roc_to_ts(x):
    s = str(x).split(".")[0].zfill(7)
    if not s.isdigit() or len(s) != 7: return pd.NaT
    y = int(s[:3]) + 1911
    mm = s[3:5] if s[3:5] not in ("00","99") else "01"
    dd = s[5:7] if s[5:7] not in ("00","99") else "01"
    try: return pd.Timestamp(f"{y}-{mm}-{dd}")
    except: return pd.NaT


def load():
    df = pd.read_csv(RAW, low_memory=False)
    df["pid"]   = df["病歷號(2)"].astype(str).str.strip()
    df["site"]  = df["腫瘤部位(47)"].astype(str).str[:3].str.upper()
    df = df[df["site"].str.match(r"C\d\d", na=False)].copy()
    df["dx"]    = df["最初診斷日(45)"].apply(roc_to_ts)
    df["age"]   = pd.to_numeric(df["診斷年齡(33)"], errors="coerce")
    df["sex"]   = df["性別(5)"].map({1:"M",2:"F","1":"M","2":"F"})
    df["dead"]  = (pd.to_numeric(df["生存狀態(27)"], errors="coerce")==0).astype(int)
    df["death_dt"]   = df["死亡日期(31)"].apply(roc_to_ts)
    df["contact_dt"] = df["最後聯絡日(30)"].apply(roc_to_ts)
    df = df.dropna(subset=["dx"])
    return df


def patient_level(df):
    """One row per (patient, site): earliest dx. Plus patient end-of-followup."""
    first = (df.sort_values("dx").groupby(["pid","site"], as_index=False)
               .agg(dx=("dx","first"), age=("age","first"), sex=("sex","first")))
    # end of follow-up per patient
    endfu = df.groupby("pid").apply(
        lambda g: g["death_dt"].dropna().min() if g["dead"].max()==1 and g["death_dt"].notna().any()
        else (g["contact_dt"].dropna().max() if g["contact_dt"].notna().any() else g["dx"].max())
    ).rename("end_fu").reset_index()
    endfu["end_fu"] = endfu["end_fu"].clip(upper=STUDY_END)
    first = first.merge(endfu, on="pid", how="left")
    return first


def age_band(a):
    if pd.isna(a): return "unk"
    for lo in [0,40,50,60,70,80]:
        if a < lo+10 if lo<80 else True:
            pass
    if a < 40: return "<40"
    if a < 50: return "40-49"
    if a < 60: return "50-59"
    if a < 70: return "60-69"
    if a < 80: return "70-79"
    return "80+"


def compute_sir(first):
    """Internal-reference SIR for index -> later target primaries.
    Person-years from (index dx + washout) to end of follow-up, stratified by
    sex x age-at-index band. Background = 2nd-primary incidence per PY in each stratum."""
    # index cancer = each patient's earliest primary
    idx = first.loc[first.groupby("pid")["dx"].idxmin()].copy()
    idx["band"]  = idx["age"].apply(age_band)
    idx["start"] = idx["dx"] + pd.to_timedelta(int(WASHOUT_MO*30.44), "D")
    idx["py"]    = (idx["end_fu"] - idx["start"]).dt.days / 365.25
    idx = idx[idx["py"] > 0]
    idx = idx.rename(columns={"site":"index_site"}).set_index("pid")

    # later primaries (metachronous, after washout) per patient
    fp = first.merge(idx[["index_site","start","band","sex","py","end_fu"]],
                     left_on="pid", right_index=True, suffixes=("","_idx"))
    later = fp[fp["dx"] > fp["start"]].copy()       # strictly after washout
    later = later[later["site"] != later["index_site"]]

    strata = ["sex","band"]
    # total PY per stratum (each index patient contributes once)
    py_str = idx.groupby(strata)["py"].sum()
    total_py = idx["py"].sum()

    # background: 2nd-primary events of target T per stratum / total PY of that stratum
    bg = (later.groupby(strata+["site"]).size()
                .rename("events").reset_index())
    bg = bg.merge(py_str.rename("py").reset_index(), on=strata, how="left")
    bg["rate"] = bg["events"] / bg["py"]
    rate_lookup = {(r.sex, r.band, r.site): r.rate for r in bg.itertuples()}
    target_sites = sorted(later["site"].unique())

    # for each index site, expected per target = sum over its patients' PY * stratum rate
    idx_py_by_str = idx.groupby(["index_site","sex","band"])["py"].sum().reset_index()
    obs = later.groupby(["index_site","site"]).size().rename("O").reset_index()

    rows = []
    for isite in idx["index_site"].unique():
        sub_py = idx_py_by_str[idx_py_by_str["index_site"]==isite]
        n_idx  = (idx["index_site"]==isite).sum()
        for tsite in target_sites:
            if tsite == isite: continue
            E = sum(r.py * rate_lookup.get((r.sex, r.band, tsite), 0.0) for r in sub_py.itertuples())
            o_row = obs[(obs["index_site"]==isite)&(obs["site"]==tsite)]
            O = int(o_row["O"].iloc[0]) if len(o_row) else 0
            if O < MIN_OBS or E <= 0: continue
            sir = O / E
            lo  = 0.5*chi2.ppf(0.025, 2*O)/E
            hi  = 0.5*chi2.ppf(0.975, 2*(O+1))/E
            # two-sided Poisson p
            p   = 2*min(poisson.cdf(O, E), 1-poisson.cdf(O-1, E))
            p   = min(p, 1.0)
            rows.append({"index":isite,"index_label":lab(isite),"n_index":n_idx,
                         "target":tsite,"target_label":lab(tsite),
                         "O":O,"E":round(E,2),"SIR":round(sir,2),
                         "CI_low":round(lo,2),"CI_high":round(hi,2),"p":p})
    res = pd.DataFrame(rows)
    if len(res):
        res["FDR"] = multipletests(res["p"], method="fdr_bh")[1]
        res = res.sort_values("SIR", ascending=False)
    return res, total_py


def fig_sir_forest(sir, fname="sir_forest.png", top=25):
    s = sir[sir["FDR"]<0.05].nlargest(top, "SIR").copy()
    if len(s)==0: s = sir.nlargest(top,"SIR").copy()
    s["pair"] = s["index_label"].str[:13] + " → " + s["target_label"].str[:13]
    s = s.sort_values("SIR")
    y = range(len(s))
    fig, ax = plt.subplots(figsize=(9, len(s)*0.38+1))
    ax.hlines(list(y), s["CI_low"], s["CI_high"], color="#2980b9", lw=1.8, zorder=2)
    ax.scatter(s["SIR"], list(y), color="#c0392b", s=34, zorder=3, edgecolors="white", lw=0.5)
    ax.axvline(1, color="#333", ls="--", lw=0.9)
    ax.set_yticks(list(y)); ax.set_yticklabels(s["pair"], fontsize=8)
    ax.set_xscale("log")
    ax.set(title=f"Second-Primary SIR (index → later primary, FDR<0.05)",
           xlabel="Standardized Incidence Ratio (95% CI, log scale)")
    ax.spines[["top","right"]].set_visible(False)
    for i,(_,r) in enumerate(s.iterrows()):
        ax.text(ax.get_xlim()[1]*1.05, i, f"O={r.O}, SIR={r.SIR:.1f}", va="center", fontsize=6.5, color="#555")
    fig.tight_layout(); fig.savefig(OUT/fname, dpi=300, bbox_inches="tight"); plt.close(fig)


def trajectories(first):
    """Directional A->B (metachronous) + synchronous/metachronous split + 3-step paths."""
    # all co-occurring ordered pairs by dx date
    g = first.groupby("pid")
    pair_dir = {}      # (A,B) ordered A before B -> count
    sync_cnt = {}      # unordered frozenset -> [sync, meta]
    triples  = {}
    for pid, grp in g:
        recs = grp.sort_values("dx")[["site","dx"]].values
        sites_dates = [(s,d) for s,d in recs]
        # dedup keep earliest per site
        seen={}
        for s,d in sites_dates:
            if s not in seen: seen[s]=d
        items = sorted(seen.items(), key=lambda kv: kv[1])
        sl = [s for s,_ in items]; dl=[d for _,d in items]
        for i in range(len(sl)):
            for j in range(i+1, len(sl)):
                a,b = sl[i], sl[j]
                if a==b: continue
                gap = abs((dl[j]-dl[i]).days)
                key = frozenset((a,b))
                sync_cnt.setdefault(key,[0,0])
                if gap <= SYNC_MO*30.44: sync_cnt[key][0]+=1
                else:
                    sync_cnt[key][1]+=1
                    pair_dir[(a,b)] = pair_dir.get((a,b),0)+1   # a strictly before b
        # 3-step ordered paths (metachronous chain)
        for i in range(len(sl)-2):
            t = (sl[i],sl[i+1],sl[i+2])
            if len(set(t))==3:
                triples[t] = triples.get(t,0)+1

    # directional significance: among metachronous A<->B, is A->B != B->A? binomial
    rows=[]
    done=set()
    for (a,b),nab in pair_dir.items():
        if frozenset((a,b)) in done: continue
        nba = pair_dir.get((b,a),0)
        tot = nab+nba
        if tot < MIN_OBS: continue
        done.add(frozenset((a,b)))
        # binomial test direction
        from scipy.stats import binomtest
        if nab>=nba:
            major,mn = (a,b),nab; minor=nba
        else:
            major,mn = (b,a),nba; minor=nab
        p = binomtest(mn, tot, 0.5).pvalue
        sc = sync_cnt.get(frozenset((a,b)),[0,0])
        rows.append({"from":major[0],"from_label":lab(major[0]),
                     "to":major[1],"to_label":lab(major[1]),
                     "n_forward":mn,"n_reverse":minor,"n_total_metach":tot,
                     "frac_forward":round(mn/tot,2),
                     "n_synchronous":sc[0],"n_metachronous":sc[1],
                     "pct_synchronous":round(100*sc[0]/(sc[0]+sc[1]),1) if (sc[0]+sc[1])>0 else np.nan,
                     "dir_p":p})
    traj = pd.DataFrame(rows)
    if len(traj):
        traj["dir_FDR"] = multipletests(traj["dir_p"], method="fdr_bh")[1]
        traj = traj.sort_values("n_total_metach", ascending=False)

    trip = pd.DataFrame([{"step1":lab(t[0]),"step2":lab(t[1]),"step3":lab(t[2]),"n":n}
                         for t,n in triples.items() if n>=3]).sort_values("n",ascending=False) \
           if triples else pd.DataFrame()
    return traj, trip


def fig_trajectory_graph(traj, sir, fname="trajectory_graph.png", min_n=15):
    """Directed graph: edges = significant metachronous trajectories, sized by n."""
    t = traj[(traj["n_total_metach"]>=min_n) & (traj["dir_FDR"]<0.10)].copy()
    if len(t)==0: t = traj.nlargest(15,"n_total_metach")
    G = nx.DiGraph()
    for _,r in t.iterrows():
        G.add_edge(r["from_label"], r["to_label"], w=r["n_forward"],
                   sync=r["pct_synchronous"])
    if G.number_of_edges()==0: return
    fig, ax = plt.subplots(figsize=(13,9))
    pos = nx.spring_layout(G, k=1.5, seed=42, iterations=200)
    deg = dict(G.degree())
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[300+deg[n]*250 for n in G.nodes()],
                           node_color="#aed6f1", edgecolors="#2471a3", linewidths=1.2)
    ws = [G[u][v]["w"] for u,v in G.edges()]
    wmax = max(ws)
    nx.draw_networkx_edges(G, pos, ax=ax, width=[0.8+3.5*w/wmax for w in ws],
                           edge_color=[G[u][v]["sync"] for u,v in G.edges()],
                           edge_cmap=plt.cm.RdYlBu_r, edge_vmin=0, edge_vmax=100,
                           arrowsize=16, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.08", alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_weight="bold")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(0,100))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("% synchronous (red = found together, blue = sequential)", fontsize=8)
    ax.set_title("Cancer Disease Trajectories (directional, metachronous; edge width = n patients)",
                 fontsize=12, fontweight="bold")
    ax.axis("off"); fig.tight_layout()
    fig.savefig(OUT/fname, dpi=300, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    print("Loading registry…")
    df = load()
    print(f"  {df['pid'].nunique():,} patients, {len(df):,} cancer records")

    print("Building patient-level first-diagnosis table…")
    first = patient_level(df)

    print("\n[1/2] Standardized Incidence Ratios (second primary, internal reference)…")
    sir, total_py = compute_sir(first)
    sir.to_csv(OUT/"sir_second_primary.csv", index=False, encoding="utf-8-sig")
    print(f"  Total person-years at risk: {total_py:,.0f}")
    print(f"  SIR pairs (O>={MIN_OBS}): {len(sir)}  |  FDR<0.05: {(sir['FDR']<0.05).sum()}")
    print("\n  Top 15 elevated second-primary risks (SIR, FDR<0.05):")
    sig = sir[sir["FDR"]<0.05].nlargest(15,"SIR")
    print(f"  {'index':<16} -> {'target':<16}{'O':>4}{'E':>7}{'SIR':>7}  95% CI")
    for _,r in sig.iterrows():
        print(f"  {r.index_label:<16} -> {r.target_label:<16}{r.O:>4}{r.E:>7.1f}{r.SIR:>7.1f}  "
              f"({r.CI_low:.1f}-{r.CI_high:.1f})")
    fig_sir_forest(sir)

    print("\n[2/2] Disease-trajectory mining…")
    traj, trip = trajectories(first)
    traj.to_csv(OUT/"trajectories_directional.csv", index=False, encoding="utf-8-sig")
    if len(trip): trip.to_csv(OUT/"trajectories_3step.csv", index=False, encoding="utf-8-sig")
    print(f"  Directional pairs (n>={MIN_OBS}): {len(traj)}  |  dir FDR<0.10: {(traj['dir_FDR']<0.10).sum()}")
    print("\n  Top 12 trajectories (by n, with sync/metach split):")
    print(f"  {'A -> B':<34}{'n_fwd':>6}{'n_rev':>6}{'%sync':>7}{'dirFDR':>9}")
    for _,r in traj.nlargest(12,"n_total_metach").iterrows():
        arrow = f"{r.from_label[:15]} → {r.to_label[:15]}"
        print(f"  {arrow:<34}{r.n_forward:>6}{r.n_reverse:>6}{r.pct_synchronous:>6.0f}%{r.dir_FDR:>9.3f}")
    if len(trip):
        print("\n  Top 8 three-step trajectories:")
        for _,r in trip.head(8).iterrows():
            print(f"    {r.step1} → {r.step2} → {r.step3}  (n={r.n})")
    fig_trajectory_graph(traj, sir)

    print(f"\nDone. Results in {OUT}")
