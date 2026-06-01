"""
Deep learning discovery for esophageal cancer (C15):
1. Autoencoder — latent representation learning
2. UMAP + cluster visualization (patient subtype discovery)
3. k-means clustering — novel subgroup identification
4. DeepSurv-style MLP — survival prediction & feature importance
Outputs: results/04_deep_learning/
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

DATA = Path(__file__).parent.parent / "data/c15_enriched.csv"
OUT  = Path(__file__).parent.parent / "results/04_deep_learning"
OUT.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.0)
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")


# ── Feature engineering ────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame):
    feat = pd.DataFrame()

    feat["age"] = pd.to_numeric(df["age"], errors="coerce")
    feat["male"] = (df["sex"] == "Male").astype(float)

    # Subsite (one-hot)
    for s in df["subsite"].unique():
        feat[f"subsite_{s}"] = (df["subsite"] == s).astype(float)

    # Histology group (one-hot)
    for h in df["histology_group"].unique():
        feat[f"hist_{h}"] = (df["histology_group"] == h).astype(float)

    # Grade (ordinal)
    feat["grade"] = pd.to_numeric(df["grade"], errors="coerce").replace(9, np.nan)

    # Stage group (ordinal)
    feat["stage_ord"] = df["stage_group"].map({"I": 1, "II": 2, "III": 3, "IV": 4})

    # Treatment flags
    feat["surgery"] = df["surgery"].astype(float)
    feat["radiation"] = df["radiation"].astype(float)
    feat["chemo"] = df["chemo"].astype(float)
    feat["immunotherapy"] = df["immunotherapy"].astype(float)
    feat["targeted"] = df["targeted"].astype(float)

    # Lifestyle
    feat["smoker"] = (df["smoker"] == "Yes").astype(float)
    feat["betel_nut"] = (df["betel_nut"] == "Yes").astype(float)
    feat["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    # Confirmation method (ordinal proxy for diagnostic certainty)
    feat["confirm"] = pd.to_numeric(df["confirm"], errors="coerce")

    feat.index = df.index
    return feat


def preprocess(feat: pd.DataFrame):
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(feat)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, feat.columns.tolist(), imp, scaler


# ── Autoencoder ──────────���────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_autoencoder(X: np.ndarray, latent_dim=8, epochs=200, batch_size=128, lr=1e-3):
    Xt = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=True)
    model = Autoencoder(X.shape[1], latent_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            recon, _ = model(batch)
            loss = nn.MSELoss()(recon, batch)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * len(batch)
        losses.append(epoch_loss / len(Xt))
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs}  loss={losses[-1]:.4f}")
    return model, losses


def get_latent(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32)
        _, z = model(Xt)
    return z.cpu().numpy()


# ── DeepSurv-style MLP ──────────────���─────────────────────────────────────────
class DeepSurv(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def cox_partial_likelihood_loss(risk, t, e):
    """Breslow approximation of partial likelihood loss."""
    order = torch.argsort(t, descending=True)
    risk = risk[order]; t = t[order]; e = e[order]
    log_risk = torch.logcumsumexp(risk, dim=0)
    loss = -torch.mean((risk - log_risk)[e == 1])
    return loss


def train_deepsurv(X: np.ndarray, t: np.ndarray, e: np.ndarray,
                   epochs=300, batch_size=128, lr=1e-3):
    valid = ~(np.isnan(t) | np.isnan(e))
    X, t, e = X[valid], t[valid], e[valid]
    Xt = torch.tensor(X, dtype=torch.float32)
    tt = torch.tensor(t, dtype=torch.float32)
    et = torch.tensor(e, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, tt, et), batch_size=batch_size, shuffle=True)
    model = DeepSurv(X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for bx, bt, be in loader:
            risk = model(bx.to(DEVICE))
            loss = cox_partial_likelihood_loss(risk, bt.to(DEVICE), be.to(DEVICE))
            if torch.isnan(loss): continue
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss)
    return model, losses, valid


def deepsurv_feature_importance(model, X, feat_names, n_perms=30):
    """Permutation importance via change in Cox loss."""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        baseline_risk = model(Xt).cpu().numpy()
    baseline_var = np.var(baseline_risk)
    importances = []
    for i in range(X.shape[1]):
        perturbed_var = []
        for _ in range(n_perms):
            Xp = X.copy()
            np.random.shuffle(Xp[:, i])
            Xpt = torch.tensor(Xp, dtype=torch.float32)
            with torch.no_grad():
                pr = model(Xpt).cpu().numpy()
            perturbed_var.append(np.var(pr))
        importance = baseline_var - np.mean(perturbed_var)
        importances.append(importance)
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    return imp_df.sort_values("importance", ascending=False)


# ── Plots ──���──────────────────────────────────────────────────────────────────
def plot_ae_loss(losses, fname="ae_training_loss.png"):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(losses, color="steelblue")
    ax.set(title="Autoencoder Training Loss", xlabel="Epoch", ylabel="MSE Loss")
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150); plt.close(fig)


def plot_umap_clusters(embedding, labels, color_col, title, fname, palette="tab10"):
    fig, ax = plt.subplots(figsize=(8, 6))
    unique = sorted(set(labels[~pd.isna(labels)]))
    colors = sns.color_palette(palette, len(unique))
    for i, g in enumerate(unique):
        mask = labels == g
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[colors[i]], label=str(g), alpha=0.6, s=15)
    ax.set(title=title, xlabel="UMAP-1", ylabel="UMAP-2")
    ax.legend(markerscale=2, fontsize=8, loc="best", bbox_to_anchor=(1, 1))
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_cluster_km(df_sub, cluster_col, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    colors = sns.color_palette("tab10")
    sub = df_sub.dropna(subset=["os_months","event"])
    clusters = sorted(sub[cluster_col].unique())
    for i, c in enumerate(clusters):
        g = sub[sub[cluster_col] == c]
        kmf.fit(g["os_months"], g["event"], label=f"Cluster {c} (n={len(g)})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[i % len(colors)])
    res = multivariate_logrank_test(sub["os_months"], sub[cluster_col], sub["event"])
    ax.text(0.65, 0.85, f"Log-rank p={res.p_value:.4f}", transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.set(title=f"KM Curves by {cluster_col}", xlabel="Months", ylabel="Survival probability", ylim=(0,1.05))
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150); plt.close(fig)


def plot_feature_importance(imp_df, fname="deepsurv_feature_importance.png"):
    top = imp_df.head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tomato" if v >= 0 else "steelblue" for v in top["importance"]]
    ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="DeepSurv Feature Importance (Permutation)", xlabel="Importance score")
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=150); plt.close(fig)
    imp_df.to_csv(OUT / "deepsurv_feature_importance.csv", index=False, encoding="utf-8-sig")


def cluster_profile(df_merged, cluster_col, feat_names):
    rows = []
    for c in sorted(df_merged[cluster_col].unique()):
        sub = df_merged[df_merged[cluster_col] == c]
        row = {"cluster": c, "n": len(sub)}
        for f in feat_names[:10]:
            if f in df_merged.columns:
                row[f"mean_{f}"] = round(sub[f].mean(), 3)
        if "os_months" in df_merged:
            row["median_os_months"] = round(sub["os_months"].median(), 1) if sub["os_months"].notna().any() else np.nan
        rows.append(row)
    profile = pd.DataFrame(rows)
    profile.to_csv(OUT / f"{cluster_col}_profile.csv", index=False, encoding="utf-8-sig")
    print(f"\n  {cluster_col} profile:\n{profile.to_string(index=False)}")


# ── Main ──────────────────────────────────────────────────────────────────���───
if __name__ == "__main__":
    print("Loading enriched data...")
    df = pd.read_csv(DATA, low_memory=False)
    # Derive survival columns (Taiwan registry: 生存狀態=0 → Dead, event=1)
    df["event"] = (df["生存狀態(27)"].astype(str).str.strip() == "0").astype(int)
    df["os_months"] = pd.to_numeric(df["os_days"], errors="coerce") / 30.44
    print(f"  {len(df)} cases  (events={df['event'].sum()})")

    print("\nBuilding feature matrix...")
    feat = build_features(df)
    X, feat_names, imp, scaler = preprocess(feat)
    print(f"  Feature matrix: {X.shape}")

    # ── 1. Autoencoder ──
    print("\n[1/4] Training autoencoder (latent_dim=8)...")
    ae_model, ae_losses = train_autoencoder(X, latent_dim=8, epochs=200)
    plot_ae_loss(ae_losses)
    Z = get_latent(ae_model, X)
    np.save(OUT / "latent_representations.npy", Z)
    print(f"  Latent space shape: {Z.shape}")

    # ── 2. UMAP ──
    print("\n[2/4] UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2,
                        metric="euclidean", random_state=42)
    embedding = reducer.fit_transform(Z)
    np.save(OUT / "umap_embedding.npy", embedding)

    # Color UMAP by clinical variables
    for col, title, fname in [
        ("stage_group", "UMAP colored by AJCC Stage", "umap_stage.png"),
        ("histology_group", "UMAP colored by Histology", "umap_histology.png"),
        ("sex", "UMAP colored by Sex", "umap_sex.png"),
    ]:
        if col in df.columns:
            plot_umap_clusters(embedding, df[col].values, col, title, fname)

    # ── 3. k-means clustering ──
    print("\n[3/4] k-means clustering on latent space...")
    sil_scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(Z)
        sil = silhouette_score(Z, labels)
        sil_scores[k] = sil
        print(f"  k={k}: silhouette={sil:.4f}")

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"  Best k={best_k} (silhouette={sil_scores[best_k]:.4f})")

    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    df["cluster_kmeans"] = km_best.fit_predict(Z)

    plot_umap_clusters(embedding, df["cluster_kmeans"].values,
                       "cluster_kmeans", f"UMAP — k-means Clusters (k={best_k})",
                       f"umap_kmeans_k{best_k}.png")

    # Silhouette plot
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(list(sil_scores.keys()), list(sil_scores.values()), "o-", color="steelblue")
    ax.axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
    ax.set(title="Silhouette Score vs. k (k-means)", xlabel="Number of clusters k",
           ylabel="Silhouette score")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "silhouette_scores.png", dpi=150); plt.close(fig)
    pd.DataFrame({"k": list(sil_scores.keys()), "silhouette": list(sil_scores.values())}).to_csv(
        OUT / "silhouette_scores.csv", index=False)

    # Merge features back for profiling
    feat_df = pd.DataFrame(
        imp.transform(feat),
        columns=feat_names, index=feat.index
    )
    df_profile = df[["cluster_kmeans","os_months","event"]].join(feat_df)
    cluster_profile(df_profile, "cluster_kmeans", feat_names)

    # KM by cluster
    df_km = df[["cluster_kmeans","os_months","event"]].dropna(subset=["os_months","event"])
    if len(df_km) > 20:
        plot_cluster_km(df_km, "cluster_kmeans", f"km_clusters_k{best_k}.png")

    # ── 4. DeepSurv ──
    print("\n[4/4] Training DeepSurv survival model...")
    t_arr = df["os_months"].values
    e_arr = df["event"].values.astype(float)
    ds_model, ds_losses, valid_mask = train_deepsurv(X, t_arr, e_arr, epochs=300)

    print("  Computing feature importance (permutation)...")
    imp_df = deepsurv_feature_importance(ds_model, X[valid_mask], feat_names)
    plot_feature_importance(imp_df)
    print(f"  Top 10 prognostic features:\n{imp_df.head(10).to_string(index=False)}")

    # DeepSurv risk score quartiles → KM
    ds_model.eval()
    with torch.no_grad():
        risk_scores = ds_model(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
    df["deepsurv_risk"] = risk_scores
    df["risk_quartile"] = pd.qcut(risk_scores, 4, labels=["Q1 (Low)","Q2","Q3","Q4 (High)"])

    plot_umap_clusters(embedding, df["risk_quartile"].values,
                       "risk_quartile", "UMAP — DeepSurv Risk Quartile", "umap_deepsurv_risk.png",
                       palette="RdYlGn_r")

    df_risk = df[["risk_quartile","os_months","event"]].dropna(subset=["os_months","event"])
    if len(df_risk) > 20:
        plot_cluster_km(df_risk, "risk_quartile", "km_deepsurv_risk_quartiles.png")

    # Save final annotated dataset
    df.to_csv(OUT / "c15_final_annotated.csv", index=False, encoding="utf-8-sig")
    print(f"\nDone. All results saved in {OUT}")
