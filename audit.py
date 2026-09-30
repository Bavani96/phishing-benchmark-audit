#!/usr/bin/env python3
"""
Audit code for:
  "Temporal Confounds and Transfer Failure in Phishing E-mail Benchmarks:
   An Empirical Audit of Two Widely Used Corpora"
  Bavani K., Sathyabama Institute of Science and Technology.

A single script regenerates every table (CSV) and figure (PNG) in Section 5.

Usage:
    python audit.py --ceas data/CEAS_08.csv --nazario data/Nazario_5.csv --out outputs

Expected input: the CSV files of the curated phishing corpora (Champa et al., 2024),
with at least the columns `body` and `label` (1 = phishing/spam, 0 = legitimate),
and optionally `subject`, `date`, `urls`. The corpora are NOT redistributed here.
"""
import argparse
import hashlib
import json
import re
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

SEED = 42
TEST_SIZE = 0.30
PLAUSIBLE_YEARS = (1990, 2026)      # outside this range a Date header is "implausible"
DUP_TOKENS = 60                     # near-duplicate hash: first 60 normalised tokens
PROBE_CHARS = 40                    # structural probe: first 40 body characters
URL_RE = re.compile(r"(?:https?://|www\.)", re.I)
CT_RE = re.compile(r"^content-type:.*$", re.I | re.M)

# Domain-content filter (Section 5.6). Keep this list identical to the one used
# for the manuscript so that Table 7 reproduces.
HEALTH_TERMS = [
    "patient", "hospital", "clinic", "clinical", "medical", "medicine", "doctor",
    "physician", "nurse", "pharmacy", "prescription", "health", "healthcare",
    "insurance", "medicare", "medicaid", "diagnosis", "treatment", "drug", "pill",
]
HEALTH_RE = re.compile(r"\b(?:" + "|".join(HEALTH_TERMS) + r")\b", re.I)


# --------------------------------------------------------------------- loading
def load_corpus(path, name):
    df = pd.read_csv(path, encoding_errors="replace", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("body", "label"):
        if col not in df.columns:
            raise ValueError(f"{name}: required column '{col}' missing")
    for col in ("subject", "date", "urls"):
        if col not in df.columns:
            df[col] = ""
    df["body"] = df["body"].fillna("").astype(str)
    df["subject"] = df["subject"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    df["text"] = df["subject"] + "\n" + df["body"]
    df["year"] = df["date"].apply(parse_year)
    df["dup_hash"] = df["body"].apply(dup_hash)
    df["content_type"] = df["body"].apply(lambda b: (CT_RE.search(b) or [""])[0])
    df["fragment"] = df["content_type"] + "\n" + df["body"].str[:PROBE_CHARS]
    df["text_minus_fragment"] = df["subject"] + "\n" + df["body"].str[PROBE_CHARS:]
    df.attrs["name"] = name
    return df


def parse_year(value):
    if not isinstance(value, str) or not value.strip():
        return np.nan
    try:
        year = parsedate_to_datetime(value).year
    except Exception:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(ts):
            return np.nan
        year = ts.year
    lo, hi = PLAUSIBLE_YEARS
    return year if lo <= year <= hi else np.nan


def dup_hash(body):
    tokens = re.findall(r"[a-z0-9]+", body.lower())[:DUP_TOKENS]
    return hashlib.md5(" ".join(tokens).encode()).hexdigest()


# --------------------------------------------------------------------- models
def tfidf_lr():
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=60000),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
    )


def tfidf_svm():
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=60000),
        LinearSVC(class_weight="balanced", random_state=SEED),
    )


def tfidf_rf():
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=60000),
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               n_jobs=-1, random_state=SEED),
    )


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def metrics(y, pred, score):
    return {
        "F1": round(f1_score(y, pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y, score), 4),
        "Recall": round(recall_score(y, pred, zero_division=0), 4),
        "Precision": round(precision_score(y, pred, zero_division=0), 4),
    }


def split(df):
    return train_test_split(df, test_size=TEST_SIZE, stratify=df["label"],
                            random_state=SEED)


def in_corpus(df, column="text", factory=tfidf_lr, dedup_train=True):
    """Stratified 70/30 split; duplicates removed from the training partition
    after splitting (the 'de-duplicated after split' baseline of Table 6)."""
    train, test = split(df)
    if dedup_train:
        train = train.drop_duplicates("dup_hash")
    model = factory().fit(train[column], train["label"])
    return metrics(test["label"], model.predict(test[column]), scores(model, test[column]))


# ------------------------------------------------------------------ diagnostics
def profile(df):
    n = len(df)
    pos, neg = df[df.label == 1], df[df.label == 0]
    has_link = (df["body"].str.contains(URL_RE)
                | (pd.to_numeric(df["urls"], errors="coerce").fillna(0) > 0))
    dups = df["dup_hash"].duplicated(keep="first").sum()
    missing = df["year"].isna().sum()

    def yr(d):
        y = d["year"].dropna()
        return "n/a" if y.empty else (f"{int(y.min())}" if y.min() == y.max()
                                      else f"{int(y.min())}-{int(y.max())}")
    pct = lambda k: f"{k:,} ({100 * k / n:.2f}%)"
    return {
        "Messages": f"{n:,}",
        "Phishing or spam class": pct(len(pos)),
        "Legitimate class": pct(len(neg)),
        "Mean body length (characters)": f"{df.body.str.len().mean():.1f}",
        "Median body length (characters)": f"{df.body.str.len().median():.0f}",
        "Messages containing a hyperlink": pct(int(has_link.sum())),
        "Near-duplicate bodies": pct(int(dups)),
        "Missing or implausible Date header": pct(int(missing)),
        "  of which phishing class": f"{int(pos['year'].isna().sum()):,}",
        "Phishing class year range (parsable dates)": yr(pos),
        "Legitimate class year range (parsable dates)": yr(neg),
    }


def year_only_auc(df):
    d = df.dropna(subset=["year"])
    if d["label"].nunique() < 2:
        return {"n": len(d), "AUC": float("nan")}
    X = d[["year"]].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    prob = cross_val_predict(LogisticRegression(), X, d["label"], cv=cv,
                             method="predict_proba")[:, 1]
    return {"n": len(d), "AUC": round(roc_auc_score(d["label"], prob), 4)}


def cross_corpus(src, tgt):
    model = tfidf_lr().fit(src["text"], src["label"])
    prob = model.predict_proba(tgt["text"])[:, 1]
    pred = (prob >= 0.5).astype(int)
    return metrics(tgt["label"], pred, prob), prob, pred


def dedup_effect(df):
    train, test = split(df)
    leaked = int(test["dup_hash"].isin(set(train["dup_hash"])).sum())
    after = in_corpus(df)["F1"]
    before = in_corpus(df.drop_duplicates("dup_hash"))["F1"]
    return {
        "Test messages with a near-duplicate in training":
            f"{leaked:,} ({100 * leaked / len(test):.2f}%)",
        "F1 (de-duplicated after split)": after,
        "F1 (de-duplicated before split)": before,
        "Difference": round(after - before, 4),
    }


def domain_content(df):
    hits = df[df["text"].str.contains(HEALTH_RE)]
    return {
        "Keyword matches": f"{len(hits):,}",
        "Share of corpus": f"{100 * len(hits) / len(df):.2f}%",
        "Of which phishing or spam": f"{int(hits.label.sum()):,}",
    }, hits


# --------------------------------------------------------------------- figures
def fig_years(ceas, naz, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.arange(1990, 2027, 2)
    for ax, df, title in ((axes[0], naz, "Nazario"), (axes[1], ceas, "CEAS-2008")):
        d = df.dropna(subset=["year"])
        ax.hist(d[d.label == 0].year, bins=bins, alpha=.75, color="#4c78a8", label="Legitimate")
        ax.hist(d[d.label == 1].year, bins=bins, alpha=.75, color="#e45756", label="Phishing/spam")
        ax.set(title=title, xlabel="Year of Date header", ylabel="E-mails")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig1_date_years.png", dpi=300); plt.close(fig)


def fig_shortcut(ceas, prob, naz, auc_year, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 1, 41)
    axes[0].hist(prob[ceas.label.values == 0], bins=bins, alpha=.75, color="#4c78a8", label="CEAS legitimate")
    axes[0].hist(prob[ceas.label.values == 1], bins=bins, alpha=.75, color="#e45756", label="CEAS spam/phishing")
    axes[0].axvline(0.5, ls="--", c="k", lw=1); axes[0].set_yscale("log")
    axes[0].set(title="Nazario-trained model scoring 2008 mail",
                xlabel="Phishing probability assigned", ylabel="E-mails (log scale)")
    axes[0].legend(fontsize=8)
    d = naz.dropna(subset=["year"])
    rng = np.random.default_rng(SEED)
    jitter = rng.uniform(-0.12, 0.12, len(d))
    colors = np.where(d.label == 1, "#e45756", "#4c78a8")
    axes[1].scatter(d.year + rng.uniform(-.4, .4, len(d)), d.label + jitter, s=4, c=colors, alpha=.6)
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["Legitimate", "Phishing"])
    axes[1].set(title=f"Nazario: year alone separates the classes (AUC = {auc_year:.4f})",
                xlabel="Year of Date header")
    fig.tight_layout(); fig.savefig(out / "fig2_age_shortcut.png", dpi=300); plt.close(fig)


def fig_probe(probe, out):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(probe)); w = .38
    full = [r["Full text AUC"] for r in probe]; frag = [r["Structural fragment AUC"] for r in probe]
    for xs, vals, lab, col in ((x - w / 2, full, "Full message text", "#4c78a8"),
                               (x + w / 2, frag, "Structural fragment only", "#f58518")):
        bars = ax.bar(xs, vals, w, label=lab, color=col)
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.axhline(0.5, ls="--", c="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([r["Corpus"] for r in probe])
    ax.set(ylim=(0.45, 1.05), ylabel="ROC-AUC", title="Structural-fragment probe fires on both corpora")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig3_structural_probe.png", dpi=300); plt.close(fig)


def fig_transfer(matrix, out):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
    names = ["CEAS-08", "Nazario"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(names); ax.set_yticks([0, 1]); ax.set_yticklabels(names)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i][j]:.3f}", ha="center", va="center", fontweight="bold")
    ax.set(xlabel="Evaluated on", ylabel="Trained on", title="Cross-corpus transfer (F1)")
    fig.colorbar(im, ax=ax, label="F1")
    fig.tight_layout(); fig.savefig(out / "fig4_transfer_matrix.png", dpi=300); plt.close(fig)


def fig_integrity(ceas, naz, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = ["CEAS-08", "Nazario"]
    dup = [100 * d.dup_hash.duplicated().mean() for d in (ceas, naz)]
    miss = [100 * d.year.isna().mean() for d in (ceas, naz)]
    for ax, vals, title, ylab in ((axes[0], dup, "Duplication before de-duplication", "% near-duplicate bodies"),
                                  (axes[1], miss, "Date header integrity", "% missing/implausible Date header")):
        bars = ax.bar(names, vals, color=["#4c78a8", "#f58518"])
        ax.bar_label(bars, fmt="%.2f%%", fontsize=8)
        ax.set(title=title, ylabel=ylab)
    fig.tight_layout(); fig.savefig(out / "fig5_integrity.png", dpi=300); plt.close(fig)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ceas", required=True, help="CEAS_08.csv")
    ap.add_argument("--nazario", required=True, help="Nazario_5.csv (phishing + legitimate)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--skip-rf", action="store_true", help="skip the random-forest probe (slow)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ceas = load_corpus(args.ceas, "CEAS-2008")
    naz = load_corpus(args.nazario, "Nazario")
    results = {}

    # Table 1 -------------------------------------------------------------
    t1 = pd.DataFrame({"CEAS-2008": profile(ceas), "Nazario": profile(naz)})
    t1.to_csv(out / "table1_profile.csv"); print("\nTable 1\n", t1)

    # Section 5.1: year-only classifier -----------------------------------
    results["year_only"] = {"Nazario": year_only_auc(naz), "CEAS-2008": year_only_auc(ceas)}
    print("\nYear-only AUC:", results["year_only"])

    # Table 5 and Table 2: in-corpus and cross-corpus ----------------------
    rows = [
        {"Training corpus": "CEAS-2008", "Evaluation corpus": "CEAS-2008 (held out)", **in_corpus(ceas)},
        {"Training corpus": "Nazario", "Evaluation corpus": "Nazario (held out)", **in_corpus(naz)},
    ]
    m_cn, _, _ = cross_corpus(ceas, naz)
    m_nc, prob_nc, pred_nc = cross_corpus(naz, ceas)
    rows += [{"Training corpus": "CEAS-2008", "Evaluation corpus": "Nazario", **m_cn},
             {"Training corpus": "Nazario", "Evaluation corpus": "CEAS-2008", **m_nc}]
    t5 = pd.DataFrame(rows); t5.to_csv(out / "table5_transfer.csv", index=False)
    print("\nTable 5\n", t5.to_string(index=False))

    t2 = []
    for label, mask in (("Spam / phishing class", ceas.label.values == 1),
                        ("Legitimate class", ceas.label.values == 0),
                        ("Entire corpus", np.ones(len(ceas), bool))):
        t2.append({"CEAS-2008 subset": label, "Messages": int(mask.sum()),
                   "Flagged as phishing": f"{100 * pred_nc[mask].mean():.2f}%",
                   "Mean assigned probability": round(float(prob_nc[mask].mean()), 3)})
    t2 = pd.DataFrame(t2); t2.to_csv(out / "table2_nazario_on_ceas.csv", index=False)
    print("\nTable 2\n", t2.to_string(index=False))

    # Table 3 and Table 4: structural probe --------------------------------
    t3 = []
    for df, contemp in ((naz, "No"), (ceas, "Yes")):
        full = in_corpus(df, "text"); frag = in_corpus(df, "fragment")
        t3.append({"Corpus": df.attrs["name"], "Classes contemporaneous": contemp,
                   "Full text AUC": full["ROC-AUC"], "Structural fragment AUC": frag["ROC-AUC"],
                   "Structural F1": frag["F1"],
                   "F1 change, fragment removed": round(in_corpus(df, "text_minus_fragment")["F1"] - full["F1"], 4)})
    t3 = pd.DataFrame(t3); t3.to_csv(out / "table3_probe.csv", index=False)
    print("\nTable 3\n", t3.to_string(index=False))

    ct = naz[naz.content_type.str.len() > 0]
    results["content_type_only_nazario"] = (
        in_corpus(naz, "content_type")["ROC-AUC"] if len(ct) else
        "not computable: no Content-Type lines present in body text")

    families = [("Logistic regression", tfidf_lr), ("Linear support vector machine", tfidf_svm)]
    if not args.skip_rf:
        families.append(("Random forest", tfidf_rf))
    t4 = pd.DataFrame([{"Classifier": n, **{k: v for k, v in in_corpus(naz, "fragment", f).items()
                                             if k in ("ROC-AUC", "F1")}} for n, f in families])
    t4.to_csv(out / "table4_probe_families.csv", index=False)
    print("\nTable 4\n", t4.to_string(index=False))

    # Table 6: de-duplication order ----------------------------------------
    t6 = pd.DataFrame([{"Corpus": d.attrs["name"], **dedup_effect(d)} for d in (ceas, naz)])
    t6.to_csv(out / "table6_dedup.csv", index=False)
    print("\nTable 6\n", t6.to_string(index=False))

    # Table 7: domain content -----------------------------------------------
    t7 = []
    for df in (ceas, naz):
        summary, hits = domain_content(df)
        t7.append({"Corpus": df.attrs["name"], **summary})
        hits[["label", "subject", "body"]].assign(body=hits.body.str[:300]).to_csv(
            out / f"health_matches_{df.attrs['name']}.csv", index=False)  # for manual inspection
    t7 = pd.DataFrame(t7); t7.to_csv(out / "table7_domain.csv", index=False)
    print("\nTable 7\n", t7.to_string(index=False))

    # Figures -------------------------------------------------------------
    fig_years(ceas, naz, out)
    fig_shortcut(ceas, prob_nc, naz, results["year_only"]["Nazario"]["AUC"], out)
    fig_probe(t3.to_dict("records"), out)
    fig_transfer([[rows[0]["F1"], m_cn["F1"]], [m_nc["F1"], rows[1]["F1"]]], out)
    fig_integrity(ceas, naz, out)

    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nAll tables and figures written to {out.resolve()}")


if __name__ == "__main__":
    main()
