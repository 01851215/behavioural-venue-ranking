"""
Phase 4: Cross-Domain Coffee → Hotel Transfer Analysis

Analyses whether coffee shop behavioural patterns predict hotel preferences
for the 59,668 users who reviewed both domains.

Analyses:
  1. Correlation: do coffee Loyalists become hotel Road Warriors?
  2. Contingency table: coffee archetype × hotel archetype (chi-squared test)
  3. Classifier: can coffee features predict hotel segment?
  4. Transfer priors: for users with coffee but no hotel history, infer hotel
     archetype to use as a BiRank prior weight

Outputs:
  cross_domain_analysis.csv    — per-user coffee + hotel archetype labels
  cross_domain_priors.csv      — hotel prior weights for all users
  cross_domain_summary.txt     — human-readable findings
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent

print("=" * 65)
print("  PHASE 4: CROSS-DOMAIN COFFEE → HOTEL TRANSFER")
print("=" * 65)

# ── Load data ──────────────────────────────────────────────────────────────────
print("\n  Loading coffee and hotel user features...")

# Coffee user features
coffee_feat = pd.read_csv(DATA_DIR / "coffee_user_features_v3.csv")
# Coffee user groups (from phase3_taskA_define_groups.py)
coffee_groups_file = DATA_DIR / "coffee_user_groups.csv"
if coffee_groups_file.exists():
    coffee_groups = pd.read_csv(coffee_groups_file)
else:
    # Fall back: assign groups from coffee features using same thresholds as v5
    coffee_groups = coffee_feat[["user_id"]].copy()
    coffee_groups["coffee_archetype"] = "Casual Weekenders"
    if "revisit_ratio" in coffee_feat.columns:
        coffee_groups.loc[coffee_feat["revisit_ratio"].values >= 0.25, "coffee_archetype"] = "Loyalists"
        coffee_groups.loc[
            (coffee_feat["total_visits"].values <= 2) & (coffee_feat["revisit_ratio"].values < 0.25),
            "coffee_archetype"
        ] = "Infrequent Visitors"
        coffee_groups.loc[
            (coffee_feat["total_visits"].values >= 4) & (coffee_feat["revisit_ratio"].values < 0.25),
            "coffee_archetype"
        ] = "Weekday Regulars"

# Hotel user groups
hotel_groups = pd.read_csv(DATA_DIR / "hotel_user_groups.csv")
hotel_feat   = pd.read_csv(DATA_DIR / "hotel_user_features.csv")

print(f"    Coffee users: {len(coffee_feat):,}")
print(f"    Hotel users:  {len(hotel_feat):,}")

# ── Find overlap ───────────────────────────────────────────────────────────────
coffee_ids = set(coffee_feat["user_id"].astype(str))
hotel_ids  = set(hotel_feat["user_id"].astype(str))
overlap    = coffee_ids & hotel_ids

print(f"    Overlap (both domains): {len(overlap):,}")

# Merge on overlap
coffee_feat["user_id"] = coffee_feat["user_id"].astype(str)
hotel_feat["user_id"]  = hotel_feat["user_id"].astype(str)
hotel_groups["user_id"] = hotel_groups["user_id"].astype(str)

if "user_id" in coffee_groups.columns:
    coffee_groups["user_id"] = coffee_groups["user_id"].astype(str)
    coffee_labeled = coffee_groups[["user_id","coffee_archetype"]] if "coffee_archetype" in coffee_groups.columns else None
else:
    coffee_labeled = None

# Build cross-domain df
cross = (hotel_groups[["user_id","archetype"]]
         .rename(columns={"archetype": "hotel_archetype"})
         .merge(hotel_feat, on="user_id", how="left"))

# Add coffee archetype if available
if coffee_labeled is not None and len(coffee_labeled) > 0:
    cross = cross.merge(coffee_labeled, on="user_id", how="left")
else:
    cross["coffee_archetype"] = "Unknown"

# Add coffee features for overlap users
coffee_feat_sub = coffee_feat[coffee_feat["user_id"].isin(overlap)].copy()
cross_overlap = cross[cross["user_id"].isin(overlap)].merge(
    coffee_feat_sub, on="user_id", how="left", suffixes=("", "_coffee")
)

print(f"\n  Overlap users with full features: {len(cross_overlap):,}")

# Coffee numeric feature columns (original names, now in cross_overlap)
coffee_num_cols = [c for c in coffee_feat.columns
                   if c != "user_id" and coffee_feat[c].dtype in [np.float64, np.int64, float, int]]

# ── Analysis 1: Archetype contingency table ────────────────────────────────────
print("\n  Analysis 1: Coffee archetype → Hotel archetype")
print("  " + "─" * 60)

if "coffee_archetype" in cross_overlap.columns and cross_overlap["coffee_archetype"].notna().any():
    contingency = pd.crosstab(
        cross_overlap["coffee_archetype"],
        cross_overlap["hotel_archetype"],
        margins=True
    )
    print(contingency.to_string())

    ct_data = pd.crosstab(
        cross_overlap["coffee_archetype"].dropna(),
        cross_overlap["hotel_archetype"].dropna()
    )
    if ct_data.shape[0] > 1 and ct_data.shape[1] > 1:
        chi2, p, dof, expected = chi2_contingency(ct_data)
        print(f"\n  Chi-squared: {chi2:.2f}  dof={dof}  p={p:.4f}")
        if p < 0.05:
            print("  → Coffee and hotel archetypes are significantly associated (p<0.05)")
        else:
            print("  → No significant association between domains")
else:
    print("  (Coffee archetype labels not available — skipping contingency)")

# ── Analysis 2: Feature correlations ──────────────────────────────────────────
print("\n  Analysis 2: Cross-domain feature correlations")
print("  " + "─" * 60)

hotel_num_cols = ["total_hotel_reviews","n_states_visited","weekday_fraction",
                  "hotel_frequency","hotel_city_diversity"]

corr_results = []
for cc in coffee_num_cols[:8]:
    for hc in hotel_num_cols:
        if cc in cross_overlap.columns and hc in cross_overlap.columns:
            valid = cross_overlap[[cc, hc]].dropna()
            if len(valid) > 50:
                r, p_val = spearmanr(valid[cc], valid[hc])
                if abs(r) > 0.05:
                    corr_results.append({"coffee_feat": cc, "hotel_feat": hc,
                                         "spearman_r": round(r, 3), "p": round(p_val, 4)})

if corr_results:
    corr_df = pd.DataFrame(corr_results).sort_values("spearman_r", key=abs, ascending=False)
    print(corr_df.head(10).to_string(index=False))
else:
    print("  (No correlations above threshold)")

# ── Analysis 3: Classifier — can coffee features predict hotel archetype? ──────
print("\n  Analysis 3: Coffee features → Hotel archetype classifier")
print("  " + "─" * 60)

feature_cols = [c for c in coffee_num_cols if c in cross_overlap.columns]
X = cross_overlap[feature_cols].copy()
y = cross_overlap["hotel_archetype"].copy()

valid_mask = y.notna() & X.notna().all(axis=1)
X, y = X[valid_mask], y[valid_mask]

clf_accuracy = None
if len(X) > 200 and y.nunique() > 1:
    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    X_proc = scl.fit_transform(imp.fit_transform(X))

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    scores = cross_val_score(clf, X_proc, y_enc, cv=5, scoring="accuracy")
    clf_accuracy = scores.mean()
    baseline = 1.0 / y.nunique()

    print(f"  5-fold CV accuracy: {clf_accuracy:.3f}  (baseline={baseline:.3f})")
    if clf_accuracy > baseline + 0.05:
        print("  → Coffee behaviour meaningfully predicts hotel preference")
    else:
        print("  → Coffee behaviour does NOT reliably predict hotel preference")
        print("     (domains appear largely independent)")

    # Fit full model for transfer priors
    clf.fit(X_proc, y_enc)
    clf_model  = clf
    clf_scaler = scl
    clf_imputer = imp
    clf_le     = le
    clf_features = feature_cols
else:
    print("  (Insufficient data for classifier)")
    clf_model = None

# ── Analysis 4: Build transfer priors ─────────────────────────────────────────
print("\n  Analysis 4: Building transfer priors for hotel BiRank")
print("  " + "─" * 60)

# For all coffee users (including those without hotel history):
# predict their hotel archetype → use as prior weight
# Prior weight: Road Warriors get higher weight (stronger signal),
#               One-Time Tourists get weight 1.0 (neutral)
archetype_weights = {
    "Road Warriors":           1.5,
    "Budget Explorers":        1.3,
    "Leisure Travelers":       1.1,
    "One-Time Tourists":       1.0,
    "One-Time Tourists (Business)": 1.0,
    "One-Time Tourists (Leisure)":  1.0,
}

# For users already in hotel data, use actual archetype
known_priors = hotel_groups[["user_id","archetype"]].copy()
known_priors["prior_weight"] = known_priors["archetype"].map(archetype_weights).fillna(1.0)
known_priors["source"] = "hotel_archetype"

# For coffee-only users with classifier available, predict
transferred = pd.DataFrame(columns=["user_id","archetype","prior_weight","source"])
if clf_model is not None:
    coffee_only = coffee_feat[~coffee_feat["user_id"].isin(hotel_ids)].copy()
    if len(coffee_only) > 0:
        X_co = coffee_only[clf_features].copy()
        X_co_proc = clf_scaler.transform(clf_imputer.transform(X_co))
        pred_labels = clf_le.inverse_transform(clf_model.predict(X_co_proc))
        transferred = pd.DataFrame({
            "user_id":     coffee_only["user_id"].values,
            "archetype":   pred_labels,
            "prior_weight": [archetype_weights.get(l, 1.0) for l in pred_labels],
            "source":      "coffee_transfer",
        })
        print(f"  Predicted hotel archetypes for {len(transferred):,} coffee-only users")
        print("  Predicted distribution:")
        print(transferred["archetype"].value_counts().to_string())

# Combine
all_priors = pd.concat([known_priors, transferred], ignore_index=True)

all_priors.to_csv(DATA_DIR / "cross_domain_priors.csv", index=False)
print(f"\n  Saved: cross_domain_priors.csv  ({len(all_priors):,} users)")

# Save cross-domain analysis
cross_overlap[["user_id","hotel_archetype","coffee_archetype"]].to_csv(
    DATA_DIR / "cross_domain_analysis.csv", index=False
)
print(f"  Saved: cross_domain_analysis.csv  ({len(cross_overlap):,} overlap users)")

# Summary text
with open(DATA_DIR / "cross_domain_summary.txt", "w") as f:
    f.write("CROSS-DOMAIN TRANSFER ANALYSIS\n")
    f.write(f"Coffee users: {len(coffee_feat):,}  |  Hotel users: {len(hotel_feat):,}  |  Overlap: {len(overlap):,}\n\n")
    if clf_accuracy is not None:
        f.write(f"Classifier accuracy: {clf_accuracy:.3f}  (chance={1.0/y.nunique():.3f})\n")
        f.write(f"Result: {'Meaningful transfer' if clf_accuracy > 1.0/y.nunique() + 0.05 else 'Domains largely independent'}\n\n")
    f.write(f"Transfer priors built for {len(all_priors):,} users\n")

print(f"  Saved: cross_domain_summary.txt")
print("\n  Phase 4 complete.\n")
