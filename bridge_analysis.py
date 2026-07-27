import json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
import statsmodels.api as sm

OUT_JSON = Path("bridge_results.json")
OUT_CSV = Path("bridge_panel.csv")

URL_PLAYERS_2425 = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/players_raw.csv"
URL_GWS_2425 = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv"
URL_MAP_2526 = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data/2025-2026/By%20Gameweek/GW10/players.csv"
URL_GW10_2526 = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data/2025-2026/By%20Gameweek/GW10/playerstats.csv"

def corr(func, x, y):
    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    if len(x) < 4 or x.nunique() < 2 or y.nunique() < 2:
        return {"r": None, "p": None, "n": int(len(x))}
    r, p = func(x, y)
    return {"r": float(r), "p": float(p), "n": int(len(x))}

def cv_r2(X, y):
    n = len(y)
    k = min(10, max(3, n // 20))
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")
    return {"mean": float(scores.mean()), "sd": float(scores.std()), "folds": k}

def segment_stats(df, label):
    d = df.copy()
    y = d["f10_points"].astype(float)
    out = {
        "segment": label,
        "n": int(len(d)),
        "correlations": {
            "full_season_2425": {"spearman": corr(spearmanr, d["points_2425"], y), "pearson": corr(pearsonr, d["points_2425"], y)},
            "gw1_28": {"spearman": corr(spearmanr, d["gw1_28"], y), "pearson": corr(pearsonr, d["gw1_28"], y)},
            "gw29_38": {"spearman": corr(spearmanr, d["gw29_38"], y), "pearson": corr(pearsonr, d["gw29_38"], y)},
            "acceleration": {"spearman": corr(spearmanr, d["acceleration"], y), "pearson": corr(pearsonr, d["acceleration"], y)},
        }
    }
    X0 = sm.add_constant(d[["early_rate"]].astype(float))
    X1 = sm.add_constant(d[["early_rate","late_rate"]].astype(float))
    m0 = sm.OLS(y, X0).fit()
    m1 = sm.OLS(y, X1).fit()
    out["regression"] = {
        "early_only_r2": float(m0.rsquared),
        "early_plus_late_r2": float(m1.rsquared),
        "incremental_r2": float(m1.rsquared - m0.rsquared),
        "late_coefficient": float(m1.params["late_rate"]),
        "late_p": float(m1.pvalues["late_rate"]),
        "early_coefficient": float(m1.params["early_rate"]),
        "early_p": float(m1.pvalues["early_rate"]),
    }
    out["cross_validated_r2"] = {
        "early_only": cv_r2(d[["early_rate"]].values, y.values),
        "early_plus_late": cv_r2(d[["early_rate","late_rate"]].values, y.values),
    }
    late_model = LinearRegression().fit(d[["early_rate"]], d["late_rate"])
    d["late_residual"] = d["late_rate"] - late_model.predict(d[["early_rate"]])
    out["residual_late"] = {"spearman": corr(spearmanr, d["late_residual"], y), "pearson": corr(pearsonr, d["late_residual"], y)}

    ql = float(d["gw29_38"].quantile(.75))
    qf = float(d["f10_points"].quantile(.75))
    qs = float(d["points_2425"].quantile(.75))
    d["top_late"] = d["gw29_38"] >= ql
    d["top_f10"] = d["f10_points"] >= qf
    d["top_full"] = d["points_2425"] >= qs
    base = float(d["top_f10"].mean())
    hit_late = float(d.loc[d["top_late"],"top_f10"].mean())
    hit_full = float(d.loc[d["top_full"],"top_f10"].mean())
    out["quartiles"] = {
        "late_threshold": ql,
        "f10_threshold": qf,
        "full_season_threshold": qs,
        "top_f10_base_rate": base,
        "top_late_count": int(d["top_late"].sum()),
        "top_late_and_top_f10_count": int((d["top_late"] & d["top_f10"]).sum()),
        "top_late_hit_rate": hit_late,
        "top_late_lift_vs_base": hit_late/base if base else None,
        "top_full_count": int(d["top_full"].sum()),
        "top_full_and_top_f10_count": int((d["top_full"] & d["top_f10"]).sum()),
        "top_full_hit_rate": hit_full,
        "top_full_lift_vs_base": hit_full/base if base else None,
        "mean_f10_top_late": float(d.loc[d["top_late"],"f10_points"].mean()),
        "mean_f10_others": float(d.loc[~d["top_late"],"f10_points"].mean()),
        "difference_in_means": float(d.loc[d["top_late"],"f10_points"].mean() - d.loc[~d["top_late"],"f10_points"].mean()),
    }
    return out, d

players = pd.read_csv(URL_PLAYERS_2425)
gws = pd.read_csv(URL_GWS_2425)
mapping = pd.read_csv(URL_MAP_2526)
gw10 = pd.read_csv(URL_GW10_2526)

gw_col = next((c for c in ["GW","round","event"] if c in gws.columns), None)
if gw_col is None:
    raise RuntimeError(f"No gameweek column found: {list(gws.columns)}")
if "element" not in gws.columns or "total_points" not in gws.columns:
    raise RuntimeError(f"Expected element and total_points in merged_gw: {list(gws.columns)}")

weekly = gws.groupby(["element", gw_col], as_index=False)["total_points"].sum()
early = weekly.loc[weekly[gw_col] <= 28].groupby("element")["total_points"].sum().rename("gw1_28")
late = weekly.loc[weekly[gw_col] >= 29].groupby("element")["total_points"].sum().rename("gw29_38")
season = players[["id","code","first_name","second_name","web_name","element_type","total_points"]].copy()
season = season.rename(columns={"id":"element","total_points":"points_2425"})
season = season.join(early, on="element").join(late, on="element").fillna({"gw1_28":0,"gw29_38":0})
season["player"] = season["first_name"].fillna("") + " " + season["second_name"].fillna("")
pos_map = {1:"GK",2:"DEF",3:"MID",4:"FWD"}
season["position"] = season["element_type"].map(pos_map)
season["early_rate"] = season["gw1_28"]/28.0
season["late_rate"] = season["gw29_38"]/10.0
season["acceleration"] = season["late_rate"] - season["early_rate"]
season = season[season["points_2425"] >= 30].copy()

if "player_code" not in mapping.columns:
    mapping = mapping.rename(columns={"code":"player_code"})
if "player_id" not in mapping.columns:
    mapping = mapping.rename(columns={"id":"player_id"})
gw10 = gw10.rename(columns={"id":"player_id","total_points":"f10_points","minutes":"f10_minutes"})

bridge = season.merge(mapping[["player_code","player_id"]], left_on="code", right_on="player_code", how="left")
bridge = bridge.merge(gw10[["player_id","f10_points","f10_minutes"]], on="player_id", how="left")
matched = bridge[bridge["f10_points"].notna()].copy()
matched["position_group"] = np.where(matched["position"].isin(["MID","FWD"]),"MID/FWD","DEF/GK")

all_s, all_d = segment_stats(matched, "All positions")
att_s, att_d = segment_stats(matched[matched["position_group"]=="MID/FWD"], "MID/FWD")
def_s, def_d = segment_stats(matched[matched["position_group"]=="DEF/GK"], "DEF/GK")

d = all_d.copy()
good = d[d["top_late"] & d["top_f10"]].sort_values(["f10_points","gw29_38"], ascending=False)
fp = d[d["top_late"] & ~d["top_f10"]].sort_values(["gw29_38","f10_points"], ascending=[False,True])
fn = d[~d["top_late"] & d["top_f10"]].sort_values(["f10_points","gw29_38"], ascending=[False,True])
top = d.sort_values(["gw29_38","f10_points"], ascending=False).head(25)
cols = ["player","web_name","position","points_2425","gw1_28","gw29_38","acceleration","f10_points","f10_minutes"]
def rec(df,n=15):
    return df[cols].head(n).replace({np.nan:None}).to_dict("records")

rho_late = all_s["correlations"]["gw29_38"]["spearman"]["r"]
rho_full = all_s["correlations"]["full_season_2425"]["spearman"]["r"]
rho_acc = all_s["correlations"]["acceleration"]["spearman"]["r"]
p_acc = all_s["correlations"]["acceleration"]["spearman"]["p"]
late_p = all_s["regression"]["late_p"]
inc = all_s["regression"]["incremental_r2"]
cv0 = all_s["cross_validated_r2"]["early_only"]["mean"]
cv1 = all_s["cross_validated_r2"]["early_plus_late"]["mean"]

level = "moderate" if rho_late >= .35 else ("modest" if rho_late >= .20 else "weak")
independent = "retained independent predictive value" if late_p < .05 and inc > 0 else "did not retain convincing independent predictive value"
accel = "showed a positive relationship" if p_acc < .05 and rho_acc > 0 else "did not show a reliable positive relationship"
interpretation = (
    f"The direct ten-to-ten bridge showed a {level} relationship: Spearman rho was {rho_late:.3f} for "
    f"2024/25 GW29-38 points against 2025/26 GW1-10 points, versus {rho_full:.3f} for the full 2024/25 season. "
    f"Late-season level {independent} after controlling for GW1-28; raw acceleration {accel}. "
    f"Cross-validated R-squared changed from {cv0:.3f} to {cv1:.3f} when the final-ten rate was added. "
    f"The evidence supports Exit Velocity as a modest recency and role signal, not a claim that momentum itself survives the summer."
)

result = {
    "sources":{"players_2425":URL_PLAYERS_2425,"weekly_2425":URL_GWS_2425,"map_2526":URL_MAP_2526,"gw10_2526":URL_GW10_2526},
    "sample":{"players_2425_at_least_30":int(len(season)),"matched_to_gw10":int(len(matched)),"excluded_not_in_gw10_snapshot":int(len(season)-len(matched))},
    "segments":{"all":all_s,"mid_fwd":att_s,"def_gk":def_s},
    "examples":{"successful_persistence":rec(good),"false_positives":rec(fp),"false_negatives":rec(fn),"top_final_ten":rec(top,25)},
    "interpretation":interpretation
}
OUT_JSON.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
matched[cols+["code","player_id","position_group"]].sort_values("gw29_38",ascending=False).to_csv(OUT_CSV,index=False)
print(json.dumps({"sample":result["sample"],"interpretation":interpretation},indent=2))
