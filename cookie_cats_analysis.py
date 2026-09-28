import json, hashlib
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260927)
OUT = {}

df = pd.read_csv("cookie_cats.csv")
OUT["raw_shape"] = list(df.shape)
OUT["raw_columns"] = list(df.columns)
OUT["nulls"] = {c: int(df[c].isna().sum()) for c in df.columns}
OUT["duplicated_userid"] = int(df["userid"].duplicated().sum())
OUT["version_counts"] = {k: int(v) for k, v in df["version"].value_counts().items()}
OUT["gamerounds_min"] = int(df["sum_gamerounds"].min())
OUT["gamerounds_max"] = int(df["sum_gamerounds"].max())
OUT["gamerounds_top10"] = df.nlargest(10, "sum_gamerounds")[["userid", "version", "sum_gamerounds"]].to_dict("records")
OUT["zero_rounds"] = {k: int(v) for k, v in df.groupby("version")["sum_gamerounds"].apply(lambda s: (s == 0).sum()).items()}
OUT["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

n30 = int((df["version"] == "gate_30").sum())
n40 = int((df["version"] == "gate_40").sum())
N = n30 + n40
chi2_srm, p_srm = stats.chisquare([n30, n40], f_exp=[N / 2, N / 2])
OUT["srm"] = {"n30": n30, "n40": n40, "N": N, "ratio40_over_30": n40 / n30,
              "chi2": float(chi2_srm), "p": float(p_srm),
              "srm_alert": bool(p_srm < 0.001)}

q1, q3 = df["sum_gamerounds"].quantile([0.25, 0.75])
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
outlier_mask = df["sum_gamerounds"] > upper_fence
OUT["outliers"] = {"q1": float(q1), "q3": float(q3), "iqr": float(iqr),
                   "upper_fence": float(upper_fence),
                   "count": int(outlier_mask.sum()),
                   "pct": float(outlier_mask.mean() * 100),
                   "by_version": {k: int(v) for k, v in df[outlier_mask]["version"].value_counts().items()}}

df_clean = df[~outlier_mask].copy()
OUT["clean_shape"] = list(df_clean.shape)

def wilson(k, n, z=1.959963985):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def wald_se(k, n):
    p = k / n
    return np.sqrt(p * (1 - p) / n)

desc = {}
for g in ["gate_30", "gate_40"]:
    sub = df[df["version"] == g]
    ng = len(sub)
    ent = {"n": ng}
    for m in ["retention_1", "retention_7"]:
        k = int(sub[m].sum())
        lo, hi = wilson(k, ng)
        ent[m] = {"k": k, "rate": k / ng, "ci_low": float(lo), "ci_high": float(hi),
                  "se": float(wald_se(k, ng))}
    for m in ["sum_gamerounds"]:
        s = sub[m]
        ent[m] = {"mean": float(s.mean()), "std": float(s.std(ddof=1)),
                  "median": float(s.median()), "q1": float(s.quantile(.25)),
                  "q3": float(s.quantile(.75)), "p90": float(s.quantile(.90)),
                  "p99": float(s.quantile(.99)), "max": float(s.max()),
                  "mean_trimmed_winsor": float(s.clip(upper=q3 + 1.5 * iqr).mean())}
    desc[g] = ent
OUT["desc"] = desc
OUT["overall_retention"] = {"retention_1": float(df["retention_1"].mean()),
                            "retention_7": float(df["retention_7"].mean())}
OUT["overall_gamerounds_mean"] = float(df["sum_gamerounds"].mean())
OUT["overall_gamerounds_median"] = float(df["sum_gamerounds"].median())

def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se_pool
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    se_unpool = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p2 - p1
    return {"p1": p1, "p2": p2, "diff": diff, "se": float(se_unpool), "z": float(z), "p": float(p),
            "ci_low": float(diff - 1.959963985 * se_unpool),
            "ci_high": float(diff + 1.959963985 * se_unpool),
            "rel_lift": diff / p1,
            "cohens_h": float(2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1)))}

freq = {}
for m in ["retention_1", "retention_7"]:
    a = df[df["version"] == "gate_30"]; b = df[df["version"] == "gate_40"]
    freq[m] = two_prop_z(int(a[m].sum()), len(a), int(b[m].sum()), len(b))
    tab = pd.crosstab(df["version"], df[m])
    c2, pc, dof, _ = stats.chi2_contingency(tab)
    freq[m]["chi2"] = float(c2); freq[m]["chi2_p"] = float(pc)
OUT["freq"] = freq

a = df.loc[df["version"] == "gate_30", "sum_gamerounds"].to_numpy()
b = df.loc[df["version"] == "gate_40", "sum_gamerounds"].to_numpy()
u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
n1, n2 = len(a), len(b)
# rank-biserial correlation
rbc = 1 - 2 * u_stat / (n1 * n2)
prob_sup = u_stat / (n1 * n2)
OUT["mannwhitney"] = {"u": float(u_stat), "p": float(u_p),
                      "rank_biserial": float(rbc),
                      "prob_superiority": float(prob_sup),
                      "n1": n1, "n2": n2}

t_stat, t_p = stats.ttest_ind(b, a, equal_var=False) 
OUT["welch_raw"] = {"t": float(t_stat), "p": float(t_p),
                    "mean30": float(a.mean()), "mean40": float(b.mean()),
                    "diff": float(b.mean() - a.mean())}
ac = df_clean.loc[df_clean["version"] == "gate_30", "sum_gamerounds"].to_numpy()
bc = df_clean.loc[df_clean["version"] == "gate_40", "sum_gamerounds"].to_numpy()
tc_stat, tc_p = stats.ttest_ind(bc, ac, equal_var=False)

sp = np.sqrt(((len(ac) - 1) * ac.std(ddof=1) ** 2 + (len(bc) - 1) * bc.std(ddof=1) ** 2) / (len(ac) + len(bc) - 2))
d_cohen = (bc.mean() - ac.mean()) / sp
J = 1 - 3 / (4 * (len(ac) + len(bc)) - 9)
OUT["welch_clean"] = {"t": float(tc_stat), "p": float(tc_p),
                      "mean30": float(ac.mean()), "mean40": float(bc.mean()),
                      "diff": float(bc.mean() - ac.mean()),
                      "mean30_median": float(np.median(ac)), "mean40_median": float(np.median(bc)),
                      "hedges_g": float(d_cohen * J),
                      "n30": len(ac), "n40": len(bc)}

B = 5000
diffs = np.empty(B)
for i in range(0, B, 250):
    m = min(250, B - i)
    ia = RNG.integers(0, len(ac), size=(m, len(ac)))
    ib = RNG.integers(0, len(bc), size=(m, len(bc)))
    diffs[i:i + m] = bc[ib].mean(axis=1) - ac[ia].mean(axis=1)
OUT["boot_mean"] = {"n_iter": B, "diff_obs": float(bc.mean() - ac.mean()),
                    "ci_low": float(np.percentile(diffs, 2.5)),
                    "ci_high": float(np.percentile(diffs, 97.5)),
                    "mean": float(diffs.mean()), "sd": float(diffs.std(ddof=1))}

B2 = 20000
p30 = desc["gate_30"]["retention_7"]["rate"]; p40 = desc["gate_40"]["retention_7"]["rate"]
d7 = RNG.binomial(n40, p40, B2) / n40 - RNG.binomial(n30, p30, B2) / n30
lo7, hi7 = np.percentile(d7, [2.5, 97.5])
OUT["boot_ret7"] = {"n_iter": B2, "diff_obs": float(p40 - p30),
                    "ci_low": float(lo7), "ci_high": float(hi7),
                    "mean": float(d7.mean()), "sd": float(d7.std(ddof=1)),
                    "p_less_0": float((d7 < 0).mean())}
cnt, edges = np.histogram(d7 * 100, bins=60)
OUT["boot_ret7"]["hist"] = {"labels": [round(float((edges[i] + edges[i + 1]) / 2), 4) for i in range(len(cnt))],
                            "values": [int(x) for x in cnt]}

p30_1 = desc["gate_30"]["retention_1"]["rate"]; p40_1 = desc["gate_40"]["retention_1"]["rate"]
d1 = RNG.binomial(n40, p40_1, B2) / n40 - RNG.binomial(n30, p30_1, B2) / n30
OUT["boot_ret1"] = {"n_iter": B2, "diff_obs": float(p40_1 - p30_1),
                    "ci_low": float(np.percentile(d1, 2.5)), "ci_high": float(np.percentile(d1, 97.5)),
                    "sd": float(d1.std(ddof=1))}

bayes = {}
for m in ["retention_1", "retention_7"]:
    k1 = desc["gate_30"][m]["k"]; k2 = desc["gate_40"][m]["k"]
    al1, be1 = 1 + k1, 1 + n30 - k1
    al2, be2 = 1 + k2, 1 + n40 - k2
    s1 = RNG.beta(al1, be1, 400000)
    s2 = RNG.beta(al2, be2, 400000)
    delta = s2 - s1
    post = {
        "prior": "Beta(1,1)",
        "alpha30": al1, "beta30": be1, "alpha40": al2, "beta40": be2,
        "mean30": al1 / (al1 + be1), "mean40": al2 / (al2 + be2),
        "ci30": [float(np.percentile(s1, 2.5)), float(np.percentile(s1, 97.5))],
        "ci40": [float(np.percentile(s2, 2.5)), float(np.percentile(s2, 97.5))],
        "ci_delta": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
        "p_treat_better": float((delta > 0).mean()),
        "p_control_better": float((delta < 0).mean()),
        "expected_lift_abs": float(delta.mean()),
        "expected_lift_rel": float((s2 / s1 - 1).mean()),
        "p_rope_noninf": float((np.abs(delta) < 0.005).mean()),
        "prob_x2_worse": float((delta < -0.01).mean()),
    }

    grid = np.linspace(min(s1.min(), s2.min()) - 0.002, max(s1.max(), s2.max()) + 0.002, 220)
    post["curve_x"] = [round(float(x), 5) for x in grid]
    post["curve30"] = [float(stats.beta.pdf(x, al1, be1)) for x in grid]
    post["curve40"] = [float(stats.beta.pdf(x, al2, be2)) for x in grid]

    cnt2, ed2 = np.histogram(delta * 100, bins=60)
    post["delta_hist"] = {"labels": [round(float((ed2[i] + ed2[i + 1]) / 2), 4) for i in range(len(cnt2))],
                          "values": [int(v) for v in cnt2]}
    bayes[m] = post
OUT["bayes"] = bayes

cuped = {"note": "原始数据集不含实验前字段，以下为以真实方差为基准的仿真演示（非真实 CUPED 结果）",
         "rows": []}
for rho in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    Nsim = 200000
    trt = RNG.integers(0, 2, Nsim)
    base_p = np.where(trt == 1, p40, p30)
    y = RNG.binomial(1, base_p).astype(float)
    e = RNG.standard_normal(Nsim)
    x = rho * (y - y.mean()) + np.sqrt(1 - rho ** 2) * e
    x = (x - x.mean()) / x.std()
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x.mean())
    var_y = np.var(y, ddof=1)
    var_adj = np.var(y_adj, ddof=1)
    cuped["rows"].append({"rho": rho, "theta": float(theta),
                          "var_reduction_pct": float((1 - var_adj / var_y) * 100),
                          "eff_sample_multiplier": float(var_y / var_adj),
                          "se_shrink_pct": float((1 - np.sqrt(var_adj / var_y)) * 100)})
sub = df[["version", "sum_gamerounds", "retention_7", "retention_1"]].copy()
sub["trt"] = (sub["version"] == "gate_40").astype(int)
sub["x"] = np.log1p(sub["sum_gamerounds"])
for m in ["retention_7", "retention_1"]:
    X = np.column_stack([np.ones(len(sub)), sub["trt"], sub["x"]])
    yv = sub[m].to_numpy(float)
    # OLS
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    dof = len(sub) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    OUT.setdefault("cuped_adj", {})[m] = {
        "beta_trt": float(beta[1]), "se_trt": float(se[1]),
        "t": float(beta[1] / se[1]),
        "p": float(2 * (1 - stats.t.cdf(abs(beta[1] / se[1]), dof))),
        "ci": [float(beta[1] - 1.96 * se[1]), float(beta[1] + 1.96 * se[1])],
        "beta_x": float(beta[2])}
OUT["cuped"] = cuped

def htest(sd, m="retention_7"):
    a = sd[sd["version"] == "gate_30"]; b = sd[sd["version"] == "gate_40"]
    if len(a) == 0 or len(b) == 0 or a[m].sum() == 0 or b[m].sum() == 0:
        return None
    r = two_prop_z(int(a[m].sum()), len(a), int(b[m].sum()), len(b))
    r.update({"n30": len(a), "n40": len(b),
              "rate30": float(a[m].mean()), "rate40": float(b[m].mean())})
    return r

segments = []
try:
    df["gr_q"] = pd.qcut(df["sum_gamerounds"].rank(method="first"), 4,
                         labels=["Q1 (0-4 回合)", "Q2", "Q3", "Q4 (高活跃)"])
except Exception:
    df["gr_q"] = pd.qcut(df["sum_gamerounds"], 4, duplicates="drop")
for q in df["gr_q"].cat.categories:
    sd = df[df["gr_q"] == q]
    for m in ["retention_7", "retention_1"]:
        r = htest(sd, m)
        if r:
            r.update({"dim": "游戏活跃度四分位", "seg": str(q), "metric": m}); segments.append(r)
for lab, sd in [("首日 0 回合", df[df["sum_gamerounds"] == 0]), ("首日有游玩", df[df["sum_gamerounds"] > 0])]:
    for m in ["retention_7", "retention_1"]:
        r = htest(sd, m)
        if r:
            r.update({"dim": "是否产生游玩行为", "seg": lab, "metric": m}); segments.append(r)
med = df["sum_gamerounds"].median()
for lab, sd in [(f"低活跃 (≤{med:.0f})", df[df["sum_gamerounds"] <= med]), (f"高活跃 (>{med:.0f})", df[df["sum_gamerounds"] > med])]:
    for m in ["retention_7"]:
        r = htest(sd, m)
        if r:
            r.update({"dim": "活跃度中位数切分", "seg": lab, "metric": m}); segments.append(r)

def sim_bucket(uid, salt):
    h = hashlib.md5(f"{salt}-{uid}".encode()).hexdigest()
    return int(h[:8], 16)

df["sim_device"] = df["userid"].map(lambda u: "iOS" if sim_bucket(u, "dev") % 100 < 55 else "Android")
df["sim_channel"] = df["userid"].map(
    lambda u: "自然量 (Organic)" if sim_bucket(u, "ch") % 100 < 62 else ("买量 (Paid)" if sim_bucket(u, "ch") % 100 < 92 else "交叉推广"))
for dim, col in [("设备（模拟）", "sim_device"), ("获客渠道（模拟）", "sim_channel")]:
    for v in sorted(df[col].unique()):
        sd = df[df[col] == v]
        for m in ["retention_7", "retention_1"]:
            r = htest(sd, m)
            if r:
                r.update({"dim": dim, "seg": str(v), "metric": m}); segments.append(r)

for m in ["retention_7", "retention_1"]:
    grp = [s for s in segments if s["metric"] == m]
    ps = np.array([s["p"] for s in grp])
    order = np.argsort(ps)
    mtest = len(ps)
    # Bonferroni
    bonf = np.minimum(ps * mtest, 1.0)
    # BH-FDR
    bh = np.empty(mtest)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        r = mtest - rank
        val = min(prev, ps[idx] * mtest / r)
        bh[idx] = val
        prev = val
    for i, s in enumerate(grp):
        s["p_bonf"] = float(bonf[i]); s["p_fdr"] = float(bh[i])
        s["sig_naive"] = bool(s["p"] < 0.05)
        s["sig_bonf"] = bool(bonf[i] < 0.05)
        s["sig_fdr"] = bool(bh[i] < 0.05)
OUT["segments"] = segments
OUT["multi_test"] = {
    "retention_7": {"k": int(sum(1 for s in segments if s["metric"] == "retention_7")),
                    "sig_naive": int(sum(1 for s in segments if s["metric"] == "retention_7" and s["sig_naive"])),
                    "sig_fdr": int(sum(1 for s in segments if s["metric"] == "retention_7" and s["sig_fdr"])),
                    "sig_bonf": int(sum(1 for s in segments if s["metric"] == "retention_7" and s["sig_bonf"]))},
    "retention_1": {"k": int(sum(1 for s in segments if s["metric"] == "retention_1")),
                    "sig_naive": int(sum(1 for s in segments if s["metric"] == "retention_1" and s["sig_naive"])),
                    "sig_fdr": int(sum(1 for s in segments if s["metric"] == "retention_1" and s["sig_fdr"])),
                    "sig_bonf": int(sum(1 for s in segments if s["metric"] == "retention_1" and s["sig_bonf"]))}}

guard = {}
for name, base_c, base_t in [("崩溃率 (crash rate)", 0.0312, 0.0325), ("卸载率 (uninstall rate)", 0.1235, 0.1251)]:
    k_c = int(RNG.binomial(n30, base_c)); k_t = int(RNG.binomial(n40, base_t))
    r = two_prop_z(k_c, n30, k_t, n40)
    r.update({"name": name, "k30": k_c, "k40": k_t, "rate30_sim": k_c / n30, "rate40_sim": k_t / n40})
    guard[name] = r
OUT["guardrail_sim"] = guard

def power_2prop(p1, p2, n1, n2, alpha=0.05):
    pbar = (p1 * n1 + p2 * n2) / (n1 + n2)
    se0 = np.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
    se1 = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_a = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf((abs(p2 - p1) - z_a * se0) / se1))

def mde_2prop(p, n_per, alpha=0.05, power=0.8):
    z_a = stats.norm.ppf(1 - alpha / 2); z_b = stats.norm.ppf(power)
    d = (z_a + z_b) * np.sqrt(2 * p * (1 - p) / n_per)
    return float(d)

p_ref = OUT["overall_retention"]["retention_7"]
n_per_actual = min(n30, n40)
OUT["power"] = {
    "baseline_ret7": float(p_ref),
    "alpha": 0.05, "power_target": 0.8,
    "n_per_arm": int(n_per_actual),
    "mde_abs_pp": float(mde_2prop(p_ref, n_per_actual) * 100),
    "mde_rel_pct": float(mde_2prop(p_ref, n_per_actual) / p_ref * 100),
    "mde_1pct_abs_pp": float(mde_2prop(0.01, n_per_actual) * 100),
    "observed_power_ret7": power_2prop(p30, p40, n30, n40),
    "observed_power_ret1": power_2prop(p30_1, p40_1, n30, n40),
    "n_per_arm_for_2pp": int(np.ceil(2 * p_ref * (1 - p_ref) * ((1.959963985 + 0.841621234) / 0.02) ** 2)),
}

roi = {"delta_ret7_pp": float((p40 - p30) * 100),
       "delta_ret7_pct": float((p40 - p30) / p30 * 100),
       "ci_pp": [float((p40 - p30 - 1.959963985 * freq["retention_7"]["se"]) * 100),
                 float((p40 - p30 + 1.959963985 * freq["retention_7"]["se"]) * 100)],
       "ltv_options": [5.0, 10.0, 20.0, 40.0],
       "installs_options": [300000, 1000000, 3000000],
       "dev_cost": 45000,
       "annual_opportunity_cost": []}
for inst in roi["installs_options"]:
    row = {"installs_per_period": inst,
           "lost_retained_users_per_cohort": float(-(p40 - p30) * inst),
           "lost_users_ci_low": float(-(p40 - p30 - 1.959963985 * freq["retention_7"]["se"]) * inst),
           "lost_users_ci_high": float(-(p40 - p30 + 1.959963985 * freq["retention_7"]["se"]) * inst)}
    for ltv in roi["ltv_options"]:
        row[f"annual_lost_rev_ltv{ltv}"] = float(-(p40 - p30) * inst * ltv * 12)
    roi["annual_opportunity_cost"].append(row)
OUT["roi"] = roi

bins = np.array([0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 350, 600, 1000, 2000, 5000, 100000])
h30, _ = np.histogram(df.loc[df["version"] == "gate_30", "sum_gamerounds"], bins=bins)
h40, _ = np.histogram(df.loc[df["version"] == "gate_40", "sum_gamerounds"], bins=bins)
OUT["hist_rounds"] = {
    "labels": ["0", "1", "2", "3-4", "5-7", "8-11", "12-19", "20-29", "30-49", "50-79", "80-119",
               "120-199", "200-349", "350-599", "600-999", "1000-1999", "2000-4999", "5000+"],
    "gate_30": [int(x) for x in h30], "gate_40": [int(x) for x in h40]}

def box_stats(s):
    s = np.asarray(s)
    return {"min": float(s.min()),
            "p1": float(np.percentile(s, 1)), "p5": float(np.percentile(s, 5)),
            "q1": float(np.percentile(s, 25)), "med": float(np.median(s)),
            "q3": float(np.percentile(s, 75)),
            "p95": float(np.percentile(s, 95)), "p99": float(np.percentile(s, 99)),
            "mean": float(s.mean()), "max": float(s.max())}
OUT["box_rounds"] = {
    "gate_30": box_stats(df.loc[df["version"] == "gate_30", "sum_gamerounds"]),
    "gate_40": box_stats(df.loc[df["version"] == "gate_40", "sum_gamerounds"])}

OUT["funnel"] = {"gate_30": {"install": 100.0, "d1": p30_1 * 100, "d7": p30 * 100},
                 "gate_40": {"install": 100.0, "d1": p40_1 * 100, "d7": p40 * 100}}

with open("stats.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1, default=str)

print("DONE")
print("rows", OUT["raw_shape"], "n30", n30, "n40", n40, "SRM p=", round(p_srm, 4))
print("ret1", round(p30_1 * 100, 2), round(p40_1 * 100, 2), "p=", round(freq["retention_1"]["p"], 4))
print("ret7", round(p30 * 100, 2), round(p40 * 100, 2), "p=", round(freq["retention_7"]["p"], 5))
print("MWU p=", u_p, "rbc", round(rbc, 4))
print("boot7 CI", round(lo7 * 100, 3), round(hi7 * 100, 3))
print("bayes P(t>c) ret7", round(bayes["retention_7"]["p_treat_better"], 4))
print("mde pp", OUT["power"]["mde_abs_pp"])
print("segments", len(segments), OUT["multi_test"])
print("--- ret7 segments ---")
for s in segments:
    if s["metric"] == "retention_7":
        print(f'{s["dim"][:12]:14s}|{s["seg"][:16]:18s}| {s["rate30"]*100:6.2f} {s["rate40"]*100:6.2f} | d={s["diff"]*100:+6.3f}pp | p={s["p"]:.4f} fdr={s["p_fdr"]:.4f} bonf={s["p_bonf"]:.4f}')
print("welch raw", OUT["welch_raw"])
print("welch clean", OUT["welch_clean"])
print("boot mean", OUT["boot_mean"])
print("cuped_adj", OUT["cuped_adj"])
print("guard", {k: (round(v["rate30_sim"],4), round(v["rate40_sim"],4), round(v["p"],4)) for k,v in guard.items()})
print("power", OUT["power"])
print("outliers", OUT["outliers"])
