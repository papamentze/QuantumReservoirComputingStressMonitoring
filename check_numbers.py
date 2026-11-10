import pandas as pd
M = "Output/V2/monitor/"
d = pd.read_csv(M + "V3_contrasts.csv")
A, W = "all (subject-averaged)", "wesad (subject-averaged)"
rows = [("auroc", "qrc-esn", W, "p_tost_5pp"), ("auroc", "qrc-esn", A, "p_tost_5pp"),
        ("auroc_ewma", "qrc-esn", A, "p_tost_5pp"), ("auroc", "qrc-memoryless", W, "p_tost_5pp"),
        ("auroc", "qrc-memoryless", A, "p_tost_5pp"), ("auprc", "qrc-memoryless", W, "p"),
        ("auprc", "qrc-memoryless", A, "p"), ("auroc", "qrc_nomem-qrc", A, "p_tost_5pp"),
        ("score_lag1_autocorr", "qrc-memoryless", A, "p"), ("score_lag1_autocorr", "qrc-esn", A, "p"),
        ("score_lag1_autocorr", "qrc_nomem-qrc", A, "p")]
print("== Table 2 ==")
for mt, c, u, pc in rows:
    r = d[(d.metric == mt) & (d.contrast == c) & (d.unit == u)].iloc[0]
    print(f"{c:15s} {u[:5]:5s} {mt:20s} diff={r['diff']:+.4f} "
          f"CI90=[{r['ci90_lo']:+.4f},{r['ci90_hi']:+.4f}] {pc}={r[pc]:.3g} n={int(r['n'])}")
cell = ~d.unit.str.contains("subject")
print("\n== P2: lag-1 per cell ==")
print(d[cell & (d.metric == "score_lag1_autocorr") & (d.contrast == "qrc_nomem-qrc")]
      [["unit", "diff", "p", "p_holm"]].to_string(index=False))
print("\n== Smoothed QRC-ESN per cell ==")
print(d[cell & (d.metric == "auroc_ewma") & (d.contrast == "qrc-esn")]
      [["unit", "diff", "p", "p_holm"]].to_string(index=False))
m = pd.read_csv(M + "V3_alarm_matched.csv")
print("\n== FA/h at matched FAR ==")
print(m[(m.outcome == "false_alarms_per_h") & m.contrast.isin(["qrc-esn", "qrc-memoryless", "qrc_nomem-qrc"])]
      [["scope", "m", "contrast", "budget", "diff", "p_holm"]].to_string(index=False))
n = pd.read_csv(M + "V3_alarm_null.csv")
print("\n== Null-adjusted detection, WESAD, m=3 ==")
print(n[(n.m == 3) & (n.scope == "wesad")][["row", "name", "budget", "excess", "p_holm"]].to_string(index=False)) 