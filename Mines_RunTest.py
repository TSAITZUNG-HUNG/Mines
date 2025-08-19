# -*- coding: utf-8 -*-
"""
Mines 5x5 模擬驗證（m ∈ {1..24}；k=1..(25-m)）
- 每組合以二項分佈抽樣做批次模擬（每批 N=trials_per_batch）
- 早停：連續 consec_ok 批 |rtp - target_rtp| <= tol_abs 即停止
- 每組合輸出：每批 rtp / 標準差 / 中獎率 + 最後 TOTAL 彙總列（含總試次、總成功數）
- 加印每個組合的實際賠率（賠率 multiplier）及理論中獎率（理論中獎率 p_theory）
- 另輸出：
    * Summary_TOTALS：彙整各組合 TOTAL 指標
    * Test_HighOdds：僅賠率 >= 10 的組合做 z-test（p-value ≤ alpha 視為通過）
"""
import math
import numpy as np
import pandas as pd
from pathlib import Path

# --------- 欄位名稱（集中管理，避免不一致） ---------
COL_BATCH = "模擬批次（每批100萬次）"
COL_RTP = "rtp"
COL_STD = "標準差"
COL_HIT = "中獎率"
COL_M = "炸彈數"
COL_K = "關卡數"
COL_PTHEORY = "理論中獎率"
COL_MULTI = "賠率"
COL_TOTAL_TRIALS = "總試次"
COL_TOTAL_SUCC = "總成功數"

# ----------------------------
# 組合、機率與賠率
# ----------------------------
def mines_combinations():
    combos = []
    for m in range(1, 25):  # m = 1..24
        for k in range(1, 25):
            if k <= 25 - m:     # 超過 25-m 步不可能成功，不模擬
                combos.append((m, k))
    return combos

def survival_prob(m, k):
    # P_alive_to_k = C(25-m, k) / C(25, k) （連續猜中 k 格的機率）
    from math import comb
    return comb(25 - m, k) / comb(25, k)

def payout_multiplier(m, k, target_rtp):
    # 設計的實際賠率（multiplier），使得 E[RTP]=target_rtp
    p = survival_prob(m, k)
    return target_rtp / p

# ----------------------------
# 模擬（單一組合）
# ----------------------------
def simulate_batches_for_combo(
    m,
    k,
    *,
    target_rtp=0.95,
    tol_abs=0.005,
    consec_ok=10,
    trials_per_batch=1_000_000,
    max_batches=100,
    rng=None,
):
    """
    針對單一 (m,k) 以二項分佈抽樣做批次模擬。
    每批輸出：COL_BATCH, COL_RTP, COL_STD, COL_HIT, COL_M, COL_K, COL_PTHEORY, COL_MULTI
    最後追加一列 'TOTAL'，彙總所有已跑批次（總試次、總成功數計算得到的總中獎率、總RTP、總std）。
    """
    if rng is None:
        rng = np.random.default_rng()

    p_true = survival_prob(m, k)                   # 理論中獎率（存活到第 k 步）
    M = payout_multiplier(m, k, target_rtp)        # 實際賠率（multiplier）

    rows = []
    streak = 0
    total_trials = 0
    total_successes = 0

    for batch_idx in range(1, max_batches + 1):
        # 成功次數（中獎數）~ Binomial(N, p_true)
        s = rng.binomial(trials_per_batch, p_true)
        p_hat = s / trials_per_batch
        rtp = p_hat * M

        # 兩點分佈 {0, M} 的單一局標準差
        # Var = p_hat * (1 - p_hat) * M^2
        std_per_trial = math.sqrt(max(p_hat * (1 - p_hat) * (M ** 2), 0.0))

        rows.append({
            COL_BATCH: batch_idx,
            COL_RTP: rtp,
            COL_STD: std_per_trial,
            COL_HIT: p_hat,
            COL_M: m,
            COL_K: k,
            COL_PTHEORY: p_true,
            COL_MULTI: M,
            COL_TOTAL_TRIALS: None,
            COL_TOTAL_SUCC: None,
        })

        # 累計做 TOTAL
        total_trials += trials_per_batch
        total_successes += s

        # 早停條件（連續 consec_ok 批都在容忍範圍內）
        if abs(rtp - target_rtp) <= tol_abs:
            streak += 1
        else:
            streak = 0
        if streak >= consec_ok:
            break

    # TOTAL 彙總列
    if total_trials > 0:
        p_hat_total = total_successes / total_trials
        rtp_total = p_hat_total * M
        std_per_trial_total = math.sqrt(max(p_hat_total * (1 - p_hat_total) * (M ** 2), 0.0))
        rows.append({
            COL_BATCH: "TOTAL",
            COL_RTP: rtp_total,
            COL_STD: std_per_trial_total,
            COL_HIT: p_hat_total,
            COL_M: m,
            COL_K: k,
            COL_PTHEORY: p_true,
            COL_MULTI: M,
            COL_TOTAL_TRIALS: total_trials,
            COL_TOTAL_SUCC: total_successes,
        })

    return pd.DataFrame(rows)

# ----------------------------
# 模擬（所有組合）
# ----------------------------
def simulate_all_combos(
    *,
    target_rtp=0.95,
    tol_abs=0.005,
    consec_ok=10,
    trials_per_batch=1_000_000,
    max_batches=100,
    seed=777,
):
    """
    對所有 (m,k) 執行 simulate_batches_for_combo，回傳 dict: {(m,k): DataFrame}
    """
    rng = np.random.default_rng(seed)
    results = {}
    for m, k in mines_combinations():
        df = simulate_batches_for_combo(
            m,
            k,
            target_rtp=target_rtp,
            tol_abs=tol_abs,
            consec_ok=consec_ok,
            trials_per_batch=trials_per_batch,
            max_batches=max_batches,
            rng=rng,
        )
        results[(m, k)] = df
    return results

# ----------------------------
# 匯出（精簡模式 + 分組模式 + 只總覽）
# ----------------------------
def _round_df(df: pd.DataFrame, ndigits: int) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=["float", "float64", "float32", "int", "int64", "int32"]).columns
    return df.copy().round({c: ndigits for c in num_cols})

def _filter_by_multiplier(results: dict, min_multiplier: float):
    if min_multiplier <= 1:
        return results
    filt = {}
    for key, df in results.items():
        total_row = df[df[COL_BATCH] == "TOTAL"]
        if total_row.empty:
            continue
        if float(total_row.iloc[0][COL_MULTI]) >= min_multiplier:
            filt[key] = df
    return filt

# ----------------------------
# 高賠率檢定（z-test）
# ----------------------------
def runtest_high_odds(results: dict, alpha: float = 0.001) -> pd.DataFrame:
    """
    針對賠率 >= 10 的 TOTAL 結果做 z-test，檢定模擬命中率與理論值是否顯著不同。
    p <= alpha 視為通過（fail=False）。
    """
    rows = []
    from math import sqrt
    from scipy.stats import norm

    for (m, k), df in results.items():
        total_row = df[df[COL_BATCH] == "TOTAL"]
        if total_row.empty:
            continue
        row = total_row.iloc[0]
        M = row[COL_MULTI]
        if M < 10:
            continue  # 只做賠率 ≥ 10 的

        p_theory = row[COL_PTHEORY]
        n = row[COL_TOTAL_TRIALS]
        succ = row[COL_TOTAL_SUCC]
        p_hat = succ / n

        # z 檢定（單一比例檢定）
        se = sqrt(p_theory * (1 - p_theory) / n)
        if se == 0:
            pval = 0.0 if abs(p_hat - p_theory) < 1e-12 else 1.0
        else:
            z = (p_hat - p_theory) / se
            # 雙尾檢定
            pval = 2 * (1 - norm.cdf(abs(z)))

        rows.append({
            COL_M: m,
            COL_K: k,
            COL_MULTI: M,
            COL_PTHEORY: p_theory,
            "模擬命中率": p_hat,
            "總試次": n,
            "總成功數": succ,
            "z值": z if se > 0 else None,
            "p-value": pval,
            "根據權重均勻分配": pval > alpha
        })

    return pd.DataFrame(rows)


def export_to_excel(
    results,
    path,
    include_summary=True,
    include_test=True,
    alpha=0.001,
    export_mode: str = "per_combo",  # "per_combo" | "grouped" | "summary"
    ndigits: int = 6,
    min_multiplier: float = 1.0,
):
    """
    export_mode:
      - per_combo: 每個 (m,k) 一個 sheet（原行為，檔案最大）
      - grouped:   每批資料彙整到單一 `All_Batches`，再加 `Summary_TOTALS`、`Test_HighOdds`
      - summary:   只輸出 `Summary_TOTALS`（與可選的 `Test_HighOdds`）
    """
    # 先依賠率過濾
    results = _filter_by_multiplier(results, min_multiplier)

    path = Path(path)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_rows = []
        all_batches_rows = []  # for grouped

        # 蒐集 summary 與（若需要）grouped 的 rows
        for (m, k), df in results.items():
            # rounding
            df = _round_df(df, ndigits)

            # Summary row
            total_row = df[df[COL_BATCH] == "TOTAL"].copy()
            if not total_row.empty:
                num_batches = int((df[COL_BATCH] != "TOTAL").sum())
                total_row = total_row.assign(num_batches=num_batches)
                summary_rows.append(total_row)

            if export_mode == "per_combo":
                sheet_name = f"m{m}_k{k}"
                # 防止工作表過多造成開啟緩慢：印出進度
                # print(f"[匯出] {sheet_name}", flush=True)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            elif export_mode == "grouped":
                # 加上 (m,k) 欄位，合併到 all_batches_rows
                gdf = df.copy()
                # 已有 COL_M, COL_K，所以直接使用
                all_batches_rows.append(gdf)

        # Summary_TOTALS
        if include_summary and summary_rows:
            summary_df = pd.concat(summary_rows, ignore_index=True)
            summary_cols = [
                COL_M, COL_K,
                COL_RTP, COL_STD, COL_HIT,
                COL_PTHEORY, COL_MULTI,
                COL_TOTAL_TRIALS, COL_TOTAL_SUCC,
                "num_batches",
            ]
            summary_df = summary_df[summary_cols].sort_values([COL_M, COL_K]).reset_index(drop=True)
            summary_df = _round_df(summary_df, ndigits)
            summary_df.to_excel(writer, sheet_name="Summary_TOTALS", index=False)

        # Grouped 模式下，輸出 All_Batches 單一表
        if export_mode == "grouped" and all_batches_rows:
            all_batches = pd.concat(all_batches_rows, ignore_index=True)
            all_batches = all_batches.sort_values([COL_M, COL_K, COL_BATCH], key=lambda s: s.map(lambda v: (10**9 if v=="TOTAL" else v)))
            all_batches.to_excel(writer, sheet_name="All_Batches", index=False)

        # 高賠率檢定
        if include_test:
            test_df = runtest_high_odds(results, alpha=alpha)
            if ndigits is not None:
                test_df = _round_df(test_df, ndigits)
            test_df.to_excel(writer, sheet_name="Test_HighOdds", index=False)

    return str(path)

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Mines 5x5 模擬驗證（含早停、TOTAL、實際賠率、總覽與高倍率檢定）")
    ap.add_argument("--target_rtp", type=float, default=0.95, help="目標 RTP（例如 0.985 = 98.5%）")
    ap.add_argument("--tol_abs", type=float, default=0.005, help="RTP 絕對容忍（±），例如 0.005 = ±0.5 個百分點")
    ap.add_argument("--consec_ok", type=int, default=10, help="連續達標批數後停止")
    ap.add_argument("--trials_per_batch", type=int, default=1_000_000, help="每批試次數")
    ap.add_argument("--max_batches", type=int, default=100, help="最多批數上限（保險）")
    ap.add_argument("--seed", type=int, default=777, help="隨機種子")
    ap.add_argument("--out", type=str, default="Mines_RunTest.xlsx", help="輸出 Excel 檔名")
    ap.add_argument("--no_summary", action="store_true", help="不輸出 Summary_TOTALS 總覽工作表")
    ap.add_argument("--no_test", action="store_true", help="不輸出 Test_HighOdds 檢定工作表")
    ap.add_argument("--alpha", type=float, default=0.001, help="z-test 雙尾顯著水準（預設 0.001）")

    # 🔹 新增參數：輸出模式 / 四捨五入 / 最小賠率
    ap.add_argument("--export_mode", type=str, default="summary", choices=["summary", "grouped", "per_combo"],
                    help="輸出模式：summary=只總覽、grouped=單一彙整表+總覽、per_combo=每組合一個工作表（最大）")
    ap.add_argument("--round", dest="ndigits", type=int, default=6, help="數值四捨五入位數（預設 6）")
    ap.add_argument("--min_multiplier", type=float, default=1.0, help="只輸出賠率 >= 此值的組合（預設 1.0）")
    args = ap.parse_args()

    results = simulate_all_combos(
        target_rtp=args.target_rtp,
        tol_abs=args.tol_abs,
        consec_ok=args.consec_ok,
        trials_per_batch=args.trials_per_batch,
        max_batches=args.max_batches,
        seed=args.seed,
    )
    out_path = export_to_excel(
        results,
        args.out,
        include_summary=(not args.no_summary),
        include_test=(not args.no_test),
        alpha=args.alpha,
        export_mode=args.export_mode,
        ndigits=args.ndigits,
        min_multiplier=args.min_multiplier
    )
    print(f"Saved to {out_path}")
