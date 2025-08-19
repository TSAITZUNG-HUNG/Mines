# -*- coding: utf-8 -*-
"""
Mines 5x5 模擬驗證（m ∈ {1,2,3,4}；k=1..(25-m)）
- 每組合以二項分佈抽樣做批次模擬（每批 N=trials_per_batch）
- 符合早停條件：連續 consec_ok 批 |rtp - target_rtp| <= tol_abs 即停止
- 每組合輸出：每批 rtp / 標準差 / 中獎率 + 最後 TOTAL 彙總列
- 加印每個組合的實際賠率（賠率 multiplier）及理論中獎率（理論中獎率 p_theory）
- 另輸出 Summary_TOTALS 工作表彙整各組合的 TOTAL 指標
"""
import math
import numpy as np
import pandas as pd
from pathlib import Path

# --------- 欄位名稱（集中管理，避免不一致） ---------
COL_BATCH = "模擬批次（每批1000萬次）"
COL_RTP = "rtp"
COL_STD = "標準差"
COL_HIT = "中獎率"
COL_M = "炸彈數"
COL_K = "關卡數"
COL_PTHEORY = "理論中獎率"
COL_MULTI = "賠率"

# ----------------------------
# 組合、機率與賠率
# ----------------------------
def mines_combinations():
    combos = []
    for m in [1, 2, 3, 4]:
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
    target_rtp=0.985,
    tol_abs=0.005,
    consec_ok=10,
    trials_per_batch=10_000_000,
    max_batches=100000,
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
        })

    return pd.DataFrame(rows)

# ----------------------------
# 模擬（所有組合）
# ----------------------------
def simulate_all_combos(
    *,
    target_rtp=0.985,
    tol_abs=0.005,
    consec_ok=10,
    trials_per_batch=10_000_000,
    max_batches=100000,
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
# 匯出 Excel
# ----------------------------
def export_to_excel(results, path, include_summary=True):
    """
    - 每個 (m,k) 一個 sheet：包含每批資料與最後 TOTAL 列
    - 可選擇輸出 Summary_TOTALS：彙整各組合的 TOTAL 指標 + num_batches（早停前實際批數）
    """
    path = Path(path)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_rows = []
        for (m, k), df in results.items():
            sheet_name = f"m{m}_k{k}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            if include_summary:
                # 依照本檔一致的欄位名稱擷取 TOTAL 列
                total_row = df[df[COL_BATCH] == "TOTAL"].copy()
                if not total_row.empty:
                    # 早停前實際批數：非 TOTAL 列的筆數
                    num_batches = int((df[COL_BATCH] != "TOTAL").sum())
                    total_row = total_row.assign(num_batches=num_batches)
                    summary_rows.append(total_row)

        if include_summary and summary_rows:
            summary_df = pd.concat(summary_rows, ignore_index=True)
            # 只保留常用欄位並排序（以中文欄名為準）
            summary_cols = [
                COL_M, COL_K,
                COL_RTP, COL_STD, COL_HIT,
                COL_PTHEORY, COL_MULTI,
                "num_batches",
            ]
            summary_df = summary_df[summary_cols].sort_values([COL_M, COL_K]).reset_index(drop=True)
            summary_df.to_excel(writer, sheet_name="Summary_TOTALS", index=False)

    return str(path)

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Mines 5x5 模擬驗證（含早停、TOTAL、實際賠率與總覽）")
    ap.add_argument("--target_rtp", type=float, default=0.985, help="目標 RTP（例如 0.985 = 98.5%）")
    ap.add_argument("--tol_abs", type=float, default=0.005, help="RTP 絕對容忍（±），例如 0.005 = ±0.5 個百分點")
    ap.add_argument("--consec_ok", type=int, default=10, help="連續達標批數後停止")
    ap.add_argument("--trials_per_batch", type=int, default=10_000_000, help="每批試次數")
    ap.add_argument("--max_batches", type=int, default=100000, help="最多批數上限（保險）")
    ap.add_argument("--seed", type=int, default=777, help="隨機種子")
    ap.add_argument("--out", type=str, default="mines模擬程式結果.xlsx", help="輸出 Excel 檔名")
    ap.add_argument("--no_summary", action="store_true", help="不輸出 Summary_TOTALS 總覽工作表")
    args = ap.parse_args()

    results = simulate_all_combos(
        target_rtp=args.target_rtp,
        tol_abs=args.tol_abs,
        consec_ok=args.consec_ok,
        trials_per_batch=args.trials_per_batch,
        max_batches=args.max_batches,
        seed=args.seed,
    )
    out_path = export_to_excel(results, args.out, include_summary=(not args.no_summary))
    print(f"Saved to {out_path}")
