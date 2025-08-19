# -*- coding: utf-8 -*-
"""
Mines 5x5 模擬驗證（m ∈ {1..24}；k=1..(25-m)）
- 每組合以二項分佈抽樣做批次模擬（每批 N=trials_per_batch）
- 符合早停條件：連續 consec_ok 批 |rtp - target_rtp| <= tol_abs 即停止
- 每組合輸出：每批 rtp / 標準差 / 中獎率 + 最後 TOTAL 彙總列
- 加印每個組合的實際賠率（multiplier）及理論中獎率（p_theory）
- 另輸出 Summary_TOTALS 工作表彙整各組合的 TOTAL 指標
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

# ----------------------------
# 組合、機率與賠率
# ----------------------------
def mines_combinations():
    combos = []
    for m in range(1, 25):  # 🔹 改成 1..24
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
    if rng is None:
        rng = np.random.default_rng()

    p_true = survival_prob(m, k)
    M = payout_multiplier(m, k, target_rtp)

    rows = []
    streak = 0
    total_trials = 0
    total_successes = 0

    for batch_idx in range(1, max_batches + 1):
        s = rng.binomial(trials_per_batch, p_true)
        p_hat = s / trials_per_batch
        rtp = p_hat * M
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

        total_trials += trials_per_batch
        total_successes += s

        if abs(rtp - target_rtp) <= tol_abs:
            streak += 1
        else:
            streak = 0
        if streak >= consec_ok:
            break

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
# 模擬（所有組合，新增進度顯示）
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
    rng = np.random.default_rng(seed)
    results = {}
    combos = mines_combinations()
    total_combos = len(combos)

    for idx, (m, k) in enumerate(combos, start=1):
        print(f"[進度] ({idx}/{total_combos}) 模擬炸彈數 m={m}, 關卡數 k={k} ...", flush=True)

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

    print("[完成] 所有組合模擬結束 ✅")
    return results


# ----------------------------
# 匯出 Excel
# ----------------------------
def export_to_excel(results, path, include_summary=True, only_summary=False):
    path = Path(path)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_rows = []
        for (m, k), df in results.items():
            total_row = df[df[COL_BATCH] == "TOTAL"].copy()
            if not total_row.empty:
                num_batches = int((df[COL_BATCH] != "TOTAL").sum())
                total_row = total_row.assign(num_batches=num_batches)
                summary_rows.append(total_row)

            if not only_summary:  # 🔹 控制是否輸出每批資料
                df.to_excel(writer, sheet_name=f"m{m}_k{k}", index=False)

        if include_summary and summary_rows:
            summary_df = pd.concat(summary_rows, ignore_index=True)
            summary_df.to_excel(writer, sheet_name="Summary_TOTALS", index=False)

    return str(path)


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Mines 5x5 模擬驗證（含早停、TOTAL、實際賠率與總覽）")
    ap.add_argument("--target_rtp", type=float, default=0.95, help="目標 RTP（例如 0.95 = 95%）")
    ap.add_argument("--tol_abs", type=float, default=0.005, help="RTP 絕對容忍（±），例如 0.005 = ±0.5 個百分點")
    ap.add_argument("--consec_ok", type=int, default=10, help="連續達標批數後停止")
    ap.add_argument("--trials_per_batch", type=int, default=1_000_000, help="每批試次數")
    ap.add_argument("--max_batches", type=int, default=100, help="最多批數上限（保險）")
    ap.add_argument("--seed", type=int, default=777, help="隨機種子")
    ap.add_argument("--out", type=str, default="Mine統計結果.xlsx", help="輸出 Excel 檔名")
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
