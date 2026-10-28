#!/usr/bin/env python3
"""
compare_det.py - Vẽ chồng nhiều đường DET (từ các thư mục kết quả của evaluate_oww.py)
để so sánh: model A vs model B, bật/tắt VAD, bật/tắt Speex, ...

Ví dụ:
  python compare_det.py --runs results/base results/vad05 results/hey_robot \
      --labels "hey_jarvis" "hey_jarvis + VAD 0.5" "hey_robot" --out results/compare.png

Lưu ý: chỉ so sánh công bằng khi các lần chạy dùng CÙNG bộ positive và CÙNG bộ negative.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_results(run_dir):
    with open(Path(run_dir) / "results.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [(float(r["threshold"]), float(r["fa_per_hour"]), float(r["frr"]) * 100) for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="Các thư mục kết quả (mỗi thư mục có results.csv)")
    ap.add_argument("--labels", nargs="+", default=None, help="Nhãn cho từng lần chạy (mặc định = tên thư mục)")
    ap.add_argument("--out", default="compare_det.png", help="File ảnh đầu ra")
    args = ap.parse_args()

    labels = args.labels or [Path(r).name for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("Số --labels phải bằng số --runs")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    table = []
    for run, lab in zip(args.runs, labels):
        pts = read_results(run)
        ax.plot([p[1] for p in pts], [p[2] for p in pts], "-o", ms=2.5, lw=1.3, label=lab)
        s = json.loads((Path(run) / "summary.json").read_text(encoding="utf-8"))
        ops = s["operating_points_min_FRR_given_FA_per_hour_leq_target"]
        table.append((lab, s["data"]["n_positive_clips"], s["data"]["negative_hours"], ops))

    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("False accepts per hour (FA/giờ, thang symlog)")
    ax.set_ylabel("False reject rate - FRR (%)")
    ax.set_title("So sánh đường DET (càng gần góc dưới-trái càng tốt)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(-2, 102)
    ax.set_xlim(left=-0.02)
    ax.legend()
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Đã lưu: {args.out}\n")

    # Bảng FRR tại các mức FA/giờ
    targets = sorted({k for _, _, _, ops in table for k in ops}, key=float)
    header = f"{'Run':<28}{'#pos':>6}{'neg(h)':>8}" + "".join(f"{'FRR@FA<=' + t:>16}" for t in targets)
    print(header)
    print("-" * len(header))
    for lab, npos, nh, ops in table:
        cells = ""
        for t in targets:
            op = ops.get(t)
            txt = "%.1f%%" % (op["frr"] * 100) if op else "n/a"
            cells += f"{txt:>16}"
        print(f"{lab:<28}{npos:>6}{nh:>8.2f}{cells}")


if __name__ == "__main__":
    main()