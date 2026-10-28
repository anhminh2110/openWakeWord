#!/usr/bin/env python3
"""
evaluate_oww.py - Đo FRR, FA/giờ và vẽ đường DET cho một model openWakeWord.

Cách đo:
  * Positive: các clip ngắn, MỖI clip chứa ĐÚNG MỘT lần nói wake word.
      -> clip được coi là "phát hiện đúng" nếu điểm số cao nhất >= threshold.
      -> FRR(t) = (số clip bị bỏ lỡ) / (tổng số clip positive)
  * Negative: các file audio dài KHÔNG chứa wake word (podcast, TV, hội thoại, tiếng ồn...).
      -> chạy streaming từng khung 80 ms, đếm số lần kích hoạt nhầm.
      -> FA/giờ(t) = (số lần kích hoạt nhầm) / (tổng số giờ negative)
  * Quét nhiều threshold để vẽ đường DET (FRR theo FA/giờ).

Ví dụ:
  python evaluate_oww.py --model hey_jarvis --pos_dir data/pos --neg_dir data/neg --out_dir results/hey_jarvis_base
"""
import argparse
import csv
import json
import sys
import time
from math import gcd, sqrt
from pathlib import Path

import numpy as np

SR = 16000
FRAME = 1280  # 80 ms @ 16 kHz, đúng khung mà openWakeWord xử lý
AUDIO_EXT = {".wav", ".flac", ".ogg"}


# --------------------------------------------------------------------------- #
# Đọc audio
# --------------------------------------------------------------------------- #
def load_audio_16k(path):
    """Đọc file audio, đổi sang mono 16 kHz int16 (tự resample nếu cần)."""
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767).astype(np.int16)


def list_audio(folder):
    folder = Path(folder)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    return files


# --------------------------------------------------------------------------- #
# Chạy model
# --------------------------------------------------------------------------- #
def build_model(args):
    from openwakeword.model import Model

    kwargs = dict(wakeword_models=[args.model], inference_framework=args.framework)
    if args.vad_threshold > 0:
        kwargs["vad_threshold"] = args.vad_threshold
    if args.speex:
        kwargs["enable_speex_noise_suppression"] = True
    model = Model(**kwargs)
    keys = list(model.models.keys())
    if args.key:
        if args.key not in keys:
            sys.exit(f"--key '{args.key}' không có trong model. Các key hiện có: {keys}")
        key = args.key
    else:
        key = keys[0]
        if len(keys) > 1:
            print(f"[!] Model có nhiều key {keys}, dùng '{key}'. Đổi bằng --key.")
    return model, key


def score_audio(model, key, audio, pad_start_s, pad_end_s):
    """Chạy streaming từng khung 80 ms, trả về mảng điểm số mỗi khung."""
    model.reset()
    x = np.concatenate([
        np.zeros(int(pad_start_s * SR), dtype=np.int16),
        audio,
        np.zeros(int(pad_end_s * SR), dtype=np.int16),
    ])
    rem = (-len(x)) % FRAME
    if rem:
        x = np.concatenate([x, np.zeros(rem, dtype=np.int16)])
    n = len(x) // FRAME
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        pred = model.predict(x[i * FRAME:(i + 1) * FRAME])
        scores[i] = pred[key]
    return scores


# --------------------------------------------------------------------------- #
# Tính metrics
# --------------------------------------------------------------------------- #
def apply_patience(s, p):
    """Khung chỉ tính là 'vượt ngưỡng' khi p khung liên tiếp cùng vượt (min trượt)."""
    if p <= 1:
        return s
    padded = np.concatenate([np.zeros(p - 1, dtype=s.dtype), s])
    return np.lib.stride_tricks.sliding_window_view(padded, p).min(axis=1)


def count_events(s, t, refractory_frames):
    """Đếm số lần kích hoạt: sau mỗi lần kích hoạt bỏ qua `refractory_frames` khung."""
    idx = np.flatnonzero(s >= t)
    events, last = [], -10 ** 9
    for i in idx:
        if i - last >= refractory_frames:
            events.append(int(i))
            last = int(i)
    return events


def wilson(k, n, z=1.96):
    """Khoảng tin cậy 95% (Wilson) cho tỉ lệ k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def fmt_time(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="Tên model có sẵn (vd: hey_jarvis) hoặc đường dẫn tới file .onnx/.tflite")
    ap.add_argument("--pos_dir", required=True, help="Thư mục clip positive (mỗi clip 1 lần nói wake word)")
    ap.add_argument("--neg_dir", required=True, help="Thư mục audio negative dài (KHÔNG chứa wake word)")
    ap.add_argument("--out_dir", required=True, help="Thư mục lưu kết quả")
    ap.add_argument("--framework", default="onnx", choices=["onnx", "tflite"], help="Mặc định onnx (chạy được trên Windows)")
    ap.add_argument("--key", default=None, help="Tên key điểm số nếu model có nhiều đầu ra")
    ap.add_argument("--vad_threshold", type=float, default=0.0, help="Bật Silero VAD (0 = tắt, thử 0.5)")
    ap.add_argument("--speex", action="store_true", help="Bật Speex noise suppression (chỉ Linux)")
    ap.add_argument("--patience", type=int, default=1, help="Số khung 80ms liên tiếp phải vượt ngưỡng (mặc định 1)")
    ap.add_argument("--refractory", type=float, default=2.0,
                    help="Sau mỗi lần kích hoạt, bỏ qua bấy nhiêu giây (mặc định 2.0) để không đếm trùng")
    ap.add_argument("--report_threshold", type=float, default=0.5,
                    help="Ngưỡng dùng cho bảng chi tiết/danh sách lỗi (mặc định 0.5 = ngưỡng mặc định của openWakeWord)")
    ap.add_argument("--fa_targets", type=float, nargs="+", default=[1.0, 0.5, 0.2, 0.1],
                    help="Các mức FA/giờ mục tiêu để báo cáo FRR (mặc định 1 0.5 0.2 0.1)")
    ap.add_argument("--reuse", action="store_true",
                    help="Dùng lại raw_scores.npz đã có trong out_dir (không chạy lại model). "
                         "Dùng khi chỉ muốn đổi --patience/--refractory/--report_threshold")
    ap.add_argument("--max_neg_files", type=int, default=0, help="(Thử nhanh) chỉ dùng N file negative đầu tiên")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_npz, raw_meta = out / "raw_scores.npz", out / "raw_meta.json"

    t0 = time.time()
    # ------------------------------------------------------------------ #
    # 1) Chạy model (hoặc đọc lại điểm số đã lưu)
    # ------------------------------------------------------------------ #
    if args.reuse and raw_npz.exists() and raw_meta.exists():
        print(f"[reuse] Đọc lại điểm số từ {raw_npz}")
        meta = json.loads(raw_meta.read_text(encoding="utf-8"))
        data = np.load(raw_npz)
        pos_names, pos_groups = meta["pos_names"], meta["pos_groups"]
        neg_names, neg_durs = meta["neg_names"], meta["neg_durations_s"]
        pos_scores = [data[f"pos_{i}"] for i in range(len(pos_names))]
        neg_scores = [data[f"neg_{i}"] for i in range(len(neg_names))]
        pad_pos = meta["pad_pos_s"]
        model_name = meta["model"]
    else:
        pos_files, neg_files = list_audio(args.pos_dir), list_audio(args.neg_dir)
        if args.max_neg_files:
            neg_files = neg_files[:args.max_neg_files]
        if not pos_files:
            sys.exit(f"Không tìm thấy file audio (.wav/.flac/.ogg) trong {args.pos_dir}")
        if not neg_files:
            sys.exit(f"Không tìm thấy file audio (.wav/.flac/.ogg) trong {args.neg_dir}")
        print(f"Positive: {len(pos_files)} clip | Negative: {len(neg_files)} file")

        model, key = build_model(args)
        model_name = args.model
        print(f"Model: {args.model} (key='{key}', framework={args.framework}, "
              f"vad={args.vad_threshold}, speex={args.speex})")

        pad_pos = 1.0  # đệm 1s im lặng đầu/cuối clip positive cho model "ấm" lên
        pos_names, pos_groups, pos_scores = [], [], []
        pos_root = Path(args.pos_dir)
        for i, f in enumerate(pos_files, 1):
            audio = load_audio_16k(f)
            pos_scores.append(score_audio(model, key, audio, pad_pos, pad_pos))
            rel = f.relative_to(pos_root)
            pos_names.append(str(rel))
            pos_groups.append(str(rel.parent) if str(rel.parent) != "." else "(root)")
            if i % 20 == 0 or i == len(pos_files):
                print(f"  positive {i}/{len(pos_files)}", flush=True)

        neg_names, neg_durs, neg_scores = [], [], []
        neg_root = Path(args.neg_dir)
        total_audio = 0.0
        for i, f in enumerate(neg_files, 1):
            audio = load_audio_16k(f)
            dur = len(audio) / SR
            ts = time.time()
            neg_scores.append(score_audio(model, key, audio, 0.0, 0.0))
            neg_names.append(str(f.relative_to(neg_root)))
            neg_durs.append(dur)
            total_audio += dur
            print(f"  negative {i}/{len(neg_files)}: {f.name} ({dur / 60:.1f} phút audio, "
                  f"xử lý {time.time() - ts:.0f}s) | tổng {total_audio / 3600:.2f} giờ", flush=True)

        np.savez_compressed(
            raw_npz,
            **{f"pos_{i}": s for i, s in enumerate(pos_scores)},
            **{f"neg_{i}": s for i, s in enumerate(neg_scores)},
        )
        raw_meta.write_text(json.dumps({
            "model": model_name, "pos_names": pos_names, "pos_groups": pos_groups,
            "neg_names": neg_names, "neg_durations_s": neg_durs, "pad_pos_s": pad_pos,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # 2) Tính metrics trên mọi threshold
    # ------------------------------------------------------------------ #
    pos_scores = [apply_patience(s, args.patience) for s in pos_scores]
    neg_scores = [apply_patience(s, args.patience) for s in neg_scores]
    refr = max(1, int(round(args.refractory / (FRAME / SR))))
    neg_hours = float(sum(neg_durs)) / 3600.0
    n_pos = len(pos_scores)
    pos_max = np.array([s.max() for s in pos_scores])
    pos_arg = np.array([int(s.argmax()) for s in pos_scores])

    thresholds = np.unique(np.round(np.concatenate([np.linspace(0.01, 0.99, 99), [0.995, 0.999]]), 4))
    rows = []
    for t in thresholds:
        missed = int((pos_max < t).sum())
        fa = sum(len(count_events(s, t, refr)) for s in neg_scores)
        rows.append({
            "threshold": float(t),
            "frr": missed / n_pos,
            "n_missed": missed,
            "n_pos": n_pos,
            "fa_events": fa,
            "fa_per_hour": fa / neg_hours,
            "neg_hours": round(neg_hours, 4),
        })

    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Điểm vận hành: FRR thấp nhất mà FA/giờ <= mục tiêu
    op_points = {}
    for target in args.fa_targets:
        ok = [r for r in rows if r["fa_per_hour"] <= target]
        if ok:
            best = min(ok, key=lambda r: (r["frr"], -r["threshold"]))
            op_points[str(target)] = {"fa_target_per_hour": target, **best}
        else:
            op_points[str(target)] = None

    # Bảng tại ngưỡng báo cáo
    rt = args.report_threshold
    r_rep = min(rows, key=lambda r: abs(r["threshold"] - rt))
    lo, hi = wilson(r_rep["n_missed"], n_pos)

    # FRR theo nhóm (thư mục con của pos_dir) tại ngưỡng báo cáo và ngưỡng ở mức FA<=1/giờ
    groups = {}
    for name, g, m in zip(pos_names, pos_groups, pos_max):
        groups.setdefault(g, []).append(m)
    thr_1fa = op_points.get("1.0")["threshold"] if op_points.get("1.0") else None
    group_rows = []
    for g, ms in sorted(groups.items()):
        ms = np.array(ms)
        row = {"group": g, "n": len(ms), f"frr_at_{r_rep['threshold']}": float((ms < r_rep["threshold"]).mean())}
        if thr_1fa is not None:
            row[f"frr_at_{thr_1fa}_(FA<=1/h)"] = float((ms < thr_1fa).mean())
        group_rows.append(row)

    # Danh sách clip positive bị bỏ lỡ
    with open(out / "missed_positives.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "group", "max_score", f"threshold={r_rep['threshold']}"])
        for name, g, m in sorted(zip(pos_names, pos_groups, pos_max), key=lambda z: z[2]):
            if m < r_rep["threshold"]:
                w.writerow([name, g, f"{m:.4f}", "MISSED"])

    # Danh sách lần kích hoạt nhầm (để mở audio nghe lại)
    fa_list = []
    for name, s in zip(neg_names, neg_scores):
        for i in count_events(s, r_rep["threshold"], refr):
            fa_list.append((name, i * FRAME / SR, float(s[i])))
    with open(out / "false_accepts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "time_hh:mm:ss", "time_seconds", "score", f"threshold={r_rep['threshold']}"])
        for name, ts_, sc in fa_list[:1000]:
            w.writerow([name, fmt_time(ts_), f"{ts_:.2f}", f"{sc:.4f}", "FALSE_ACCEPT"])

    # ------------------------------------------------------------------ #
    # 3) summary.json + report.txt
    # ------------------------------------------------------------------ #
    summary = {
        "model": model_name,
        "settings": {
            "vad_threshold": args.vad_threshold, "speex": args.speex, "patience_frames": args.patience,
            "refractory_s": args.refractory, "framework": args.framework,
        },
        "data": {
            "n_positive_clips": n_pos, "n_negative_files": len(neg_names),
            "negative_hours": round(neg_hours, 3), "positive_groups": {g: len(v) for g, v in groups.items()},
        },
        "at_report_threshold": {
            "threshold": r_rep["threshold"], "frr": r_rep["frr"], "frr_ci95": [lo, hi],
            "n_missed": r_rep["n_missed"], "fa_events": r_rep["fa_events"], "fa_per_hour": r_rep["fa_per_hour"],
        },
        "operating_points_min_FRR_given_FA_per_hour_leq_target": op_points,
        "frr_by_group": group_rows,
        "note_resolution": f"Với {neg_hours:.2f} giờ negative, FA/giờ nhỏ nhất đo được khác 0 là {1 / neg_hours:.3f}. "
                           f"Nếu đếm được 0 lần nhầm thì chỉ kết luận được FA/giờ < ~{3 / neg_hours:.2f} (độ tin cậy 95%).",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    L = []
    L.append(f"=== BÁO CÁO openWakeWord: {model_name} ===")
    L.append(f"Cài đặt: vad={args.vad_threshold}, speex={args.speex}, patience={args.patience}, "
             f"refractory={args.refractory}s, framework={args.framework}")
    L.append(f"Dữ liệu: {n_pos} clip positive, {len(neg_names)} file negative = {neg_hours:.2f} giờ")
    L.append("")
    L.append(f"Tại threshold = {r_rep['threshold']}:")
    L.append(f"  FRR = {r_rep['frr'] * 100:.1f}%  ({r_rep['n_missed']}/{n_pos} bị bỏ lỡ; KTC 95% {lo * 100:.1f}%–{hi * 100:.1f}%)")
    L.append(f"  FA  = {r_rep['fa_events']} lần = {r_rep['fa_per_hour']:.2f} lần/giờ")
    L.append("")
    L.append("FRR thấp nhất đạt được với từng mức FA/giờ cho phép:")
    for target in args.fa_targets:
        op = op_points[str(target)]
        if op is None:
            L.append(f"  FA <= {target}/giờ: không threshold nào đạt (kể cả 0.999)")
        else:
            warn = ""
            if neg_hours * target < 3:
                warn = f"  [CẢNH BÁO: chưa đủ negative để kết luận, cần >= {3 / target:.0f} giờ]"
            L.append(f"  FA <= {target}/giờ: threshold = {op['threshold']}, FRR = {op['frr'] * 100:.1f}% "
                     f"(FA thực đo = {op['fa_per_hour']:.2f}/giờ){warn}")
    L.append("")
    L.append("FRR theo nhóm (thư mục con của pos_dir):")
    for gr in group_rows:
        vals = ", ".join(f"{k}={v * 100:.1f}%" if k.startswith("frr") else f"{k}={v}" for k, v in gr.items() if k != "group")
        L.append(f"  {gr['group']}: {vals}")
    L.append("")
    L.append(summary["note_resolution"])
    L.append(f"(Thời gian chạy: {time.time() - t0:.0f}s)")
    report = "\n".join(L)
    (out / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)

    # ------------------------------------------------------------------ #
    # 4) Vẽ hình
    # ------------------------------------------------------------------ #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fa = np.array([r["fa_per_hour"] for r in rows])
    frr = np.array([r["frr"] for r in rows]) * 100
    th = np.array([r["threshold"] for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(fa, frr, "-o", ms=3, lw=1.3, color="tab:blue")
    ax[0].set_xscale("symlog", linthresh=0.1)
    ax[0].set_xlabel("False accepts per hour (FA/giờ, thang symlog)")
    ax[0].set_ylabel("False reject rate - FRR (%)")
    ax[0].set_title(f"DET: {model_name}  ({n_pos} pos, {neg_hours:.2f} giờ neg)")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].plot([r_rep["fa_per_hour"]], [r_rep["frr"] * 100], "r*", ms=14, label=f"threshold={r_rep['threshold']}")
    for target in args.fa_targets:
        ax[0].axvline(target, color="gray", ls=":", lw=0.8)
    ax[0].legend()
    ax[0].set_ylim(-2, 102)
    ax[0].set_xlim(left=-0.02)

    ax[1].plot(th, frr, color="tab:red", label="FRR (%)")
    ax[1].set_xlabel("Threshold")
    ax[1].set_ylabel("FRR (%)", color="tab:red")
    ax[1].grid(True, alpha=0.3)
    ax2 = ax[1].twinx()
    ax2.plot(th, fa, color="tab:green", label="FA/giờ")
    ax2.set_yscale("symlog", linthresh=0.1)
    ax2.set_ylabel("FA/giờ (symlog)", color="tab:green")
    ax[1].set_title("FRR và FA/giờ theo threshold")
    ax[1].axvline(r_rep["threshold"], color="k", ls="--", lw=0.8)
    plt.tight_layout()
    fig.savefig(out / "det_curve.png", dpi=150)
    plt.close(fig)

    print(f"\nĐã lưu kết quả vào: {out.resolve()}")
    for name in ["report.txt", "summary.json", "results.csv", "det_curve.png",
                 "missed_positives.csv", "false_accepts.csv", "raw_scores.npz", "raw_meta.json"]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()