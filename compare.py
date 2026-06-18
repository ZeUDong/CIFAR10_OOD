"""
Final comparison. For each backbone, reduce all trained models to the three
rows that matter:

    base            -- no FD          (train_cifar_base.py,                method=base)
    fd              -- FD, no pruning  (train_cifar_fd_sparse --mask none, method=dense)
    fd + compress   -- FD + the BEST compression method (highest OOD acc among
                       imp/lth/lrr/ep/bp at any sparsity)

so you can read off, per backbone: does FD help (fd vs base), does compression
help (fd+compress vs fd), and which backbone wins overall.

Example
-------
python compare.py --glob "./cards/*/summary.csv" --out comparison.csv
python compare.py --glob "./cards/*/summary.csv" --all   # also dump every row
"""

import csv
import glob
import argparse

COMPRESS_METHODS = {"imp", "lth", "lrr", "ep", "bp"}


def load_rows(pattern):
    rows = []
    for fp in sorted(glob.glob(pattern)):
        with open(fp) as f:
            for row in csv.DictReader(f):
                row["_path"] = fp
                rows.append(row)
    return rows


def _f(row, key):
    try:
        return float(row.get(key, 0) or 0)
    except ValueError:
        return 0.0


def condensed(rows):
    backbones = sorted({r.get("backbone", "?") for r in rows})
    out = []
    for bb in backbones:
        rb = [r for r in rows if r.get("backbone") == bb]
        base = next((r for r in rb if r.get("method") == "base"), None)
        fd = next((r for r in rb if r.get("method") == "dense"), None)
        comp = [r for r in rb if r.get("method") in COMPRESS_METHODS]
        best = max(comp, key=lambda r: _f(r, "ood_mean_acc")) if comp else None

        if base:
            out.append(_mk(bb, "base (no FD)", base))
        if fd:
            out.append(_mk(bb, "FD (dense)", fd))
        if best:
            tag = f"FD + {best.get('method')} (best, s={best.get('target_sparsity','')})"
            out.append(_mk(bb, tag, best))
    return out


def _mk(bb, label, r):
    return {
        "backbone": bb,
        "config": label,
        "method": r.get("method", ""),
        "target_sparsity": r.get("target_sparsity", ""),
        "encoder_sparsity": r.get("encoder_sparsity", ""),
        "clean_acc": r.get("clean_acc", ""),
        "ood_mean_acc": r.get("ood_mean_acc", ""),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="./cards/*/summary.csv")
    p.add_argument("--out", default="comparison.csv")
    p.add_argument("--all", action="store_true", help="also write comparison_full.csv with every model")
    args = p.parse_args()

    rows = load_rows(args.glob)
    if not rows:
        print("No summary.csv files matched", args.glob)
        return

    table = condensed(rows)
    fields = ["backbone", "config", "method", "target_sparsity",
              "encoder_sparsity", "clean_acc", "ood_mean_acc"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(table)

    print(f"\n=== Final comparison (base / FD / FD+best-compression) -> {args.out} ===\n")
    print(f"{'backbone':>8} {'config':>34} {'clean':>7} {'ood':>7} {'sparsity':>9}")
    last_bb = None
    for r in table:
        bb = r["backbone"] if r["backbone"] != last_bb else ""
        last_bb = r["backbone"]
        print(f"{bb:>8} {r['config']:>34} {str(r['clean_acc']):>7} "
              f"{str(r['ood_mean_acc']):>7} {str(r['encoder_sparsity']):>9}")

    if args.all:
        full = sorted(rows, key=lambda r: _f(r, "ood_mean_acc"), reverse=True)
        ff = [k for k in full[0].keys() if k != "_path"]
        with open("comparison_full.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ff, extrasaction="ignore")
            w.writeheader(); w.writerows(full)
        print("\n(full per-model table -> comparison_full.csv)")


if __name__ == "__main__":
    main()
