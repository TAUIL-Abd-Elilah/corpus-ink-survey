"""Merge the per-shard corpus survey scores into one table, repairing undefined ratios.

The survey runs as N independent shards, each writing `_fl/corpus_scores_{i}of{N}.json`. This
merges them, applies one repair, and reports the ranking that matters.

**The repair.** An early version of the scorer divided by `max(g075, 1e-9)` when computing the
confidence ratio and the control comparison. For a mesh with no pixel above 0.75 that turned
"nothing here" into `ratio 156688.97` and `36,940,000x below control` — noise dressed as a
measurement. Both quantities are simply *undefined* when `unanimous_gt075 == 0`, and that
condition is recorded, so the repair is lossless: wherever `unanimous_gt075` is 0, both derived
fields become null. Meshes scored after the fix already store null and pass through unchanged.

Reading the output: a mesh is only interesting when it has **both** high-confidence coverage
(`vs_control` small) **and** a crisp profile (`conf_ratio` near the known-ink control's 2.07).
Real ink is crisp — what clears 0.5 mostly clears 0.75. Fibre texture and the "kolleisis juice"
at sheet joins are diffuse and run 4–15. Either signal alone is common and means little.
"""
import glob
import json
import os
import sys

P = r"D:/Competition/Vesuvius progress prizes"
CONTROL = {"unanimous_gt05": 0.07640, "unanimous_gt075": 0.03694, "conf_ratio": 2.07}


def repair(sc):
    """Null the derived fields where they were divided by the 1e-9 floor."""
    if not sc:
        return sc
    if not sc.get("unanimous_gt075"):
        sc["conf_ratio"] = None
        sc["vs_control"] = None
    return sc


def main():
    out = {}
    for f in sorted(glob.glob(f"{P}/_fl/corpus_scores*.json")):
        for k, v in json.load(open(f)).items():
            for d in ("forward", "reverse"):
                if d in v:
                    v[d] = repair(v[d])
            out[k] = v
    json.dump(out, open(f"{P}/_fl/corpus_scores_merged.json", "w"), indent=1)

    done = [v for v in out.values() if v.get("status") == "done"]
    scored = [v for v in done if (v.get("forward") or {}).get("vs_control") is not None]
    blank = len(done) - len(scored)
    area = sum((v.get("forward") or {}).get("area_cm2", 0) for v in done)
    print(f"{len(out)} meshes attempted, {len(done)} scored, {area:.1f} cm2")
    print(f"  {blank} had nothing above 0.75 at all (ratio undefined)")
    for st in ("render_fail", "no_ct_support"):
        n = sum(1 for v in out.values() if v.get("status") == st)
        if n:
            print(f"  {n} {st}")
    if not scored:
        print("\nnothing scored yet")
        return 0

    print(f"\ncontrol (known ink): >0.75 {CONTROL['unanimous_gt075']:.5f}, "
          f"ratio {CONTROL['conf_ratio']:.2f}")
    print(f"\n{'mesh':28s} {'align':>6s} {'area':>6s} {'>0.75':>9s} {'ratio':>7s} {'vs ctrl':>9s}")
    rank = sorted(scored, key=lambda v: v["forward"]["vs_control"])
    for v in rank[:15]:
        f = v["forward"]
        print(f"{v['name'][:28]:28s} {v['angle']:5.1f}° {f['area_cm2']:6.2f} "
              f"{f['unanimous_gt075']:9.5f} {f['conf_ratio']:7.2f} {f['vs_control']:8.1f}x")

    # the combination that would matter: coverage AND crispness together
    hits = [v for v in scored
            if v["forward"]["vs_control"] < 5.0 and v["forward"]["conf_ratio"] < 3.0]
    print(f"\nmeshes with BOTH coverage <5x below control AND ratio <3.0: {len(hits)}")
    for v in hits:
        f = v["forward"]
        print(f"  ** {v['name']} {f['area_cm2']:.2f} cm2 "
              f"ratio {f['conf_ratio']:.2f} {f['vs_control']:.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
