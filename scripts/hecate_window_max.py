"""Fair comparison: the candidate's densest window against every corpus mesh's densest window.

The candidate's hecate score (0.1353) was measured on a 3.56 cm-wide window picked because ink_9um was
densest there, while the corpus baseline scores whole meshes. Picking the best window inflates any
score, so the fair comparison gives every corpus mesh the same advantage: for each mesh, slide a
window of the candidate's width across its hecate forward map and keep the best >0.75 fraction.
The window is chosen by hecate itself here, which favours the corpus, not the candidate.

Support: the survey's own support (CT in the 16 central planes) needs the render, which is deleted
after scoring. The kept ink_9um predictions give the rendered footprint instead (raw output > 0),
resampled from native spacing to the 9.6 um hecate grid and eroded 64 px. The candidate window is
scored with the same approximate support, so both sides are measured identically.
"""
import glob, json, os, sys
import numpy as np
import cv2, tifffile

P = r"D:/Competition/Vesuvius progress prizes"
sys.path.insert(0, f"{P}/_fl")
import corpus_survey as cs  # noqa: E402

WIN = 3706            # the candidate window's width in 9.6 um pixels (grid cols 170-360)
MIN_SUPPORT = 0.25    # a window must be at least this fraction supported to count


def support_from_ink(name, tag, shape96, vox):
    fs = [f"{P}/_fl/pred/{tag}/{name}_s{sd}_{st}.tif" for sd, st in cs.CKPTS]
    if not all(os.path.exists(f) for f in fs):
        return None
    cov = np.stack([tifffile.imread(f) for f in fs]).max(0) > 0
    cov = cv2.resize(cov.astype(np.uint8), (shape96[1], shape96[0]), interpolation=cv2.INTER_NEAREST)
    k = 2 * cs.ERODE_PX + 1
    return cv2.erode(cov, np.ones((k, k), np.uint8)).astype(bool)


def best_window(prob, sup):
    hot = ((prob > 0.75) & sup).sum(0).astype(np.float64)
    tot = sup.sum(0).astype(np.float64)
    w = min(WIN, prob.shape[1])
    kern = np.ones(w)
    h, t = np.convolve(hot, kern, "valid"), np.convolve(tot, kern, "valid")
    ok = t >= MIN_SUPPORT * prob.shape[0] * w
    if not ok.any():
        return None, None
    frac = np.where(ok, h / np.maximum(t, 1), -1)
    i = int(np.argmax(frac))
    return float(frac[i]), i


def main():
    rows = {}
    for f in glob.glob(f"{P}/_fl/corpus_scores*.json"):
        if "merged" in f or "badmask" in f:
            continue
        rows.update(json.load(open(f)))
    out = {}
    for name, v in sorted(rows.items()):
        png = f"{P}/_fl/pred/corpus/{name}_hecate_forward.png"
        if v.get("status") != "done" or not os.path.exists(png):
            continue
        prob = cv2.imread(png, cv2.IMREAD_UNCHANGED) / 255.0
        sup = support_from_ink(name, "corpus", prob.shape, cs.voxel_um(v["scroll"]))
        if sup is None:
            continue
        frac, x0 = best_window(prob, sup)
        if frac is not None:
            out[name] = dict(best_window_gt075=round(frac, 5), x0=x0, width=min(WIN, prob.shape[1]))
    # the candidate, scored the same way on its own window render
    lead = cv2.imread(f"{P}/_fl/pred/hecate_s64/lead_forward.png", cv2.IMREAD_UNCHANGED) / 255.0
    R = np.stack([tifffile.imread(f"{P}/_fl/pred/corpus/PHerc0813_z12496_w060_s{sd}_{st}.tif")[:, 3400:7200]
                  for sd, st in cs.CKPTS])
    cov = cv2.resize((R.max(0) > 0).astype(np.uint8), (lead.shape[1], lead.shape[0]), interpolation=cv2.INTER_NEAREST)
    k = 2 * cs.ERODE_PX + 1
    lsup = cv2.erode(cov, np.ones((k, k), np.uint8)).astype(bool)
    lead_frac = float(((lead > 0.75) & lsup).sum() / max(lsup.sum(), 1))
    vals = np.array([x["best_window_gt075"] for x in out.values()])
    top = sorted(out.items(), key=lambda kv: -kv[1]["best_window_gt075"])[:5]
    res = dict(n_meshes=len(out), window_px=WIN, candidate_window_gt075=round(lead_frac, 5),
               corpus_best_window_median=round(float(np.median(vals)), 5),
               corpus_best_window_p90=round(float(np.percentile(vals, 90)), 5),
               corpus_best_window_max=round(float(vals.max()), 5),
               top5=[(k, x["best_window_gt075"]) for k, x in top], per_mesh=out)
    json.dump(res, open(f"{P}/_fl/hecate_window_max.json", "w"), indent=1)
    print(f"{len(out)} meshes, each at its densest {WIN}-px window under hecate:")
    print(f"  median {res['corpus_best_window_median']:.4f}  p90 {res['corpus_best_window_p90']:.4f}  "
          f"max {res['corpus_best_window_max']:.4f}")
    print("  top 5:", [(k, round(x, 4)) for k, x in res["top5"]])
    print(f"candidate window, same support rule: {lead_frac:.4f} "
          f"= {lead_frac / res['corpus_best_window_max']:.1f}x the best corpus window")
    return 0


if __name__ == "__main__":
    sys.exit(main())
