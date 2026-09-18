"""Fair comparison: the candidate's densest window against every corpus mesh's densest window,
in either direction.

The candidate's hecate score (0.1353) was measured on a 3.56 cm-wide window picked because ink_9um was
densest there, while the corpus baseline scores whole meshes. Picking the best window inflates any
score, so the fair comparison gives every corpus mesh the same advantage: for each mesh, slide a
window of the candidate's width across its hecate map in each direction and keep the best >0.75
fraction.
The window is chosen by hecate itself here, which favours the corpus, not the candidate.

Support: the mesh's own footprint (valid tifxyz cells) on the 9.6 um grid, eroded 64 px -- which
matches the survey's CT-based support to within 1.3%. The candidate window is scored the same way.
"""
import glob, json, os, sys
import numpy as np
import cv2, tifffile

P = r"D:/Competition/Vesuvius progress prizes"
sys.path.insert(0, f"{P}/_fl")
import corpus_survey as cs  # noqa: E402

WIN = 3706            # the candidate window's width in 9.6 um pixels (grid cols 170-360)
MIN_SUPPORT = 0.25    # a window must be at least this fraction supported to count
LEAD = "PHerc0813_z12496_w060"   # the candidate itself never counts as its own competitor


def footprint96(mesh_dir, shape96, cols=None):
    """The mesh's own footprint (valid tifxyz cells, 20x corner-aligned), on the 9.6 um hecate grid,
    eroded 64 px. An earlier version used ink_9um raw output > 0, which also counts ~20% tile padding
    beyond the mesh where hecate outputs zero, diluting every window."""
    X, Y, Z = (tifffile.imread(f"{mesh_dir}/{a}.tif") for a in "xyz")
    ok = ((X >= 0) & (Y >= 0) & (Z >= 0)).astype(np.uint8)
    if cols is not None:
        ok = ok[:, cols[0]:cols[1]]
    f = cv2.resize(np.kron(ok, np.ones((20, 20), np.uint8)), (shape96[1], shape96[0]),
                   interpolation=cv2.INTER_NEAREST)
    k = 2 * cs.ERODE_PX + 1
    return cv2.erode(f, np.ones((k, k), np.uint8)).astype(bool)


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
    # Both directions: which face a mesh's "forward" points at depends on how that mesh was fitted,
    # so a strong response in reverse counts exactly as much as one in forward. (A first version of
    # this comparison used forward maps only and overstated the candidate's lead.)
    out = {}
    for name, v in sorted(rows.items()):
        if v.get("status") != "done" or name == LEAD:
            continue
        best = {}
        for d in ("forward", "reverse"):
            png = f"{P}/_fl/pred/corpus/{name}_hecate_{d}.png"
            if not os.path.exists(png):
                continue
            prob = cv2.imread(png, cv2.IMREAD_UNCHANGED) / 255.0
            sup = footprint96(f"{P}/_fl/meshes_eligible/{v['mesh']}", prob.shape)
            frac, x0 = best_window(prob, sup)
            if frac is not None:
                best[d] = round(frac, 5)
        if best:
            d = max(best, key=best.get)
            out[name] = dict(best_window_gt075=best[d], direction=d, **{f"{k}_gt075": x for k, x in best.items()})
    # the candidate, scored the same way on its own window render
    lead = cv2.imread(f"{P}/_fl/pred/hecate_s64/lead_forward.png", cv2.IMREAD_UNCHANGED) / 255.0
    lsup = footprint96(f"{P}/_fl/meshes_eligible/meshes/PHerc0813/z12496_w060", lead.shape, cols=(170, 360))
    lead_frac = float(((lead > 0.75) & lsup).sum() / max(lsup.sum(), 1))
    vals = np.array([x["best_window_gt075"] for x in out.values()])
    top = sorted(out.items(), key=lambda kv: -kv[1]["best_window_gt075"])[:5]
    second = top[0][1]["best_window_gt075"]
    res = dict(n_meshes=len(out), window_px=WIN, candidate_window_gt075=round(lead_frac, 5),
               corpus_best_window_median=round(float(np.median(vals)), 5),
               corpus_best_window_p90=round(float(np.percentile(vals, 90)), 5),
               corpus_best_window_max=round(float(vals.max()), 5),
               top5=[(k, x["best_window_gt075"], x["direction"]) for k, x in top], per_mesh=out)
    json.dump(res, open(f"{P}/_fl/hecate_window_max.json", "w"), indent=1)
    print(f"{len(out)} meshes, each at its densest {WIN}-px window under hecate, best of both directions:")
    print(f"  median {res['corpus_best_window_median']:.4f}  p90 {res['corpus_best_window_p90']:.4f}  "
          f"max {res['corpus_best_window_max']:.4f}")
    print("  top 5:", [(k, round(x, 4), d) for k, x, d in res["top5"]])
    print(f"candidate window, same support rule: {lead_frac:.4f} "
          f"= {lead_frac / res['corpus_best_window_max']:.2f}x the best other window "
          f"({res['top5'][0][0]} {res['top5'][0][2]}); third place {res['top5'][1][1]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
