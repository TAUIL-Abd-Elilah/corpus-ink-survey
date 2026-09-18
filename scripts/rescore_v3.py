"""Rescore every ink_9um survey mesh on the mesh's own footprint.

History of the coverage mask:
  v1 (until 16 Sept)  rescale(p).max > 0      -- dropped confidently blank papyrus (37% of the control)
  v2 (16-19 Sept)     raw model output > 0    -- counts tile padding beyond the mesh: 19-24% of the
                                                 "covered" pixels lie outside the mesh, where the model
                                                 outputs ~0.01-0.02 on empty input
  v3 (this file)      the tifxyz grid's valid cells, upsampled 20x corner-aligned, eroded 64 px

v3 is what "the surface" means. hecate's CT-based support already matches it to within 1.3%, so hecate
scores are unaffected. The control is a crop of a published surface volume that is data everywhere, so
its footprint is the whole crop and its value is unchanged (0.04269).

Reads the kept prediction TIFFs (--keep-all since 16 Sept); writes _fl/ink_v3_footprint.json and never
touches the shard score files the running backfill is writing.
"""
import glob, json, os, sys
import numpy as np
import cv2, tifffile

P = r"D:/Competition/Vesuvius progress prizes"
sys.path.insert(0, f"{P}/_fl")
import corpus_survey as cs  # noqa: E402

K = np.ones((2 * cs.ERODE_PX + 1,) * 2, np.uint8)


def footprint(mesh_dir, shape):
    X, Y, Z = (tifffile.imread(f"{mesh_dir}/{a}.tif") for a in "xyz")
    ok = ((X >= 0) & (Y >= 0) & (Z >= 0)).astype(np.uint8)
    f = np.kron(ok, np.ones((20, 20), np.uint8))
    out = np.zeros(shape, np.uint8)
    h, w = min(shape[0], f.shape[0]), min(shape[1], f.shape[1])
    out[:h, :w] = f[:h, :w]
    return cv2.erode(out, K).astype(bool)


def score_on(files, cov, vox):
    S = np.stack([cs.rescale(tifffile.imread(f)) for f in files])
    if cov.sum() < 1000:
        return None
    v = S.min(0)[cov]
    g05, g075 = float((v > 0.5).mean()), float((v > 0.75).mean())
    return dict(area_cm2=round(float(cov.sum() * (vox / 10000.0) ** 2), 2),
                unanimous_gt05=round(g05, 5), unanimous_gt075=round(g075, 5),
                conf_ratio=round(g05 / g075, 2) if g075 > 0 else None,
                vs_control=round(cs.CONTROL["unanimous_gt075"] / g075, 1) if g075 > 0 else None)


def main():
    rows = {}
    for f in sorted(glob.glob(f"{P}/_fl/corpus_scores_*of2.json")):
        rows.update(json.load(open(f)))
    out = {}
    for name, v in sorted(rows.items()):
        if v.get("status") != "done" or v.get("scoring") != "v2":
            continue
        vox = cs.voxel_um(v["scroll"])
        rec = dict(scroll=v["scroll"], wrap=v["wrap"], z=v["z"], angle=v["angle"], mesh=v["mesh"])
        for d, suf in (("forward", ""), ("reverse", "_reverse")):
            fs = [f"{P}/_fl/pred/corpus/{name}_s{sd}_{st}{suf}.tif" for sd, st in cs.CKPTS]
            if not all(os.path.exists(x) for x in fs):
                rec[d] = None
                continue
            shape = tifffile.imread(fs[0]).shape
            rec[d] = score_on(fs, footprint(f"{P}/_fl/meshes_eligible/{v['mesh']}", shape), vox)
            rec[d + "_v2_rawmask"] = v.get(d)
        out[name] = rec
    json.dump(out, open(f"{P}/_fl/ink_v3_footprint.json", "w"), indent=1)
    pairs = sorted((s["vs_control"], k, d) for k, r in out.items() for d in ("forward", "reverse")
                   if (s := r.get(d)) and s.get("vs_control") is not None)
    print(f"{len(out)} meshes rescored on their footprint; strongest mesh-directions:")
    for x in pairs[:8]:
        print("  %.1fx below control  %s %s" % x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
