"""Second opinion on the PHerc0813 lead from the team's new 9.6 um ink model, hecate.

The corpus survey's strongest region, PHerc0813_z12496_w060, is 3.1x below the known-ink control
under ink_9um and is unresolved (corpus-ink-survey, correction of 2026-09-16). hecate
(huggingface.co/scrollprize/hecate, published 2026-09-15, checkpoint sha256 809f4f10...fe5d) is a
different model: distilled from the 2.4 um canonical detector, trained partly on native coarse
scans, predicting ink on the middle sheet. If the lead is ink, hecate should light it up
relative to blank meshes the way it lights up known ink.

Cases, each rendered at native spacing then resampled to 9.6 um in all three axes (hecate's model
card requires it; the script does not resample):
  control  PHerc0139 w043, the same 3 x 3 cm crop of the published surface volume used as the
           ink_9um control
  lead     PHerc0813_z12496_w060, grid columns 170-360 (covers the densest ink_9um window)
  neg_a    PHerc0813_z6496_w060, middle 190 columns (ink_9um: 161x below control, pre-fix)
  neg_b    PHerc0813_z13696_w060, middle 190 columns (ink_9um: 110x below control, pre-fix)

hecate runs in both depth directions. Scores: fraction of supported pixels above 0.5 and 0.75,
support = CT present in the central planes, eroded 64 px -- the same shape of metric as the survey.
ink_9um on the identical pixel window is reported alongside for the three corpus cases.
"""
import json, os, shutil, subprocess, sys, time
import numpy as np
import tifffile, cv2, zarr
from scipy import ndimage as ndi

P = r"D:/Competition/Vesuvius progress prizes"
PY = r"C:/Users/PC/miniconda3/envs/vesuvius/python.exe"
B = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
HEC = f"{P}/_fl/ckpt/hecate"
W = f"{P}/_fl/work/hecate"
OUT = f"{P}/_fl/pred/hecate"
TARGET_UM = 9.6
sys.path.insert(0, f"{P}/_fl")
import corpus_survey as cs  # noqa: E402

CASES = {
    "lead": dict(mesh="meshes/PHerc0813/z12496_w060", scroll="PHerc0813", cols=(170, 360),
                 survey="PHerc0813_z12496_w060"),
    "neg_a": dict(mesh="meshes/PHerc0813/z6496_w060", scroll="PHerc0813", cols="middle190",
                  survey="PHerc0813_z6496_w060"),
    "neg_b": dict(mesh="meshes/PHerc0813/z13696_w060", scroll="PHerc0813", cols="middle190",
                  survey="PHerc0813_z13696_w060"),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def resample_to_target(src_arr, native_um, dst_path):
    f = native_um / TARGET_UM
    vol = np.asarray(src_arr)
    out = np.stack([ndi.zoom(vol[k], f, order=1) for k in range(vol.shape[0])])   # YX per layer
    zi = np.linspace(0, vol.shape[0] - 1, int(round(vol.shape[0] * f)))           # Z by linear interp
    lo = np.floor(zi).astype(int); hi = np.minimum(lo + 1, vol.shape[0] - 1); t = (zi - lo)[:, None, None]
    out = np.rint(out[lo] * (1 - t) + out[hi] * t).astype(np.uint8)
    g = zarr.open_group(dst_path, mode="w", zarr_format=2)
    g.create_array("0", shape=out.shape, chunks=(out.shape[0], 256, 256), dtype="u1")[:] = out
    g.attrs["spacing_um"] = TARGET_UM
    g.attrs["resampled_from_um"] = native_um
    return out.shape


def build_control():
    dst = f"{W}/control_9.6.zarr"
    if os.path.isdir(dst):
        return dst
    import fsspec
    src = ("s3://vesuvius-challenge-open-data/PHerc0139/segments/20260112000000-w043_2026011217/"
           "surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr")
    a = zarr.open(fsspec.filesystem("s3", anon=True).get_mapper(src), mode="r")["0"]
    y, x, s = 1500, 2000, int(round(3.0 * 10000 / 9.362))       # identical to _fl/mk_ctrl_crop.py
    log("control: fetching", a.shape[0], "x", s, "x", s)
    blk = np.asarray(a[:, y:y + s, x:x + s])
    log("control: resampling", resample_to_target(blk, 9.362, dst))
    return dst


def build_mesh(name, c):
    dst = f"{W}/{name}_9.6.zarr"
    if os.path.isdir(dst):
        return dst
    src = f"{P}/_fl/meshes_eligible/{c['mesh']}"
    X, Y, Z = (tifffile.imread(f"{src}/{k}.tif") for k in "xyz")
    H, Wd = X.shape
    c0, c1 = c["cols"] if c["cols"] != "middle190" else ((Wd - 190) // 2, (Wd - 190) // 2 + 190)
    c["cols"] = (c0, c1)
    crop = f"{W}/{name}_tifxyz"
    os.makedirs(crop, exist_ok=True)
    for k, arr in zip("xyz", (X, Y, Z)):
        tifffile.imwrite(f"{crop}/{k}.tif", np.ascontiguousarray(arr[:, c0:c1]))
    meta = json.load(open(f"{src}/meta.json"))
    json.dump({"format": "tifxyz", "type": "seg", "scale": meta["scale"], "uuid": name},
              open(f"{crop}/meta.json", "w"), indent=1)
    open(f"{crop}/volume_source.txt", "w").write(f"{B}/{c['scroll']}/volumes/{cs.VOLUMES[c['scroll']]}\n")
    rz = f"{W}/{name}_native.zarr"
    log(name, "rendering grid cols", (c0, c1))
    t = time.time()
    rc = subprocess.call([PY, f"{P}/_fl/render_tifxyz.py", crop, rz, "--layers", "31", "--tile", "384",
                          "--workers", "8", "--cache-chunks", "900"],
                         stdout=open(f"{W}/{name}.render.log", "w"), stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"{name} render failed rc={rc}")
    log(name, f"rendered in {time.time() - t:.0f}s; resampling")
    log(name, "->", resample_to_target(zarr.open_group(rz, mode="r")["0"], cs.voxel_um(c["scroll"]), dst))
    shutil.rmtree(rz, ignore_errors=True)
    return dst


def run_hecate(name, vol):
    outs = {}
    for d, flag in (("forward", []), ("reverse", ["--reverse"])):
        png = f"{OUT}/{name}_{d}.png"
        if not os.path.exists(png):
            t = time.time()
            rc = subprocess.call([PY, f"{HEC}/hecate.py", "--checkpoint", f"{HEC}/hecate_9.6um.pth",
                                  "--input", vol, "--spacing-um", "9.6", "--output", png,
                                  "--device", "cuda", "--precision", "bf16", "--batch-size", "16"] + flag,
                                 stdout=open(f"{W}/{name}_{d}.hecate.log", "w"), stderr=subprocess.STDOUT)
            if rc != 0:
                raise RuntimeError(f"hecate {name} {d} rc={rc}")
            log(name, d, f"hecate {time.time() - t:.0f}s")
        outs[d] = png
    return outs


def support_mask(vol):
    a = zarr.open_group(vol, mode="r")["0"]
    z0 = a.shape[0] // 2 - 8
    sup = np.asarray(a[z0:z0 + 16]).any(0)
    k = 2 * 64 + 1
    return cv2.erode(sup.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def frac(p, sup):
    v = p[sup]
    g05, g075 = float((v > 0.5).mean()), float((v > 0.75).mean())
    return dict(gt05=round(g05, 5), gt075=round(g075, 5), ratio=round(g05 / g075, 2) if g075 else None,
                area_cm2=round(float(sup.sum()) * (TARGET_UM / 1e4) ** 2, 2))


def ink9_window(c):
    """ink_9um unanimous minimum (fixed mask) on the identical pixel window, native spacing."""
    base = f"{P}/_fl/pred/corpus/{c['survey']}"
    out = {}
    for d, suf in (("forward", ".tif"), ("reverse", "_reverse.tif")):
        fs = [f"{base}_s{sd}_{st}{suf}" for sd, st in cs.CKPTS]
        if not all(os.path.exists(f) for f in fs):
            out[d] = "predictions not on disk"
            continue
        R = np.stack([tifffile.imread(f) for f in fs])
        c0, c1 = (x * 20 for x in c["cols"])
        R = R[:, :, c0:c1]
        S = np.stack([cs.rescale(r) for r in R])
        k = 2 * cs.ERODE_PX + 1
        cov = cv2.erode((R.max(0) > 0).astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        v = S.min(0)[cov]
        g05, g075 = float((v > 0.5).mean()), float((v > 0.75).mean())
        out[d] = dict(gt05=round(g05, 5), gt075=round(g075, 5), ratio=round(g05 / g075, 2) if g075 else None,
                      vs_control=round(cs.CONTROL["unanimous_gt075"] / g075, 1) if g075 else None)
    return out


def main():
    os.makedirs(W, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    res = json.load(open(f"{P}/_fl/hecate_eval.json")) if os.path.exists(f"{P}/_fl/hecate_eval.json") else {}
    for name in ["control"] + list(CASES):
        vol = build_control() if name == "control" else build_mesh(name, CASES[name])
        if name in CASES and not isinstance(CASES[name]["cols"], tuple):
            H, Wd = tifffile.imread(f"{P}/_fl/meshes_eligible/{CASES[name]['mesh']}/x.tif").shape
            CASES[name]["cols"] = ((Wd - 190) // 2, (Wd - 190) // 2 + 190)
        pngs = run_hecate(name, vol)
        sup = support_mask(vol)
        r = {"hecate_" + d: frac(cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0, sup)
             for d, p in pngs.items()}
        if name in CASES:
            r["grid_cols"] = list(CASES[name]["cols"])
            r["ink_9um_same_window"] = ink9_window(CASES[name])
        res[name] = r
        log(name, json.dumps(r))
        json.dump(res, open(f"{P}/_fl/hecate_eval.json", "w"), indent=1)
    log("->", f"{P}/_fl/hecate_eval.json")


if __name__ == "__main__":
    main()
