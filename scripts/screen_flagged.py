"""Run the full four-check rule on every region the corpus survey flagged, and on the control.

Section 18 of firstletters.md added checks 3 (row periodicity) and 4 (stroke morphology) after one
false positive, and ran them inline on that one region only. Two things were never checked:

1. Does the known-ink control itself pass checks 3 and 4? ink_9um localises ink but does not
   resolve letters, so real ink may fail a stroke-size test too. A check the control fails cannot
   be used to reject candidates.
2. Which grid axis is scroll-vertical? Text lines stack along the scroll's z. The inline version
   summed along axis 1 and assumed axis 0 is vertical. Here the vertical axis is read from z.tif
   and both axes are reported.

Scoring matches corpus_survey.py exactly (same checkpoints, rescale, 64 px erosion, threshold),
and the control is scored at the SAME 4 checkpoints -- the survey had compared its 4-checkpoint
scores against an 8-checkpoint control figure.
"""
import glob, json, os, sys
import numpy as np
import tifffile, cv2
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus_survey as cs  # noqa: E402

P = r"D:/Competition/Vesuvius progress prizes"
STROKE_MM2 = (0.3, 2.0)


def unanimous(files):
    R = np.stack([tifffile.imread(f) for f in files])
    S = np.stack([cs.rescale(r) for r in R])
    cov = R.max(0) > 0          # same fixed mask as corpus_survey.score()
    k = 2 * cs.ERODE_PX + 1
    cov = cv2.erode(cov.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    return S.min(0), cov


def vertical_axis(mesh_dir):
    """Axis of the tifxyz grid along which z changes most = scroll-vertical."""
    z = tifffile.imread(f"{mesh_dir}/z.tif").astype(np.float32)
    ok = z >= 0
    g0 = np.nanmedian(np.abs(np.diff(np.where(ok, z, np.nan), axis=0)))
    g1 = np.nanmedian(np.abs(np.diff(np.where(ok, z, np.nan), axis=1)))
    return (0 if g0 > g1 else 1), float(g0), float(g1)


def periodicity(hot, vox, axis):
    """Peak line pitch of the hot-pixel profile taken along `axis` (lines stacked along it)."""
    prof = hot.sum(1 - axis).astype(float)
    if len(prof) < 800 or prof.sum() == 0:
        return None
    prof = prof - ndi.uniform_filter1d(prof, 400)
    F = np.abs(np.fft.rfft(prof * np.hanning(len(prof))))
    fr = np.fft.rfftfreq(len(prof), d=vox / 10000.0)          # cycles per cm
    band = (fr > 1.0) & (fr < 4.0)                            # 2.5-10 mm pitch
    if band.sum() < 3:
        return None
    pk = int(np.argmax(F[band]))
    return dict(pitch_mm=round(10.0 / fr[band][pk], 2),
                prominence=round(float(F[band][pk] / np.median(F[band])), 2))


def morphology(hot, vox):
    lab, n = ndi.label(hot)
    if n == 0:
        return dict(n=0)
    sz = np.bincount(lab.ravel())[1:] * (vox / 1000.0) ** 2   # mm2
    stroke = (sz >= STROKE_MM2[0]) & (sz <= STROKE_MM2[1])
    return dict(n=int(n), p90_mm2=round(float(np.percentile(sz, 90)), 4),
                largest_mm2=round(float(sz.max()), 2), n_stroke_sized=int(stroke.sum()),
                stroke_area_share=round(float(sz[stroke].sum() / sz.sum()), 3))


def edge(hot, cov):
    d = ndi.distance_transform_edt(cov)
    return dict(hot_within_64px=round(float((d[hot] < 64).mean()), 3),
                covered_within_64px=round(float((d[cov] < 64).mean()), 3))


def strips(hot, vox, axis, n=3):
    """Row periodicity in n strips across the page, to see whether one peak is stable."""
    cols = np.where(hot.any(axis))[0] if hot.any() else []
    if len(cols) == 0:
        return []
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    w = (c1 - c0) // n
    out = []
    for i in range(n):
        sub = np.zeros_like(hot)
        sl = [slice(None), slice(None)]
        sl[1 - axis] = slice(c0 + i * w, c0 + (i + 1) * w)
        sub[tuple(sl)] = hot[tuple(sl)]
        out.append(periodicity(sub, vox, axis))
    return out


def screen(files, vox, axis, ctrl_g075=None):
    mn, cov = unanimous(files)
    v = mn[cov]
    g05, g075 = float((v > 0.5).mean()), float((v > 0.75).mean())
    hot = (mn > 0.75) & cov
    out = dict(area_cm2=round(float(cov.sum() * (vox / 10000.0) ** 2), 2),
               gt05=round(g05, 5), gt075=round(g075, 5),
               conf_ratio=round(g05 / g075, 2) if g075 else None,
               vertical_axis=axis,
               rows_vertical=periodicity(hot, vox, axis),
               rows_other_axis=periodicity(hot, vox, 1 - axis),
               rows_vertical_strips=strips(hot, vox, axis),
               morphology=morphology(hot, vox), edge=edge(hot, cov))
    if ctrl_g075 is not None and g075:
        out["vs_control_4ckpt"] = round(ctrl_g075 / g075, 2)
    return out, mn, cov


def main():
    res = {}
    four = [f"{P}/_fl/pred/ens/ctrl_s{sd}_{st}.tif" for sd, st in cs.CKPTS]
    # The w043 prediction grid is not a tifxyz we hold here, so report both axes for the control.
    # Control orientation: the w043 tifxyz (_fl/proxy/w043_ref) has z changing 19.4 vx per grid
    # step along axis 0 vs 1.3 along axis 1, and the prediction grid keeps that orientation.
    ctrl, mn, _ = screen(four, 9.362, 0)
    res["CONTROL PHerc0139 w043 (known ink)"] = ctrl
    np.save(f"{P}/_fl/screen_ctrl_min.npy", mn[::4, ::4])
    cg = ctrl["gt075"]
    print("control:", json.dumps(ctrl), flush=True)

    rows = {}
    for f in glob.glob(f"{P}/_fl/corpus_scores_*of2.json"):
        rows.update({k: v for k, v in json.load(open(f)).items() if v.get("status") == "done"})
    for name, v in sorted(rows.items()):
        for d in ("forward", "reverse"):
            if not cs.interesting(v.get(d)):
                continue
            suf = "_reverse.tif" if d == "reverse" else ".tif"
            files = [f"{P}/_fl/pred/corpus/{name}_s{sd}_{st}{suf}" for sd, st in cs.CKPTS]
            if not all(os.path.exists(x) for x in files):
                print(name, d, "predictions missing", flush=True)
                continue
            axis, g0, g1 = vertical_axis(f"{P}/_fl/meshes_eligible/{v['mesh']}")
            r, mn, _ = screen(files, cs.voxel_um(v["scroll"]), axis, cg)
            r.update(scroll=v["scroll"], direction=d, alignment_deg=round(v["angle"], 1),
                     z_step_axis0=round(g0, 3), z_step_axis1=round(g1, 3))
            res[f"{name} {d}"] = r
            np.save(f"{P}/_fl/screen_{name}_{d}_min.npy", mn[::4, ::4])
            print(f"{name} {d}:", json.dumps(r), flush=True)
    json.dump(res, open(f"{P}/_fl/screen_flagged.json", "w"), indent=1, default=float)
    figure()
    print("->", f"{P}/_fl/screen_flagged.json")


def figure():
    """Known ink and the strongest flagged region side by side, same scale, same contrast."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c_mn, c_cov = unanimous([f"{P}/_fl/pred/ens/ctrl_s{sd}_{st}.tif" for sd, st in cs.CKPTS])
    base = f"{P}/_fl/pred/corpus/PHerc0813_z12496_w060"
    f_mn, f_cov = unanimous([f"{base}_s{sd}_{st}.tif" for sd, st in cs.CKPTS])
    r_mn, r_cov = unanimous([f"{base}_s{sd}_{st}_reverse.tif" for sd, st in cs.CKPTS])
    H, W = f_mn.shape[0], c_mn.shape[1]
    hot = (f_mn > 0.75) & f_cov
    x0 = int(np.argmax(np.convolve(hot.sum(0).astype(float), np.ones(W), "valid")))
    cmap = matplotlib.colormaps["gray"].copy()
    cmap.set_bad("#2b4c7e")                       # blue = no model output (off the mesh)
    fig, ax = plt.subplots(3, 1, figsize=(11, 12.5))
    panels = [(c_mn, c_cov, slice(0, H), slice(0, W),
               "KNOWN INK - PHerc0139 w043, 1.5 x 3 cm of the control"),
              (f_mn, f_cov, slice(None), slice(x0, x0 + W),
               "PHerc0813 z12496_w060 forward - densest 1.5 x 3 cm (3.1x below control, ratio 2.52 vs 2.07)"),
              (r_mn, r_cov, slice(None), slice(x0, x0 + W), "same window, reverse")]
    for a, (mn, cov, ys, xs, title) in zip(ax, panels):
        a.imshow(np.where(cov, mn, np.nan)[ys, xs], cmap=cmap, vmin=0, vmax=1)
        a.set_title(title, fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        a.plot([40, 40 + 5000 / 9.362], [H - 50, H - 50], color="#ff3b30", lw=3)
        a.text(40, H - 80, "5 mm", color="#ff3b30", fontsize=9)
    fig.text(0.5, 0.005, "Unanimous minimum of the survey's 4 ink_9um checkpoints, 0-1, identical contrast. "
             "Blue = off the mesh. At 9 um neither panel shows letterforms.", ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.savefig(f"{P}/_fl/FIG_control_vs_flagged.png", dpi=110)
    print("->", f"{P}/_fl/FIG_control_vs_flagged.png")


if __name__ == "__main__":
    main()
