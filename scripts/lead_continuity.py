"""Does the PHerc0813 band continue up and down the scroll, or is it a local patch?

A kollesis (sheet join) runs vertically up the roll, so its footprint keeps nearly the same (x, y)
while z changes -- no umbilicus needed, because the scroll axis is z. A short run of text would not
repeat at the same (x, y) in the meshes above and below.

Only meshes of the SAME wrap are informative: wrap labels on this scroll step ~2.84 mm in radius,
which is a different sheet entirely, so "no hot pixels" on another wrap says nothing about this band.

For each mesh: hot = ink_9um unanimous minimum > 0.75 over the fixed coverage mask (eroded 64 px);
hot pixels are mapped through the tifxyz grid to scroll coordinates. Reported per mesh:
  - the (x, y) centroid and spread of hot pixels, and their z range
  - whether the mesh even COVERS the lead band's (x, y) column, because absence where there is no
    coverage proves nothing
"""
import glob, json, os, sys
import numpy as np
import tifffile, cv2

P = r"D:/Competition/Vesuvius progress prizes"
sys.path.insert(0, f"{P}/_fl")
import corpus_survey as cs  # noqa: E402

LEAD = "PHerc0813_z12496_w060"
MESHES = [LEAD, "PHerc0813_z11904_w060", "PHerc0813_z13088_w060",
          "PHerc0813_z12496_w040", "PHerc0813_z13088_w040", "PHerc0813_z12496_w080"]
# Measured on this scroll: consecutive wrap labels (w040 -> w060) are a median 303 voxels
# (2.84 mm) apart, so a 300-voxel radius silently includes the neighbouring sheet. Keep the
# column well inside one wrap.
RADIUS = 100.0      # voxels (~0.94 mm)


def scores():
    rows = {}
    for f in glob.glob(f"{P}/_fl/corpus_scores*.json"):
        if "merged" in f or "badmask" in f:
            continue
        try:
            rows.update(json.load(open(f)))
        except Exception:
            pass
    return rows


def preds_for(name):
    for tag in ("corpus", "lead_nbr"):
        fs = [f"{P}/_fl/pred/{tag}/{name}_s{sd}_{st}.tif" for sd, st in cs.CKPTS]
        if all(os.path.exists(f) for f in fs):
            return fs
    return None


def mesh_hot(name, rows):
    fs = preds_for(name)
    if not fs:
        return None
    v = rows.get(name)
    if not v or v.get("status") != "done":
        return None
    R = np.stack([tifffile.imread(f) for f in fs])
    S = np.stack([cs.rescale(r) for r in R])
    k = 2 * cs.ERODE_PX + 1
    cov = cv2.erode((R.max(0) > 0).astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    hot = (S.min(0) > 0.75) & cov
    src = f"{P}/_fl/meshes_eligible/{v['mesh']}"
    X, Y, Z = (tifffile.imread(f"{src}/{a}.tif").astype(np.float32) for a in "xyz")
    ok = (X >= 0) & (Y >= 0) & (Z >= 0)
    gy, gx = (np.arange(s) for s in (hot.shape[0], hot.shape[1]))
    # pred pixel -> grid cell: the render upsamples the grid 20x, corner-aligned
    def to_coords(mask):
        yy, xx = np.nonzero(mask)
        gi, gj = np.minimum(yy // 20, X.shape[0] - 1), np.minimum(xx // 20, X.shape[1] - 1)
        keep = ok[gi, gj]
        return X[gi, gj][keep], Y[gi, gj][keep], Z[gi, gj][keep]
    return dict(hot=to_coords(hot), cov=to_coords(cov), n_hot=int(hot.sum()), n_cov=int(cov.sum()))


def main():
    rows = scores()
    got = {}
    for m in MESHES:
        r = mesh_hot(m, rows)
        if r is None:
            print(f"{m:28s} not available yet")
            continue
        got[m] = r
        hx, hy, hz = r["hot"]
        cx, cy, cz = r["cov"]
        print(f"{m:28s} hot {r['n_hot']:8d} px  centroid x {np.mean(hx) if len(hx) else float('nan'):7.0f} "
              f"y {np.mean(hy) if len(hy) else float('nan'):7.0f}  z {np.min(cz):.0f}-{np.max(cz):.0f}  "
              f"covered x {np.min(cx):.0f}-{np.max(cx):.0f} y {np.min(cy):.0f}-{np.max(cy):.0f}")
    if LEAD not in got:
        return 1
    # The band lies along a curved sheet, so its (x, y) centroid sits OFF the sheet (the chord-vs-arc
    # trap). Define the column by distance to the band's own point set instead.
    from scipy.spatial import cKDTree
    lx, ly, _ = got[LEAD]["hot"]
    step = max(1, len(lx) // 20000)
    tree = cKDTree(np.c_[lx[::step], ly[::step]])
    print(f"\nband of the lead: {len(lx)} hot points, x {lx.min():.0f}-{lx.max():.0f} "
          f"y {ly.min():.0f}-{ly.max():.0f}")
    print(f"\nwithin {RADIUS:.0f} voxels (~{RADIUS * 9.362 / 1000:.1f} mm) of the band footprint:")
    out = {}
    for m, r in got.items():
        cx, cy, cz = r["cov"]
        hx, hy, hz = r["hot"]
        dc = tree.query(np.c_[cx, cy])[0] if len(cx) else np.array([])
        dh = tree.query(np.c_[hx, hy])[0] if len(hx) else np.array([])
        near_cov, near_hot = int((dc <= RADIUS).sum()), int((dh <= RADIUS).sum())
        frac = near_hot / near_cov if near_cov else None
        out[m] = dict(near_covered_px=near_cov, near_hot_px=near_hot,
                      hot_fraction_in_column=round(frac, 5) if frac is not None else None,
                      z_range=[float(np.min(cz)), float(np.max(cz))])
        print(f"  {m:28s} covered {near_cov:8d} px, hot {near_hot:7d} px -> "
              f"{'%.4f' % frac if frac is not None else 'no coverage in this column'}")
    json.dump(out, open(f"{P}/_fl/lead_continuity.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
