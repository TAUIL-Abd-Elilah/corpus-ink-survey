"""Which way does each mesh's surface normal point, and what does "forward" therefore mean?

For every grid row of a tifxyz mesh (one z band of the sheet) a least-squares circle is fitted through
the row's (x, y) points; its centre is the local centre of curvature, on the side of the roll's core.
The normal is cross(d/dv, d/du), the convention the survey renderer shares with the published surface
volumes. A row votes "outward" if most normals point away from its curvature centre; the mesh's score is
the median share over rows. Nearly flat rows (radius > 50,000 voxels) give no reliable centre and are
skipped.

Run on the published PHerc0139 w043 mesh (the known-ink control, whose ink reads forward) and on every
survey mesh, this says whether "forward" is the same physical face everywhere. It also measures how
much each model's reverse direction runs above forward on ordinary papyrus.
"""
import glob, json, os, sys
import numpy as np
import tifffile

P = os.environ.get("SURVEY_ROOT", r"D:/Competition/Vesuvius progress prizes")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def outward_share(d):
    X, Y, Z = (tifffile.imread(f"{d}/{a}.tif").astype(np.float64) for a in "xyz")
    ok = (X >= 0) & (Y >= 0) & (Z >= 0)
    du = np.stack([np.gradient(A, axis=1) for A in (X, Y, Z)], -1)
    dv = np.stack([np.gradient(A, axis=0) for A in (X, Y, Z)], -1)
    n = np.cross(dv, du)
    votes = []
    for i in range(0, X.shape[0], max(1, X.shape[0] // 12)):
        m = ok[i]
        if m.sum() < 20:
            continue
        x, y = X[i, m], Y[i, m]
        cx, cy, c = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones_like(x)], x ** 2 + y ** 2, rcond=None)[0]
        R = np.sqrt(c + cx ** 2 + cy ** 2)
        if not np.isfinite(R) or R > 50000:
            continue
        votes.append(float(np.mean(n[i, m, 0] * (x - cx) + n[i, m, 1] * (y - cy) > 0)))
    return float(np.median(votes)) if votes else None


def main():
    rows = {}
    for f in glob.glob(f"{P}/_fl/corpus_scores*.json"):
        if "merged" in f or "badmask" in f:
            continue
        rows.update(json.load(open(f)))
    control = outward_share(f"{P}/_fl/proxy/w043_ref")
    per = {}
    for k, v in sorted(rows.items()):
        if v.get("status") == "done":
            o = outward_share(f"{P}/_fl/meshes_eligible/{v['mesh']}")
            if o is not None:
                per[k] = round(o, 4)
    o = np.array(list(per.values()))
    H, I = [], []
    for v in rows.values():
        if v.get("status") != "done":
            continue
        h = v.get("hecate") if isinstance(v.get("hecate"), dict) else None
        if h and (h.get("forward") or {}).get("gt075") is not None and (h.get("reverse") or {}).get("gt075") is not None:
            H.append((h["forward"]["gt075"], h["reverse"]["gt075"]))
        f_, r_ = (v.get("forward") or {}).get("unanimous_gt075"), (v.get("reverse") or {}).get("unanimous_gt075")
        if f_ is not None and r_ is not None:
            I.append((f_, r_))
    H, I, e = np.array(H), np.array(I), 1e-4
    out = dict(control_outward_share=round(control, 4), n_meshes=len(per),
               n_outward=int((o > 0.5).sum()), n_inward=int((o <= 0.5).sum()),
               median_outward_share=round(float(np.median(o)), 3),
               hecate=dict(n=len(H), share_rev_gt_fwd=round(float((H[:, 1] > H[:, 0]).mean()), 3),
                           median_rev_over_fwd=round(float(np.median((H[:, 1] + e) / (H[:, 0] + e))), 2)),
               ink_9um=dict(n=len(I), share_rev_gt_fwd=round(float((I[:, 1] > I[:, 0]).mean()), 3),
                            median_rev_over_fwd=round(float(np.median((I[:, 1] + e) / (I[:, 0] + e))), 2)),
               per_mesh=per)
    json.dump(out, open(os.path.join(HERE, "data", "hecate", "mesh_orientation.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_mesh"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
