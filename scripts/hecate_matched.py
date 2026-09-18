"""Is hecate's response on the lead more than 'a sheet is centred here'?

On the lead window hecate's hot pixels coincide with a papyrus sheet sitting at the render's centre
(depth-profile peak), and the unresponsive pixels with a flat profile. A model that simply reacts to
a well-centred sheet would light up blank meshes the same way. So compare hecate's >0.75 fraction
at MATCHED sheet centring across known ink, the lead, and meshes ink_9um scored blank.

Centring per pixel = mean CT of the central 8 planes minus mean of the outer planes (the 7 at
each end), in the hecate input's own 9.6 um render. Pixels outside CT support (eroded 64 px) are
dropped. Bins are pooled quantiles over all cases so every case is read at the same centring.
"""
import json, sys
import numpy as np, cv2, zarr

P = r"D:/Competition/Vesuvius progress prizes"
sys.path.insert(0, f"{P}/_fl")
import hecate_eval as he  # noqa: E402

CASES = ["control", "lead", "neg_a", "neg_b"]


def centring(vol):
    V = np.asarray(zarr.open_group(vol, mode="r")["0"]).astype(np.float32)
    n = V.shape[0]
    mid = V[n // 2 - 4:n // 2 + 4].mean(0)
    outer = np.concatenate([V[:7], V[-7:]]).mean(0)
    return mid - outer


def main():
    data = {}
    for c in CASES:
        vol = f"{P}/_fl/work/hecate/{c}_9.6.zarr"
        try:
            sup = he.support_mask(vol)
        except Exception:
            continue
        cen = centring(vol)
        out = {}
        for d in ("forward", "reverse"):
            p = cv2.imread(f"{P}/_fl/pred/hecate/{c}_{d}.png", cv2.IMREAD_UNCHANGED)
            if p is None:
                break
            out[d] = p[sup] / 255.0
        if len(out) == 2:
            data[c] = dict(cen=cen[sup], **out)
    edges = np.quantile(np.concatenate([v["cen"] for v in data.values()]), np.linspace(0, 1, 6))
    res = {"bin_edges_centring": [round(float(e), 1) for e in edges]}
    print("centring bins (pooled quintiles):", res["bin_edges_centring"])
    print("%-8s %s" % ("case", "  ".join("q%d fwd>0.75 (n)" % (i + 1) for i in range(5))))
    for c, v in data.items():
        row = []
        for i in range(5):
            m = (v["cen"] >= edges[i]) & (v["cen"] <= edges[i + 1])
            row.append(dict(n=int(m.sum()),
                            fwd_gt075=round(float((v["forward"][m] > 0.75).mean()), 4) if m.sum() > 500 else None,
                            rev_gt075=round(float((v["reverse"][m] > 0.75).mean()), 4) if m.sum() > 500 else None))
        res[c] = dict(median_centring=round(float(np.median(v["cen"])), 1), bins=row)
        print("%-8s " % c + "  ".join("%s (%d)" % (r["fwd_gt075"], r["n"]) for r in row),
              "| median centring %.1f" % np.median(v["cen"]))
    json.dump(res, open(f"{P}/_fl/hecate_matched.json", "w"), indent=1)


if __name__ == "__main__":
    main()
