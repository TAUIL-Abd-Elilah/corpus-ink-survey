"""Where in depth does hecate put the ink -- on the sheet, or spread through the render?

hecate's `--output-3d` is meant to localise ink on the middle sheet of a surface-conditioned render.
That is a test the 2D map cannot do: real ink sits as a thin layer on one face of the papyrus, while a
response to bulk material has no reason to.

For each case (known ink, the PHerc0813 candidate, two ink_9um-blank meshes of that scroll) this runs
the 3D head on the same 9.6 um volume used for the 2D maps, then, over the pixels the 2D map calls hot
(>0.75), compares two depth profiles inside the evaluated z window:

  CT              - where the papyrus sheet is
  3D ink logit    - where hecate thinks the ink is

and reports the offset between their peaks in um, plus the thickness (FWHM) of the ink layer. Known ink
sets the scale: whatever offset and thickness it shows is what ink looks like to this model at 9.6 um.
"""
import json, os, subprocess, sys, time
import numpy as np
import cv2, zarr

P = r"D:/Competition/Vesuvius progress prizes"
sys.path.insert(0, f"{P}/_fl")
import corpus_survey as cs  # noqa: E402

HEC = f"{P}/_fl/ckpt/hecate"
OUT = f"{P}/_fl/pred/hecate3d"
CASES = ("control", "lead", "neg_a", "neg_b")


def run_3d(name):
    vol = f"{P}/_fl/work/hecate/{name}_9.6.zarr"
    out = f"{OUT}/{name}_forward.zarr"
    if not os.path.isdir(out):
        t = time.time()
        with cs.GpuLock():                     # share the GPU with the running survey
            rc = subprocess.call([cs.PY, f"{HEC}/hecate.py", "--checkpoint", f"{HEC}/hecate_9.6um.pth",
                                  "--input", vol, "--spacing-um", "9.6", "--output-3d", out,
                                  "--device", "cuda", "--precision", "bf16", "--batch-size", "16",
                                  "--stride", "64"],
                                 stdout=open(f"{OUT}/{name}.log", "w"), stderr=subprocess.STDOUT)
        if rc != 0:
            raise RuntimeError(f"hecate 3d {name} rc={rc}")
        print(f"{name}: 3D volume in {time.time() - t:.0f}s", flush=True)
    return vol, out


def fwhm(profile):
    p = profile - profile.min()
    pk = int(np.argmax(p))
    half = p[pk] / 2.0
    lo = hi = pk
    while lo > 0 and p[lo] > half:
        lo -= 1
    while hi < len(p) - 1 and p[hi] > half:
        hi += 1
    return pk, hi - lo


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for name in CASES:
        vol, out3d = run_3d(name)
        ct = zarr.open_group(vol, mode="r")["0"]
        ink = zarr.open_array(out3d, mode="r")
        z0, z1 = zarr.open_array(out3d, mode="r").attrs["evaluated_z_interval"]
        hot2d = cv2.imread(f"{P}/_fl/pred/hecate/{name}_forward.png", cv2.IMREAD_UNCHANGED)
        if hot2d is None:
            print(name, "2D map missing"); continue
        hot = hot2d > 191                                   # >0.75 probability
        if hot.sum() < 2000:
            print(name, "too few hot pixels"); continue
        ys, xs = np.nonzero(hot)
        take = np.linspace(0, len(ys) - 1, min(40000, len(ys))).astype(int)
        ys, xs = ys[take], xs[take]
        # read the two volumes plane by plane over the evaluated window
        ct_prof, ink_prof = [], []
        for z in range(z0, z1):
            ct_prof.append(float(np.asarray(ct[z])[ys, xs].mean()))
            ink_prof.append(float(np.asarray(ink[z])[ys, xs].mean()) / 255.0)
        ct_prof, ink_prof = np.array(ct_prof), np.array(ink_prof)
        ct_pk, ct_w = fwhm(ct_prof)
        ink_pk, ink_w = fwhm(ink_prof)
        res[name] = dict(evaluated_z=[z0, z1], n_hot_pixels_sampled=int(len(ys)),
                         ct_peak_plane=ct_pk, ct_fwhm_um=round(ct_w * 9.6, 1),
                         ink_peak_plane=ink_pk, ink_fwhm_um=round(ink_w * 9.6, 1),
                         offset_um=round((ink_pk - ct_pk) * 9.6, 1),
                         ink_max=round(float(ink_prof.max()), 4),
                         ct_profile=[round(v, 1) for v in ct_prof],
                         ink_profile=[round(v, 4) for v in ink_prof])
        print(f"{name:8s} ink peak plane {ink_pk} vs CT sheet peak {ct_pk} "
              f"-> offset {res[name]['offset_um']:+.0f} um | ink layer FWHM {res[name]['ink_fwhm_um']:.0f} um "
              f"(sheet {res[name]['ct_fwhm_um']:.0f} um) | peak p {res[name]['ink_max']:.3f}", flush=True)
        json.dump(res, open(f"{P}/_fl/hecate_3d.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
