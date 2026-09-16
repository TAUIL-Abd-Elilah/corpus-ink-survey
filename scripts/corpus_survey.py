"""Ink-survey the published eligible-scroll mesh corpus, best-aligned surfaces first.

Input is pscamillo/vesuvius-eligible-meshes (MIT, 1 September 2026): 340 tifxyz meshes,
1,935 cm^2, eight of the thirteen originally First-Letters-eligible volumes.

**Ordering is by measured alignment, not by any proxy.** `_fl/mesh_alignment.json` carries the
angle between each mesh's normal and the local sheet normal, measured for 338 of the 340. Two
things follow:

  - meshes at >= 30 deg are **skipped**. On PHerc1447 every published surface sits at 42-81 deg
    and the ink signal there is uninterpretable -- a surface that shears across windings feeds
    the model a blend of sheet and gap, which suppresses signal for reasons unrelated to ink.
    11 meshes / 77 cm^2 are dropped on this rule.
  - the rest run in ascending angle, ties broken by descending area, so if the run is cut short
    the surfaces most likely to carry a fair test are already done.

This replaces the corpus author's eye verdict as the ranking key, which the alignment
measurement showed does not predict geometry: `reprova` meshes median 9.4 deg and 30/30 within
30 deg, against `aprova` at 5.3 deg.

Per mesh: render 31 layers with the validated renderer, gate on CT support (villa #1254 -- a
surface on zero CT returns a constant map that reads as "no ink" and means the opposite), run
the ink checkpoints, **score immediately**, then delete. Disk is the binding constraint at ~5 GB
free: 327 meshes x 8 prediction TIFFs would be ~9 GB, so predictions are kept only for meshes
that score interestingly and discarded otherwise. Scores are appended to
`_fl/corpus_scores.json` as they are produced, so the run is resumable and partial results are
never lost.
"""
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np
import tifffile
import zarr
import cv2

P = r"D:/Competition/Vesuvius progress prizes"
PY = r"C:/Users/PC/miniconda3/envs/vesuvius/python.exe"
ENV = dict(os.environ, PYTHONPATH=rf"{P}/villa/_worktrees/ink9um/vesuvius/src")
CORPUS = rf"{P}/_fl/meshes_eligible"
BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
ALIGN = rf"{P}/_fl/mesh_alignment.json"

CKPTS = [("42", "030000"), ("42", "075000"), ("43", "030000"), ("43", "075000")]

VOLUMES = {
    "PHerc0125": "20250821151825-9.362um-1.2m-113keV-masked.zarr",
    "PHerc0211": "20250821151803-9.362um-1.2m-113keV-masked.zarr",
    "PHerc0257": "20250821151750-9.362um-1.2m-113keV-masked.zarr",
    "PHerc0268": "20251110183117-8.640um-1.2m-116keV-masked.zarr",
    "PHerc0358": "20250821151737-9.362um-1.2m-113keV-masked.zarr",
    "PHerc0800": "20250521135224-8.640um-1.2m-116keV-masked.zarr",
    "PHerc0813": "20250821151723-9.362um-1.2m-113keV-masked.zarr",
    "PHerc0826": "20250821151701-9.362um-1.2m-113keV-masked.zarr",
}

# firstletters.md section 4.3 standing rule, and the same-day control it is read against
ERODE_PX = 64
# PHerc0139 w043 scored by score() itself at the SAME 4 checkpoints as CKPTS (fixed mask).
# Until 2026-09-16 this held the 8-checkpoint figure (0.03694), which is not comparable.
CONTROL = {"unanimous_gt05": 0.08834, "unanimous_gt075": 0.04269, "conf_ratio": 2.07}


def voxel_um(scroll):
    return 8.64 if "8.640um" in VOLUMES[scroll] else 9.362


def rescale(p):
    """ink_9um trains with BCE label smoothing 0.5, so confident no-ink sits at 0.25, not 0."""
    return np.clip((p.astype(np.float32) / 255.0 - 0.25) / 0.5, 0.0, 1.0)


def score(pred_paths, vox):
    """Unanimous minimum across checkpoints, over covered pixels eroded by the model's radius."""
    R = np.stack([tifffile.imread(f) for f in sorted(pred_paths)])
    S = np.stack([rescale(r) for r in R])
    # Coverage is where the model produced output (raw > 0). It must not be taken after rescale:
    # rescale maps every confident no-ink value (<= 0.25) to 0, so `S.max(0) > 0` dropped blank
    # papyrus from the denominator and the 64 px erosion then cut square holes around it -- 37% of
    # the known-ink control's area. Fixed 2026-09-16; scores before that date used the bad mask.
    cov = R.max(0) > 0
    k = 2 * ERODE_PX + 1
    cov = cv2.erode(cov.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    if cov.sum() < 1000:
        return None
    mn = S.min(0)
    v = mn[cov]
    g05 = float((v > 0.5).mean())
    g075 = float((v > 0.75).mean())
    # With no pixel above 0.75 both derived numbers are undefined, not astronomical: a
    # 1e-9 floor turns "nothing here" into ratios of 10^5 and "36,940,000x below control",
    # which are noise dressed as measurements. Report null and let the caller say "none".
    return dict(area_cm2=round(float(cov.sum() * (vox / 10000.0) ** 2), 2),
                unanimous_gt05=round(g05, 5), unanimous_gt075=round(g075, 5),
                conf_ratio=round(g05 / g075, 2) if g075 > 0 else None,
                vs_control=round(CONTROL["unanimous_gt075"] / g075, 1) if g075 > 0 else None)


def interesting(sc):
    """Keep the prediction TIFFs only when a mesh could plausibly carry text.

    Two ways in: high-confidence coverage within 3x of the known-ink control, or a
    confidence-ratio profile close to the control's 2.07 (real ink is crisp -- what clears 0.5
    mostly clears 0.75 -- while fibre and kolleisis are diffuse and run 4-15).
    """
    if sc is None or sc["vs_control"] is None:
        return False
    return sc["vs_control"] < 3.0 or sc["conf_ratio"] < 3.2


def plan(limit, scrolls, max_angle):
    align = json.load(open(ALIGN))
    rows = []
    for name, v in align.items():
        if v.get("status") != "ok" or v.get("median_angle_deg", 99) >= max_angle:
            continue
        if scrolls and v["scroll"] not in scrolls:
            continue
        rows.append(dict(name=name, scroll=v["scroll"], mesh=v["mesh"], wrap=v["wrap"],
                         z=v["z"], area=v["area_cm2"], angle=v["median_angle_deg"]))
    rows.sort(key=lambda r: (r["angle"], -r["area"]))
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="corpus")
    ap.add_argument("--scrolls", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-angle", type=float, default=30.0)
    ap.add_argument("--layers", type=int, default=31)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min-support", type=float, default=0.15)
    # A render can consume ~2 GB while it runs, so a 2 GB floor checked *before* starting lets
    # the write cross it mid-flight and die with rc=1. That is what produced 19 spurious
    # render_fail entries on 2026-09-14 as the disk filled. Leave room for a render plus slack.
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-attempt meshes recorded render_fail (they are usually "
                         "environmental -- disk or network -- not bad geometry)")
    ap.add_argument("--shard", default="0/1",
                    help="i/N -- run only meshes whose rank %% N == i. The render is CPU-bound "
                         "and single-threaded (0.3%% chunk-miss rate on a 16-core box), so "
                         "several shards in parallel multiply throughput almost linearly. "
                         "Each shard writes to its own score file; merge with --tag.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescore", action="store_true",
                    help="re-run meshes whose scores predate the 2026-09-16 coverage-mask fix")
    ap.add_argument("--keep-all", action="store_true",
                    help="keep every prediction TIFF, so any mesh can be re-scored later")
    a = ap.parse_args()

    rows = plan(a.limit, [s for s in a.scrolls.split(",") if s], a.max_angle)
    si, sn = (int(x) for x in a.shard.split("/"))
    if sn > 1:
        rows = [r for k, r in enumerate(rows) if k % sn == si]
    print(f"shard {a.shard}: {len(rows)} meshes, {sum(r['area'] for r in rows):.0f} cm2, "
          f"alignment {rows[0]['angle']:.1f}-{rows[-1]['angle']:.1f} deg", flush=True)
    if a.dry_run:
        for r in rows[:20]:
            print(f"  {r['scroll']:11s} z{r['z']:<6s} {r['wrap']:5s} {r['area']:5.2f} cm2 "
                  f"{r['angle']:5.1f} deg", flush=True)
        return 0

    outdir = f"{P}/_fl/pred/{a.tag}"
    workdir = f"{P}/_fl/work/{a.tag}"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(workdir, exist_ok=True)
    scorep = (f"{P}/_fl/corpus_scores.json" if sn == 1
              else f"{P}/_fl/corpus_scores_{si}of{sn}.json")
    done = json.load(open(scorep)) if os.path.exists(scorep) else {}
    if a.retry_failed:
        drop = [k for k, v in done.items() if v.get("status") == "render_fail"]
        for k in drop:
            del done[k]
        print(f"retrying {len(drop)} previously failed meshes", flush=True)
    if a.rescore:
        # Scores written before the 2026-09-16 mask fix cannot be recomputed: their predictions were
        # deleted. Re-run those meshes end to end. The old file is backed up by the caller.
        stale = [k for k, v in done.items() if v.get("status") == "done" and v.get("scoring") != "v2"]
        for k in stale:
            del done[k]
        print(f"re-running {len(stale)} meshes scored with the pre-fix mask", flush=True)

    t_start = time.time()
    for n, r in enumerate(rows, 1):
        name = r["name"]
        head = (f"[{n}/{len(rows)}] {name:26s} {r['area']:5.2f} cm2 {r['angle']:4.1f} deg")
        if name in done:
            continue
        free_gb = shutil.disk_usage(P).free / 1e9
        if free_gb < a.min_free_gb:
            print(f"{head}: STOPPING -- only {free_gb:.1f} GB free", flush=True)
            break

        src = f"{CORPUS}/{r['mesh']}"
        crop, rz = f"{workdir}/{name}", f"{workdir}/{name}.zarr"
        shutil.rmtree(crop, ignore_errors=True)
        os.makedirs(crop, exist_ok=True)
        for nm in ("x", "y", "z"):
            shutil.copy(f"{src}/{nm}.tif", f"{crop}/{nm}.tif")
        meta = json.load(open(f"{src}/meta.json"))
        json.dump({"format": "tifxyz", "type": "seg", "scale": meta["scale"], "uuid": name},
                  open(f"{crop}/meta.json", "w"), indent=1)
        url = f"{BUCKET}/{r['scroll']}/volumes/{VOLUMES[r['scroll']]}"
        open(f"{crop}/volume_source.txt", "w").write(url + "\n")

        t = time.time()
        rc = subprocess.call([PY, f"{P}/_fl/render_tifxyz.py", crop, rz,
                              "--layers", str(a.layers), "--tile", "384",
                              "--workers", str(a.workers), "--cache-chunks", "900"],
                             stdout=open(f"{workdir}/{name}.render.log", "w"),
                             stderr=subprocess.STDOUT, env=ENV)
        if rc != 0:
            print(f"{head}: RENDER FAIL rc={rc}", flush=True)
            done[name] = dict(status="render_fail", rc=rc, **r)
            json.dump(done, open(scorep, "w"), indent=1)
            shutil.rmtree(crop, ignore_errors=True); shutil.rmtree(rz, ignore_errors=True)
            continue

        try:
            arr = zarr.open_group(rz, mode="r")["0"]
            mid = np.asarray(arr[arr.shape[0] // 2])
            support = float((mid > 0).mean())
        except Exception:
            support = -1.0
        if support < a.min_support:
            print(f"{head}: no CT support ({support:.3f}) -- skipped (villa #1254)", flush=True)
            done[name] = dict(status="no_ct_support", support=round(support, 4), **r)
            json.dump(done, open(scorep, "w"), indent=1)
            shutil.rmtree(crop, ignore_errors=True); shutil.rmtree(rz, ignore_errors=True)
            continue

        preds = []
        for sd, st in CKPTS:
            out = f"{outdir}/{name}_s{sd}_{st}.tif"
            subprocess.call([PY, "-m", "vesuvius.ink_detection.inference.infer", rz,
                             f"{P}/_fl/ckpt/hybrid_3d2d-seed{sd}/step-{st}.pth", out,
                             "--overlap", "0.5", "--blend-mode", "hann", "--batch-size", "4",
                             "--direction", "both", "--no-compile"],
                            stdout=open(f"{workdir}/{name}.ink.log", "a"),
                            stderr=subprocess.STDOUT, env=ENV)
            preds += [p for p in (out, out.replace(".tif", "_reverse.tif")) if os.path.exists(p)]
        shutil.rmtree(rz, ignore_errors=True)
        shutil.rmtree(crop, ignore_errors=True)

        vox = voxel_um(r["scroll"])
        fwd = [p for p in preds if not p.endswith("_reverse.tif")]
        rev = [p for p in preds if p.endswith("_reverse.tif")]
        rec = dict(status="done", scoring="v2", support=round(support, 4),
                   render_s=round(time.time() - t), **r)
        keep = False
        for lab, group in (("forward", fwd), ("reverse", rev)):
            if len(group) < 2:
                continue
            sc = score(group, vox)
            rec[lab] = sc
            keep = keep or interesting(sc)
        done[name] = rec
        json.dump(done, open(scorep, "w"), indent=1)

        if not keep and not a.keep_all:
            for p in preds:
                try:
                    os.remove(p)
                except OSError:
                    pass
        f = rec.get("forward") or {}
        g = f.get("unanimous_gt075")
        if g:
            tail = f"ratio {f['conf_ratio']:.2f} ({f['vs_control']:.1f}x below control)"
        else:
            tail = "nothing above 0.75"
        print(f"{head}: >0.75 {g if g is not None else float('nan'):.5f} {tail}"
              f"{'  KEPT' if keep else ''}  [{(time.time() - t_start) / 60:.0f} min]", flush=True)

    okd = [v for v in done.values() if v.get("status") == "done"]
    area = sum((v.get("forward") or {}).get("area_cm2", 0) for v in okd)
    print(f"\n{len(okd)} meshes surveyed, {area:.0f} cm2 scored -> {scorep}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
