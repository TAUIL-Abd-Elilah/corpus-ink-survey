"""Recompute data/hecate/corpus_baseline.json from the survey score files.

The survey (corpus_survey.py --hecate) writes each mesh's hecate scores next to its ink_9um scores.
This collects the forward >0.75 fractions so the corpus baseline can be checked against the claim in
the README: the strongest region reads far above every other mesh.
"""
import glob, json, os, sys
import numpy as np

P = os.environ.get("SURVEY_ROOT", r"D:/Competition/Vesuvius progress prizes")
CONTROL, LEAD = 0.06847, 0.13525      # hecate forward >0.75, stride 64 (data/hecate/eval_stride64.json)


def main():
    rows = {}
    for f in glob.glob(f"{P}/_fl/corpus_scores*.json"):
        if "merged" in f or "badmask" in f:
            continue
        rows.update(json.load(open(f)))
    hec = {k: v["hecate"]["forward"]["gt075"] for k, v in rows.items()
           if v.get("status") == "done" and v.get("scoring") == "v2"
           and isinstance(v.get("hecate"), dict)
           and (v["hecate"].get("forward") or {}).get("gt075") is not None}
    if not hec:
        print("no hecate scores found under", P)
        return 1
    h = np.array(list(hec.values()))
    out = dict(n_meshes=len(hec), median=round(float(np.median(h)), 5),
               p90=round(float(np.percentile(h, 90)), 5), max=round(float(h.max()), 5),
               max_mesh=max(hec, key=hec.get),
               settings="stride 64, batch 16, bf16; support = CT in the 16 central planes eroded 64 px",
               control_gt075=CONTROL, lead_gt075=LEAD, per_mesh=dict(sorted(hec.items())))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "data", "hecate", "corpus_baseline.json"), "w"), indent=1)
    print(f"{len(hec)} meshes: median {out['median']}, p90 {out['p90']}, max {out['max']} "
          f"({out['max_mesh']}); control {CONTROL}, strongest region {LEAD} "
          f"= {LEAD / out['max']:.1f}x the corpus max")
    return 0


if __name__ == "__main__":
    sys.exit(main())
