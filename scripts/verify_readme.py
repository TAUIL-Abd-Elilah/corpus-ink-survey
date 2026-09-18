"""Re-derive the README's numbers from data/ and check its claims still hold.

This repository has published several statements it later had to correct (see the README's corrections
log). Every figure on the page is recomputed here from the JSON in data/ and compared with the literal
string in README.md, and the claims the text rests on -- the candidate is the highest window, every mesh
points toward the core, each correction is logged -- are checked against the data too. Exit code 1 on any
mismatch.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = open(os.path.join(HERE, "README.md"), encoding="utf-8").read()
load = lambda p: json.load(open(os.path.join(HERE, "data", p)))
fails, checks = [], 0


def ok(label, cond, detail=""):
    global checks
    checks += 1
    print("  %-60s %s" % (label, "ok" if cond else "FAIL " + detail))
    if not cond:
        fails.append(label)


def has(label, needle):
    ok(label, needle in README, "missing " + repr(needle[:70]))


def f4(x):
    return "%.4f" % x


def main():
    S = load("summary_v3.json")
    ink, hw = S["ink"], S["hecate_window"]
    v3 = load("ink_v3_footprint.json")

    print("survey (ink_9um on the mesh footprint)")
    ok("summary matches ink_v3_footprint.json mesh count", ink["n_meshes"] == len(v3))
    total = sum((r.get("forward") or {}).get("area_cm2", 0) for r in v3.values())
    ok("total footprint area re-derived", abs(total - ink["total_cm2"]) < 0.06, f"{total:.1f} vs {ink['total_cm2']}")
    has("mesh count", f"**{ink['n_meshes']}**")
    has("total area", f"{ink['total_cm2']:,.1f}")
    for s, (n, a) in ink["per_scroll"].items():
        got_n = sum(1 for r in v3.values() if r["scroll"] == s)
        got_a = sum((r.get("forward") or {}).get("area_cm2", 0) for r in v3.values() if r["scroll"] == s)
        ok(f"{s} re-derived", got_n == n and abs(got_a - a) < 0.06)
        has(f"{s} row", f"| {s} | {n} | {a:,.1f} |")
    pairs = sorted((s["vs_control"], k, d) for k, r in v3.items() for d in ("forward", "reverse")
                   if (s := r.get(d)) and s.get("vs_control") is not None)
    ok("top list re-derived from v3", [p[1] for p in pairs[:6]] == [t[1] for t in ink["top"]])
    for i, (x, k, d, r) in enumerate(ink["top"]):
        has(f"rank {i + 1} row", f"| {i + 1} | `{k}` | {d} | {x:.1f}x | {r} |")
    has("median below control", f"**{ink['fwd_vs_control_median']:.0f}x**")
    has("nothing above 0.75", f"{ink['meshes_nothing_above_075']} meshes have\nnothing above 0.75")

    print("\nhecate calibration")
    s64 = load("hecate/eval_stride64.json")
    for case in ("control", "neg_a", "neg_b"):
        has(f"{case} fwd >0.75", f4(s64[case]["forward"]["gt075"]))
        has(f"{case} rev >0.75", f4(s64[case]["reverse"]["gt075"]))
        has(f"{case} fwd >0.5", f4(s64[case]["forward"]["gt05"]))

    print("\nfair comparison")
    wm = load("hecate/window_max.json")
    ok("window file matches summary", wm["n_meshes"] == hw["n"] and abs(wm["candidate_window_gt075"] - hw["candidate"]) < 1e-9)
    ok("candidate is the highest window (either direction)", hw["candidate"] > wm["corpus_best_window_max"],
       f"{hw['candidate']} vs {wm['corpus_best_window_max']}")
    fwd = sorted(((x.get("forward_gt075") or 0) for x in wm["per_mesh"].values()), reverse=True)
    ok("candidate is the highest forward window", hw["candidate"] > fwd[0])
    has("candidate window", f4(hw["candidate"]))
    has("median", f4(hw["median"]))
    has("p90", f4(hw["p90"]))
    for k, v, d in hw["top5"]:
        has(f"{k} row", f"| `{k}` | {d} | {f4(v)} |")

    print("\norientation")
    mo = load("hecate/mesh_orientation.json")
    ok("no mesh points outward", mo["n_outward"] == 0)
    has("all inward stated", f"all {mo['n_inward']} of {mo['n_meshes']} corpus meshes")
    has("control outward share", "%.3f" % mo["control_outward_share"])
    has("hecate reverse tilt", "median ratio %.2f" % mo["hecate"]["median_rev_over_fwd"])

    print("\ncandidate 1")
    c = ink["candidate"]
    has("fwd >0.75", "%.5f" % c["forward"]["unanimous_gt075"])
    has("fwd below control", f"**{c['forward']['vs_control']}x**")
    has("rev below control", f"{c['reverse']['vs_control']}x")
    mc = load("hecate/matched_centring.json")
    for case in ("control", "lead", "neg_a", "neg_b"):
        has(f"centring row {case}", " | ".join("%.4f" % b["fwd_gt075"] for b in mc[case]["bins"]))
    lc = load("hecate/lead_continuity.json")
    col = [lc[m]["hot_fraction_in_column"] for m in ("PHerc0813_z12496_w060", "PHerc0813_z11904_w060", "PHerc0813_z13088_w060")]
    ok("column fades below and above", col[0] > col[1] > col[2])
    for x in col:
        has(f"column {x}", "%.4f" % x)
    d3 = load("hecate/depth_3d.json")
    ok("3D offsets all within 30 um", all(abs(d3[k]["offset_um"]) <= 30 for k in d3))
    for k in ("control", "lead", "neg_a", "neg_b"):
        has(f"3D {k} offset", "%+.0f um" % d3[k]["offset_um"])

    print("\nclaims and corrections")
    for label, needle in [
        ("candidate, not discovery", "a candidate location, not a discovery"),
        ("no letterforms", "There are no letterforms"),
        ("ink, stain, glue", "Ink, stain and glue all remain consistent"),
        ("3D does not separate", "does not separate the\ncandidate from blank papyrus"),
        ("candidate 2 structural reading", "more like a structural\nfeature"),
        ("tiling disclosed", "tiles do not overlap"),
        ("log: mask v1", "the coverage mask dropped blank papyrus"),
        ("log: control basis", "the control was on the wrong basis"),
        ("log: four-check rule", "the \"four-check rule\" is withdrawn"),
        ("log: false positive retracted", "had been called a false positive"),
        ("log: 5.8x", "\"5.8x the corpus maximum\""),
        ("log: 3.3x", "\"3.3x the best corpus window\""),
        ("log: orientation", "was false. Measured,\n   every mesh has the same orientation"),
        ("log: verifier", "failed 8 of its checks"),
        ("log: padding", "the fixed coverage mask counted tile padding"),
    ]:
        has(label, needle)
    ok("retracted rationale not asserted", "Which face a mesh's \"forward\" points at depends" not in README)

    print("\n%d checks, %d failed" % (checks, len(fails)))
    for f in fails:
        print("  FAILED:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
