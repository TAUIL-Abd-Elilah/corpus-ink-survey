"""Re-assert the README's hecate numbers against the JSON in data/hecate/.

This repository has already published one overstated claim (a "false positive" that the data did not
support), so every figure in the hecate section is recomputed here and compared with the literal
string in README.md. Exit code 1 if anything disagrees.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = open(os.path.join(HERE, "README.md"), encoding="utf-8").read()
D = os.path.join(HERE, "data", "hecate")
load = lambda n: json.load(open(os.path.join(D, n)))
fails, checks = [], 0


def present(label, needle):
    global checks
    checks += 1
    ok = needle in README
    print("  %-56s %s" % (label, "ok" if ok else "MISSING " + repr(needle)))
    if not ok:
        fails.append(label)


def num(label, value, decimals=4):
    """The README must contain this number, printed to `decimals`."""
    present(label, ("%." + str(decimals) + "f") % value)


def main():
    global checks
    s64, s32 = load("eval_stride64.json"), load("eval_stride32.json")
    mc, lc, bl = load("matched_centring.json"), load("lead_continuity.json"), load("corpus_baseline.json")

    print("calibration (stride 64, the survey's setting)")
    for case in ("control", "lead", "neg_a", "neg_b"):
        for d in ("forward", "reverse"):
            num(f"{case} {d} >0.75", s64[case][d]["gt075"])
        num(f"{case} forward >0.5", s64[case]["forward"]["gt05"], 4)
        present(f"{case} forward ratio", str(s64[case]["forward"]["ratio"]))

    print("\nink_9um on the same windows (stride-32 run recorded both)")
    for case, expect in (("neg_a", 49), ("neg_b", 205)):
        got = s32[case]["ink_9um_same_window"]["forward"]["vs_control"]
        checks_ok = round(got) == expect
        print("  %-56s %s" % (f"{case} ink_9um {got}x -> README {expect}x", "ok" if checks_ok else "MISMATCH"))
        if not checks_ok:
            fails.append(f"{case} ink_9um vs_control")
        present(f"{case} {expect}x in README", f"{expect}x below")

    print("\nmatched sheet centring (stride 32)")
    for case in ("control", "lead", "neg_a", "neg_b"):
        for b in mc[case]["bins"]:
            v = b["fwd_gt075"]
            # the table prints 3 or 4 decimals depending on the value; accept either rounding
            forms = {"%.4f" % v, "%.3f" % v, str(round(v, 4)), str(round(v, 3))}
            checks += 1
            ok = any(f in README for f in forms)
            print("  %-56s %s" % (f"{case} bin {v}", "ok" if ok else "MISSING one of " + repr(sorted(forms))))
            if not ok:
                fails.append(f"{case} bin {v}")

    print("\ncontinuity in the band's own column, wrap w060")
    for mesh in ("PHerc0813_z12496_w060", "PHerc0813_z11904_w060", "PHerc0813_z13088_w060"):
        num(f"{mesh} hot fraction", lc[mesh]["hot_fraction_in_column"])

    print("\ncorpus baseline")
    present("n meshes stated", f"{bl['n_meshes']} of 327")
    num("median", bl["median"])
    num("p90", bl["p90"])
    num("max", bl["max"])
    wm = load("window_max.json")
    print("\nfair comparison: every mesh at its densest same-width window, either direction")
    present("snapshot size", f"{wm['n_meshes']} meshes (18 September")
    num("window median", wm["corpus_best_window_median"])
    num("window p90", wm["corpus_best_window_p90"])
    num("candidate window", wm["candidate_window_gt075"])
    top = wm["top5"]
    for name, val, d in top[:2]:
        present(f"{name} {d} named", name)
        num(f"{name} value", val)
    present("everything else bounded", "at most %.4f" % top[2][1])
    global_ok = wm["candidate_window_gt075"] > top[0][1]
    checks_line = "candidate is the highest window" if global_ok else "candidate is NOT the highest window"
    print("  %-56s %s" % (checks_line, "ok" if global_ok else "MISMATCH"))
    if not global_ok:
        fails.append("candidate is not the highest window any more -- README says it is")
    present("not alone stated", "it is not alone")
    present("both overstatements named", '"5.8x the corpus maximum"')
    present("forward-only overstatement named", '"3.3x the best corpus window" counted forward maps only')
    present("second region is a candidate, not a result", "a second candidate, not a result")
    present("tiling artifact disclosed", "tiles do not overlap")
    checks += 1
    if wm["candidate_window_gt075"] <= bl["control_gt075"]:
        fails.append("candidate window is not above known ink")
        print("  candidate window %.4f vs control %.4f" % (wm["candidate_window_gt075"], bl["control_gt075"]))
    else:
        print("  %-56s ok" % "candidate window above known ink")

    print("\n3D depth (the test that did not separate the cases)")
    d3 = load("depth_3d.json")
    for case in ("control", "lead", "neg_a", "neg_b"):
        present(f"{case} offset", "%+.0f um" % d3[case]["offset_um"])
        present(f"{case} ink FWHM", "%.0f um" % d3[case]["ink_fwhm_um"])
        present(f"{case} peak p", "%.2f" % d3[case]["ink_max"])
    present("3D does not separate", "does not separate the candidate from blank papyrus")

    print("\nclaims that must stay in the README")
    for label, needle in [("candidate not discovery", "not a discovery"),
                          ("no letterforms", "no\nletterforms"),
                          ("known ink shows none either", "known ink at 9 um shows no letterforms either"),
                          ("glue and stain still possible", "Ink, stain and glue are all\nstill consistent"),
                          ("not a seam", "should\nnot fade 5x below and 35x above"),
                          ("resampling requirement", "resampled to 9.6 um in all three axes")]:
        present(label, needle)

    print("\n%d checks, %d failed" % (checks, len(fails)))
    for f in fails:
        print("  FAILED:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
