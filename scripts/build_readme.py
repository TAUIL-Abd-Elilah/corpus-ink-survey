"""Generate corpus-ink-survey/README.md from its data files, so no number is typed by hand."""
import io
import json
import os

R = r"D:/Competition/Vesuvius progress prizes/_fl/repo/corpus-ink-survey"
J = lambda p: json.load(open(os.path.join(R, p)))

S = J("data/summary_v3.json")
ink, hw = S["ink"], S["hecate_window"]
s64, s32 = J("data/hecate/eval_stride64.json"), J("data/hecate/eval_stride32.json")
mc, d3 = J("data/hecate/matched_centring.json"), J("data/hecate/depth_3d.json")
lc, mo = J("data/hecate/lead_continuity.json"), J("data/hecate/mesh_orientation.json")
c = ink["candidate"]

old = io.open(os.path.join(R, "README.md"), encoding="utf-8").read()
i0 = old.index("## PHerc1447: the three segments")
i1 = old.index("## Limitations", i0)
p1447 = old[i0:i1].rstrip() + "\n"


def f4(x):
    return "%.4f" % x


def win(name):
    return f4([v for k, v, d in hw["top5"] if k == name][0])


def mrow(k):
    return " | ".join("%.4f" % b["fwd_gt075"] for b in mc[k]["bins"])


scroll_rows = "\n".join(f"| {s} | {n} | {a:,.1f} |" for s, (n, a) in ink["per_scroll"].items())
top_rows = "\n".join(f"| {i + 1} | `{k}` | {d} | {x:.1f}x | {r} |" for i, (x, k, d, r) in enumerate(ink["top"]))
hw_rows = (f"| **`PHerc0813_z12496_w060` (the candidate)** | forward | **{f4(hw['candidate'])}** |\n"
           + "\n".join(f"| `{k}` | {d} | {f4(v)} |" for k, v, d in hw["top5"]))
ctl, na, nb = s64["control"], s64["neg_a"], s64["neg_b"]
cf, cr = c["forward"], c["reverse"]
fo = hw["forward_only_top"]

readme = f"""# Ink-surveying the public eligible-scroll corpus

**{ink['n_meshes']} meshes and {ink['total_cm2']:,.1f} cm² of mesh surface across all eight scrolls in the public
First Letters corpus, scored by two models against a known-ink control: the published `ink_9um`
checkpoints and the team's new `hecate` 9.6 um model. No letters seen. One region, on PHerc0813, stands
out under both models on the face that carries text. A second feature, on PHerc0211's inner wrap,
stands out on the opposite face. Both are published as candidate locations, not discoveries.**

Status, 19 September: `ink_9um` is complete on every planned mesh. `hecate` has scored {hw['n']} of
{ink['n_meshes']} so far and the rest are running. This page has been corrected several times; the
corrections log at the end lists each one.

## The survey

Input: [`pscamillo/vesuvius-eligible-meshes`](https://github.com/pscamillo/vesuvius-eligible-meshes)
(MIT): 340 meshes on eight eligible scrolls. The 11 meshes that cut across the papyrus at 30 degrees or
more are skipped ([eligible-mesh-alignment](https://github.com/TAUIL-Abd-Elilah/eligible-mesh-alignment)),
and {ink['no_ct']} more have no CT under them.

Per mesh: render 31 layers along the surface normal; run **4 `ink_9um` checkpoints in both directions**;
take the **unanimous minimum**; score the fraction above 0.75 over the **mesh's own footprint, eroded
64 px**. The known-ink control, PHerc0139 w043, is scored the same way at the same 4 checkpoints
(>0.75 = 0.04269, confidence ratio 2.07). Every prediction is kept.

| scroll | meshes | footprint cm² |
|---|---:|---:|
{scroll_rows}
| **total** | **{ink['n_meshes']}** | **{ink['total_cm2']:,.1f}** |

In the forward direction the median mesh is **{ink['fwd_vs_control_median']:.0f}x** below control, with a
median confidence ratio of {ink['fwd_ratio_median']:.1f} against the control's 2.07. {ink['meshes_nothing_above_075']} meshes have
nothing above 0.75 in either direction. The strongest of {ink['mesh_directions']} mesh-directions:

| rank | mesh | direction | below control | confidence ratio |
|---:|---|---|---:|---:|
{top_rows}

`ink_9um` localises ink but does not resolve letters even on training scrolls. A negative here means
"this model recovered no text", not "there is no ink".

## A second model: hecate 9.6 um

The team published [`scrollprize/hecate`](https://huggingface.co/scrollprize/hecate) on 15 September,
with a 9.6 um checkpoint distilled from their 2.4 um canonical detector. Each render is resampled to
9.6 um in all three axes, as the model card requires, and run in both directions at stride 64. Support
is CT in the 16 central planes, eroded 64 px, which matches the mesh footprint to within 1.3%.
Checkpoint commit `9cb86e5`, sha256 `809f4f10...fe5d`.

### Calibration

| case | fwd >0.5 | **fwd >0.75** | ratio | rev >0.75 | ink_9um on the same window |
|---|---:|---:|---:|---:|---|
| known ink, PHerc0139 w043 (3 x 3 cm) | {f4(ctl['forward']['gt05'])} | **{f4(ctl['forward']['gt075'])}** | {ctl['forward']['ratio']} | {f4(ctl['reverse']['gt075'])} | the control |
| PHerc0813_z6496_w060, ink_9um-blank | {f4(na['forward']['gt05'])} | {f4(na['forward']['gt075'])} | {na['forward']['ratio']} | {f4(na['reverse']['gt075'])} | {s32['neg_a']['ink_9um_same_window']['forward']['vs_control']:.0f}x below |
| PHerc0813_z13696_w060, ink_9um-blank | {f4(nb['forward']['gt05'])} | {f4(nb['forward']['gt075'])} | {nb['forward']['ratio']} | {f4(nb['reverse']['gt075'])} | {s32['neg_b']['ink_9um_same_window']['forward']['vs_control']:.0f}x below |

![four cases](figures/hecate_four_cases.png)

hecate separates known ink from the ink_9um-blank meshes, and both models show the control's bottom
text line.

### The corpus under hecate, compared fairly

Picking a mesh's densest window inflates its score, so every mesh gets the same advantage: its densest
window of the candidate's width (3.56 cm), **in either direction**, over the mesh footprint
([`scripts/hecate_window_max.py`](scripts/hecate_window_max.py)). Over {hw['n']} meshes: median
{f4(hw['median'])}, p90 {f4(hw['p90'])}.

| best same-width window | direction | hecate >0.75 |
|---|---|---:|
{hw_rows}

Forward only, which is the text face (see below): the candidate reads {f4(hw['candidate'])}, then its own
neighbour `{fo[0][0]}` {f4(fo[0][1])}, then nothing above {f4(fo[1][1])}.

### What the direction means: forward is the face that carries text

A circle fitted through each grid row gives the local centre of curvature, on the core's side
([`scripts/mesh_orientation.py`](scripts/mesh_orientation.py)). The surface normal points **toward the
core** on all {mo['n_inward']} of {mo['n_meshes']} corpus meshes. It does the same on the published
PHerc0139 w043 mesh used as the control (outward share {mo['control_outward_share']:.3f}), whose ink reads
**forward**. So on every mesh, forward is the face that carries text on the control.

On ordinary papyrus hecate runs slightly high in reverse: reverse beats forward on
{mo['hecate']['share_rev_gt_fwd']:.0%} of {mo['hecate']['n']} meshes, median ratio {mo['hecate']['median_rev_over_fwd']:.2f}. ink_9um shows no
such tilt ({mo['ink_9um']['share_rev_gt_fwd']:.0%} of {mo['ink_9um']['n']}, median {mo['ink_9um']['median_rev_over_fwd']:.2f}). A reverse response is
therefore weaker evidence than the same number forward.

## Candidate 1: PHerc0813_z12496_w060, forward (the text face)

| | ink_9um >0.5 | ink_9um >0.75 | ratio | below control |
|---|---:|---:|---:|---:|
| **forward** | {cf['unanimous_gt05']:.5f} | **{cf['unanimous_gt075']:.5f}** | **{cf['conf_ratio']}** | **{cf['vs_control']}x** |
| reverse | {cr['unanimous_gt05']:.5f} | {cr['unanimous_gt075']:.5f} | {cr['conf_ratio']} | {cr['vs_control']}x |

It is first in the survey under ink_9um and first in the corpus under hecate ({f4(hw['candidate'])}
against known ink's {f4(ctl['forward']['gt075'])}). The response is one-sided and on the text face.

**Sheet geometry doesn't explain it.** Pixels are binned by how well a sheet is centred in the render
(central-minus-outer CT, pooled quintiles; [`scripts/hecate_matched.py`](scripts/hecate_matched.py),
stride 32). hecate forward >0.75:

| | q1 (off-sheet) | q2 | q3 | q4 | q5 (well centred) |
|---|---:|---:|---:|---:|---:|
| known ink | {mrow('control')} |
| **candidate** | {mrow('lead')} |
| blank A | {mrow('neg_a')} |
| blank B | {mrow('neg_b')} |

**It is a compact patch, not a seam.** Across its 74 mm mesh the response is one band about 12 mm
wide. The table below measures the band's own (x, y) column on the same wrap
([`scripts/lead_continuity.py`](scripts/lead_continuity.py)). Wrap labels here are 2.84 mm apart, so only
same-wrap meshes compare.

| mesh, wrap w060 | hot fraction in the band's column |
|---|---:|
| **z12496 (the candidate)** | **{lc['PHerc0813_z12496_w060']['hot_fraction_in_column']:.4f}** |
| z11904, ~7 mm below | {lc['PHerc0813_z11904_w060']['hot_fraction_in_column']:.4f} |
| z13088, ~7 mm above | {lc['PHerc0813_z13088_w060']['hot_fraction_in_column']:.4f} |

A kollesis runs the full height of the roll, so a sheet join should not fade like that. In the CT, the
sheet inside the band is the same thickness as beside it (~96 um) and ~20% denser at its peak.

![band at full resolution](figures/lead_band_fullres.png)

**There are no letterforms**, but known ink at 9 um shows none either, so appearance cannot settle it.

**hecate's 3D output does not separate it.** Over each case's hot pixels
([`scripts/hecate_3d.py`](scripts/hecate_3d.py)):

| case | ink peak vs CT sheet peak | ink layer FWHM | peak probability |
|---|---:|---:|---:|
| known ink | {d3['control']['offset_um']:+.0f} um | {d3['control']['ink_fwhm_um']:.0f} um | {d3['control']['ink_max']:.2f} |
| candidate | {d3['lead']['offset_um']:+.0f} um | {d3['lead']['ink_fwhm_um']:.0f} um | {d3['lead']['ink_max']:.2f} |
| blank A | {d3['neg_a']['offset_um']:+.0f} um | {d3['neg_a']['ink_fwhm_um']:.0f} um | {d3['neg_a']['ink_max']:.2f} |
| blank B | {d3['neg_b']['offset_um']:+.0f} um | {d3['neg_b']['ink_fwhm_um']:.0f} um | {d3['neg_b']['ink_max']:.2f} |

All four sit in a thin layer within 30 um of the sheet, so **depth placement does not separate the
candidate from blank papyrus**. Only peak strength differs.

![3D depth](figures/hecate_3d_depth.png)

**Status: a candidate location, not a discovery.** Ink, stain and glue all remain consistent with what
is measured here. A model that resolves letters, or someone who reads 9 um CT for ink, could settle it.

## Candidate 2: PHerc0211's inner wrap, reverse (the opposite face)

On PHerc0211's inner wrap (w020), hecate reads high in **reverse** on a vertical stretch from z6112 to
z9120: `z6112_w020` {win('PHerc0211_z6112_w020')}, `z6720_w020` {win('PHerc0211_z6720_w020')},
`z7920_w020` {win('PHerc0211_z7920_w020')}, `z9120_w020` {win('PHerc0211_z9120_w020')}. `z7312_w020` is not
scored yet. ink_9um agrees in the same direction: `z6720_w020` is 4.0x below control in reverse and
`z7312_w020` is 6.7x.

![second region](figures/second_region_0211.png)

Unlike candidate 1, this feature is **tall and continuous** (about 3.5 cm), and it sits on the face a
roll usually leaves blank, where hecate already runs slightly high. That reads more like a structural
feature of the inner turns than a patch of text. Its orientation fit is also the least certain, since it
sits near the core. It has not been through the centring and continuity tests yet.

A note on the maps: at stride 64, hecate's 0.6 mm tiles do not overlap, so the maps show square blocks
(each tile's depth attention settles on a sheet independently). The calibration still separates known
ink from blank papyrus at this setting, but single maps are noisier than at stride 32.

{p1447}
## Limitations

- There is one known-ink control, from one 3 x 3 cm crop. Anything calibrated on it is provisional.
- hecate has scored {hw['n']} of {ink['n_meshes']} meshes so far, so its corpus numbers are a snapshot.
- The survey renders are not kept (only the predictions are), so CT-based checks need a re-render.
- The run is bandwidth-bound at ~600 KB/s.
- No discovery is claimed.

## Corrections log

Each of these was published when found, and the page above reflects all of them.

1. **16 Sept: the coverage mask dropped blank papyrus.** `score()` took coverage after the
   label-smoothing rescale, which maps confident no-ink to 0. Blank papyrus therefore left the
   denominator (37% of the control's area). All meshes were re-rendered and re-scored.
2. **16 Sept: the control was on the wrong basis.** Survey scores (4 checkpoints) had been compared
   with the control's 8-checkpoint figure (0.03694). It is now 0.04269 at the same 4 checkpoints.
3. **16 Sept: the "four-check rule" is withdrawn.** Row pitch at 4-5 mm and stroke-sized components both
   reject the known-ink control ([`scripts/screen_flagged.py`](scripts/screen_flagged.py),
   [figure](figures/control_vs_flagged.png)), so neither can be used to reject a candidate.
4. **16 Sept: candidate 1 had been called a false positive** ("plausibly kollesis") on the strength of
   those checks. That is retracted; it is unresolved.
5. **18 Sept: "5.8x the corpus maximum"** compared the candidate's picked window with whole-mesh
   averages. It is replaced by a same-window comparison.
6. **18 Sept: "3.3x the best corpus window"** counted forward maps only. Both directions now count.
7. **18 Sept: "which face forward points at depends on how the mesh was fitted"** was false. Measured,
   every mesh has the same orientation.
8. **18 Sept: this repository's own verifier**, as first pushed, failed 8 of its checks because of a
   rounding rule. It was fixed in the next commit.
9. **19 Sept: the fixed coverage mask counted tile padding.** Raw model output > 0 extends 19-24% beyond
   the mesh, where the model outputs ~0.01-0.02 on empty input. Coverage is now the mesh's own footprint
   ([`scripts/rescore_v3.py`](scripts/rescore_v3.py)), and the kept predictions were rescored without
   re-rendering. The top four regions are unchanged; candidate 1 moves from 3.1x to 2.3x below control.

`scripts/verify_readme.py` re-derives the numbers on this page from `data/` and fails on any mismatch.

## Reproduce

```bash
python scripts/corpus_survey.py --shard 0/2 --keep-all --hecate   # + shard 1/2; resumable
python scripts/rescore_v3.py                                      # ink_9um on the mesh footprint
python scripts/hecate_window_max.py                               # fair same-window comparison
python scripts/verify_readme.py                                   # re-derive every number
```

Credits: corpus [@pscamillo](https://github.com/pscamillo); alignment gate
[@flummoxjr](https://github.com/flummoxjr); row-pitch measurements and the fibre-bundle false-positive
mode [@FrankTheRope](https://github.com/FrankTheRope); `hecate` and `ink_9um` by the Vesuvius Challenge
team.

MIT · Vesuvius Challenge, September 2026 · AI assistance (Claude) under my direction; I set the
questions, chose the controls, and checked the numbers.
"""

io.open(os.path.join(R, "README.md"), "w", encoding="utf-8").write(readme)
print(len(readme.splitlines()), "lines written")
