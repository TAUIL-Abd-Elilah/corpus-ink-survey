# Ink-surveying the public eligible-scroll corpus

**1,667 cm² of the best-aligned public First Letters surface, 301 meshes across all eight corpus
scrolls, run through `ink_9um` against a known-ink control. No letters seen.**

**Being re-run.** On 16 September I found a scoring bug that affects every score-derived number
in the earlier versions of this page. Fixing it also overturned the page's headline claim. All
meshes are being re-scored from scratch; this page will be refreshed when that finishes.

## Correction — 16 September 2026 (read this first)

Four things were wrong. I found them by drawing the strongest flagged region next to the known-ink
control at the same scale, which the earlier write-up never did.

### 1. The coverage mask dropped blank papyrus

`score()` took "covered" pixels *after* the label-smoothing rescale, which maps every confident
no-ink value (≤ 0.25) to 0. So confidently blank papyrus left the denominator, and the 64 px
erosion then cut square holes around it. Coverage is now taken from the raw model output.

| | area scored | >0.75 | confidence ratio |
|---|---:|---:|---:|
| control PHerc0139 w043, old mask | 62.5% | 0.04211 | 2.23 |
| **control, fixed** | **100%** | **0.04269** | **2.07** |
| PHerc0813_z12496_w060 forward, old mask | 5.91 cm² | 0.01598 | 2.44 |
| **same, fixed** | **6.79 cm²** | **0.01392** | **2.52** |

The error does not run in one direction. On the 55 prediction sets still on disk, fixing the mask
moved high-confidence coverage by **0.73× to 2.25×**, and some ratios fell by more than 40% (12.35 →
6.81). So a mesh that was *not* flagged under the old mask is not known to be clean. Predictions for
unflagged meshes were deleted after scoring, so they cannot be re-scored in place. **All meshes are
being re-rendered and re-scored with the fix, and every prediction is now kept** (`--rescore
--keep-all`).

### 2. The control was on a different basis

Survey meshes are scored at 4 checkpoints; they were compared against the control's
**8-checkpoint** figure (0.03694). "Unanimous" over 4 is a weaker test than over 8. The control is
now scored by the same function, at the same 4 checkpoints (0.04269).

### 3. The "four-check rule" is withdrawn

The earlier page proposed rejecting candidates on **row periodicity at 4–5 mm** and **stroke-sized
components (0.3–2 mm²)**. Neither check was run on the known-ink control. Run now (fixed mask,
[`scripts/screen_flagged.py`](scripts/screen_flagged.py), scroll-vertical axis read from each
mesh's `z.tif`):

| | row pitch, whole region | row pitch, three 1 cm strips | p90 component | area in 0.3–2 mm² components |
|---|---|---|---:|---:|
| **known ink**, PHerc0139 w043 | 5.00 mm @ 1.62× | 2.73 @ 1.68 · 4.29 @ 2.12 · 3.00 @ 2.39 | **0.094 mm²** | 71% |
| PHerc0813_z12496_w060 fwd | 3.07 mm @ 1.16× | 2.56 @ 1.21 · 3.84 @ 1.29 · 2.56 @ 1.68 | 0.006 mm² | 50% |

- **Stroke size:** known ink's p90 component is 0.094 mm², under the 0.3 mm² floor, so this check
  rejects real ink under `ink_9um`. The share of area at stroke size (71% vs 50%) does not separate
  the two either.
- **Row pitch:** on known ink the strongest pitch wanders from 2.7 to 5.0 mm depending on where you
  look, so a 4–5 mm window rejects real ink too. Prominence overlaps at 1.68×. And the
  PHerc0813 mesh is only 1.5 cm tall, which leaves about four frequency bins between 2.5 and 10 mm.
  Its "3.07 mm" could hardly have come out any other way.

**Neither check can reject a candidate.** What remains of the rule is coverage and confidence ratio
against a matched control, then a look.

### 4. PHerc0813_z12496_w060 is unresolved, not "not ink"

![known ink vs the flagged region](figures/control_vs_flagged.png)

The earlier page called this region a false positive, "plausibly kollesis", on the strength of
checks 3 and 4 and of how it looks. Checks 3 and 4 are withdrawn above. And at 9 µm, `ink_9um` on
**known ink** shows no letterforms or rows either, so the picture cannot decide it.

What is measured, with the fixed mask:

| direction | >0.5 | >0.75 | ratio | vs control |
|---|---:|---:|---:|---:|
| **forward** | 0.03503 | **0.01392** | **2.52** | **3.1×** |
| reverse | 0.00400 | 0.00075 | 5.35 | 57.1× |

It is still the strongest region the survey has found on an eligible scroll: 3.1× below known ink,
a confidence ratio close to the control's, and one-sided (forward 18.6× reverse), which is what ink
on one face of a sheet would look like. That makes it **a weak lead, not a find and not a known
false positive.** A model that resolves letters, or someone who reads 9 µm CT for ink and kollesis,
could settle it. I cannot, from these outputs.

The other seven flagged regions sit **15–124× below control** with the fixed mask, and fail on
coverage alone ([`data/screen_flagged.json`](data/screen_flagged.json)).

## The survey

Input: [`pscamillo/vesuvius-eligible-meshes`](https://github.com/pscamillo/vesuvius-eligible-meshes)
(MIT) — 340 meshes, 1,935 cm², eight eligible scrolls. Ordered by **measured** mesh-vs-sheet
alignment from
[eligible-mesh-alignment](https://github.com/TAUIL-Abd-Elilah/eligible-mesh-alignment), best
first; the 11 meshes at ≥30° are skipped.

Per mesh: render 31 layers, gate CT support (villa #1254), run **4 `ink_9um` checkpoints × both
directions**, score the **unanimous minimum** at threshold 0.75 over the model's output, eroded
64 px, against the known-ink control PHerc0139 w043 at the same 4 checkpoints.

Coverage before the re-run (pre-fix snapshot, [`data/prefix_badmask/`](data/prefix_badmask/); its
score fields are superseded):

| scroll | meshes | cm² |
|---|---:|---:|
| PHerc0800 | 90 | 478.8 |
| PHerc0211 | 72 | 401.1 |
| PHerc0125 | 60 | 379.9 |
| PHerc0813 | 64 | 356.0 |
| PHerc0257 | 5 | 17.1 |
| PHerc0826 | 3 | 13.2 |
| PHerc0358 | 3 | 11.8 |
| PHerc0268 | 4 | 9.3 |
| **total** | **301** | **1,667.3** |

**On failures.** An earlier pass logged 110 render failures. Their logs show every one was
environmental: 61 DNS outages (`getaddrinfo failed`), 22 processes killed at logoff, the rest
SSL/stall timeouts. Retried on a healthy link, 1 render failure remains, plus 3 surfaces with no CT
support.

## PHerc1447: the three segments a maintainer would not rule out

FrankTheRope flagged row-like banding at 3.5–4 mm in raw meshes `z_dbg_gen_00215/00260/00701`;
Bruniss replied he "would not necessarily say those _arent_ letters/text". Re-scored with the fixed
mask ([`data/pherc1447_flagged_scores.json`](data/pherc1447_flagged_scores.json)):

| segment | >0.75 fwd / rev | ratio fwd / rev | vs control fwd / rev |
|---|---:|---:|---:|
| 00215 | 0.00613 / 0.00563 | 5.72 / 6.20 | 7.0× / 7.6× |
| 00260 | 0.00577 / 0.00489 | 5.22 / 6.32 | 7.4× / 8.7× |
| 00701 | 0.00499 / 0.00473 | 6.21 / 5.93 | 8.6× / 9.0× |

All diffuse, 7–9× below control. **Read this with the geometry**
([eligible-mesh-alignment](https://github.com/TAUIL-Abd-Elilah/eligible-mesh-alignment)): the
published PHerc1447 surfaces sit on nonzero CT a median 46% of the time, and those lying entirely on
CT cut across the sheets at 50–71°. `00260` and `00701` are among them (49.9° and 68.2°). These
surfaces are not a fair test of whether PHerc1447 has text.

## Limitations

- **Every score-derived number is pending the re-run**, except the control, the 8 flagged regions
  and the 3 PHerc1447 segments, which are re-scored above from kept predictions.
- One known-ink control, one 3 × 3 cm crop. Anything calibrated on it is provisional.
- `ink_9um` localises the ink field but does not resolve letters even on training scrolls, so a
  negative is "this model recovered no text", not "there is no ink".
- Bandwidth-bound at ~600 KB/s (~0.41 GB/cm²).
- No discovery is claimed.

## Reproduce

```bash
python scripts/corpus_survey.py --shard 0/2 --rescore --keep-all   # + shard 1/2; resumable
python scripts/corpus_aggregate.py                                  # merge shards, rank
python scripts/screen_flagged.py                                    # checks vs known ink + figure
```

Credits: corpus [@pscamillo](https://github.com/pscamillo); alignment gate
[@flummoxjr](https://github.com/flummoxjr); row-pitch measurements and the fibre-bundle
false-positive mode [@FrankTheRope](https://github.com/FrankTheRope).

MIT · Vesuvius Challenge, September 2026 · AI assistance (Claude) under my direction; I set the
questions, chose the controls, and checked the numbers.
