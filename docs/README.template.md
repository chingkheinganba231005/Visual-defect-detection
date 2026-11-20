# Visual Defect Detection for Product Inspection

A model that looks at a photo of a product and decides whether a human
inspector needs to check it, with a heatmap showing where the suspected
defect is. It is trained only on photos of good parts, the usual situation
on a real line where defects are rare and varied, and it is served as an HTTP
API in a Docker container.

I used the 15 product categories of MVTec AD (bottles, cables, screws, pills,
fabrics and so on) and evaluated everything on held-out images.

- PatchCore separates defective from good parts with a mean image-level
  AUROC of {{auroc_patchcore}} across the {{n_categories}} categories (95% CI
  {{auroc_patchcore_ci}}), against {{auroc_gaussian}} for a Gaussian model on
  CNN features and {{auroc_mean_image}} for comparing pixels with an average
  good image.
- Used to triage parts, it lets {{tri_saved}} of production pass without a
  manual check at a {{defect_rate}} defect rate, while catching {{tri_catch}}
  of the defects in held-out images. The target was {{catch_target}}; why it
  fell short is explained below.
- With {{cost_inspection}} per manual check and {{cost_escape}} per defect that
  reaches a customer, the cost-based threshold works out at {{cost_model}} per
  unit, against {{cost_all}} for checking everything by hand.
- The service takes {{lat_p50}} per image on 4 CPU cores ({{lat_p95}} at the
  95th percentile).

![Heatmaps on held-out defective parts](figures/heatmaps.png)

## Data

[MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
(Bergmann et al., 2021): {{n_train_images}} training photos of good parts and
{{n_test_images}} test photos ({{n_test_defective}} of them defective, with
pixel masks of the defects) across 15 categories of objects and textures.
`scripts/01_download.py` downloads the official archives from MVTec and checks
each against a recorded SHA-256. The data is licensed CC BY-NC-SA 4.0, for
non-commercial use, which is why this is a research prototype and not a
product.

## Method

**Splits.** For each category, 80% of the good training photos build the
model and 20% are kept back. The test photos are split 50/50 by defect type
into a calibration half and a held-out half. Thresholds are chosen on the
calibration images (the kept-back good photos plus the calibration half) and
judged once on the held-out half. MVTec's test sets have few good photos, so
the kept-back training photos give the calibration step more good examples
to set false-alarm rates on.

**PatchCore** (Roth et al., 2022). A WideResNet-50 pretrained on ImageNet
describes each cell of a 28 x 28 grid over the 224 x 224 image with a
1,536-number feature vector, taken from the network's second and third stages
and averaged over the cell's neighbours. The patches of
the good training photos go into a memory bank, thinned to {{coreset_pct}}
by greedy coreset selection so it covers the variety of good patches without
storing all of them. A new photo's patches are compared with the bank: the
distance to the nearest stored patch is that patch's anomaly score, the
largest one is the image score, and the scores upsampled and smoothed are the
heatmap. Nothing is trained by gradient descent, so a new product line needs
only a few hundred good photos and a few minutes on a CPU.

**Baselines.** A single Gaussian fitted to the globally pooled features of
the same network and scored by Mahalanobis distance (Rippel et al., 2020),
and the per-pixel difference from the average good photo, which uses no
learned features at all.

**Uncertainty.** Every AUROC and every held-out rate comes with a 95%
interval from a stratified bootstrap ({{n_boot}} resamples). The average over
categories is bootstrapped by averaging the per-category resamples.

## Results

### Detection

| | Mean image AUROC (95% CI) | Categories at 0.99 or above |
|---|---|---|
| PatchCore | {{auroc_patchcore}} ({{auroc_patchcore_ci}}) | {{n99_patchcore}} of {{n_categories}} |
| Gaussian on CNN features | {{auroc_gaussian}} ({{auroc_gaussian_ci}}) | {{n99_gaussian}} of {{n_categories}} |
| Mean-image difference | {{auroc_mean_image}} ({{auroc_mean_image_ci}}) | {{n99_mean_image}} of {{n_categories}} |

PatchCore is ahead of the Gaussian model by {{gain_vs_gaussian}} AUROC
(paired 95% CI {{gain_vs_gaussian_ci}}) and of the pixel baseline by
{{gain_vs_mean_image}} ({{gain_vs_mean_image_ci}}). It also localises
defects: the heatmaps reach a mean pixel-level AUROC of {{pixel_patchcore}}
against the masks, compared with {{pixel_mean_image}} for the pixel baseline.
The hardest category is {{worst_cat}} (AUROC {{worst_auroc}}, CI
{{worst_ci}}).

{{auroc_table}}

![AUROC by category](figures/auroc_by_category.png)

### From scores to inspection decisions

On a real line most parts are good, while MVTec's test sets are mostly
defective. So I measure the catch rate on defective photos and the
false-alarm rate on good photos separately, then weight them by an assumed
defect rate of {{defect_rate}}. Flagged parts go to a human inspector;
everything else passes.

**Triage policy.** The threshold is the highest one that still catches
{{catch_target}} of the defects in the calibration images. On the held-out
images:

- defects caught: {{tri_catch}} (95% CI {{tri_catch_ci}})
- good parts flagged: {{tri_false_alarm}} ({{tri_false_alarm_ci}})
- units sent to an inspector: {{tri_flagged}}, so {{tri_saved}} of manual
  checks are avoided ({{tri_saved_ci}})
- categories that fell short of the {{catch_target}} target on held-out
  data: {{tri_below}} of {{n_categories}} (worst: {{tri_worst_cat}},
  {{tri_worst_catch}} of {{tri_worst_n}} defects)

The shortfall is mostly a sample-size problem, though categories where the
model is weaker (see the AUROC table) contribute too. Each calibration half
has only {{cal_def_min}} to {{cal_def_max}} defective photos,
so "catch 99%" in practice means "catch all of them", and the next unseen
defect can easily fall just below that threshold. On a real line I would set
the threshold on many more defect examples, or lower it by a safety margin
and accept a few more checks.

With the Gaussian baseline instead, the same policy catches
{{tri_g_catch}} and avoids {{tri_g_saved}} of checks.

{{triage_table}}

![Triage by category](figures/triage_by_category.png)

**Cost-based policy.** If a manual check costs {{cost_inspection}} and a
defective unit that reaches the customer costs {{cost_escape}}, checking every
unit costs {{cost_all}} per unit and checking none costs {{cost_none}}. The
threshold that minimises cost on the calibration images gives {{cost_model}}
per unit on the held-out images ({{cost_model_ci}}) while catching
{{cost_catch}} of defects. That is {{cost_saving}} cheaper than the better
blanket policy, or about {{cost_per_10k_saved}} per 10,000 units.

The heatmap below re-chooses the threshold for defect rates from 0.5% to 10%
and escape costs from 10 to 1,000 times the cost of a check. The model is
cheaper than both blanket policies in {{sens_wins}} of the {{sens_total}}
settings, saving up to {{sens_best}}. It loses when defects are common and
escapes are very expensive ({{sens_losses}}): there, missing even a few
percent of defects costs more than checking every unit, so checking
everything is the right call.

![Cost sensitivity](figures/cost_sensitivity.png)

## The service

```bash
make demo                 # downloads the bottle category (156 MB) and fits it
make serve                # http://localhost:8000/docs
curl -F "file=@some_bottle.png" "localhost:8000/inspect/bottle?heatmap=true"
```

`POST /inspect/{category}` returns the anomaly score, the threshold, the
decision (`pass` or `send to inspector`) and, if asked, the heatmap as a
base64 PNG. `?policy=cost_based` switches to the cost-based threshold.
`GET /health` lists the categories with a fitted model. `make docker` builds
and runs the same thing in a container with the backbone weights baked in.

Measured on a 4-core Linux machine with {{lat_threads}} CPU threads, over
the {{lat_n}} held-out {{lat_cat}} images: {{lat_p50}} median model time
({{lat_p95}} at the 95th percentile) and {{req_p50}} for the whole request
including the heatmap, about {{lat_ips}} images per second.

## Limitations

- MVTec AD photos are taken under controlled lighting with the part centred.
  A real inspection line would need its own photos and a check that the good
  parts used for fitting cover normal variation.
- The defect rate, inspection cost and escape cost are assumptions. The
  sensitivity grid shows how the saving changes with them.
- The held-out halves are small (some categories have under 10 good test
  photos), so the per-category rates have wide intervals.
- One threshold per category. A real deployment would monitor the score
  distribution and refit when the product or camera changes.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
make install
make all      # download (4.9 GB), fit all categories (about {{fit_minutes}} min on 4 cores), evaluate, decide, benchmark, README
make test     # tests use a few real images in tests/data and run offline
```

Each step writes `results/*.json` and `figures/`. This README is generated
from those files by `scripts/render_readme.py`, and CI fails if it falls out
of date. Seeds are fixed in `src/defects/config.py` and the dependencies are
pinned.

## Data source and licence

MVTec AD: P. Bergmann, K. Batzner, M. Fauser, D. Sattlegger and C. Steger,
"The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for
Unsupervised Anomaly Detection", International Journal of Computer Vision,
2021. MVTec Software GmbH, CC BY-NC-SA 4.0. PatchCore: K. Roth et al.,
"Towards Total Recall in Industrial Anomaly Detection", CVPR 2022. Gaussian
baseline: O. Rippel, P. Mertens and D. Merhof, "Modeling the Distribution of
Normal Data in Pre-Trained Deep Features for Anomaly Detection", ICPR 2020.
