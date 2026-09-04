# JCP readiness campaign v4

Target: Journal of Computational Physics. Author-approved on 2026-09-04.

This is a new experimental lineage. All legacy checkpoints, raw CFD and the
author-derived reviewed-v2 manuscript remain unchanged. This directory is not
a claim that the manuscript has passed independent validation.

## Four workstreams

1. Recompute signed vorticity, two-dimensional deviatoric Q and swirling
   strength from the same stored-grid velocity-gradient operator in both
   airfoil and cylinder data. Preserve original weak masks, exclude unobserved
   raster padding from supervision, and train from scratch. Check analytical
   rotation, shear, compression and Q/lambda identities before training.
2. Freeze complete development groups and their exposure history. Create blank,
   model-blind expert annotation packs. An actual named expert must supply
   decisions. A new whole CFD family is pending; already inspected f180/Re1e6
   cases cannot be renamed untouched. Independent annotation and untouched
   model-development exposure are different requirements.
3. Use identical training patches/updates for joint Qdev, no-PCGrad,
   primitive-only, official SegFormer-B0, and temporal-target ablations, with
   three initialization seeds. All start from scratch, without ImageNet or
   legacy pretrained weights. This matches data and update budgets, not model
   capacity or compute time, and is not a claim of fully optimized SOTA.
   The temporal variant uses frozen LEGACY teacher proposals on TRAIN ONLY;
   its teacher's known limitations remain. Validation targets are never
   replaced by temporal pseudo-labels. Report final-step models, validation-only
   thresholds, fixed-threshold controls, and separate learned/physics/hybrid
   products. Current scores are weak-label agreement until independent masks
   are frozen. Physics compared with its own proposal is not accuracy.
4. Assemble a versioned, hash-verified source/config/checkpoint/results bundle,
   with dependency lock and exact commands. Do not include private email,
   unrelated papers, commercial materials or bulk raw archives. Public-release
   destination/license and any third-party redistribution require explicit
   approval. The author has now approved the independent MIT repository
   `Ehsan-Roohi/ShockVortexML` and two numerically checked sample fields.
   This one-model bootstrap is not the completed 15-model study.

## Fixed campaign

`configs/jcp_v4_campaign.json` is the preregistered development configuration.
Changing budgets/threshold grids after inspecting control results requires a
new version. Individual review ambiguity is recorded as ignore, not silently
converted into a negative. Shock/vortex overlap remains valid.

SegFormer uses the official Hugging Face Transformers 4.55.4 implementation,
B0 widths [32,64,160,256], depths [2,2,2,2], decoder width 256, seven CFD inputs
and six independent logits. Only the input/output dimensions and multi-label
loss are adapted. Reference:
https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/segformer

## Full-data commands

```text
python -m pip install -e '.[research]'
python -m unittest discover -s tests -p test_jcp_v4.py -v
python -m jcp2026.prepare
python -m jcp2026.patch_bank
python -m jcp2026.train --variant joint_qdev --seed 20260904
```

Use Python 3.12 and the pinned packages. Preparation and training additionally
require the registered full data/proposals; two sample snapshots cannot
reproduce the complete study. Portable sample inference is in the root README.

## Submission gates, not yet satisfied

- Named expert masks/identities and agreement/adjudication record.
- Untouched family registered before field inspection and run once after freeze.
- Completed paired ablations, uncertainty/negative-case audits and overlays.
- Bounded claim of generalization, not correctness for every geometry.
- Immutable public ML release and manuscript update from the author's lineage.

Machine-vision accuracy on stored CFD is separate from convergence of the CFD
solution. Physical Euler shedding remains an unvalidated physical claim, but
does not prevent expert labeling of structures actually present in a snapshot.
