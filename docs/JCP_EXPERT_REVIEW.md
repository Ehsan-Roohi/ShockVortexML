# JCP expert review: author-volunteered, prediction-blind

Ehsan Roohi Golkhatmi volunteered to perform the review on 2026-09-04. This
records a future reviewer, not a completed annotation or independent verdict.

The v4 review pack contains six time-selected frames per case, 48 frames in
total. All editable masks are initially empty. The exported field archives
contain only primitive flow fields, recomputed diagnostics, coordinates and
the observation/body masks. They contain no weak labels or model predictions.
Separate raster-size packs are necessary for the existing annotation UI.

## Review protocol

1. Do not open the model figures or weak proposals while annotating these
   frames. Record any prior familiarity with the case in the review notes.
2. Delineate the shock envelope and, separately, its centerline using
   compression plus pressure/density changes normal to the candidate. Do not
   label body edges, wakes or high entropy alone as shocks.
3. Delineate coherent vortex cores using the velocity field and circulation
   evidence. Vorticity alone is insufficient. Q and lambda-ci are algebraically
   related in this 2-D formulation and are not two independent confirmations.
   Shock/core overlap is allowed. Mark genuinely ambiguous areas as ignore.
4. Enter reviewer name, notes, a decision for each head and reviewed status.
   An empty mask is acceptable only with an explicit decision and explanation.
5. Complete both shape packs before comparing labels with model predictions.
   Preserve labels and their hashes before interpreting model disagreement.

These are previously inspected development cases. Even a prediction-blind
review does not turn them into an untouched final benchmark. Train-case labels
must not contribute to held-case scores. A genuinely new whole case must be
registered before its fields are opened and before predictions are examined.

## Commands

Use the exact manifest and output paths listed in
`data/processed/jcp_v4/expert_review/packs.json`:

```text
python -m annotation_server --manifest <manifest> --output-dir <working_output> --port 8771
python -m jcp2026.review check --pack <height>x<width>
```

These commands require the full review packs, which are not included in the
two-snapshot bootstrap. Install the repository's `research` extra first.

The checker refuses to treat blank/unreviewed masks as an independent
reference. Its structural QA is necessary, not proof of scientific truth.
Tracking accuracy additionally requires independently assigned object IDs over
consecutive frames; sparse time-selected masks alone cannot validate tracks.
