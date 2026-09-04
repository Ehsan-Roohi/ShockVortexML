# ShockVortexML

ML-first shock and vortex-core segmentation in compressible CFD, with
explicit physical-input contracts and separate physics audits.

**Development release, not an independently validated benchmark.** The current
checkpoint is the first completed model in a planned five-variant, three-seed
campaign. It uses weak training labels. Expert annotation and a genuinely
untouched final case remain outstanding. Targeting the *Journal of
Computational Physics* does not imply journal acceptance or submission readiness.

## What is included

- A portable neural-inference CLI that consumes primitive flow fields, not
  shock/vortex reference masks.
- Separate shock, vortex-core and background/other outputs. Shock/core overlap
  is allowed; frame-local vortex component IDs are retained.
- The harmonized v4 joint-model checkpoint and validation-frozen thresholds.
- Two author-approved numerical-QC-passed sample snapshots: Mach-3, alpha-40,
  Reynolds-10,000 airfoil at t=6 and Mach-2.7 Euler cylinder at t=8.
- Training/evaluation source, exact dependency versions, analytic tests,
  source/model/data hashes and a prediction-blind expert-review workflow.

The samples are **not** human ground truth, untouched test cases or
grid-converged CFD references. Failed viscous-cylinder data are excluded.
See [sample-data scope and checks](sample_data/DATA_CARD.md) and
[model limitations](models/jcp_v4_joint_qdev_seed20260904/MODEL_CARD.md).

## Quick start: included cylinder sample

Use Python 3.12. The tested host was CPU-only. From the cloned repository:

```bash
python -m pip install -e .
shockvortexml \
  --input sample_data/cylinder_euler_m2p7_t8.npz \
  --reference sample_data/cylinder_euler_m2p7_t8_reference.json \
  --checkpoint models/jcp_v4_joint_qdev_seed20260904/checkpoint.pt \
  --thresholds models/jcp_v4_joint_qdev_seed20260904/thresholds.json \
  --output predictions/cylinder.npz
```

In PowerShell, put the `shockvortexml` command on one line instead of using
the Bash line continuations. For the airfoil, substitute
`sample_data/airfoil_m3_a40_re10000_t6.npz` and its `_reference.json` file.

Output includes `p_shock`, `p_vortex_core`, `p_background_head`, binary masks,
frame-local `vortex_instances`, body/observation masks and coordinates.
Sigmoid scores are **not claimed to be independently calibrated
probabilities**. The three independent heads are not a softmax partition.
`background_other` is the observed-fluid complement of the accepted shock and
core masks. Threshold files must match the checkpoint hash.

## Your own flow fields

Provide an NPZ with two-dimensional `rho`, `pressure`, `u`, `v` arrays and
strictly increasing one-dimensional `x`, `y` coordinate vectors. Rows follow
`y`, columns follow `x`. Supply `geometry` and `observation_mask` when a body
or unobserved raster padding is present. Observed density and pressure must
be positive. Freestream density, pressure, velocity components, gamma and
reference length are explicit in a separate JSON. Do not reuse the sample
freestream metadata for an unrelated flow.

The code is not tied to an airfoil/cylinder coordinate lookup, but this is
**not a guarantee of correct segmentation on every geometry or regime**.
In particular, rarefied/DSMC fields, unfamiliar resolutions and noisy
gradients require separate validation. Frame-local IDs are not temporal tracks.

## Research workflow

```bash
python -m pip install -e '.[research]'
python -m unittest discover -s tests -p 'test_jcp*.py' -v
```

The nine focused tests cover analytic differential identities, nonuniform
coordinates, leakage rejection, model forward/backward execution and hybrid
overlap/support behavior. They do not establish segmentation accuracy.

The full 197-frame preparation and 2,400-patch training assets are not part of
this lightweight bootstrap. Full-data replication therefore requires the
registered additional data; the two samples are sufficient for inference
smoke tests, not for repeating the entire training study. The complete
matched-update campaign is still running. See
[the four workstreams](docs/JCP_READINESS_V4.md),
[reproducibility contract](docs/JCP_REPRODUCIBILITY.md) and
[expert-review protocol](docs/JCP_EXPERT_REVIEW.md).

No human annotation is fabricated, no physics-only output is presented as a
network prediction, and weak-proposal agreement is not called independent
accuracy. CFD convergence is distinct from recognizing structures in a stored
snapshot.

## License and attribution

See [MIT License](LICENSE). Installed dependencies retain their own licenses.
PCGrad and SegFormer are established methods, not inventions of this project.
The SegFormer comparison uses the official Transformers implementation, from
scratch, with adapted input/output channels. No third-party pretrained weights,
private manuscripts, emails or bulk raw CFD archives are included.
