# Sample CFD data card

Two author-approved, previously inspected TRAINING snapshots are included to
exercise portable inference. They are not a hidden test set.

| File stem | Case | Stored time | Raster |
|---|---|---:|---|
| airfoil_m3_a40_re10000_t6 | MFC airfoil, M=3, alpha=40 degrees, Reynolds case label 10,000 | 6 | 512 x 512 |
| cylinder_euler_m2p7_t8 | MFC Euler/slip cylinder, M=2.7, f90 baseline | 8 | 900 x 990 |

Each NPZ includes primitive fields, coordinates, an exact stored body mask and
an observation mask. It does **not** include neural predictions, weak targets,
human segmentation labels or claimed ground-truth vortex IDs. Pressure and
density are positive in observed fluid. Every array is finite, dimensions and
coordinate ordering are checked, the solid/fluid masks are disjoint, and the
2-D Q identity is checked. Exported arrays are verified against their source
arrays after round-trip loading. SHA-256 and numerical ranges are recorded in
[DATA_QC.json](DATA_QC.json).

These checks establish file integrity and basic numerical usability for a
machine-vision smoke test. They do **not** establish grid/time convergence,
accurate shock stand-off, physically converged shedding, turbulence validity,
or experimental agreement. The Euler wake is an observed numerical field,
not a certified physical inviscid-shedding benchmark. No failed viscous-cylinder
state is included.

The original CFD files remain preserved. These derived sample arrays use the
stored harmonized cache precision (float32 primitive fields); derivatives are
recomputed in float64 at inference. Small rounding differences from derivatives
computed directly from a higher-precision upstream export are possible.
Neither derivatives nor the unknown sample labels are supplied to the model
as externally accepted truth.

Full training data, expert labels and a genuinely untouched case are separate
future releases. Do not train on these examples and then report them as an
independent benchmark.
