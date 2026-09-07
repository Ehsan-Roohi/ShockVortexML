"""Inference-runtime guard for an optional training-only augmentation.

The prospective composite imports :mod:`wake_auxiliary`, whose training path
references ``noisy_patch``.  Frozen inference never calls that path.  The full
training implementation is intentionally absent from the compact Unity
runtime; fail explicitly if training is attempted here.
"""


def noisy_patch(*_args, **_kwargs):
    raise RuntimeError(
        "noisy_patch is training-only and is unavailable in the compact "
        "prospective inference runtime"
    )
