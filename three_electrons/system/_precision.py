"""Working precision, selected by the environment BEFORE jax.numpy is used.

`jax.config.update("jax_enable_x64", ...)` has to run before any array is
created, so it cannot be a command-line flag parsed in `main`. Set
`ML_PRECISION=c64` in the environment for a single-precision arm; anything
else (or unset) gives the x64 default that every checkpoint in `ckpt/` was
relaxed in.
"""

import os

import jax

X64 = os.environ.get("ML_PRECISION", "x64").lower() != "c64"
jax.config.update("jax_enable_x64", X64)

import jax.numpy as jnp  # noqa: E402

DTYPE = jnp.complex128 if X64 else jnp.complex64
