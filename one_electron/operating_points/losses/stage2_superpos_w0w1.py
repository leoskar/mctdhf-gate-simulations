"""Stage 2: psi_1 and psi_2 become superpositions of |1000> and |0100>.
Other configs get lower-weight high-fidelity targets via masked Hungarian.
Loss math lives in `losses/_superpos.py`."""
from losses._superpos import (
    ALPHA_BOUNDS_GHZ,
    ALPHA_ORDERING,
    SEEK_GLOBAL_MINIMUM,
    MAXITER,
    make_compute,
)

A, B = 1, 2  # |1000>, |0100>
compute = make_compute(A, B)
