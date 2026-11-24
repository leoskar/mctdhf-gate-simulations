"""Stage 4: psi_3 and psi_4 become superpositions of |0010> and |0001>.
Other configs get lower-weight high-fidelity targets via masked Hungarian.
Loss math lives in `losses/_superpos.py`."""
from losses._superpos import (
    ALPHA_BOUNDS_GHZ,
    ALPHA_ORDERING,
    SEEK_GLOBAL_MINIMUM,
    MAXITER,
    make_compute,
)

A, B = 3, 4  # |0010>, |0001>
compute = make_compute(A, B)
