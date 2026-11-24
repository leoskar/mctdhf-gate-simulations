"""Stage 3: psi_2 and psi_3 become superpositions of |0100> and |0010>.
Other configs get lower-weight high-fidelity targets via masked Hungarian.
Loss math lives in `losses/_superpos.py`."""
from losses._superpos import (
    ALPHA_BOUNDS_GHZ,
    ALPHA_ORDERING,
    SEEK_GLOBAL_MINIMUM,
    MAXITER,
    make_compute,
)

A, B = 2, 3  # |0100>, |0010>
compute = make_compute(A, B)
