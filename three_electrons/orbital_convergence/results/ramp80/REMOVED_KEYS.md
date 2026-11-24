# Removed arrays

The .npz files in this folder are the original result files with the
arrays below removed to keep the data set small. All other keys are
unchanged, and no script in this repository reads the removed ones.

| key | dtype | shape(s) | content |
|---|---|---|---|
| `y0` | complex128 | (1666,); (2274,); (3106,) ... | initial state vector; equal to the packed relaxed state the run starts from |
| `y_t` | complex128 | (6, 1666); (6, 2274); (6, 3106) ... | state vectors at the saved times |
