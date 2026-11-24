# Removed arrays

The .npz files in this folder are the original result files with the
arrays below removed to keep the data set small. All other keys are
unchanged, and no script in this repository reads the removed ones.

| key | dtype | shape(s) | content |
|---|---|---|---|
| `y` | complex128 | (22626,); (28898,) | final state vector of the propagation (all layers packed) |
| `y0` | complex128 | (22626,); (28898,) | initial state vector; equal to the packed relaxed state the run starts from |
