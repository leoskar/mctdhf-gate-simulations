# Removed arrays

The .npz files in this folder are the original result files with the
arrays below removed to keep the data set small. All other keys are
unchanged, and no script in this repository reads the removed ones.

| key | dtype | shape(s) | content |
|---|---|---|---|
| `y` | complex128 | (5154,) | final state vector of the propagation (all layers packed) |
