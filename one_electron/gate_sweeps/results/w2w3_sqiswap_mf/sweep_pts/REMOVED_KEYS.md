# Removed arrays

The .npz files in this folder are the original result files with the
arrays below removed to keep the data set small. All other keys are
unchanged, and no script in this repository reads the removed ones.

| key | dtype | shape(s) | content |
|---|---|---|---|
| `segB_ts` | float64 | (141,); (239,); (403,) ... | times of `segB_ys` |
| `segB_ys` | complex64 | (141, 704); (239, 704); (403, 704) ... | state along the ramp-down (segment B), every `traj_stride`-th step |
