"""The sweep grid (exact values).

Values are already in t.u. (tau_ramp, tau_hold); no arithmetic at run time.
Emits manifests for the SLURM arrays and self-verifies the expected counts.
"""
from __future__ import annotations
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# leg -> device, dt_div, [(tau_ramp, th_max, [tau_hold, ...]), ...]
GRID = {
    "w0w1": dict(dev="cpu", dt_div="0.1", T0=129.2, dT=26.0, step_fine=2.7, rows=[
        (12.6, 169.0, [64.6, 73.3, 82, 90.7, 99.4, 108.1, 116.8, 125.5, 134.2, 142.9, 151.6, 160.3, 169]),
        (18.3, 163.3, [58.9, 67.6, 76.3, 85, 93.7, 102.4, 111.1, 119.8, 128.5, 137.2, 145.9, 154.6, 163.3]),
        (26.6, 155.0, [50.6, 59.3, 68, 76.7, 85.4, 94.1, 102.8, 111.5, 120.2, 128.9, 137.6, 146.3, 155]),
        (38.8, 142.8, [38.4, 47.1, 55.8, 64.5, 73.2, 81.9, 90.6, 99.3, 108, 116.7, 125.4, 134.1, 142.8]),
        (56.5, 125.1, [20.7, 29.4, 38.1, 46.8, 55.5, 64.2, 72.9, 81.6, 90.3, 99, 107.7, 116.4, 125.1]),
        (82.3, 99.3, [3.6, 12.3, 21, 29.7, 38.4, 47.1, 55.8, 64.5, 73.2, 81.9, 90.6, 99.3]),
    ]),
    "w1w2": dict(dev="cpu", dt_div="0.1", T0=142.9, dT=28.8, step_fine=3.0, rows=[
        (8.9, 191.7, [76.5, 86.1, 95.7, 105.3, 114.9, 124.5, 134.1, 143.7, 153.3, 162.9, 172.5, 182.1, 191.7]),
        (14.2, 186.4, [71.2, 80.8, 90.4, 100, 109.6, 119.2, 128.8, 138.4, 148, 157.6, 167.2, 176.8, 186.4]),
        (22.6, 178.0, [62.8, 72.4, 82, 91.6, 101.2, 110.8, 120.4, 130, 139.6, 149.2, 158.8, 168.4, 178]),
        (36.0, 164.6, [49.4, 59, 68.6, 78.2, 87.8, 97.4, 107, 116.6, 126.2, 135.8, 145.4, 155, 164.6]),
        (57.2, 143.4, [28.2, 37.8, 47.4, 57, 66.6, 76.2, 85.8, 95.4, 105, 114.6, 124.2, 133.8, 143.4]),
        (91.0, 109.6, [4, 13.6, 23.2, 32.8, 42.4, 52, 61.6, 71.2, 80.8, 90.4, 100, 109.6]),
    ]),
    "w2w3": dict(dev="cpu", dt_div="0.1", T0=154.2, dT=31.0, step_fine=3.3, rows=[
        (7.0, 208.7, [85.1, 95.4, 105.7, 116, 126.3, 136.6, 146.9, 157.2, 167.5, 177.8, 188.1, 198.4, 208.7]),
        (11.9, 203.8, [80.2, 90.5, 100.8, 111.1, 121.4, 131.7, 142, 152.3, 162.6, 172.9, 183.2, 193.5, 203.8]),
        (20.1, 195.6, [72, 82.3, 92.6, 102.9, 113.2, 123.5, 133.8, 144.1, 154.4, 164.7, 175, 185.3, 195.6]),
        (34.1, 181.6, [58, 68.3, 78.6, 88.9, 99.2, 109.5, 119.8, 130.1, 140.4, 150.7, 161, 171.3, 181.6]),
        (57.9, 157.8, [34.2, 44.5, 54.8, 65.1, 75.4, 85.7, 96, 106.3, 116.6, 126.9, 137.2, 147.5, 157.8]),
        (98.2, 117.5, [4.2, 14.5, 24.8, 35.1, 45.4, 55.7, 66, 76.3, 86.6, 96.9, 107.2, 117.5]),
    ]),
    "w0w3": dict(dev="cuda", dt_div="0.25", T0=2100.9, dT=422.9, step_fine=44.6, rows=[
        (12.6, 2934.4, [1242.4, 1383.4, 1524.4, 1665.4, 1806.4, 1947.4, 2088.4, 2229.4, 2370.4, 2511.4, 2652.4, 2793.4, 2934.4]),
        (31.9, 2915.1, [1223.1, 1364.1, 1505.1, 1646.1, 1787.1, 1928.1, 2069.1, 2210.1, 2351.1, 2492.1, 2633.1, 2774.1, 2915.1]),
        (81.3, 2865.7, [1173.7, 1314.7, 1455.7, 1596.7, 1737.7, 1878.7, 2019.7, 2160.7, 2301.7, 2442.7, 2583.7, 2724.7, 2865.7]),
    ]),
}

EXPECTED_POINTS = {"w0w1": 77, "w1w2": 77, "w2w3": 77, "w0w3": 39}
CPU_LEGS = ["w0w1", "w1w2", "w2w3"]
GPU_LEGS = ["w0w3"]


def verify():
    total = 0
    for leg, g in GRID.items():
        n = 0
        for tr, thmax, holds in g["rows"]:
            assert len(set(holds)) == len(holds), f"{leg} tr={tr}: duplicate hold"
            assert max(holds) == thmax, f"{leg} tr={tr}: th_max {thmax} != max hold {max(holds)}"
            for h in holds:
                assert h > 0, f"{leg} tr={tr}: non-positive hold {h}"
                assert abs(round(h * 10) - h * 10) < 1e-9, f"{leg}: hold {h} off the 0.1 grid"
            assert abs(round(tr * 10) - tr * 10) < 1e-9, f"{leg}: ramp {tr} off the 0.1 grid"
            n += len(holds)
        assert n == EXPECTED_POINTS[leg], f"{leg}: {n} points, expected {EXPECTED_POINTS[leg]}"
        total += n
    assert total == 270, f"{total} points, expected 270"
    assert sum(len(g["rows"]) for g in GRID.values()) == 21
    return total


def cost_tu():
    """t.u. of propagation, split segA / segB, per leg (segB length = 2*tau_ramp)."""
    out = {}
    for leg, g in GRID.items():
        a = sum(thmax for _, thmax, _ in g["rows"])          # t_end = th_max (delta=tau_ramp)
        b = sum(2 * tr * len(h) for tr, _, h in g["rows"])
        out[leg] = (a, b)
    return out


def write_manifests():
    a_cpu, a_gpu, b_cpu, b_gpu = [], [], [], []
    for leg, g in GRID.items():
        A, B = (a_gpu, b_gpu) if g["dev"] == "cuda" else (a_cpu, b_cpu)
        for tr, thmax, holds in g["rows"]:
            A.append(f"{leg} {tr:g} {thmax:g}")
            for h in holds:
                B.append(f"{leg} {tr:g} {h:g}")
    for name, rows in [("manifest_segA_cpu.txt", a_cpu), ("manifest_segA_gpu.txt", a_gpu),
                       ("manifest_segB_cpu.txt", b_cpu), ("manifest_segB_gpu.txt", b_gpu)]:
        with open(os.path.join(HERE, name), "w") as f:
            f.write("\n".join(rows) + "\n")
        print(f"{name}: {len(rows)} lines")


if __name__ == "__main__":
    n = verify()
    print(f"grid verified: {n} points, 21 segment-A runs")
    c = cost_tu()
    for leg, (a, b) in c.items():
        print(f"  {leg}: segA {a:8.1f} t.u.   segB {b:8.1f} t.u.   total {a+b:8.1f}")
    print(f"  CPU total {sum(sum(c[l]) for l in CPU_LEGS):.1f} t.u.   "
          f"GPU total {sum(sum(c[l]) for l in GPU_LEGS):.1f} t.u.")
    write_manifests()
