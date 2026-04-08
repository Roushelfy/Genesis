import numpy as np


def line_line_distance2(ea0: np.ndarray, ea1: np.ndarray, eb0: np.ndarray, eb1: np.ndarray) -> float:
    """Current dim==4 EE formula used by the kernel path."""
    u = ea1 - ea0
    v = eb1 - eb0
    b = np.cross(u, v)
    denom = np.dot(b, b)
    if denom == 0.0:
        return float("nan")
    aTb = np.dot(eb0 - ea0, b)
    return float((aTb * aTb) / denom)


def point_segment_distance2(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = np.dot(ab, ab)
    if denom == 0.0:
        d = p - a
        return float(np.dot(d, d))
    t = np.dot(p - a, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    q = a + t * ab
    d = p - q
    return float(np.dot(d, d))


def segment_segment_distance2(ea0: np.ndarray, ea1: np.ndarray, eb0: np.ndarray, eb1: np.ndarray) -> float:
    """Robust fallback used for quick verification in near-parallel case."""
    return min(
        point_segment_distance2(ea0, eb0, eb1),
        point_segment_distance2(ea1, eb0, eb1),
        point_segment_distance2(eb0, ea0, ea1),
        point_segment_distance2(eb1, ea0, ea1),
    )


def report_case(name: str, ea0, ea1, eb0, eb1) -> None:
    ea0 = np.array(ea0, dtype=np.float64)
    ea1 = np.array(ea1, dtype=np.float64)
    eb0 = np.array(eb0, dtype=np.float64)
    eb1 = np.array(eb1, dtype=np.float64)

    u = ea1 - ea0
    v = eb1 - eb0
    uxv2 = float(np.dot(np.cross(u, v), np.cross(u, v)))
    ac = float(np.dot(u, u) * np.dot(v, v))
    rel = uxv2 / ac if ac > 0.0 else float("inf")

    d2_line = line_line_distance2(ea0, ea1, eb0, eb1)
    d2_seg = segment_segment_distance2(ea0, ea1, eb0, eb1)

    print(f"=== {name} ===")
    print(f"parallel_ratio = |u x v|^2 / (|u|^2 |v|^2) = {rel:.3e}")
    print(f"line_line_distance2   = {d2_line:.12e}")
    print(f"segment_segment_dist2 = {d2_seg:.12e}")
    print(f"segment_segment_dist  = {np.sqrt(d2_seg):.12e}")
    print()


if __name__ == "__main__":
    # Printed from kernel:
    # [IPC][EE][corner-low-dist] EE=(35,36,37,38)
    report_case(
        "EE=(35,36,37,38)",
        (-9.048450e-14, 1.363250e-01, -6.164740e-12),
        (2.016403e-13, 1.403250e-01, -5.310797e-12),
        (2.024192e-12, 1.443250e-01, -1.499360e-12),
        (3.119794e-12, 1.483250e-01, 2.146554e-13),
    )

    # Printed from kernel:
    # [IPC][EE][corner-low-dist] EE=(34,35,36,37)
    report_case(
        "EE=(34,35,36,37)",
        (-9.260670e-15, 1.323250e-01, -6.043656e-12),
        (-9.048450e-14, 1.363250e-01, -6.164740e-12),
        (2.016403e-13, 1.403250e-01, -5.310797e-12),
        (2.024192e-12, 1.443250e-01, -1.499360e-12),
    )
