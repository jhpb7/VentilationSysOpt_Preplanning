import numpy as np
from typing import Callable
from underestimating_hyperplanes import (
    compute_tangential_planes_equation,
    tangential_planes_to_array,
    tangential_planes_to_dict,
)


def compute_planes_nd(f: Callable, grads: list[Callable], points: np.ndarray, variables: list[str]) -> dict:
    """Compute tangential planes for an nD function and return YAML dict."""
    planes = compute_tangential_planes_equation(points, f, grads)
    planes = tangential_planes_to_array(planes)
    return tangential_planes_to_dict(planes, variables)


def compute_planes_1d(f: Callable, grad: Callable, points: np.ndarray, variables: list[str]) -> dict:
    """Special case for 1D tangential planes."""
    planes = compute_tangential_planes_equation(points, f, grad)
    planes = tangential_planes_to_array(planes)
    return tangential_planes_to_dict(planes, variables)


def plane_dict_from_array(index, data: dict, prefix: str, variables: list[str]) -> dict:
    """Convert tangential plane data into dict with prefixed keys."""
    out = {f"{prefix}_hyperplanes_set": index}
    out[f"{prefix}_intercept"] = dict(zip(index, data["intercept"]))
    for var in variables:
        out[f"{prefix}_slope_{var}"] = dict(zip(index, data[f"grad_{var}"]))
        out[f"{prefix}_point_{var}"] = dict(zip(index, data[f"point_{var}"]))
    return out
