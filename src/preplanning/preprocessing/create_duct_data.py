# %%
import logging
import numpy as np
from src.preplanning.preprocessing.hyperplane_utils import compute_planes_nd, compute_planes_1d, plane_dict_from_array
from pyomo2h5.yaml_handler import construct_yaml, convert_numpy_to_native

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# %% Advanced spacing of points
def compute_advanced_spacing(min_dim: float, max_dim: float) -> np.ndarray:
    """Generate advanced log-polar spacing of (width, height) points."""
    r2 = np.logspace(np.log10(min_dim), np.log10(max_dim), 15)
    phi2 = np.array(
        [
            np.arctan(1 / 3),
            np.pi / 4 + 5 / 180 * np.pi - np.arctan(1 / 3),
            np.pi / 4 - 5 / 180 * np.pi + np.arctan(1 / 3),
            np.pi / 2 - np.arctan(1 / 3),
        ]
    )

    x2 = np.array([(ri) * np.sin(phii) for ri in r2 for phii in phi2])
    y2 = np.array([(ri) * np.cos(phii) for ri in r2 for phii in phi2])

    n_diag = 60
    r1 = np.logspace(np.log10(min_dim), np.log10(max_dim), n_diag)

    x = np.concatenate([r1, x2])
    y = np.concatenate([r1, y2])

    points = np.array(
        [(xi, yi) for xi, yi in zip(x, y) if (xi >= min_dim) and (yi >= min_dim)]
    )
    return points


OUTFOLDER = "data/duct_data/"
MIN_DIM, MAX_DIM = 0.2, 2.0


def main():
    logging.info("Computing advanced spacing of duct points...")
    points_duct = compute_advanced_spacing(MIN_DIM, MAX_DIM)
    points_inv = np.logspace(np.log10(MIN_DIM), np.log10(MAX_DIM), 10)

    # %% Compute hyperplanes
    fric_dict = compute_planes_nd(
        f_fric, [df_dx_fric, df_dy_fric], points_duct, ["width", "height"]
    )
    dp_dict = compute_planes_nd(
        f_dp, [df_dx_dp, df_dy_dp], points_duct, ["width", "height"]
    )
    inv_dict = compute_planes_1d(f_inv, df_dx_inv, points_inv, ["width"])

    # indices
    friction_index = range(1, len(fric_dict["intercept"]) + 1)
    dp_index = range(1, len(dp_dict["intercept"]) + 1)
    inv_index = range(1, len(inv_dict["intercept"]) + 1)

    # assemble
    out_dict = {
        **plane_dict_from_array(
            friction_index, fric_dict, "duct_friction", ["width", "height"]
        ),
        **plane_dict_from_array(dp_index, dp_dict, "duct_area2", ["width", "height"]),
        **plane_dict_from_array(inv_index, inv_dict, "inverse", ["width"]),
    }

    # %% Save YAML
    yaml = construct_yaml()
    out_dict = convert_numpy_to_native(out_dict)
    yaml.indent(mapping=2, sequence=4, offset=2)

    outfile = OUTFOLDER + "duct_hyperplanes.yml"
    with open(outfile, "w") as f:
        yaml.dump(out_dict, f)

    logging.info(f"Saved duct hyperplanes to {outfile}")


# %% Duct formulas
def f_fric(x, y):
    return (y + x) / (y**3 * x**3)


def df_dx_fric(x, y):
    return (-3 * y - 2 * x) / (y**3 * x**4)


def df_dy_fric(x, y):
    return (-3 * x - 2 * y) / (y**4 * x**3)


def f_dp(x, y):
    return 1 / (x**2 * y**2)


def df_dx_dp(x, y):
    return -2 / (x**3 * y**2)


def df_dy_dp(x, y):
    return -2 / (y**3 * x**2)


def f_inv(x):
    return 1 / x


def df_dx_inv(x):
    return -1 / x**2


if __name__ == "__main__":
    main()
