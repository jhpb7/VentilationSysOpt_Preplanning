import logging
import numpy as np
from underestimating_hyperplanes import (
    get_pel_n_func,
    sample_over_grid_nd,
    compute_tangential_planes_equation,
    remove_intersecting_hyperplanes,
    tangential_planes_to_array,
    tangential_planes_to_dict,
    calculate_max_values,
    lower_hull_planes,
)
from pyomo2h5 import load_yaml, save_yaml
from src.preplanning.preprocessing.domain_utils import (
    dimensionalise_coeffs,
    get_max_values,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# -------------------------------
# General helpers (fan-specific math)
# -------------------------------
def underestimating_hyperplanes(
    a: list[float], b: list[float], max_values: dict
) -> np.ndarray:
    """Compute underestimating hyperplanes for fan power loss.

    Args:
        a (list[float]): Pressure polynomial coefficients.
        b (list[float]): Power polynomial coefficients.
        max_values (dict): Dictionary with max flow (q) and pressure (dp).

    Returns:
        np.ndarray: Array of tangential plane coefficients.
    """
    Qmax, dpmax = max_values["q"], max_values["dp"]
    ploss_func, ploss_func_grad, n_func = get_pel_n_func(a, b)

    q_vals = np.linspace(Qmax * 1e-9, Qmax, 20)
    dp_vals = np.linspace(dpmax * 1e-9, dpmax, 20)
    points = sample_over_grid_nd([q_vals, dp_vals], n_func)

    q_vals_fine = np.linspace(0, Qmax, 200)
    dp_vals_fine = np.linspace(0, dpmax, 200)
    finer_points = sample_over_grid_nd([q_vals_fine, dp_vals_fine], n_func)

    tangential_planes = compute_tangential_planes_equation(
        points, ploss_func, ploss_func_grad
    )
    tangential_planes = remove_intersecting_hyperplanes(
        tangential_planes, ploss_func, finer_points
    )
    tangential_planes = tangential_planes_to_array(tangential_planes)

    # clip near-zero coefficients
    tangential_planes[:, :3] = np.where(
        np.abs(tangential_planes[:, :3]) < 1e-5, 0, tangential_planes[:, :3]
    )
    return tangential_planes


def overestimating_hyperplanes(
    a: list[float], b: list[float], max_values: dict
) -> np.ndarray:
    """Compute overestimating hyperplanes for fan power loss.

    Args:
        a (list[float]): Pressure polynomial coefficients.
        b (list[float]): Power polynomial coefficients.
        max_values (dict): Dictionary with max flow (q).

    Returns:
        np.ndarray: Array of tangential plane coefficients.
    """
    Qmax = max_values["q"]
    q_points = np.linspace(Qmax * 1e-9, Qmax, 20)

    dp = lambda q: a[1] * q**2 + a[2] * q + a[3]
    ploss_expr = (
        lambda q: b[1] * q**3 + b[2] * q**2 + b[3] * q + b[4] + b[5] - dp(q) * q
    )

    points = np.column_stack((q_points, -ploss_expr(q_points)))
    tangential_planes = lower_hull_planes(points)

    # adjust orientation
    tangential_planes[:, :-1] = -tangential_planes[:, :-1]
    return tangential_planes


def main():

    OUTFOLDER = "data/fan_data/"
    INFOLDER = "data/"

    fans_on_edges = load_yaml(INFOLDER + "network_data/fans_on_edges_raw_all_fans.yml")

    outfile_path = INFOLDER + "fan_data/fan_power_loss_hyperplanes_all_fans"
    outfile_data = {}

    # collect all fans
    fans = {
        key
        for value in fans_on_edges["edges"].values()
        for key in value.keys()
        if key != "total"
    }

    # regroup by fan_name -> diameters
    fan_p_d: dict[str, list] = {}
    for fan, diameter in fans:
        fan_p_d.setdefault(fan, []).append(diameter)

    # process fans
    for fan_name, diameters in fan_p_d.items():
        fan_data = load_yaml(OUTFOLDER + fan_name + ".yml")

        for diameter in diameters:
            logging.info(f"Now fitting fan {fan_name} with diameter {diameter}")

            a = fan_data["ansatz"]["pressure"]["o2"]["ansatz_param"]
            b = fan_data["ansatz"]["power"]["o3"]["ansatz_param"]

            # get standard max values (for max diameter)
            max_values = get_max_values(fan_data)
            a, b = dimensionalise_coeffs(a, b, diameter, max_values)

            # clip max values for given diameter
            max_values = calculate_max_values(a, b)

            # compute hyperplanes
            tangential_planes_under = underestimating_hyperplanes(a, b, max_values)
            tangential_planes_over = overestimating_hyperplanes(a, b, max_values)

            yaml_dict_under = tangential_planes_to_dict(
                tangential_planes_under, ["volume_flow", "pressure"]
            )
            yaml_dict_over = tangential_planes_to_dict(
                tangential_planes_over, ["volume_flow"]
            )

            # additional parameters
            cost_model = fan_data["ansatz"]["cost"]["o1"]["ansatz_param"]
            costs = cost_model[1] * diameter + cost_model[2]

            max_values_out = {
                "fan_power_loss_max": max_values["ploss"],
                "fan_volume_flow_max": max_values["q"],
                "fan_pressure_max": max_values["dp"],
            }
            fan_coefficients = {
                "fan_pressure_coefficients": a,
                "fan_power_coefficients": b,
            }

            # assemble data
            outfile_data[(fan_name, diameter)] = {
                "underestimation": yaml_dict_under,
                "overestimation": yaml_dict_over,
                "fan_costs": costs,
                **max_values_out,
                **fan_coefficients,
            }

    # save
    save_yaml(outfile_path + ".yml", outfile_data)


if __name__ == "__main__":
    main()
