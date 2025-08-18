from typing import Dict, Tuple, Any, Union
import copy
from collections import defaultdict
import sympy as sp
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

import pyomo.environ as pyo

# from src.preplanning.debugging.debugging import calculate_IIS
from src.preplanning.postprocessing.utils import filter_dict_by_prefix

rho = 1.2


def get_symbolic_power_expression():
    """
    Returns a lambda expression for electric power consumption with fan speed eliminated.
    """
    q_, dp_, n_, *rest = sp.symbols("q dp n a1:4 b1:6")

    a = rest[:3]
    b = rest[3:]

    eq_pressure = sp.Eq(dp_, a[0] * q_**2 + a[1] * q_ * n_ + a[2] * n_**2)
    pel_expr = (
        b[0] * q_**3 + b[1] * q_**2 * n_ + b[2] * q_ * n_**2 + b[3] * n_**3 + b[4]
    )

    n_solution = sp.solve(eq_pressure, n_)[1]
    pel_exact = pel_expr.subs(n_, n_solution)

    return sp.lambdify(
        ["q", "dp", "a1", "a2", "a3", "b1", "b2", "b3", "b4", "b5"],
        pel_exact,
        modules=["numpy", {"sqrt": np.sqrt}],
    )


def get_symbolic_rot_speed_expression():
    """
    Returns a symbolic expression for the fan speed.
    """
    q_, dp_, n_, *a = sp.symbols("q dp n a1:4 b1:6")

    eq_pressure = sp.Eq(dp_, a[0] * q_**2 + a[1] * q_ * n_ + a[2] * n_**2)

    n_solution = sp.solve(eq_pressure, n_)[1]

    return sp.lambdify(
        ["q", "dp", "a1", "a2", "a3"],
        n_solution,
        modules=["numpy"],
    )


def get_fan_tuple(instance):

    n_func = get_symbolic_rot_speed_expression()
    pel_func = get_symbolic_power_expression()

    fan_tuple = defaultdict(dict)

    for s in instance.Scenarios:
        for fan in instance.fan_set:
            if instance.scenario[s].fan_ind_active[fan].value > 1e-3:
                pd_combination = fan[2:4]
                # Get dimensional values
                dp = (
                    instance.scenario[s].fan_pressure_change_dimless[fan].value
                    * instance.fan_pressure_max[pd_combination]
                )
                q = instance.scenario[s].fan_volume_flow[fan].value

                # Extract and sort alphas and betas by key index to ensure correct order
                a_coeffs = [
                    instance.fan_pressure_coefficients[
                        (pd_combination[0], pd_combination[1], i)
                    ]
                    for i in range(1, 4)
                ]
                b_coeffs = [
                    instance.fan_power_coefficients[
                        (pd_combination[0], pd_combination[1], i)
                    ]
                    for i in range(1, 6)
                ]

                # Evaluate symbolic expressions
                n_val = n_func(q, dp, *a_coeffs)
                pel_val = pel_func(q, dp, *a_coeffs, *b_coeffs)

                if not pel_val == 0:
                    efficiency = q * dp / pel_val * 100
                else:
                    efficiency = -9999

                # Store result
                fan_tuple[s][fan] = (q, n_val, dp, pel_val, efficiency)
    return fan_tuple


def preprocess_fan_tuple(instance: pyo.ConcreteModel) -> dict:
    fan_tuple = get_fan_tuple(instance)

    table = []

    for scenario, fan_data in fan_tuple.items():
        for fan_key, (q, n_val, dp, pel_val, efficiency) in fan_data.items():
            table.append(
                (
                    int(scenario),
                    str(fan_key),
                    float(q),
                    float(n_val),
                    float(dp),
                    float(pel_val),
                    float(efficiency),
                )
            )

    dtype = np.dtype(
        [
            ("Scenario", "i4"),
            ("Fan", "S100"),
            ("VolumeFlow", "f8"),
            ("RelativeRotSpeed", "f8"),
            ("PressureIncrease", "f8"),
            ("PowerConsumption", "f8"),
            ("Efficiency", "f8"),
        ]
    )

    return {
        "FanResults": {
            "Content": np.array(table, dtype=dtype),
            "Metadata": {
                "Description": "Flattened fan_tuple results",
                "Units": "Volume Flow in m3/s, Relative Rot Speed in -, Pressure Increase in Pa, Power Consumption in W, Efficiency in %",
            },
        }
    }


def preprocess_virtual_pressure_losses(instance: pyo.ConcreteModel) -> dict:
    """
    Computes and formats virtual pressure losses (zeta values) across duct edges
    based on deviations between nonlinear functions and their piecewise-linear approximations.

    This function analyzes each edge in the duct network and calculates the total
    virtual pressure loss (zeta value) resulting from discrepancies in the friction
    and area approximations. Only non-negligible losses (above a small threshold)
    are retained. The results are returned in a dictionary suitable for HDF5 serialization,
    either as a structured NumPy array or a descriptive string if no losses are found.

    Args:
        instance (pyo.ConcreteModel): A Pyomo concrete model instance containing
            duct edge definitions and related parameters (e.g., nonlinear functions,
            hyperplanes, resistance coefficients, and bending/branching zeta factors).

    Returns:
        dict: A dictionary with the key `"ZetaEdge"` containing either:
            - A structured array with columns:
                - `E_duct` (bytes): Encoded string representation of edge (e.g., "1->2")
                - `Virtual pressure loss zeta` (float): Computed zeta value for the edge
            - Or a string indicating that no virtual pressure losses were detected.
          Metadata such as units and description are included for HDF5 export.
    """

    zeta_edge = {edge: 0 for edge in instance.E_duct}

    for i, j in instance.E_duct:

        fun_friction = instance.fun_nonlinear_duct_hb_friction[i, j].value
        hyperplane_friction = max(
            instance.duct_friction_hyperplanes[i, j, t].expr()
            for k, l, t in instance.duct_friction_hyperplanes_specific_set
            if (i, j) == (k, l)
        )
        if abs(fun_friction - hyperplane_friction) > 1e-9:
            zeta_edge[i, j] += (
                (fun_friction - hyperplane_friction)
                * instance.rho
                / 4
                * instance.duct_length[i, j]
                * instance.duct_resistance_coefficient
            )

        fun_area = instance.fun_duct_nonlinear_hb_area2[i, j].value
        hyperplane_area = max(
            instance.duct_area2_hyperplanes[i, j, t].expr()
            for k, l, t in instance.duct_area2_hyperplanes_specific_set
            if (i, j) == (k, l)
        )

        if abs(fun_area - hyperplane_area) > 1e-9:
            zeta_edge[i, j] += (
                (fun_area - hyperplane_area)
                * instance.rho
                / 2
                * (
                    instance.zeta_bending[i, j]
                    + instance.zeta_e_branch[i, j]
                    + instance.zeta_t_branch[i, j]
                )
            )

    # Filter out near-zero losses
    zeta_edge = {edge: val for edge, val in zeta_edge.items() if abs(val) > 1e-9}

    if len(zeta_edge) == 0:
        return {
            "ZetaEdge": {
                "Content": "No virtual pressure losses",
                "Metadata": {
                    "Description": "There are no virtual pressure losses on any E_duct edge."
                },
            }
        }

    # Format data into structured array
    table = []
    for edge, val in zeta_edge.items():
        # Ensure edge is a tuple of strings
        edge_str = f"{edge[0]}->{edge[1]}"
        table.append((edge_str.encode("utf-8"), float(val)))  # encode for 'S' dtype

    dtype = np.dtype(
        [
            ("E_duct", "S20"),  # Edge name like "1->2"
            ("Virtual pressure loss zeta", "f8"),
        ]
    )

    return {
        "ZetaEdge": {
            "Content": np.array(table, dtype=dtype),
            "Metadata": {
                "Description": "volume flow-dependent Zeta value according to virtual pressure loss across duct edges E_duct",
                "Units": "Zeta value for volume flow in Pa/(m^3/s)^2",
            },
        }
    }


def calculate_fan_fit_accuracy(
    instance: pyo.ConcreteModel,
) -> Tuple[Dict[Any, float], float, float]:
    """
    Calculates the accuracy of the fan power approximation compared to an exact symbolic model.

    Args:
        instance: A solved Pyomo model instance containing fan and scenario data.

    Returns:
        fan_fits: Dictionary mapping (scenario, edge, fan type) or (scenario, edge) to the fit ratio.
        total_power_consumption_exact: Total exact power consumption across all scenarios and fans.
        total_power_consumption: Total approximated power consumption across all scenarios and fans.
    """
    fan_fits: Dict[Any, float] = {}
    total_power_consumption_exact = 0.0
    total_power_consumption_approx = 0.0
    ploss_expr = get_symbolic_power_expression()

    for s in instance.Scenarios:
        scen = instance.scenario[s]
        time_share = instance.time_share[s]

        for edge in instance.E_fan_station:
            pel_fs_exact = 0.0
            pel_fs = 0.0
            update_flag = False

            for fan in instance.fan_set:
                if fan[:2] != edge:
                    continue

                if scen.fan_ind_active[fan].value > 1e-3:
                    pd_combination = fan[2:4]
                    update_flag = True
                    q = scen.fan_volume_flow[fan].value
                    dp = (
                        scen.fan_pressure_change_dimless[fan].value
                        * instance.fan_pressure_max[pd_combination]
                    )
                    pel_approx = scen.fan_power_loss[fan].value

                    a = filter_dict_by_prefix(
                        instance.fan_pressure_coefficients, pd_combination
                    ).values()
                    b = filter_dict_by_prefix(
                        instance.fan_power_coefficients, pd_combination
                    ).values()
                    pel_exact = ploss_expr(q, dp, *a, *b)
                    pel_fs_exact += pel_exact
                    pel_fs += pel_approx + q * dp

            if update_flag:
                fan_fits[(s, edge)] = (
                    (pel_fs / pel_fs_exact * 100) if pel_fs_exact != 0 else 0
                )
                total_power_consumption_exact += time_share * pel_fs_exact
                total_power_consumption_approx += time_share * pel_fs

    return fan_fits, total_power_consumption_exact, total_power_consumption_approx


def calculate_duct_convex_underestimation_fit_accuracy(
    instance: pyo.ConcreteModel,
) -> Dict[Tuple[Any, ...], Tuple[float, float]]:
    """
    Calculates the accuracy of pressure loss approximations in ducts.

    For each duct (i, j) the function compares the approximated pressure loss
    (based on a nonlinear zeta formulation) with the exact pressure loss calculated.

    The result is a dictionary containing, for each (scenario, i, j), a tuple:
        (approximated pressure loss, exact pressure loss),
    both scaled by the volume flow in that scenario.

    Args:
        instance: A solved Pyomo model instance containing scenario data, duct geometry,
                  resistance parameters, and volume flows.

    Returns:
        A dictionary mapping (i, j) to a tuple:
            (see below).
    """
    pressure_loss_fits = {}
    for i, j in instance.E_duct:
        height, width = (
            instance.duct_height[i, j].value,
            instance.duct_width[i, j].value,
        )
        area = height * width

        friction_exact = (height + width) / (height * width) ** 3
        friction_approx = instance.fun_nonlinear_duct_hb_friction[i, j].value

        zeta_exact = 1 / area**2
        zeta_approx = instance.fun_duct_nonlinear_hb_area2[i, j].value

        pressure_loss_fits.update(
            {
                (i, j): {
                    "approx": friction_approx,
                    "exact: ": friction_exact,
                    "width": instance.duct_width[i, j].value,
                    "height": instance.duct_height[i, j].value,
                    "relative friction": friction_approx / friction_exact,
                    "relative dp area2": zeta_approx / zeta_exact,
                }
            }
        )
    return pressure_loss_fits


def calculate_duct_friction_fit_accuracy(
    instance: pyo.ConcreteModel,
) -> Dict[Tuple[Any, ...], Tuple[float, float]]:
    """
    Calculates the accuracy of pressure loss due to friction in ducts for all scenarios.

    For each duct (i, j) and each scenario s, the function compares the approximated pressure loss
    (based on a nonlinear zeta formulation) with the exact pressure loss calculated.

    The result is a dictionary containing, for each (scenario, i, j), a tuple:
        (approximated pressure loss, exact pressure loss),
    both scaled by the volume flow in that scenario.

    Args:
        instance: A solved Pyomo model instance containing scenario data, duct geometry,
                  resistance parameters, and volume flows.

    Returns:
        A dictionary mapping (scenario, i, j) to a tuple:
            (approximated pressure loss [Pa], exact pressure loss [Pa]).
    """
    pressure_loss_fits = {}
    for i, j in instance.E_duct:
        height, width = (
            instance.duct_height[i, j].value,
            instance.duct_width[i, j].value,
        )

        friction_exact = (height + width) / (height * width) ** 3
        friction_approx = instance.fun_nonlinear_duct_hb_friction[i, j].value

        for s in instance.Scenarios:
            volume_flow = instance.scenario[s].volume_flow[i, j]
            pressure_loss_fits.update(
                {
                    (s, i, j): {
                        "approx": friction_approx
                        * volume_flow**2
                        * rho
                        / 4
                        * instance.duct_length[i, j]
                        * instance.duct_resistance_coefficient,
                        "exact": friction_exact
                        * volume_flow**2
                        * rho
                        / 4
                        * instance.duct_length[i, j]
                        * instance.duct_resistance_coefficient,
                    }
                }
            )
    return pressure_loss_fits


def calculate_velocity_constraint_fit_accuracy(
    instance: pyo.ConcreteModel,
) -> Dict[Tuple[str, str], float]:
    """
    Checks which ducts exceed their maximum allowed air velocity across all scenarios.

    For each duct (i, j), it computes the maximum velocity observed across all scenarios and compares it
    to the specified maximum velocity constraint for that duct. If the velocity exceeds the constraint,
    the duct is flagged and reported.

    Args:
        instance: A solved Pyomo model instance containing scenario data, duct geometry, and constraints.

    Returns:
        A dictionary mapping (i, j) duct tuples to their maximum velocity (in m/s),
        but only for ducts that exceed the maximum allowed velocity.
    """
    velocity_constraint_acc: Dict[Tuple[str, str], float] = {}

    for i, j in instance.E_duct:
        max_q = max(instance.scenario[s].volume_flow[i, j] for s in instance.Scenarios)
        area = instance.duct_width[i, j].value * instance.duct_height[i, j].value
        velocity = max_q / area

        if velocity >= instance.max_velocity[i, j]:
            print(f"Max velocity exceeded at duct ({i}, {j}): {velocity:.2f} m/s")
            velocity_constraint_acc[(i, j)] = velocity

    return velocity_constraint_acc


def plot_resulting_graph(
    instance, activation_flag: str = "purchase", scenario_index: int = 4
) -> None:
    """
    Plots the graph of the ventilation system for a specific scenario.

    Nodes are labeled with their pressure values.
    Edges are drawn with widths proportional to duct cross-sectional area.
    Edges are labeled based on their type: 'Duct', 'Fan', 'VFC', or 'fixed'.

    Args:
        instance: The Pyomo model instance with all necessary sets and values.
        scenario_index: Index of the scenario to visualize (default is 4).
    """
    scenario = instance.scenario[scenario_index]

    G = nx.DiGraph()
    G.add_nodes_from(instance.V)
    G.add_edges_from(instance.E)

    pos = graphviz_layout(G, prog="dot")
    plt.figure(figsize=(15, 12))

    # --- Node Drawing ---
    nx.draw_networkx_nodes(G, pos, node_size=300)

    # --- Edge Widths (Based on Duct Area) ---
    edge_widths = {}
    for i, j in instance.E:
        if (i, j) in instance.E_duct:
            area = (
                instance.duct_width[i, j].value * instance.duct_height[i, j].value * 10
            )
        else:
            area = 0.1
        G[i][j]["weight"] = area
        edge_widths[(i, j)] = area

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=G.edges(),
        width=[edge_widths[e] * 2 for e in G.edges()],
        edge_color="black",
    )

    # --- Node Labels (Pressure) ---
    pressure_labels = {v: round(scenario.pressure[v].value, 2) for v in instance.V}
    nx.draw_networkx_labels(G, pos, labels=pressure_labels, font_size=5)

    # --- Edge Labels ---
    edge_labels = {}
    for edge in instance.E:
        label = ""
        if edge in instance.E_duct:
            label = "Duct"
        elif edge in instance.E_fan_station:
            if activation_flag == "purchase":
                if instance.ind_purchase[edge].value > 1e-3:
                    label = "Fan"
            else:
                if instance.scenario[scenario_index].pressure_change[edge].value > 1e-2:
                    label = "Fan"
        elif edge in instance.E_vfc:
            if activation_flag == "purchase":
                if instance.ind_purchase[edge].value > 1e-3:
                    label = "VFC"
            else:
                if (
                    abs(instance.scenario[scenario_index].pressure_change[edge].value)
                    > 1e-3
                ):
                    label = "VFC"
        elif edge in instance.E_fixed:
            label = "fixed"

        if label:
            edge_labels[edge] = label

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, bbox=dict(alpha=0), font_color="black"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def solve_with_fixed_geometry(
    instance: pyo.ConcreteModel, max_load_case: int | Any = None
) -> float:
    fixed_instance = optimal_preplanning_fixed_geometry(instance, max_load_case)

    solver = pyo.SolverFactory("gurobi", solver_io="python")
    # solver.options["DualReductions"] = 0

    solver.solve(fixed_instance, tee=False)

    _, exact_power_consumption, _ = calculate_fan_fit_accuracy(fixed_instance)
    return exact_power_consumption


def calculate_total_duct_pressure_loss(instance, s, e, test_flag=False):
    velocity_fun = (
        lambda e: instance.scenario[s].volume_flow[e]
        / instance.duct_width[e].value
        / instance.duct_height[e].value
    )
    i, _ = e
    w, h = (
        instance.duct_width[e].value,
        instance.duct_height[e].value,
    )
    q = instance.scenario[s].volume_flow[e]
    v = q / w / h
    friction = (w + h) / (w * h) ** 3
    dp_friction = (
        rho
        / 4
        * instance.duct_length[e]
        * friction
        * instance.duct_resistance_coefficient
        * q**2
    )

    dp_bend = compute_pressure_loss_bend(w, h, v) * instance.n_duct_bendings[e]

    if i in instance.duct_t_branch_node:
        edge_in = next((o, p) for o, p in instance.E_duct if p == i)
        dp_t_branch = compute_pressure_loss_t_branch(
            w, h, velocity_fun(edge_in), velocity_fun(e)
        )
    else:
        dp_t_branch = 0

    dp_e_branch_straight, dp_e_branch_bend = 0, 0
    for k, l, m in instance.duct_e_branch:
        if (k, l) == e:
            edge_in = next((o, p) for o, p in instance.E_duct if p == i)
            dp_e_branch_straight, _ = compute_pressure_loss_e_branch(
                velocity_fun(edge_in),
                velocity_fun((k, l)),
                velocity_fun((k, m)),
            )
            break
        if (k, m) == e:
            edge_in = next((o, p) for o, p in instance.E_duct if p == i)
            _, dp_e_branch_bend = compute_pressure_loss_e_branch(
                velocity_fun(edge_in),
                velocity_fun((k, l)),
                velocity_fun((k, m)),
            )
            break

    dp_total = (
        dp_friction + dp_bend + dp_t_branch + dp_e_branch_straight + dp_e_branch_bend
    )
    if test_flag:
        return {
            "friction": dp_friction,
            "bend": dp_bend,
            "t_branch": dp_t_branch,
            "e_branch_straight": dp_e_branch_straight,
            "e_branch_bend": dp_e_branch_bend,
        }
    return dp_total


def fix_dimensions_and_pressure_losses(instance: pyo.ConcreteModel):
    for s in instance.Scenarios:
        for e in instance.E_duct:
            dp_total = calculate_total_duct_pressure_loss(instance, s, e)

            instance.scenario[s].pressure_change[e].value = dp_total
            instance.scenario[s].pressure_change[e].fixed = True

            instance.duct_width[e].fixed = True
            instance.duct_height[e].fixed = True

            instance.scenario[s].pressure_loss_duct[e].deactivate()

    instance.duct_friction_outer_polyhedral_approx.deactivate()
    instance.duct_area2_outer_polyhedral_approx.deactivate()
    return instance


def optimal_preplanning_fixed_geometry(
    instance: pyo.ConcreteModel, max_load_case: int | Any = None
):

    fixed_instance = copy.deepcopy(instance)

    fixed_instance = fix_dimensions_and_pressure_losses(fixed_instance)

    for e in instance.E_fan_station | instance.E_vfc:
        fixed_instance.ind_purchase[e].fixed = True

    for fan in instance.fan_set:
        fixed_instance.fan_ind_purchase[fan].fixed = True

    for s in instance.Scenarios:
        for fan in instance.fan_set:
            fixed_instance.scenario[s].fan_ind_active[fan].fixed = True

    if max_load_case:
        # in max load case scenario, the duct size could just meet the requirements that the fan can support. thus, the max_load_case is excluded because setting the pressure loss a little higher could result in an infeasible model.
        fixed_instance.scenario[
            max_load_case
        ].pressure_propagation_vfc_and_duct.deactivate()

    return fixed_instance


def optimal_preplanning_fixed_geometry2(
    instance: pyo.ConcreteModel, max_load_case: int | Any = None
):

    fixed_instance = copy.deepcopy(instance)

    for e in instance.E_duct:
        w, h = instance.duct_width[e].value, instance.duct_height[e].value
        fixed_instance.duct_width[e].set_value(w)
        fixed_instance.duct_height[e].set_value(h)
        fixed_instance.duct_width[e].fixed = True
        fixed_instance.duct_height[e].fixed = True

        friction = (w + h) / (w * h) ** 3
        duct_dp = 1 / (w * h) ** 2

        fixed_instance.fun_nonlinear_duct_hb_friction[e].set_value(friction)
        fixed_instance.fun_nonlinear_duct_hb_friction[e].fixed = True

        fixed_instance.fun_duct_nonlinear_hb_area2[e].set_value(duct_dp)
        fixed_instance.fun_duct_nonlinear_hb_area2[e].fixed = True

    fixed_instance.duct_friction_outer_polyhedral_approx.deactivate()
    fixed_instance.duct_area2_outer_polyhedral_approx.deactivate()

    for e in instance.E_fan_station | instance.E_vfc:
        fixed_instance.ind_purchase[e].set_value(round(instance.ind_purchase[e].value))
        fixed_instance.ind_purchase[e].fixed = True

    for fan in instance.fan_set:
        fixed_instance.fan_ind_purchase[fan].set_value(
            round(instance.fan_ind_purchase[fan].value)
        )
        fixed_instance.fan_ind_purchase[fan].fixed = True

    for s in instance.Scenarios:
        for fan in instance.fan_set:
            fixed_instance.scenario[s].fan_ind_active[fan].set_value(
                round(instance.scenario[s].fan_ind_active[fan].value)
            )
            fixed_instance.scenario[s].fan_ind_active[fan].fixed = True

    if max_load_case:
        # in max load case scenario, the duct size could just meet the requirements that the fan can support. thus, the max_load_case is excluded because setting the pressure loss a little higher could result in an infeasible model.
        fixed_instance.scenario[
            max_load_case
        ].pressure_propagation_vfc_and_duct.deactivate()

    return fixed_instance


def calculate_fittings_fit_accuracy(
    instance: pyo.ConcreteModel,
) -> None:
    duct_dp_acc_exact: Dict[Tuple[str, str, str, str], float] = {}
    duct_dp_acc_approx: Dict[Tuple[str, str, str, str], float] = {}

    for s in instance.Scenarios:
        volume_flows = instance.scenario[s].volume_flow
        widths = instance.duct_width
        heights = instance.duct_height

        for edge in instance.E_duct:
            i, j = edge
            velocity_fun = (
                lambda edge: volume_flows[edge]
                / widths[edge].value
                / heights[edge].value
            )

            (
                dp_e_straight_exact,
                dp_e_straight_approx,
                dp_e_bend_exact,
                dp_e_bend_approx,
                dp_t_branch_exact,
                dp_t_branch_approx,
            ) = (0, 0, 0, 0, 0, 0)

            width = widths[edge].value
            height = heights[edge].value

            for k, l, m in instance.duct_e_branch:
                if (k, l) == edge:
                    edge_in = next((o, p) for o, p in instance.E_duct if p == i)
                    dp_e_straight_exact, _ = compute_pressure_loss_e_branch(
                        velocity_fun(edge_in),
                        velocity_fun((k, l)),
                        velocity_fun((k, m)),
                    )
                    dp_e_straight_approx = (
                        instance.zeta_e_branch_straight_val
                        * rho
                        / 2
                        * velocity_fun([k, l]) ** 2
                    )
                    break
                if (k, m) == edge:
                    edge_in = next((o, p) for o, p in instance.E_duct if p == i)
                    _, dp_e_bend_exact = compute_pressure_loss_e_branch(
                        velocity_fun(edge_in),
                        velocity_fun((k, l)),
                        velocity_fun((k, m)),
                    )
                    dp_e_bend_approx = (
                        instance.zeta_e_branch_bend_val
                        * rho
                        / 2
                        * velocity_fun([k, m]) ** 2
                    )
                    break

            for k in instance.duct_t_branch_node:
                if k == i:
                    edge_in = next((o, p) for o, p in instance.E_duct if p == i)
                    dp_t_branch_exact = compute_pressure_loss_t_branch(
                        width, height, velocity_fun(edge_in), velocity_fun(edge)
                    )
                    dp_t_branch_approx = (
                        instance.zeta_t_branch_val * velocity_fun(edge) ** 2 * rho / 2
                    )
                    break

            dp_bend_exact = instance.n_duct_bendings[edge] * compute_pressure_loss_bend(
                width, height, velocity_fun(edge)
            )

            dp_bend_approx = (
                instance.n_duct_bendings[edge]
                * rho
                / 2
                * velocity_fun(edge) ** 2
                * instance.zeta_bending_val
            )

            duct_dp_acc_exact[s, i, j] = (
                dp_e_straight_exact,
                dp_e_bend_exact,
                dp_t_branch_exact,
                dp_bend_exact,
            )
            duct_dp_acc_approx[s, i, j] = (
                dp_e_straight_approx,
                dp_e_bend_approx,
                dp_t_branch_approx,
                dp_bend_approx,
            )
    return duct_dp_acc_exact, duct_dp_acc_approx


def compute_pressure_loss_bend(width: float, height: float, velocity: float):
    alpha = 90
    A = 1.6094 - 1.60868 * np.exp(-0.01089 * alpha)
    B = 0.21 / (1.5) ** 2.5
    C = (
        -1.03663e-4 * (width / height) ** 5
        + 0.00338 * (width / height) ** 4
        - 0.04277 * (width / height) ** 3
        + 0.25496 * (width / height) ** 2
        - 0.66296 * (width / height)
        + 1.4499
    )
    return A * B * C * rho / 2 * velocity**2


def compute_pressure_loss_t_branch(
    width_out: float, height_out: float, velocity_in: float, velocity_out: float
):
    if any([velocity_in, velocity_out]) == 0:
        return -99999
    K1, K2, K3, K4 = 0.0644, 0.0727, 0.3746, -3.4885
    pressure_loss_out = (
        (K1 * (width_out / height_out) + K2)
        * (velocity_out / velocity_in) ** (K3 * np.log(width_out / height_out) + K4)
        * rho
        / 2
        * velocity_out**2
    )
    return pressure_loss_out


def compute_pressure_loss_e_branch(
    velocity_in: float, velocity_straight: float, velocity_bend: float
) -> Tuple[float, float]:
    if any([velocity_in, velocity_bend, velocity_straight]) == 0:
        return (-99999, -99999)
    K1, K2, K3 = 183.3, 0.06, 0.17
    zeta_straight = K1 * np.exp(-velocity_straight / velocity_in / K2) + K3
    pressure_loss_straight = zeta_straight * velocity_straight**2 * rho / 2

    K1, K2, K3 = 301.95, 0.06, 0.75
    zeta_bend = K1 * np.exp(-velocity_bend / velocity_in / K2) + K3
    pressure_loss_bend = zeta_bend * velocity_bend**2 * rho / 2

    return pressure_loss_straight, pressure_loss_bend


def postprocess(instance: pyo.ConcreteModel, max_load_case: float = 6) -> None:
    """
    Code is a wrapper for postprocessing everything relevant

    Args:
        instance (pyo.ConcreteModel): solved pyomo instance
    """
    exact_power_consumption = solve_with_fixed_geometry(instance, max_load_case)
    exact_lcc = (
        instance.fan_energy_costs.expr()
        / instance.fan_power_consumption.expr()
        * exact_power_consumption
        + instance.total_invest_costs.expr()
    )
    lcc_error = (exact_lcc - instance.obj.expr()) / exact_lcc * 100

    return {
        "Postprocessing": {
            **preprocess_fan_tuple(instance),
            "Exact Power Consumption": {
                "Content": exact_power_consumption,
                "Metadata": {
                    "Information": "The exact power consumption is obtained by considering the exact power loss curves and the exact duct pressure losses.",
                    "Unit": "W",
                },
            },
            "Exact Life-Cycle Costs": {
                "Content": exact_lcc,
                "Metadata": {
                    "Information": "Scaled life-cycle costs with reduced power consumption.",
                    "Unit": "€",
                },
            },
            "Life-Cycle Cost Gap": {
                "Content": lcc_error,
                "Metadata": {
                    "Information": r"(True-Approx)/Approx in % Life-cycle costs, yielding the gap to an exact solution approach."
                },
            },
            **preprocess_virtual_pressure_losses(instance),
        }
    }
