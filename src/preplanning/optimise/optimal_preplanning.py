from collections import defaultdict

import numpy as np
import pyomo.environ as pyo


def find_leafy_edges(model, switch):
    "Wrapper around initializer. Need for abstract model"

    def initializer_(m):
        """
        Returns the list of fan station or vfc edges that lead to a leaf
        through a path with no branches in a directed tree.

        Parameters:
            edges (list of tuple): All directed edges (u, v).
            fan_station_edges (set of tuple): Subset of edges (u, v).

        Returns:
            list of tuple: Valid fan station edges.
        """
        adj = defaultdict(list)
        for u, v in m.E:
            adj[u].append(v)

        def is_valid_fan_edge(v):
            visited = set()
            while v in adj and len(adj[v]) == 1:
                if v in visited:
                    return False  # cycle detection
                visited.add(v)
                v = adj[v][0]
                if len(adj[v]) > 1:
                    return False  # branch
            return len(adj[v]) == 0  # valid only if ends in leaf

        if switch == "fan_station":
            edge_searcher = m.E_fan_station
        elif switch == "vfc":
            edge_searcher = m.E_vfc
        else:
            raise ValueError("no such edge type")

        return [e for e in edge_searcher if is_valid_fan_edge(e[1])]

    return initializer_


def model(
    duct_model=1,
    fan_model=1,
    branching_constraints=0,
    velocity_constraint=1,
    pressure_target_met=1,
):
    model = pyo.AbstractModel()

    model.Scenarios = pyo.Set(
        doc="Different load cases that are considered for operation"
    )

    model.V = pyo.Set(dimen=1, doc="Set of nodes")

    if pressure_target_met == 1:
        model.V_ports = pyo.Set(within=model.V, doc="Set of source and target nodes")

    elif pressure_target_met == 0:
        model.V_out = pyo.Set(within=model.V, doc="Set of source and target nodes")
        model.V_in = pyo.Set(within=model.V, doc="Set of source nodes")

    model.E = pyo.Set(within=model.V * model.V, doc="Set of edges")

    model.E_fan_station = pyo.Set(within=model.E, doc="Set of fan station edges")

    model.E_vfc = pyo.Set(
        within=model.E - model.E_fan_station, doc="Set of volume flow controller edges"
    )
    model.E_duct = pyo.Set(
        within=model.E - model.E_fan_station - model.E_vfc, doc="Set of duct edges"
    )
    model.E_duct_vertical = pyo.Set(
        within=model.E_duct,
        doc="Set of duct edges that are oriented vertically and thus shouldn't be limited when height is limited",
    )
    model.E_fixed = pyo.Set(
        within=model.E - model.E_fan_station - model.E_vfc - model.E_duct,
        doc="Set of fixed edges, i.e. a fixed zeta value according to dp = zeta * volume_flow^2",
    )
    model.E_empty = pyo.Set(
        initialize=model.E
        - model.E_fan_station
        - model.E_vfc
        - model.E_duct
        - model.E_fixed,
        doc="Set of empty nodes, where none of the above is elements is present. This set always has to contain all edges that are not defined else!",
    )

    # Parameters

    model.time_share = pyo.Param(
        model.Scenarios, doc="Time share of each scenario. Sums up to 1"
    )

    model.max_pressure = pyo.Param(
        doc="Maximum pressure in the problem. Used for bigM constraints"
    )

    # Finances

    model.electric_energy_costs = pyo.Param(doc="Costs of electric energy in € / Wh")
    model.operating_years = pyo.Param(
        doc="Number of years the plant is designed to be operating"
    )
    model.operating_days_per_year = pyo.Param(
        doc="Number of days in a year the plant is operating"
    )
    model.operating_hours_per_day = pyo.Param(
        doc="Number of hours in a day the plant is operating"
    )

    model.component_names = pyo.Set(
        initialize=["fan", "vfc", "duct"],
        doc="Component names used for computing investment costs according to VDI 2067",
    )

    model.price_change_factor_service_maintenance = pyo.Param(
        within=pyo.PositiveReals,
        doc="Price change factor of the service and maintenance acc. to VDI 2067",
    )

    model.price_change_factor_electricity = pyo.Param(
        within=pyo.PositiveReals,
        doc="Price change factor of the energy costs acc. to VDI 2067",
    )

    model.maintenance_factor = pyo.Param(
        model.component_names,
        within=pyo.NonNegativeReals,
        doc="Factor for the cost of maintenance acc. to VDI 2067",
        initialize={"fan": 0.03, "vfc": 0.03, "duct": 0.02},
    )

    model.service_factor = pyo.Param(
        model.component_names,
        within=pyo.NonNegativeReals,
        doc="Factor for the cost of service acc. to VDI 2067",
        initialize={"fan": 0.01, "vfc": 0.01, "duct": 0},
    )

    model.deprecation_period = pyo.Param(
        model.component_names,
        within=pyo.PositiveIntegers,
        doc="Number of years of the deprecation period acc. to VDI 2067",
        initialize={"fan": 12, "vfc": 12, "duct": 30},
    )

    model.interest_rate = pyo.Param(
        within=pyo.NonNegativeReals,
        doc="Interest factor of price change acc. to VDI 2067",
    )

    model.vfc_costs = pyo.Param(
        doc="Costs of a single VFC. Assumed to be independend on height and width of the respective duct."
    )

    model.ind_purchase = pyo.Var(
        model.E_fan_station | model.E_vfc,
        within=pyo.Binary,
        doc="purchase indicator variable for fan station and VFC.",
    )

    if duct_model == 1:
        model.duct_friction_hyperplanes_set = pyo.Set(
            doc="Set of duct friction hyperplanes. Used for outer polyhedral approximation of (h+w)/(h^3*w^3)"
        )

        model.duct_friction_hyperplanes_specific_pre_set = pyo.Set(
            model.E_duct,
            doc="Pre set of duct friction hyperplanes at a specific edge. Used for outer polyhedral approximation of (h+w)/(h^3*w^3)",
        )

        def duct_friction_hyperplanes_specific_creator(model):
            pairs = []
            for i, j in model.E_duct:
                for x in model.duct_friction_hyperplanes_specific_pre_set[i, j]:
                    pairs.append((i, j, x))
            return pairs

        model.duct_friction_hyperplanes_specific_set = pyo.Set(
            dimen=3,
            initialize=duct_friction_hyperplanes_specific_creator,
            doc="Set of duct friction hyperplanes at a specific edge. Used for outer polyhedral approximation of (h+w)/(h^3*w^3)",
        )

        model.duct_area2_hyperplanes_set = pyo.Set(
            doc="Set of duct area^2 hyperplanes. Used for outer polyhedral approximation of 1/(h^2b^2)"
        )

        model.duct_area2_hyperplanes_specific_pre_set = pyo.Set(
            model.E_duct,
            doc="Pre set of duct area^2 hyperplanes at a specific edge. Used for outer polyhedral approximation of 1/(h^2b^2)",
        )

        def duct_area2_hyperplanes_specific_creator(model):
            pairs = []
            for i, j in model.E_duct:
                for x in model.duct_area2_hyperplanes_specific_pre_set[i, j]:
                    pairs.append((i, j, x))
            return pairs

        model.duct_area2_hyperplanes_specific_set = pyo.Set(
            dimen=3,
            initialize=duct_area2_hyperplanes_specific_creator,
            doc="Set of duct area2 hyperplanes at a specific edge. Used Used for outer polyhedral approximation of 1/(h^2b^2)",
        )

        model.duct_resistance_coefficient = pyo.Param(
            doc="resistance coefficient lambda of Darcy-Weisbach equation"
        )

        model.duct_area_costs = pyo.Param(doc="Costs per square meter duct in € / m^2.")

        model.duct_width_min = pyo.Param(model.E_duct, doc="Minimum duct width in m")
        model.duct_width_max = pyo.Param(model.E_duct, doc="Maximum duct width in m")

        model.duct_height_min = pyo.Param(model.E_duct, doc="Minimum duct height in m")
        model.duct_height_max = pyo.Param(model.E_duct, doc="Maximum duct height in m")

        model.duct_length = pyo.Param(model.E_duct, doc="Duct length in m")

        model.n_duct_bendings = pyo.Param(
            model.E_duct,
            initialize=0,
            doc="Number of bendings in duct (i,j), used for pressure loss calculation.",
        )

        model.duct_t_branch_node = pyo.Set(
            within=model.V,
            doc="Set of nodes that are center of a T-branch, used for pressure loss calculation.",
        )
        model.duct_e_branch = pyo.Set(
            within=model.V * model.V * model.V,
            doc="Set of node triples (k,l,m) that are part of a branching where branch l goes straight and branch m bends.",
        )

        model.zeta_bending_val = pyo.Param(
            initialize=0.0675, doc="Fixed zeta value for all bendings."
        )
        model.zeta_t_branch_val = pyo.Param(
            initialize=0.094, doc="Fixed zeta value for T-branches added to edges."
        )
        model.zeta_e_branch_straight_val = pyo.Param(
            initialize=0.17,
            doc="Fixed zeta value of the straight branch of an e-branch",
        )
        model.zeta_e_branch_bend_val = pyo.Param(
            initialize=0.75, doc="Fixed zeta value of the bend branch of an e-branch"
        )

        model.duct_friction_slope_width = pyo.Param(
            model.duct_friction_hyperplanes_set,
            doc="Slope in width direction of the friction hyperplanes \
                for outer polyhedral approximation",
        )

        model.duct_friction_slope_height = pyo.Param(
            model.duct_friction_hyperplanes_set,
            doc="Slope in height direction of the friction hyperplanes \
                for outer polyhedral approximation",
        )

        model.duct_friction_intercept = pyo.Param(
            model.duct_friction_hyperplanes_set,
            doc="Intercept of the friction hyperplanes\
                for outer polyhedral approximation",
        )

        model.duct_area2_slope_width = pyo.Param(
            model.duct_area2_hyperplanes_set,
            doc="Slope in width direction of the area^2 hyperplanes\
                for outer polyhedral approximation",
        )

        model.duct_area2_slope_height = pyo.Param(
            model.duct_area2_hyperplanes_set,
            doc="Slope in height direction of the area^2 hyperplanes\
                for outer polyhedral approximation",
        )

        model.duct_area2_intercept = pyo.Param(
            model.duct_area2_hyperplanes_set,
            doc="Intercept of the area^2 hyperplanes\
                for outer polyhedral approximation",
        )

        model.duct_width = pyo.Var(
            model.E_duct,
            bounds=lambda model, i, j: (
                model.duct_width_min[i, j],
                model.duct_width_max[i, j],
            ),
            doc="Duct width of duct (i,j)",
        )

        model.duct_height = pyo.Var(
            model.E_duct,
            bounds=lambda model, i, j: (
                model.duct_height_min[i, j],
                model.duct_height_max[i, j],
            ),
            doc="Duct height of duct (i,j)",
        )

        model.fun_nonlinear_duct_hb_friction = pyo.Var(
            model.E_duct,
            within=pyo.NonNegativeReals,
            bounds=lambda model, i, j: (
                (model.duct_width_max[i, j] + model.duct_height_max[i, j])
                / (model.duct_width_max[i, j] * model.duct_height_max[i, j]) ** 3,
                (model.duct_width_min[i, j] + model.duct_height_min[i, j])
                / (model.duct_width_min[i, j] * model.duct_height_min[i, j]) ** 3,
            ),
            doc="Friction term (w+h)/w^3h^3 of duct (i,j)",
        )

        model.fun_duct_nonlinear_hb_area2 = pyo.Var(
            model.E_duct,
            within=pyo.NonNegativeReals,
            bounds=lambda model, i, j: (
                1 / (model.duct_width_max[i, j] * model.duct_height_max[i, j]) ** 2,
                1 / (model.duct_width_min[i, j] * model.duct_height_min[i, j]) ** 2,
            ),
            doc="Area^2 term 1/w^2h^2 of duct (i,j)",
        )

        @model.Expression(
            model.duct_friction_hyperplanes_specific_set,
            doc="Hyperplane expression approximating (h+b)/h^3/b^3",
        )
        def duct_friction_hyperplanes(model, i, j, t):
            return (
                model.duct_friction_slope_width[t] * model.duct_width[i, j]
                + model.duct_friction_slope_height[t] * model.duct_height[i, j]
                + model.duct_friction_intercept[t]
            )

        @model.Expression(
            model.duct_area2_hyperplanes_specific_set,
            doc="Hyperplane expression approximating 1/h^2/b^2",
        )
        def duct_area2_hyperplanes(model, i, j, t):
            return (
                model.duct_area2_slope_width[t] * model.duct_width[i, j]
                + model.duct_area2_slope_height[t] * model.duct_height[i, j]
                + model.duct_area2_intercept[t]
            )

        @model.Constraint(
            model.duct_friction_hyperplanes_specific_set,
            doc="Friction term must be larger than respective hyperplanes",
        )
        def duct_friction_outer_polyhedral_approx(model, i, j, t):
            return (
                model.fun_nonlinear_duct_hb_friction[i, j]
                >= model.duct_friction_hyperplanes[i, j, t]
            )

        @model.Constraint(
            model.duct_area2_hyperplanes_specific_set,
            doc="Area^2 term must be larger than respective hyperplanes",
        )
        def duct_area2_outer_polyhedral_approx(model, i, j, t):
            return (
                model.fun_duct_nonlinear_hb_area2[i, j]
                >= model.duct_area2_hyperplanes[i, j, t]
            )

        @model.Param(model.E_duct, doc="Zeta value of duct-t-branch = 0.7")
        def zeta_t_branch(model, i, j):
            if i in model.duct_t_branch_node:
                return model.zeta_t_branch_val
            return 0

        @model.Param(
            model.E_duct,
            doc="zeta value of a E branch where one flow goes straight and one bends. (k,l) goes straight and the (k,m) bends.",
        )
        def zeta_e_branch(model, i, j):
            for k, l, m in model.duct_e_branch:
                if k == i and l == j:
                    return model.zeta_e_branch_straight_val
                if k == i and m == j:
                    return model.zeta_e_branch_bend_val
            return 0

        @model.Param(model.E_duct, doc="redundancy with n_duct_bendings")
        def zeta_bending(model, i, j):
            return model.n_duct_bendings[i, j] * model.zeta_bending_val

        @model.Constraint(model.E_duct, doc="limit height to width ratio")
        def limit_height_to_width_ratio1(model, i, j):
            return model.duct_width[i, j] >= 1 / 3 * model.duct_height[i, j]

        @model.Constraint(model.E_duct, doc="limit height to width ratio")
        def limit_height_to_width_ratio2(model, i, j):
            return model.duct_height[i, j] >= 1 / 3 * model.duct_width[i, j]

    # FAN MODEL
    if fan_model == 1:

        model.E_fan_station_leaf = pyo.Set(
            within=model.E_fan_station,
            initialize=find_leafy_edges(model, "fan_station"),
            doc="Set of fan stations that are connected to an air diffuser without an intermediate branch",
        )

        model.E_vfc_leaf = pyo.Set(
            within=model.E_vfc,
            initialize=find_leafy_edges(model, "vfc"),
            doc="Set of VFCs that are connected to an air diffuser without an intermediate branch",
        )

        model.max_num_fans_per_fan_station = pyo.Param(
            model.E_fan_station,
            doc="Maximum number of fans that can be placed in a fan station",
        )

        model.leaf_component_decision = pyo.Var(
            within=pyo.Binary,
            doc="Decision variable that ensures that either in front of *all* rooms/zones a fan station or a VFC is placed. Additional fans or VFCs are not hindered.",
        )

        @model.Constraint(
            model.E_fan_station_leaf,
            doc="Decision whether fan stations are purchased in front of *all* rooms/zones",
        )
        def leaves_all_fan_stations_or_all_vfcs_purchased_a(model, i, j):
            return model.ind_purchase[i, j] >= model.leaf_component_decision

        @model.Constraint(
            model.E_vfc_leaf,
            doc="Decision whether VFCs are purchased in front of *all* rooms/zones",
        )
        def leaves_all_fan_stations_or_all_vfcs_purchased_b(model, i, j):
            return model.ind_purchase[i, j] >= (1 - model.leaf_component_decision)

        model.fan_product_line = pyo.Set(doc="Set of fan product lines")

        model.fan_diameter = pyo.Set(
            model.fan_product_line, doc="Set of fan diameters in a certain product line"
        )

        def valid_pd_pairs_init(model):
            pairs = []
            for p in model.fan_product_line:
                for d in model.fan_diameter[p]:
                    pairs.append((p, d))
            return pairs

        model.p_d_combination_set = pyo.Set(
            dimen=2,
            initialize=valid_pd_pairs_init,
            doc="Set of (product line, fan diameter) of fans",
        )

        model.fan_hyperplanes_overestimation_specific_pre_set = pyo.Set(
            model.Scenarios,
            model.E_fan_station,
            model.p_d_combination_set,
            doc="Set of sets of scenarios, E_fan stations and distinct fans for overestimation of fan power loss",
        )

        model.fan_hyperplanes_overestimation_pre_set = pyo.Set(
            model.p_d_combination_set,
            doc="Set of sets of distinct fan for overestimation of fan power loss",
        )

        def fan_hyperplanes_overestimation_specific_creator(model):
            pairs = []
            for s in model.Scenarios:
                for i, j in model.E_fan_station:
                    for p, d in model.p_d_combination_set:
                        for x in model.fan_hyperplanes_overestimation_specific_pre_set[
                            s, i, j, p, d
                        ]:
                            pairs.append((s, i, j, p, d, x))
            return pairs

        def fan_hyperplanes_overestimation_creator(model):
            pairs = []
            for p, d in model.p_d_combination_set:
                for x in model.fan_hyperplanes_overestimation_pre_set[p, d]:
                    pairs.append((p, d, x))
            return pairs

        model.fan_hyperplanes_overestimation_specific_set = pyo.Set(
            dimen=6,
            initialize=fan_hyperplanes_overestimation_specific_creator,
            doc="Set of supporting hyperplanes per scenario per fan station per distinct fan (p,d) for overestimation of fan power loss",
        )

        model.fan_hyperplanes_overestimation_set = pyo.Set(
            dimen=3,
            initialize=fan_hyperplanes_overestimation_creator,
            doc="Set of supporting hyperplanes per distinct fan (p,d) for overestimation of fan power loss",
        )

        model.fan_n = pyo.Set(
            model.p_d_combination_set, doc="Set of number of distinct fans"
        )

        model.fan_set = pyo.Set(
            doc="Set of (*edge, product line, fan diameter, number) of fans",
        )

        model.fan_hyperplanes_underestimation_pre_set = pyo.Set(
            model.p_d_combination_set,
            doc="Set of sets of distinct fans for underestimation of fan power loss",
        )

        model.fan_hyperplanes_underestimation_specific_pre_set = pyo.Set(
            model.Scenarios,
            model.E_fan_station,
            model.p_d_combination_set,
            doc="Set of sets of distinct fans for underestimation of fan power loss",
        )

        def fan_hyperplanes_underestimation_set(model):
            fan_hyp_set = []
            for p, d in model.p_d_combination_set:
                for t in model.fan_hyperplanes_underestimation_pre_set[p, d]:
                    fan_hyp_set.append((p, d, t))
            return fan_hyp_set

        def fan_hyperplanes_underestimation_specific_set(model):
            fan_hyp_set = []
            for s in model.Scenarios:
                for i, j in model.E_fan_station:
                    for p, d in model.p_d_combination_set:
                        for t in model.fan_hyperplanes_underestimation_specific_pre_set[
                            s, i, j, p, d
                        ]:
                            fan_hyp_set.append((s, i, j, p, d, t))
            return fan_hyp_set

        model.fan_hyperplanes_underestimation_specific_set = pyo.Set(
            initialize=fan_hyperplanes_underestimation_specific_set,
            doc="Set of supporting hyperplanes per scenario per fan station per distinct fan (p,d) for underestimation of fan power loss",
        )

        model.fan_hyperplanes_underestimation_set = pyo.Set(
            initialize=fan_hyperplanes_underestimation_set,
            doc="Set of supporting hyperplanes per distinct fan (p,d) for underestimation of fan power loss",
        )

        model.fan_pressure_coefficients = pyo.Param(
            model.p_d_combination_set,
            pyo.RangeSet(3),
            doc="Fan pressure coefficients for dp = a1*q^2 + a2*q*n + a3*n^2 - not used in model, only added for postprocessing",
        )

        model.fan_power_coefficients = pyo.Param(
            model.p_d_combination_set,
            pyo.RangeSet(5),
            doc="Fan power coefficients for pel = b1*q^3 + b2*q^2*n + b3*q*n^2 + b4*n^3 + b5 - not used in model, only added for postprocessing",
        )

        model.fan_hyperplanes_underestimation_slope_volume_flow = pyo.Param(
            model.fan_hyperplanes_underestimation_set,
            doc="Slope in volume flow direction of the fan hyperplanes \
                for outer polyhedral approximation",
        )

        model.fan_hyperplanes_underestimation_slope_pressure = pyo.Param(
            model.fan_hyperplanes_underestimation_set,
            doc="Slope in pressure direction of the fan hyperplanes \
                for outer polyhedral approximation",
        )

        model.fan_hyperplanes_underestimation_intercept = pyo.Param(
            model.fan_hyperplanes_underestimation_set,
            doc="Intercept of the fan curve approx\
                for outer polyhedral approximation",
        )

        model.fan_hyperplanes_overestimation_slope_volume_flow = pyo.Param(
            model.fan_hyperplanes_overestimation_set,
            doc="Slope of overestimating hyperplanes for fan power loss",
        )

        model.fan_hyperplanes_overestimation_intercept = pyo.Param(
            model.fan_hyperplanes_overestimation_set,
            doc="Intercept of overestimating hyperplanes for fan power loss",
        )

        model.fan_power_loss_max = pyo.Param(
            model.p_d_combination_set,
            doc="Maximal electric power consumption of distinct fan (p,d)",
        )

        def calculate_fan_power_loss_max_of_all_fans(model):
            return max(
                model.fan_power_loss_max[p, d] for (p, d) in model.p_d_combination_set
            )

        model.fan_power_loss_max_of_all_fans = pyo.Param(
            initialize=calculate_fan_power_loss_max_of_all_fans,
            doc="Maximal electric power of all fans",
        )

        model.fan_volume_flow_max = pyo.Param(
            model.p_d_combination_set, doc="Maximal volume flow of distinct fan (p,d)"
        )
        model.fan_pressure_max = pyo.Param(
            model.p_d_combination_set, doc="Maximal pressure rise of distinct fan (p,d)"
        )

        model.fan_costs = pyo.Param(
            model.p_d_combination_set, doc="Cost of distinct fan (p,d)"
        )

        model.fan_ind_purchase = pyo.Var(
            model.fan_set,
            within=pyo.Binary,
            doc="Purchase indicator for fan (p,d,n) in fan_station (i,j)",
        )

        @model.Constraint(
            model.E_fan_station, doc="Limit number of fans per fan station"
        )
        def only_n_fans_per_fan_station(model, i, j):
            return (
                sum(
                    model.fan_ind_purchase[i, j, p, d, n]
                    for (k, l, p, d, n) in model.fan_set
                    if (i, j) == (k, l)
                )
                <= model.max_num_fans_per_fan_station[i, j]
            )

    model.rho = pyo.Param(doc="Density of air. Used in pressure loss calculation")

    model.fixed_zeta = pyo.Param(
        model.E_fixed, doc="Fixed zeta value per fixed edge (i,j)"
    )

    @model.Block(model.Scenarios)
    def scenario(m_scen, s):
        model = m_scen.parent_block()

        m_scen.volume_flow = pyo.Param(model.E, doc="Volume flow along edge in m³/s")

        m_scen.pressure = pyo.Var(
            model.V,
            bounds=(-model.max_pressure / 3, model.max_pressure),
            doc="Pressure at node in Pa",
        )

        m_scen.pressure_change = pyo.Var(
            model.E,
            within=pyo.NonNegativeReals,
            bounds=(0, model.max_pressure),
            doc="Pressure change along edge in Pa",
        )

        m_scen.ind_active = pyo.Var(
            model.E_fan_station,
            within=pyo.Binary,
            doc="Activation indicator for fan station",
        )

        if duct_model == 1:

            @m_scen.Expression(
                model.E_duct,
                doc="Zeta value of duct friction using the volume flow and not the velocity!",
            )
            def zeta_volume_flow_duct_friction(m_scen, i, j):
                return (
                    model.rho
                    / 4
                    * model.duct_length[i, j]
                    * model.fun_nonlinear_duct_hb_friction[i, j]
                    * model.duct_resistance_coefficient
                )

            @m_scen.Expression(
                model.E_duct,
                doc="Duct zeta as function of bendings, branches and friction. Equation is pressure_loss = zeta * volume_flow^2",
            )
            def duct_zeta_volume_flow_calc(m_scen, i, j):
                return (
                    model.rho
                    / 2
                    * (
                        model.fun_duct_nonlinear_hb_area2[i, j]
                        * (
                            model.zeta_bending[i, j]
                            + model.zeta_e_branch[i, j]
                            + model.zeta_t_branch[i, j]
                        )
                    )
                    + m_scen.zeta_volume_flow_duct_friction[i, j]
                )

            @m_scen.Constraint(
                model.E_duct,
                doc="Pressure change along duct is equal to zeta*volume_flow^2",
            )
            def pressure_loss_duct(m_scen, i, j):
                return (
                    m_scen.pressure_change[i, j]
                    == m_scen.duct_zeta_volume_flow_calc[i, j]
                    * m_scen.volume_flow[i, j] ** 2
                )

        elif duct_model == 0:

            @m_scen.Constraint(
                model.E_duct,
                doc="if duct_model is switched off, then pressure change is zero",
            )
            def pressure_loss_duct(m_scen, i, j):
                return m_scen.pressure_change[i, j] == 0

        if fan_model == 1:
            m_scen.fan_power_loss = pyo.Var(
                model.fan_set,
                within=pyo.NonNegativeReals,
                bounds=lambda m, i, j, p, d, n: (0, model.fan_power_loss_max[p, d]),
                doc="Electrical power consumption of fan in W",
            )

            m_scen.fan_power_loss_intermediate = pyo.Var(
                model.fan_set,
                within=pyo.NonNegativeReals,
                bounds=lambda m, i, j, p, d, n: (0, model.fan_power_loss_max[p, d]),
                doc="Intermediate value of electrical power consumption of fan in W. Necessary for activation bigM constraints.",
            )

            m_scen.fan_volume_flow = pyo.Var(
                model.fan_set,
                within=pyo.NonNegativeReals,
                bounds=lambda m, i, j, p, d, n: (
                    0,
                    min(m_scen.volume_flow[i, j], model.fan_volume_flow_max[p, d]),
                ),
                doc="Intermediate value of volume flow of fan in m³/h. Necessary for activation bigM constraints.",
            )

            m_scen.fan_volume_flow_intermediate = pyo.Var(
                model.fan_set,
                within=pyo.NonNegativeReals,
                bounds=lambda m, i, j, p, d, n: (
                    0,
                    min(m_scen.volume_flow[i, j], model.fan_volume_flow_max[p, d]),
                ),
                doc="Intermediate value of volume flow of fan in m³/h. Necessary for activation bigM constraints.",
            )

            m_scen.fan_pressure_change_dimless = pyo.Var(
                model.fan_set,
                within=pyo.NonNegativeReals,
                bounds=(0, 1),
                doc="Pressure change of fan in dimensionless form. Dimensionless form has numerical stability reasons.",
            )

            m_scen.fan_ind_active = pyo.Var(model.fan_set, within=pyo.Binary)

            @m_scen.Constraint(
                model.fan_set, doc="Only purchased fans can be activated"
            )
            def only_purchased_fans_are_active(m_scen, i, j, p, d, n):
                return (
                    m_scen.fan_ind_active[i, j, p, d, n]
                    <= model.fan_ind_purchase[i, j, p, d, n]
                )

            @m_scen.Constraint(
                model.fan_set,
                doc="For a fan to be active, the fan station also has to be active.",
            )
            def fans_active_only_if_fan_station_active(m_scen, i, j, p, d, n):
                return m_scen.ind_active[i, j] >= m_scen.fan_ind_active[i, j, p, d, n]

        @m_scen.Constraint(
            model.E_fan_station, doc="Only purchased fan stations can be activated"
        )
        def only_purchased_fan_stations_are_active(m_scen, i, j):
            return m_scen.ind_active[i, j] <= model.ind_purchase[i, j]

        if pressure_target_met == 1:

            @m_scen.Constraint(
                model.V_ports, doc="Pressure at target and source node is equal to zero"
            )
            def set_pressure_at_ports_to_zero(m_scen, v):
                return m_scen.pressure[v] == 0

        elif pressure_target_met == 0:

            @m_scen.Constraint(
                model.V_in, doc="Pressure at source node is equal to zero"
            )
            def set_pressure_at_source_to_zero(m_scen, v):
                return m_scen.pressure[v] == 0

            @m_scen.Constraint(
                model.V_out, doc="Pressure at source node is equal or larger than zero"
            )
            def set_pressure_at_targets_geq_zero(m_scen, v):
                return m_scen.pressure[v] >= 0

        @m_scen.Constraint(
            model.E_fan_station,
            doc="Pressure change of fan is zero if fan is not active",
        )
        def pressure_increase_fan(m_scen, i, j):
            return (
                m_scen.pressure_change[i, j]
                <= model.max_pressure * m_scen.ind_active[i, j]
            )

        @m_scen.Constraint(
            model.E_fixed,
            doc="Pressure change of a fixed component is zeta*volume_flow^2",
        )
        def pressure_loss_fix(m_scen, i, j):
            return (
                m_scen.pressure_change[i, j]
                == model.fixed_zeta[i, j] * m_scen.volume_flow[i, j] ** 2
            )

        @m_scen.Constraint(
            model.E_vfc, doc="Pressure loss of a VFC is zero if VFC is not active"
        )
        def pressure_loss_vfc(m_scen, i, j):
            return (
                m_scen.pressure_change[i, j]
                <= model.max_pressure * model.ind_purchase[i, j]
            )

        # pressure propagation

        @m_scen.Constraint(
            model.E_fan_station,
            doc="Pressure difference along a fan station edge is equal to the fan station's pressure rise",
        )
        def pressure_propagation_fan(m_scen, i, j):
            return (
                -m_scen.pressure[i] + m_scen.pressure[j] == m_scen.pressure_change[i, j]
            )

        @m_scen.Constraint(
            model.E_vfc | model.E_duct | model.E_fixed | model.E_empty,
            doc="Pressure difference along a VFC, duct, fixed component of empty edge is equal to the negative pressure_change of that edge",
        )
        def pressure_propagation_vfc_and_duct(m_scen, i, j):
            return (
                -m_scen.pressure[i] + m_scen.pressure[j]
                == -m_scen.pressure_change[i, j]
            )

        if fan_model == 1:

            # ALL FAN STATION CONSTRAINTS

            @m_scen.Expression(
                model.E_fan_station, doc="Hydraulic power of the fan station"
            )
            def electric_power_consumption_factor_fan_station(m_scen, i, j):
                return m_scen.volume_flow[i, j] * m_scen.pressure_change[i, j]

            @m_scen.Constraint(
                model.fan_set, doc="bigM connecting intermediate power consumption"
            )
            def power_loss_bigm_a(m_scen, i, j, p, d, n):
                return m_scen.fan_power_loss_intermediate[
                    i, j, p, d, n
                ] - m_scen.fan_power_loss[i, j, p, d, n] <= model.fan_power_loss_max[
                    p, d
                ] * (
                    1 - m_scen.fan_ind_active[i, j, p, d, n]
                )

            @m_scen.Constraint(
                model.fan_set, doc="bigM connecting intermediate power consumption"
            )
            def power_loss_bigm_b(m_scen, i, j, p, d, n):
                return m_scen.fan_power_loss_intermediate[
                    i, j, p, d, n
                ] - m_scen.fan_power_loss[i, j, p, d, n] >= -model.fan_power_loss_max[
                    p, d
                ] * (
                    1 - m_scen.fan_ind_active[i, j, p, d, n]
                )

            @m_scen.Expression(
                model.E_fan_station,
                doc="Power consumption of a fan station = fan's power loss + hydraulic power",
            )
            def electric_power_consumption_fan_station(m_scen, i, j):
                return (
                    sum(
                        m_scen.fan_power_loss[i, j, p, d, n]
                        for (k, l, p, d, n) in model.fan_set
                        if (i, j) == (k, l)
                    )
                    + m_scen.electric_power_consumption_factor_fan_station[i, j]
                )

            @m_scen.Constraint(
                model.fan_set, doc="Electric power of fan is zero if fan is not active"
            )
            def power_loss_bigm_c(m_scen, i, j, p, d, n):
                return (
                    m_scen.fan_power_loss[i, j, p, d, n]
                    <= model.fan_power_loss_max[p, d]
                    * m_scen.fan_ind_active[i, j, p, d, n]
                )

            @m_scen.Constraint(
                model.fan_set,
                doc="Pressure change of fan station and fan are equal if fan is active",
            )
            def pressure_change_connection_to_fan_station_bigm_a(m_scen, i, j, p, d, n):
                return m_scen.fan_pressure_change_dimless[
                    i, j, p, d, n
                ] * model.fan_pressure_max[p, d] - m_scen.pressure_change[
                    i, j
                ] <= model.max_pressure * (
                    1 - m_scen.fan_ind_active[i, j, p, d, n]
                )

            @m_scen.Constraint(
                model.fan_set,
                doc="Pressure change of fan station and fan are equal if fan is active",
            )
            def pressure_change_connection_to_fan_station_bigm_b(m_scen, i, j, p, d, n):
                return m_scen.fan_pressure_change_dimless[
                    i, j, p, d, n
                ] * model.fan_pressure_max[p, d] - m_scen.pressure_change[
                    i, j
                ] >= -model.max_pressure * (
                    1 - m_scen.fan_ind_active[i, j, p, d, n]
                )

            @m_scen.Constraint(
                model.E_fan_station,
                doc="Pressure change of fan station is zero if fan station is not active",
            )
            def pressure_change_connection_to_fan_station_bigm_c(m_scen, i, j):
                return (
                    m_scen.pressure_change[i, j]
                    <= model.max_pressure * m_scen.ind_active[i, j]
                )

            @m_scen.Constraint(
                model.fan_set,
                model.fan_hyperplanes_underestimation_specific_set,
                doc="Intermediate electrical power consumption >= underestimating hyperplanes",
            )
            def power_loss_lower_bound_fan(
                m_scen, i, j, p, d, n, s_, i_, j_, p_, d_, t
            ):
                if (s, i, j, p, d) == (s_, i_, j_, p_, d_):
                    return (
                        m_scen.fan_power_loss_intermediate[i, j, p, d, n]
                        >= model.fan_hyperplanes_underestimation_slope_pressure[p, d, t]
                        * m_scen.fan_pressure_change_dimless[i, j, p, d, n]
                        * model.fan_pressure_max[p, d]
                        + model.fan_hyperplanes_underestimation_slope_volume_flow[
                            p, d, t
                        ]
                        * m_scen.fan_volume_flow_intermediate[i, j, p, d, n]
                        + model.fan_hyperplanes_underestimation_intercept[p, d, t]
                    )
                return pyo.Constraint.Skip

            @m_scen.Constraint(
                model.fan_set,
                model.fan_hyperplanes_overestimation_specific_set,
                doc="Intermediate electrical power consumption <= overestimating hyperplanes",
            )
            def power_loss_upper_bound_fan(
                m_scen, i, j, p, d, n, s_, i_, j_, p_, d_, t
            ):
                if (s, i, j, p, d) == (s_, i_, j_, p_, d_):
                    return (
                        m_scen.fan_power_loss_intermediate[i, j, p, d, n]
                        <= model.fan_hyperplanes_overestimation_slope_volume_flow[
                            p, d, t
                        ]
                        * m_scen.fan_volume_flow_intermediate[i, j, p, d, n]
                        + model.fan_hyperplanes_overestimation_intercept[p, d, t]
                    )
                return pyo.Constraint.Skip

            @m_scen.Constraint(
                model.E_fan_station,
                doc="fan station volume flow = sum of fans' volume flows",
            )
            def volume_flow_connection_to_fan_station_bigm_a(m_scen, i, j):
                return (
                    sum(
                        m_scen.fan_volume_flow[i, j, p, d, n]
                        for (k, l, p, d, n) in model.fan_set
                        if (k, l) == (i, j)
                    )
                    == m_scen.volume_flow[i, j] * m_scen.ind_active[i, j]
                )

            @m_scen.Constraint(
                model.fan_set, doc="bigM connecting intermediate volume flow"
            )
            def volume_flow_connection_to_fan_station_bigm_b(m_scen, i, j, p, d, n):
                return m_scen.fan_volume_flow_intermediate[
                    i, j, p, d, n
                ] - m_scen.fan_volume_flow[i, j, p, d, n] <= m_scen.volume_flow[
                    i, j
                ] * (
                    1 - m_scen.fan_ind_active[i, j, p, d, n]
                )

            @m_scen.Constraint(
                model.fan_set, doc="bigM connecting intermediate volume flow"
            )
            def volume_flow_connection_to_fan_station_bigm_c(m_scen, i, j, p, d, n):
                return m_scen.fan_volume_flow_intermediate[
                    i, j, p, d, n
                ] - m_scen.fan_volume_flow[i, j, p, d, n] >= -m_scen.volume_flow[
                    i, j
                ] * (
                    1 - m_scen.fan_ind_active[i, j, p, d, n]
                )

            @m_scen.Constraint(
                model.fan_set, doc="Fan volume flow is zero if fan is not active"
            )
            def volume_flow_connection_to_fan_station_bigm_d(m_scen, i, j, p, d, n):
                return (
                    m_scen.fan_volume_flow[i, j, p, d, n]
                    <= m_scen.volume_flow[i, j] * m_scen.fan_ind_active[i, j, p, d, n]
                )

        elif fan_model == 0:

            @m_scen.Expression(
                model.E_fan_station,
                doc="If no fan model is used, the electrical energy consumption is P_hyd/0.6 - an efficiency of 0.6 is assumed",
            )
            def electric_power_consumption_fan_station(model, i, j):
                return m_scen.pressure_change[i, j] * m_scen.volume_flow[i, j] / 0.6

        m_scen.electric_power_consumption = pyo.Var(
            doc="Electric power consumption of all fan stations"
        )

        @m_scen.Constraint(
            doc="Electrical power consumption is sum of all fan stations'"
        )
        def def_electric_power_consumption(m_scen):
            return m_scen.electric_power_consumption == sum(
                m_scen.electric_power_consumption_fan_station[i, j]
                for (i, j) in model.E_fan_station
            )

    if duct_model == 1 and branching_constraints == 1:
        # the following for constraints are quite a conservative overestimation
        # for the relationship between the ducts of a branch
        # as such they are not used in the publication

        @model.Constraint(
            model.E_duct,
            doc="width of inbranch >= sqrt(0.6 * (q_in/q_out)) * width of bend outbranch which stems from w_A/w>=0.6",
        )
        def branch_limit_width_ratio_straight_branch(model, i, j):
            if (i, j) in [(k, l) for k, l, m in model.duct_e_branch]:
                edge_in = next((o, p) for o, p in model.E_duct if p == i)
                volume_flow_ratio = max(
                    model.scenario[s].volume_flow[i, j]
                    / model.scenario[s].volume_flow[edge_in]
                    for s in model.Scenarios
                )
                return (
                    model.duct_width[edge_in]
                    >= np.sqrt(0.6 / volume_flow_ratio) * model.duct_width[i, j]
                )
            return pyo.Constraint.Skip

        @model.Constraint(
            model.E_duct,
            doc="height of inbranch >= sqrt(0.6 * (q_in/q_out)) * height of bend outbranch which stems from w_A/w>=0.6",
        )
        def branch_limit_height_ratio_straight_branch(model, i, j):
            if (i, j) in [(k, l) for k, l, m in model.duct_e_branch]:
                edge_in = next((o, p) for o, p in model.E_duct if p == i)
                volume_flow_ratio = max(
                    model.scenario[s].volume_flow[i, j]
                    / model.scenario[s].volume_flow[edge_in]
                    for s in model.Scenarios
                )
                return (
                    model.duct_height[edge_in]
                    >= np.sqrt(0.6 / volume_flow_ratio) * model.duct_height[i, j]
                )
            return pyo.Constraint.Skip

        @model.Constraint(
            model.E_duct,
            doc="width of inbranch >= sqrt(0.6 * (q_in/q_out)) * width of bend outbranch which stems from w_A/w>=0.6",
        )
        def branch_limit_width_ratio_bending_branch(model, i, j):
            if (i, j) in [(k, m) for k, l, m in model.duct_e_branch]:
                edge_in = next((o, p) for o, p in model.E_duct if p == i)
                volume_flow_ratio = max(
                    model.scenario[s].volume_flow[i, j]
                    / model.scenario[s].volume_flow[edge_in]
                    for s in model.Scenarios
                )
                return (
                    model.duct_width[edge_in]
                    >= np.sqrt(0.6 / volume_flow_ratio) * model.duct_width[i, j]
                )
            return pyo.Constraint.Skip

        @model.Constraint(
            model.E_duct,
            doc="height of inbranch >= sqrt(0.6 * (q_in/q_out)) * height of bend outbranch which stems from w_A/w>=0.6",
        )
        def branch_limit_height_ratio_bending_branch(model, i, j):
            if (i, j) in [(k, m) for k, l, m in model.duct_e_branch]:
                edge_in = next((o, p) for o, p in model.E_duct if p == i)
                volume_flow_ratio = max(
                    model.scenario[s].volume_flow[i, j]
                    / model.scenario[s].volume_flow[edge_in]
                    for s in model.Scenarios
                )
                return (
                    model.duct_height[edge_in]
                    >= np.sqrt(0.6 / volume_flow_ratio) * model.duct_height[i, j]
                )
            return pyo.Constraint.Skip

    if duct_model == 1 and velocity_constraint == 1:

        model.inverse_hyperplanes_set = pyo.Set(
            doc="Set of duct inverse hyperplanes. Used for outer polyhedral approximation of 1/w"
        )

        model.inverse_slope_width = pyo.Param(
            model.inverse_hyperplanes_set,
            doc="Slope in width direction of the inverse hyperplanes \
                for outer polyhedral approximation",
        )
        model.inverse_intercept = pyo.Param(
            model.inverse_hyperplanes_set,
            doc="Intercept of the inverse hyperplanes \
                for outer polyhedral approximation",
        )

        model.inverse_width = pyo.Var(
            model.E_duct,
            bounds=lambda model, i, j: (
                1 / model.duct_width_max[i, j],
                1 / model.duct_width_min[i, j],
            ),
            doc="Variable set to 1/w using supporting hyperplanes",
        )

        model.max_velocity = pyo.Param(
            model.E_duct,
            initialize=5,
            mutable=True,
            doc="Maximum allowed velocity in duct (i,j)",
        )

        @model.Constraint(
            model.E_duct,
            model.inverse_hyperplanes_set,
            doc="Inverse width >= supporting hyperplanes",
        )
        def approx_inverse_width(model, i, j, t):
            return (
                model.inverse_width[i, j]
                >= model.inverse_intercept[t]
                + model.inverse_slope_width[t] * model.duct_width[i, j]
            )

        @model.Constraint(model.E_duct, doc="Limit duct area by max velocity")
        def velocity_limit(model, i, j):
            return (
                max(model.scenario[s].volume_flow[i, j] for s in model.Scenarios)
                * model.inverse_width[i, j]
            ) <= model.max_velocity[i, j] * model.duct_height[i, j]

    @model.Expression(
        model.component_names,
        doc="Compute annuity factors for three component types according to VDI 2067",
    )
    def component_annuity(model, comp_name):
        T = int(model.operating_years.value)
        T_N = model.deprecation_period[comp_name]

        Z = model.interest_rate
        R = model.price_change_factor_service_maintenance
        B_SM = (1 - (R / Z) ** T) / (Z - R)

        annuity_factor = (Z - 1) / (1 - Z ** (-T))

        cost_factor = annuity_factor * (
            1
            + (model.service_factor[comp_name] + model.maintenance_factor[comp_name])
            * B_SM
        )

        div, mod = divmod(T, T_N)

        if mod == 0:
            return (
                1
                + sum((R / Z) ** (T_N * i) for i in range(1, div))
                - R ** (T_N * div) * mod / T_N * 1 / Z**T
            ) * cost_factor
        return (
            1
            + sum((R / Z) ** (T_N * i) for i in range(1, div + 1))
            - R ** (T_N * div) * mod / T_N * 1 / Z**T
        ) * cost_factor

    if duct_model == 1:

        @model.Expression(doc="Total used duct area in m²")
        def total_duct_used(model):
            return 2 * sum(
                (model.duct_width[i, j] + model.duct_height[i, j])
                * model.duct_length[i, j]
                for (i, j) in model.E_duct
            )

        @model.Expression(doc="Total duct costs in €")
        def total_duct_costs(model):
            return (
                model.component_annuity["duct"]
                * model.operating_years
                * model.duct_area_costs
                * model.total_duct_used
            )

        def calculate_duct_losses(model, s, i, j):
            return (
                model.rho
                / 2
                * (
                    model.fun_duct_nonlinear_hb_area2[i, j]
                    * (
                        model.zeta_bending[i, j]
                        + model.zeta_e_branch[i, j]
                        + model.zeta_t_branch[i, j]
                    )
                    + model.fun_nonlinear_duct_hb_friction[i, j]
                    * model.duct_resistance_coefficient
                    * (model.duct_length[i, j] / 2)
                )
            )

    elif duct_model == 0:

        @model.Expression()
        def total_duct_costs(model, doc="If not duct model is used, the costs are 0"):
            return 0

    if fan_model == 1:

        @model.Constraint(model.fan_set, doc="Symmetry breaking constraint")
        def symmetry_breaking(model, i, j, p, d, n):
            if n > 1:
                return (
                    model.fan_ind_purchase[i, j, p, d, n]
                    <= model.fan_ind_purchase[i, j, p, d, n - 1]
                )
            return pyo.Constraint.Skip

        @model.Constraint(
            model.Scenarios,
            model.fan_set,
            doc="Identical fans have identical volume flows",
        )
        def identical_fans_operate_identically_a(model, s, i, j, p, d, n):
            if n > 1:
                return (
                    model.scenario[s].fan_volume_flow_intermediate[i, j, p, d, n]
                    == model.scenario[s].fan_volume_flow_intermediate[i, j, p, d, 1]
                )
            return pyo.Constraint.Skip

        @model.Constraint(
            model.Scenarios,
            model.fan_set,
            doc="Identical fans have identical power loss",
        )
        def identical_fans_operate_identically_b(model, s, i, j, p, d, n):
            if n > 1:
                return (
                    model.scenario[s].fan_power_loss_intermediate[i, j, p, d, n]
                    == model.scenario[s].fan_power_loss_intermediate[i, j, p, d, 1]
                )
            return pyo.Constraint.Skip

        @model.Constraint(
            model.Scenarios,
            model.fan_set,
            doc="Identical fans have identical power loss",
        )
        def identical_fans_operate_identically_c(model, s, i, j, p, d, n):
            if n > 1:
                return (
                    model.scenario[s].fan_pressure_change_dimless[i, j, p, d, n]
                    == model.scenario[s].fan_pressure_change_dimless[i, j, p, d, 1]
                )
            return pyo.Constraint.Skip

        @model.Expression(doc="Total fan costs in €")
        def total_fan_costs(model):
            return (
                model.component_annuity["fan"]
                * model.operating_years
                * sum(
                    model.fan_costs[p, d] * model.fan_ind_purchase[i, j, p, d, n]
                    for (i, j, p, d, n) in model.fan_set
                )
            )

        @model.Constraint(
            model.E_fan_station,
            doc="A fan station is only purchased if at least one fan is purchased",
        )
        def fan_station_only_purchased_if_fan_purchased_a(model, i, j):
            return model.ind_purchase[i, j] <= sum(
                model.fan_ind_purchase[i, j, p, d, n]
                for (k, l, p, d, n) in model.fan_set
                if (k, l) == (i, j)
            )

        @model.Constraint(
            model.fan_set,
            doc="A fan station is only purchased if at least one fan is purchased",
        )
        def fan_station_only_purchased_if_fan_purchase_b(model, i, j, p, d, n):
            return model.ind_purchase[i, j] >= model.fan_ind_purchase[i, j, p, d, n]

        @model.Constraint(
            model.Scenarios, doc="Lower bound of electrical power consumption"
        )
        def limit_electric_power_subproblem(model, s):

            pel_hyd_fixed = sum(
                model.scenario[s].volume_flow[edge] ** 3 * model.fixed_zeta[edge]
                for edge in model.E_fixed
            )

            if duct_model == 1:
                pel_hyd_duct = sum(
                    model.scenario[s].volume_flow[edge] ** 3
                    * calculate_duct_losses(model, s, *edge)
                    for edge in model.E_duct
                )
            else:
                pel_hyd_duct = 0

            return (
                model.scenario[s].electric_power_consumption
                >= (pel_hyd_duct + pel_hyd_fixed) / 0.7
            )

    elif fan_model == 0:

        @model.Expression(doc="If not fan model is used, fan costs are set to 1000 €")
        def total_fan_costs(model):
            return (
                model.component_annuity["fan"]
                * model.operating_years
                * sum(1000 * model.ind_purchase[i, j] for (i, j) in model.E_fan_station)
            )

    @model.Expression(doc="Fan power consumption over all scenarios")
    def fan_power_consumption(model):
        return sum(
            model.time_share[s] * model.scenario[s].electric_power_consumption
            for s in model.Scenarios
        )

    @model.Expression(
        doc="Total duct volume in m³, only used in postprocessing - is quadratic constraint (removed during solve)"
    )
    def duct_volume(model):
        return sum(
            model.duct_length[e] * model.duct_height[e] * model.duct_width[e]
            for e in model.E_duct
        )

    @model.Expression(doc="Total fan energy costs in €")
    def fan_energy_costs(model):
        Z = model.interest_rate
        annuity_factor = (Z - 1) / (1 - Z ** (-model.operating_years))
        B_E = (
            1 - (model.price_change_factor_electricity / Z) ** model.operating_years
        ) / (Z - model.price_change_factor_electricity)

        return (
            annuity_factor
            * B_E
            * model.electric_energy_costs
            * model.operating_years
            * model.operating_days_per_year
            * model.operating_hours_per_day
            * model.fan_power_consumption
        )

    @model.Expression(doc="Total VFC invest costs in €")
    def total_vfc_costs(model):
        return (
            model.component_annuity["vfc"]
            * model.operating_years
            * model.vfc_costs
            * sum(model.ind_purchase[i, j] for (i, j) in model.E_vfc)
        )

    @model.Expression(doc="Total investment costs in €")
    def total_invest_costs(model):
        return model.total_duct_costs + model.total_fan_costs + model.total_vfc_costs

    @model.Objective(
        doc="Minimal life-cycle costs consisting of energy and investment costs in €"
    )
    def obj(model):
        return model.fan_energy_costs + model.total_invest_costs

    return model
