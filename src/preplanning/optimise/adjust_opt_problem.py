import pyomo.environ as pyo
from typing import Optional, Dict, Any
import copy
from src.preplanning.preprocessing.old_utils import (
    find_fan_edge,
    find_branch_node,
)


def adjust_to_control_strategy(
    cs: str, 
    model: pyo.AbstractModel, 
    data: Dict[str, Any]
) -> pyo.ConcreteModel:
    """
    Adjusts a Pyomo model instance to reflect a given control strategy.

    Depending on the specified control strategy (`cs`), this function 
    modifies the input data and/or applies additional constraints 
    to the instantiated Pyomo model.

    Supported strategies:
        - ``"distributed"``: Creates the instance directly.
        - ``"distributed CPC"``: Adds a branch node to `V_ports` or `V_in`.
        - ``"distributed VPC"``: Fixes leaf component decision to 1 and 
          forces purchase of the central fan.
        - ``"fully distributed"``: Fixes leaf component decision to 1 and 
          forces neglect of non-leaf central fans.
        - ``"central"``: Forces central fan purchase decision.
        - ``"central CPC"``: Same as ``"central"``, but also constrains 
          scenario pre-pressures to remain equal.
        - ``"cav"``: Collapses all scenarios to a single one and adjusts 
          fan hyperplane sets accordingly.

    Args:
        cs (str):
            Control strategy identifier. Must match one of the supported values.
        model (pyo.AbstractModel):
            A Pyomo abstract model used as a template.
        data (Dict[str, Any]):
            Model input data in dictionary format, structured as required by 
            ``model.create_instance``.

    Returns:
        pyo.ConcreteModel:
            A model instance with the control strategy applied.

    Raises:
        KeyError: If a required key is missing from the input data.
        ValueError: If an unsupported control strategy is provided.
    """
    central_fan_edge = find_fan_edge(data["E"][None], data["E_fan_station"][None])

    def make_central(m, i, j):
        return m.ind_purchase[i, j] == int((i, j) == central_fan_edge)

    match cs:
        case "distributed":
            instance = model.create_instance({None: data})

        case "distributed CPC":
            branch_node = find_branch_node(data["E"][None])
            if "V_ports" in data:
                data["V_ports"][None].append(branch_node)
            elif "V_in" in data:
                data["V_in"][None].append(branch_node)
            else:
                raise KeyError("Can't add prepressure node.")
            instance = model.create_instance({None: data})

        case "distributed VPC":
            instance = model.create_instance({None: data})
            instance.leaf_component_decision.value = 1
            instance.leaf_component_decision.fixed = True

            @instance.Constraint(instance.E_fan_station)
            def force_purchase_central(m, i, j):
                if (i, j) == central_fan_edge:
                    return m.ind_purchase[i, j] == 1
                return pyo.Constraint.Skip

        case "fully distributed":
            instance = model.create_instance({None: data})
            instance.leaf_component_decision.value = 1
            instance.leaf_component_decision.fixed = True

            @instance.Constraint(instance.E_fan_station)
            def force_neglect_central(m, i, j):
                return (
                    m.ind_purchase[i, j] == int((i, j) in instance.E_fan_station_leaf)
                )

        case "central" | "central CPC":
            instance = model.create_instance({None: data})
            instance.make_central = pyo.Constraint(
                instance.E_fan_station, rule=make_central
            )

            if cs == "central CPC":

                @instance.Constraint(instance.Scenarios)
                def const_prepressure(m, s):
                    branch_node = find_branch_node(data["E"][None])
                    if s > 1:
                        return (
                            m.scenario[s].pressure[branch_node]
                            == m.scenario[1].pressure[branch_node]
                        )
                    return pyo.Constraint.Skip

        case "cav":
            n_scen = data["Scenarios"][None][-1]
            # collapse scenarios to 1
            data["scenario"][1] = copy.deepcopy(data["scenario"][n_scen])
            data["scenario"] = {1: data["scenario"][n_scen]}
            data["Scenarios"][None] = [1]
            data["time_share"] = {1: 1}

            # filter hyperplane sets
            for key in [
                "fan_hyperplanes_underestimation_specific_pre_set",
                "fan_hyperplanes_overestimation_specific_pre_set",
            ]:
                if key in data:
                    data[key] = {
                        (1, *k[1:]): v for k, v in data[key].items() if k[0] == n_scen
                    }

            instance = model.create_instance({None: data})
            instance.make_central = pyo.Constraint(
                instance.E_fan_station, rule=make_central
            )

        case _:
            raise ValueError(f"Unknown control strategy: {cs}")

    return instance



def adjust_to_duct_constraint(
    instance: pyo.ConcreteModel,
    max_velocity: Optional[float],
    max_height: Optional[float]
) -> pyo.ConcreteModel:
    """
    Adjusts a Pyomo model instance by applying duct-related constraints.

    This function modifies the given Pyomo model (`instance`) by:
      - Setting maximum velocity values for all ducts in `instance.E_duct`
        if `max_velocity` is provided.
      - Adding a constraint to enforce duct width equal to duct height 
        if `max_height` is not provided.
      - Otherwise, adding a constraint to limit duct height to `max_height`
        for all ducts except those in `instance.E_duct_vertical` and a few
        explicitly skipped duct edges.

    Args:
        instance (pyo.ConcreteModel): 
            A Pyomo model instance with sets `E_duct` and `E_duct_vertical`,
            and variables `duct_width` and `duct_height`.
        max_velocity (Optional[float]): 
            The maximum velocity to assign to all ducts. If None, no velocity
            constraint is applied.
        max_height (Optional[float]): 
            The maximum duct height. If None, width is constrained to equal 
            height instead.

    Returns:
        pyo.ConcreteModel: 
            The modified Pyomo model with the applied constraints.
    """
    if max_velocity is not None:
        for e in instance.E_duct:
            instance.max_velocity[e] = max_velocity

    if max_height is None:

        @instance.Constraint(instance.E_duct)
        def width_equals_height(m, i, j):
            return m.duct_width[i, j] == m.duct_height[i, j]
    else:

        @instance.Constraint(instance.E_duct - instance.E_duct_vertical)
        def limit_height(m, i, j):
            if (i, j) in [
                ("0~3", "0~4"),
                ("0~4", "2~1"),
                ("0~4", "1~1"),
                ("2~2", "2~3"),
                ("1~2", "1~3"),
            ]:
                return pyo.Constraint.Skip
            return m.duct_height[i, j] <= max_height
    return instance