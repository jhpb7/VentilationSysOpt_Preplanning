import os
import pyomo
import pyomo.environ as pyo
from gurobipy import read
import gurobipy as gp
from typing import Dict


def calculate_IIS(instance: pyo.ConcreteModel, filename: str) -> None:
    """Compute an Irreducible Inconsistent Subsystem (IIS) for an infeasible model.

    Writes the Pyomo model to a temporary LP file, loads it into Gurobi,
    computes the IIS, and writes the result to an ``.ilp`` file. The
    temporary LP file is deleted afterwards.

    Args:
        instance (pyo.ConcreteModel): Pyomo model instance suspected to be infeasible.
        filename (str): Base filename (without extension). The IIS is saved
            as ``filename + ".ilp"``.
    """
    infeasible_lp = filename + "infeasible.lp"
    instance.write(
        filename=infeasible_lp,
        io_options={"symbolic_solver_labels": True},
    )

    grb_model = read(infeasible_lp)
    grb_model.computeIIS()
    grb_model.write(filename + ".ilp")

    os.remove(infeasible_lp)


def get_parameters_out_of_range(
    instance: pyo.ConcreteModel, limit: float = 1e-7
) -> Dict[str, float]:
    """Find Pyomo parameters with small nonzero values.

    Iterates through all active parameters in a model and identifies values
    that are nonzero but below a given threshold.

    Args:
        instance (pyo.ConcreteModel): Pyomo model instance to analyze.
        limit (float, optional): Threshold below which nonzero parameter
            values are flagged. Defaults to ``1e-7``.

    Returns:
        Dict[str, float]: Mapping of parameter names to their values
        that are below the threshold.
    """
    small_values = {}

    for param in instance.component_objects(pyo.Param, active=True):
        for index in param:
            if not index:
                if abs(param.value) > 0 and abs(param.value) < limit:
                    small_values[param.name] = param.value
            else:
                if isinstance(param[index], pyomo.core.base.param.ParamData):
                    if abs(param[index].value) > 0 and abs(param[index].value) < limit:
                        small_values[param.name] = param[index].value
                else:
                    if abs(param[index]) > 0 and abs(param[index]) < limit:
                        small_values[param.name] = param[index]
    return small_values


def check_large_coefficient_range(model: gp.Model, threshold_ratio: float = 1e6) -> None:
    """Check constraints in a Gurobi model for large coefficient ranges.

    For each constraint, computes the ratio of the maximum to minimum
    coefficient. Prints a warning if the ratio exceeds the threshold.

    Args:
        model (gp.Model): Gurobi model instance.
        threshold_ratio (float, optional): Maximum allowed ratio between
            largest and smallest coefficients. Defaults to ``1e6``.
    """
    print("Checking constraints for large coefficient ranges...")
    for constr in model.getConstrs():
        coeffs = []
        for i in range(model.getAttr("numVars")):
            coeff = model.getCoeff(constr, model.getVar(i))
            if coeff != 0:
                coeffs.append(abs(coeff))

        if coeffs:
            max_coeff = max(coeffs)
            min_coeff = min(coeffs)
            if max_coeff / min_coeff > threshold_ratio:
                print(f"Constraint {constr.ConstrName} has large coefficient range:")
                print(
                    f"  Max coeff: {max_coeff}, Min coeff: {min_coeff}, Ratio: {max_coeff / min_coeff}"
                )


def find_small_matrix_coeffs(
    model: gp.Model, threshold: float = 1e7, direction: str = "smaller"
) -> None:
    """Check Gurobi model for extreme constraint coefficients.

    Iterates through all constraints and prints coefficients that are
    smaller or larger than a given threshold.

    Args:
        model (gp.Model): Gurobi model instance.
        threshold (float, optional): Threshold for coefficient size.
            Defaults to ``1e7``.
        direction (str, optional): Whether to check for coefficients
            "smaller" (default) or "bigger" than the threshold.
    """
    print(f"Checking matrix coefficients {direction} than {threshold}...\n")
    for constr in model.getConstrs():
        row = model.getRow(constr)
        for i in range(row.size()):
            var = row.getVar(i)
            coeff = row.getCoeff(i)
            if direction == "bigger":
                if abs(coeff) >= threshold:
                    print(f"Constraint: {constr.ConstrName}")
                    print(f"  Variable: {var.VarName}, Coefficient: {coeff}")
            else:
                if abs(coeff) <= threshold:
                    print(f"Constraint: {constr.ConstrName}")
                    print(f"  Variable: {var.VarName}, Coefficient: {coeff}")


def get_gurobimodel(instance: pyo.ConcreteModel) -> gp.Model:
    """Convert a Pyomo instance into its underlying Gurobi model.

    Uses the persistent Gurobi solver interface to attach the Pyomo
    model to Gurobi and returns the internal Gurobi model.

    Args:
        instance (pyo.ConcreteModel): Pyomo model instance.

    Returns:
        gp.Model: Gurobi model object corresponding to the Pyomo instance.
    """
    opt = pyo.SolverFactory("gurobi_persistent")  # persistent interface
    opt.set_instance(instance, symbolic_solver_labels=True)  # register the model
    opt.update()  # flush to Gurobi
    return opt._solver_model
