## this script automatically computes the pareto front for a control strategy. therefore, give the stepsize it should use when bounding the energy and invest costs.

import logging
import numpy as np
import pyomo.environ as pyo
from pyomo2h5 import load_yaml, ConstraintTracker
from src.preplanning.optimise import adjust_opt_problem, optimal_preplanning
from src.preplanning.optimise.utils import run_initial_solve, run_pareto_loop


INFILE = "merged_data/data1.yml"
OUTFOLDER = "results/"
CONTROL_STRATEGY = "VAV-VPC"
STEPSIZE_ENERGY = 500
STEPSIZE_INVEST = 1e3


MAX_VELOCITY = 5  # default is 5 m/s
MAX_HEIGHT = None  # default is None (keeps limits, will force height = width)
VERTICAL_DUCTS = [  # only needed when max_height is not None
    ("0~3", "0~4"),
    ("0~4", "2~1"),
    ("0~4", "1~1"),
    ("2~2", "2~3"),
    ("1~2", "1~3"),
]
MAX_LOAD_CASE_NUMBER = 6 # number of the maximum load case (if existing, else None)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# silence logger of gurobipy
logging.getLogger("gurobipy").propagate = False


def main():
    comment = input("Enter comment for file here:\n")
    tracker = ConstraintTracker()

    logging.info("Loading file...")
    data = load_yaml(INFILE)

    velocity_constraint = 1 if MAX_VELOCITY is not None else 0
    model = optimal_preplanning.model(
        duct_model=1,
        fan_model=1,
        velocity_constraint=velocity_constraint,
        pressure_target_met=1,
    )

    logging.info("Creating instance...")
    instance = adjust_opt_problem.adjust_to_control_strategy(
        CONTROL_STRATEGY, model=model, data=data
    )
    instance = adjust_opt_problem.adjust_to_duct_constraint(
        instance, MAX_VELOCITY, MAX_HEIGHT, VERTICAL_DUCTS
    )

    outfolder = OUTFOLDER + CONTROL_STRATEGY + "/"

    # max_load_case defines which load case is removed for postprocessing
    # -- this is necessary as the maximum load case could just barely become infeasible
    # when the system is laid out for slightly lower pressure losses.
    max_load_case = None if CONTROL_STRATEGY in ["CAV", "VAV-VPC"] else MAX_LOAD_CASE_NUMBER

    solver = pyo.SolverFactory("gurobi", solver_io="python")

    success, _ = run_initial_solve(
        instance, solver, tracker, outfolder, CONTROL_STRATEGY, comment, max_load_case
    )
    
    if not success:
        return  # infeasible, nothing more to do

    # Pareto front setup
    energy_cost_pareto = (
        np.floor(instance.fan_energy_costs.expr() / STEPSIZE_ENERGY) * STEPSIZE_ENERGY
    )
    invest_cost_pareto = (
        np.floor(instance.total_invest_costs.expr() / STEPSIZE_INVEST) * STEPSIZE_INVEST
    )

    logging.info("Calculating energy bounded part of the Pareto-Front...")

    # minimize investment costs
    instance.obj.deactivate()

    @instance.Objective()
    def min_invest_costs(m):
        return m.total_invest_costs

    run_pareto_loop(
        instance,
        tracker,
        solver,
        bound_expr=lambda m: m.fan_energy_costs,
        bound_start=energy_cost_pareto,
        stepsize=STEPSIZE_ENERGY,
        bound_name="energy costs",
        control_strategy=CONTROL_STRATEGY,
        comment=comment,
        outfolder=outfolder,
        max_load_case=max_load_case,
    )

    logging.info("Calculating invest bounded part of the Pareto-Front...")
    # minimize energy costs
    instance.min_invest_costs.deactivate()

    @instance.Objective()
    def min_energy_costs(m):
        return m.fan_energy_costs

    run_pareto_loop(
        instance,
        tracker,
        solver,
        bound_expr=lambda m: m.total_invest_costs,
        bound_start=invest_cost_pareto,
        stepsize=STEPSIZE_INVEST,
        bound_name="invest costs",
        control_strategy=CONTROL_STRATEGY,
        comment=comment,
        outfolder=outfolder,
        max_load_case=max_load_case,
    )


if __name__ == "__main__":
    main()
