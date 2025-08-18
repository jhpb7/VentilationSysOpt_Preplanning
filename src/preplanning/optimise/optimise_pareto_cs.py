import logging
import pyomo.environ as pyo
from pyomo2h5 import load_yaml, ConstraintTracker
from src.preplanning.optimise import adjust_opt_problem, optimal_preplanning
from utils import run_initial_solve, run_pareto_loop


INFILE = "opt_problems/preplanning/GPZ/standard_case.yml"
OUTFOLDER = "new_solutions/real_GPZ/preplanning/"
CONTROL_STRATEGY = "distributed"
MAX_VELOCITY = 5
MAX_HEIGHT = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


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
    instance = adjust_opt_problem.adjust_to_duct_constraints(
        instance, MAX_VELOCITY, MAX_HEIGHT
    )

    outfolder = OUTFOLDER + CONTROL_STRATEGY + "/"
    max_load_case = None if CONTROL_STRATEGY in ["cav", "central CPC"] else 6

    solver = pyo.SolverFactory("gurobi", solver_io="python")

    if not run_initial_solve(
        instance, solver, tracker, outfolder, CONTROL_STRATEGY, comment, max_load_case
    ):
        return  # infeasible, nothing more to do

    # Pareto front setup
    stepsize_energy = 500
    stepsize_invest = 1e3
    energy_cost_pareto = (
        np.floor(instance.fan_energy_costs.expr() / stepsize_energy) * stepsize_energy
    )
    invest_cost_pareto = (
        np.floor(instance.total_invest_costs.expr() / stepsize_invest) * stepsize_invest
    )

    logging.info("Calculating Pareto-Front...")

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
        stepsize=stepsize_energy,
        bound_name="energy costs",
        control_strategy=CONTROL_STRATEGY,
        comment=comment,
        outfolder=outfolder,
        max_load_case=max_load_case,
    )

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
        stepsize=stepsize_invest,
        bound_name="invest costs",
        control_strategy=CONTROL_STRATEGY,
        comment=comment,
        outfolder=outfolder,
        max_load_case=max_load_case,
    )


if __name__ == "__main__":
    main()
