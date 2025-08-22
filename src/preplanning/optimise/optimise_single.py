import logging
import pyomo.environ as pyo
from pyomo2h5 import load_yaml, ConstraintTracker
from src.preplanning.optimise import adjust_opt_problem, optimal_preplanning
from src.preplanning.optimise.utils import run_initial_solve


INFILE = "opt_problems/preplanning/GPZ/standard_case.yml"
OUTFOLDER = "new_solutions/real_GPZ/preplanning/"
CONTROL_STRATEGY = "ODS-CC"
MAX_VELOCITY = 5
MAX_HEIGHT = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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
        branching_constraints=0,
        velocity_constraint=velocity_constraint,
        pressure_target_met=1,
    )

    logging.info("Creating instance...")
    instance = adjust_opt_problem.adjust_to_control_strategy(
        CONTROL_STRATEGY, model=model, data=data
    )
    instance = adjust_opt_problem.adjust_to_duct_constraint(
        instance, MAX_VELOCITY, MAX_HEIGHT
    )

    outfolder = OUTFOLDER + CONTROL_STRATEGY + "/"
    max_load_case = None if CONTROL_STRATEGY in ["cav", "central CPC"] else 6

    solver = pyo.SolverFactory("gurobi", solver_io="python")

    success, filename = run_initial_solve(
        instance, solver, tracker, outfolder, CONTROL_STRATEGY, comment, max_load_case
    )

    if success:
        logging.info(f"Saved feasible solution to {filename}.h5")
    else:
        logging.warning(f"Problem infeasible. Results saved to {filename}.h5")


if __name__ == "__main__":
    main()
