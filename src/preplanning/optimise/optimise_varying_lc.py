import uuid
from pyomo2h5 import PyomoHDF5Saver, load_yaml, ConstraintTracker
import pyomo.environ as pyo
from src.preplanning.optimise import optimal_preplanning
from src.preplanning.postprocessing.postprocessing import postprocess
from src.preplanning.debugging.debugging import calculate_IIS

tracker = ConstraintTracker()

comment = input("Enter comment for file here:\n")

OUTFOLDER = "new_solutions/real_GPZ/preplanning/varying_lc/"

model = optimal_preplanning.model(
    duct_model=1, fan_model=1, branching_constraints=0, velocity_constraint=1
)


for n in range(1, 16):

    INFILE = (
        f"opt_problems/preplanning/GPZ/varying_n_load_cases/standard_case_n_lc_{n}.yml"
    )
    data = load_yaml(INFILE)

    FILENAME = str(uuid.uuid4())

    print("load file...")

    curr_comment = comment + f"number of load cases: {n}"

    print("create instance...")

    instance = model.create_instance({None: data})

    @instance.Constraint(instance.E_duct)
    def width_equals_height(m, i, j):
        return m.duct_width[i, j] == m.duct_height[i, j]

    solver = pyo.SolverFactory("gurobi", solver_io="python")
    solver.options["LogFile"] = OUTFOLDER + FILENAME + ".log"
    
    print("solve model...")
    results = solver.solve(instance, tee=True)

    with PyomoHDF5Saver(OUTFOLDER + FILENAME) as saver:

        if (
            results.solver.termination_condition == pyo.TerminationCondition.infeasible
        ) or (
            results.solver.termination_condition
            == pyo.TerminationCondition.infeasibleOrUnbounded
        ):
            saver.save_annotated_dict(
                {"Comment": {"Content": curr_comment + ", proven to be infeasible"}}
            )
            calculate_IIS(instance, OUTFOLDER + FILENAME + "_")

        else:
            saver.save_annotated_dict({"Comment": {"Content": curr_comment}})
            saver.save_instance(instance, results, solver_options=solver.options)

            print("postprocessing...")
            saver.save_annotated_dict(postprocess(instance, n), float_precision=4)
            saver.save_tracked_constraints(tracker, "Additional_constraints")
            saver.save_annotated_dict({"Number of Load Cases": {"Content": n}})

    print(f"done. saved as {FILENAME}.h5")
