import logging
import uuid
import pyomo.environ as pyo
from pyomo2h5 import PyomoHDF5Saver
from src.preplanning.postprocessing.postprocessing import postprocess
from src.preplanning.debugging.debugging import calculate_IIS


def run_initial_solve(
    instance, solver, tracker, outfolder, control_strategy, comment, max_load_case
):
    """Run the initial solve and save results or IIS if infeasible."""
    filename = str(uuid.uuid4())
    solver.options["LogFile"] = outfolder + filename + ".log"
    logging.info("Running initial solve. Logging into filename %s", filename)
    results = solver.solve(instance, tee=True, warmstart=False)
    with PyomoHDF5Saver(outfolder + filename) as saver:
        if results.solver.termination_condition in [
            pyo.TerminationCondition.infeasible,
            pyo.TerminationCondition.infeasibleOrUnbounded,
        ]:
            saver.save_annotated_dict(
                {"Comment": {"Content": comment + ", proven to be infeasible"}}
            )
            calculate_IIS(instance, outfolder + filename + "_")
            return False, filename
        else:
            saver.save_annotated_dict(
                {"Comment": {"Content": control_strategy + ", min lcc " + comment}}
            )
            saver.save_instance(instance, results, solver_options=solver.options)
            logging.info("Postprocessing...")
            saver.save_annotated_dict(
                postprocess(instance, max_load_case=max_load_case), float_precision=4
            )

            saver.save_tracked_constraints(tracker, "Additional_constraints")
    return True, filename


def run_pareto_loop(
    instance,
    tracker,
    solver,
    bound_expr,
    bound_start,
    stepsize,
    bound_name,
    control_strategy,
    comment,
    outfolder,
    max_load_case,
):
    """Run Pareto optimization loop for a given bound (energy or investment)."""
    while True:
        logging.info(f"Now solving with {bound_name} ub {bound_start}")
        filename = str(uuid.uuid4())
        curr_comment = f"{control_strategy}, {bound_name} ub: {bound_start} {comment}"

        @instance.Constraint()
        def pareto_limit(m):
            return bound_expr(m) <= bound_start

        tracker.add(instance.pareto_limit)
        solver.options["LogFile"] = outfolder + filename + ".log"

        results = solver.solve(instance, tee=True, warmstart=True)
        with PyomoHDF5Saver(outfolder + filename) as saver:
            if results.solver.termination_condition in [
                pyo.TerminationCondition.infeasible,
                pyo.TerminationCondition.infeasibleOrUnbounded,
            ]:
                saver.save_annotated_dict(
                    {"Comment": {"Content": curr_comment + ", proven to be infeasible"}}
                )
                tracker.delete(instance.pareto_limit)
                instance.del_component(pareto_limit)
                break
            else:
                saver.save_annotated_dict({"Comment": {"Content": curr_comment}})
                saver.save_instance(instance, results, solver_options=solver.options)
                logging.info("Postprocessing...")
                saver.save_annotated_dict(
                    postprocess(instance, max_load_case=max_load_case),
                    float_precision=4,
                )
                saver.save_tracked_constraints(tracker, "Additional_constraints")

        tracker.delete(instance.pareto_limit)
        instance.del_component(pareto_limit)

        logging.info(f"Done. Saved as {filename}.h5")
        bound_start -= stepsize
