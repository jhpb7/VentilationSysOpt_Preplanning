import logging
from importlib.resources import files
import numpy as np
from pyomo2h5 import load_yaml

from load_case_clustering import (
    data,
    compute_required_volume_flows,
    merge_rooms,
    analyze_cluster_quality,
    cluster_time_slots_by_q,
    save_scenario_data_to_yaml,
    compute_theoretical_max_q_per_zone,
    add_max_load_case,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    yaml_path = files(data).joinpath("general.yml")

    general_data = load_yaml(yaml_path)
    building_data = load_yaml("data/load_case_data/raw_GPZ_load_cases.yml")

    df = compute_required_volume_flows(general_data, building_data, overview_flag=False)
    df = merge_rooms(df, building_data)

    analysis = analyze_cluster_quality(df, 10)
    best_k = max(analysis["silhouette"], key=analysis["silhouette"].get)

    logging.info(
        f"Best #clusters according to silhouette metric is {best_k}. "
        f"Proceeding computing output with {best_k} clusters"
    )

    n_cluster_lst = np.arange(0, 15)

    for n_clusters in n_cluster_lst:
        filename_out = (
            f"data/load_case_data/processed_GPZ_{n_clusters+1}_load_cases.yml"
        )

        if n_clusters > 0:
            load_cases, time_shares = cluster_time_slots_by_q(df, n_clusters)

            max_load_case = compute_theoretical_max_q_per_zone(
                general_data, building_data, include_revision=False
            )
            load_cases, time_shares = add_max_load_case(
                max_load_case, load_cases, time_shares
            )
        else:
            max_load_case = compute_theoretical_max_q_per_zone(
                general_data, building_data, include_revision=False
            )
            load_cases, time_shares = add_max_load_case(max_load_case, {}, {})
            time_shares = {0: 1}

        out_dict = {"load_cases": load_cases, "time_share": time_shares}

        logging.info(
            f"Adding max load case with time share 0%, resulting in {n_clusters+1} load cases"
        )

        save_scenario_data_to_yaml(out_dict, filename_out)


if __name__ == "__main__":
    main()
