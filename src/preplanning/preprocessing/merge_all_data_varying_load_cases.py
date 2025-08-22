import logging
import numpy as np
from pyomo2h5 import load_yaml
from pyomo2h5.yaml_handler import construct_yaml, convert_numpy_to_native

from src.preplanning.preprocessing.general_utils import (
    prepare_load_case_yaml,
    get_max_volume_flow_in_problem,
    get_fan_edge_volume_flow,
    propagate_volume_flows
)

from src.preplanning.preprocessing.domain_utils import (
    prepare_fan_yaml,
    prepare_fan_on_edges,
    filter_hyperplanes,
    get_duct_max_dimensions,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    OUT_FOLDER = "merged_data/"
    OUT_FILENAME = "data1"

    NETWORK_DATA_FILE = "data/network_data/network.yml"
    DUCT_DATA_FILE = "data/duct_data/duct_hyperplanes.yml"
    FAN_DATA_FILE = "data/fan_data/fan_power_loss_hyperplanes.yml"
    FANS_ON_EDGES_FILE = "data/network_data/fans_on_edges.yml"

    data = load_yaml(NETWORK_DATA_FILE)
    fan_data = load_yaml(FAN_DATA_FILE)
    fans_on_edges = load_yaml(FANS_ON_EDGES_FILE)
    duct_data = load_yaml(DUCT_DATA_FILE)

    for n in np.arange(1, 16):
        logging.info(f"Starting computation with {n} load case(s)")

        scenario_data_file = f"data/load_case_data/varying_n_load_cases/processed_GPZ_{n}_load_cases.yml"
        load_case_data = load_yaml(scenario_data_file)

        data.update(prepare_load_case_yaml(load_case_data))

        data = propagate_volume_flows(data)
        max_volume_flow_in_problem = get_max_volume_flow_in_problem(data)
        max_pressure_in_problem = data["max_pressure"][None]
        fan_edge_volume_flow = get_fan_edge_volume_flow(data)

        duct_edge_max_dimensions, max_width, max_height = get_duct_max_dimensions(data)

        data.update(
            filter_hyperplanes(
                duct_data, max_width, max_height, duct_edge_max_dimensions
            )
        )

        data.update(
            prepare_fan_yaml(
                fan_data,
                fan_edge_volume_flow=fan_edge_volume_flow,
                max_pressure_loss_in_problem=max_pressure_in_problem,
                max_volume_flow_in_problem=max_volume_flow_in_problem,
            )
        )

        data.update(prepare_fan_on_edges(fans_on_edges))

        yaml = construct_yaml()
        data = convert_numpy_to_native(data)
        yaml.indent(mapping=2, sequence=4, offset=2)

        with open(OUT_FOLDER + OUT_FILENAME + f"_n_lc_{n}" + ".yml", "w") as f:
            yaml.dump(data, f)


if __name__ == "__main__":
    main()
