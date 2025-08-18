import matplotlib.pyplot as plt

from pyomo2h5 import load_yaml

from src.preplanning.preprocessing.propagate_volume_flows import propagate_volume_flows
from src.preplanning.preprocessing.old_utils import (
    prepare_load_case_yaml,
    get_max_volume_flow_in_problem,
)


plt.style.use("src/preplanning/FST.mplstyle")

NETWORK_DATA_FILE = "data/network_data/lab.yml"
FAN_DATA_FILE = "data/fan_data/fan_power_loss_hyperplanes_lab.yml"

data = load_yaml(NETWORK_DATA_FILE)

SCENARIO_DATA_FILE = "data/load_case_data/laboratory_scenarios.yml"
load_case_data = load_yaml(SCENARIO_DATA_FILE)


data.update(prepare_load_case_yaml(load_case_data))

# %%

data = propagate_volume_flows(data)
max_volume_flow_in_problem = get_max_volume_flow_in_problem(data)
max_pressure_in_problem = data["max_pressure"][None]

fan_data = load_yaml(FAN_DATA_FILE)

# %%


edges = data["E_fan_station"][None]

n_edges = len(edges) - 1

fig, ax = plt.subplots(1, 1, figsize=(5, 2))

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for s in data["Scenarios"][None]:
    idx = 0
    for edge, val in data["scenario"][s]["volume_flow"].items():
        if edge in edges and not edge == ("root", "0a"):
            if s == 1:
                ax.axvline(val, linestyle="--", color=colors[idx], label=edge)
            else:
                ax.axvline(val, linestyle="--", color=colors[idx])
            idx += 1

for fan, values in fan_data.items():
    qmax = values["fan_volume_flow_max"]
    ax.text(qmax + 0.3, 0.5, fan, rotation=90)
    ax.axvline(qmax, color="k")
ax.set_xlim(left=0)
ax.legend()
