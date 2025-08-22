from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple
import numpy as np

Edge = Tuple[str, str]


def build_graph(edges: List[Edge]) -> Tuple[Dict[str, List[str]], Dict[str, int], Set[str]]:
    """Build adjacency list, in-degree counts, and node set for a DAG."""
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    all_nodes = set()

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
        all_nodes.update([u, v])

    return graph, in_degree, all_nodes


def find_branch_node(edges: List[Edge]) -> str:
    """Return first branch/leaf node found when traversing from the root."""
    graph, in_degree, all_nodes = build_graph(edges)
    root = next((n for n in all_nodes if in_degree[n] == 0), None)
    if root is None:
        raise ValueError("No root found (graph may have cycles or no entry node).")

    current = root
    while True:
        neighbors = graph.get(current, [])
        if len(neighbors) != 1:
            return current
        current = neighbors[0]


def find_fan_edge(edges: List[Edge], E_fan_station: Set[Edge]) -> Edge | None:
    """Return first edge from root that is also in E_fan_station (or None)."""
    graph, in_degree, all_nodes = build_graph(edges)
    root = next((n for n in all_nodes if in_degree[n] == 0), None)
    if root is None:
        raise ValueError("No root found (graph may have cycles or no entry node).")

    current = root
    while True:
        neighbors = graph.get(current, [])
        if len(neighbors) != 1:
            return None
        next_node = neighbors[0]
        edge = (current, next_node)
        if edge in E_fan_station:
            return edge
        current = next_node


def get_max_volume_flow_in_problem(data: Dict) -> float:
    """Return maximum volume flow across all scenarios."""
    max_q = 0
    for s in data["Scenarios"][None]:
        max_q = max(max_q, np.max(list(data["scenario"][s]["volume_flow"].values())))
    return max_q


def get_fan_edge_volume_flow(data: Dict) -> Dict:
    """Return mapping: scenario -> edge -> volume flow for fan station edges."""
    fan_edge_volume_flow = defaultdict(dict)
    for s in data["Scenarios"][None]:
        fan_edge_volume_flow[s] = {}
        for e in data["E_fan_station"][None]:
            fan_edge_volume_flow[s][e] = data["scenario"][s]["volume_flow"][e]
    return fan_edge_volume_flow


def get_point_distance(value_lst: List[float]) -> float:
    """Return step size assuming values are from an equidistant grid."""
    sorted_lst = sorted(set(value_lst))
    diffs = np.diff(sorted_lst)
    if len(diffs) == 0:
        return 0
    if len(diffs) == 1:
        return diffs[0]
    if np.any(np.abs(np.diff(diffs)) > 1e-3):
        raise ValueError("List values are not equally spaced.")
    return np.min(diffs)


def prepare_load_case_yaml(data: Dict) -> Dict:
    """Convert raw scenario data into load-case YAML format."""
    load_case_data = {
        idx + 1: {
            "volume_flow": {
                key2: val["mean"] / 3600 for key2, val in scen["room"].items()
            }
        }
        for idx, (key, scen) in enumerate(data["scenario"].items())
    }

    time_shares = {idx + 1: v for idx, v in enumerate(data["time_share"].values())}

    return {
        "scenario": load_case_data,
        "Scenarios": {None: list(load_case_data.keys())},
        "time_share": time_shares,
    }


def propagate_volume_flows(data: Dict[Any, Any]) -> None:
    """
    Computes and stores aggregated volume flows for each edge in every scenario.

    Volume flows from leaf nodes are propagated upstream, accumulating total flow
    on each edge. The results are stored in-place in:
        data[None]["scenario"][s]["volume_flow"]

    Args:
        data: Nested dictionary containing network structure and per-scenario leaf flows.
              Required keys:
                - data[None]["E"][None]: List of edges (from_node, to_node)
                - data[None]["Scenarios"][None]: List of scenario keys
                - data[None]["scenario"][s]["volume_flow"]: Leaf node -> flow mapping for each scenario
    """

    def propagate_flows(
        edges: List[Tuple[Any, Any]], leaf_flows: Dict[Any, float]
    ) -> Dict[Tuple[Any, Any], float]:
        edge_flows = {edge: 0.0 for edge in edges}

        def add_flow_to_parents(child: Any, volume: float) -> None:
            for parent, downstream in edges:
                if downstream == child:
                    edge_flows[(parent, downstream)] += volume
                    add_flow_to_parents(parent, volume)

        for leaf, volume in leaf_flows.items():
            add_flow_to_parents(leaf, volume)

        return edge_flows

    edges = data["E"][None]

    for s in data["Scenarios"][None]:
        leaf_flows = data["scenario"][s]["volume_flow"]
        edge_flows = propagate_flows(edges, leaf_flows)
        data["scenario"][s]["volume_flow"] = edge_flows
    return data
