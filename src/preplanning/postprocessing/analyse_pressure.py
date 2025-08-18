from typing import List, Tuple, Dict
from collections import defaultdict, deque
import h5py
import ast
import numpy as np


def decode_h5_tuple_str_into_tuple(byte_str: bytes) -> Tuple[str, ...]:
    """Decodes a byte string representing a tuple into an actual Python tuple of strings."""
    decoded = byte_str.decode("utf-8").strip("()")
    return tuple(item.strip() for item in decoded.split(","))


def decode_and_eval(h5_array) -> List[Tuple[str, str]]:
    """Decodes and evaluates HDF5 byte string array into list of tuples."""
    return [ast.literal_eval(x.decode()) for x in h5_array[:]]


def decode_list(h5_array) -> List[str]:
    """Decodes HDF5 byte string array into list of strings."""
    return [x.decode() for x in h5_array[:]]


def decode_scenarios(scenarios) -> List[int]:
    """Decodes scenario identifiers into integers."""
    return [int(s) for s in scenarios]


def build_graph(edges: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Builds a directed graph as adjacency list from a list of edges."""
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
    return graph


def bfs_paths(
    graph: Dict[str, List[str]], root: str, targets: List[str]
) -> Dict[str, List[str]]:
    """Finds shortest paths from the root to each target node using BFS."""
    paths = {}
    visited = set()
    queue = deque([[root]])

    while queue and set(paths.keys()) != set(targets):
        path = queue.popleft()
        node = path[-1]
        if node in visited:
            continue
        visited.add(node)

        if node in targets and node not in paths:
            paths[node] = path

        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(path + [neighbor])

    return paths


def extract_pressure_data(
    h5_file: h5py.File,
) -> Tuple[List[int], List[Tuple[str, str]], List[str], Dict[int, Dict[str, float]]]:
    """Extracts scenario IDs, edges, V_ports, and pressure values from HDF5."""
    variable = h5_file["Optimisation Components"]["Variable"]
    scenarios = decode_scenarios(
        h5_file["Optimisation Components"]["Set"]["Scenarios"][:]
    )
    E = decode_and_eval(h5_file["Optimisation Components"]["Set"]["E"])
    V_port = decode_list(h5_file["Optimisation Components"]["Set"]["V_ports"])

    pressure = {}
    for s_key in scenarios:
        try:
            pressure[s_key] = {
                row["V"].decode(): row["value"]
                for row in variable["Scenario"][str(s_key)]["pressure"][:]
            }
        except KeyError:
            pressure[s_key] = {}

    return scenarios, E, V_port, pressure


def extract_pressure_change_data(
    h5_file: h5py.File,
) -> Tuple[
    Dict[int, Dict[Tuple[str, str], float]], Dict[int, Dict[Tuple[str, str], float]]
]:
    """Extracts pressure change data for duct and fixed components from HDF5."""
    scenarios = decode_scenarios(
        h5_file["Optimisation Components"]["Set"]["Scenarios"][:]
    )
    variable = h5_file["Optimisation Components"]["Variable"]
    E_duct = decode_and_eval(h5_file["Optimisation Components"]["Set"]["E_duct"])
    E_fixed = decode_and_eval(h5_file["Optimisation Components"]["Set"]["E_fixed"])

    pressure_changes_duct = {}
    pressure_changes_fixed = {}

    for s_key in scenarios:
        duct = {}
        fixed = {}
        try:
            for row in variable["Scenario"][str(s_key)]["pressure_change"][:]:
                edge = decode_h5_tuple_str_into_tuple(row["E"])
                if edge in E_duct:
                    duct[edge] = row["value"]
                elif edge in E_fixed:
                    fixed[edge] = row["value"]
        except KeyError:
            pass
        pressure_changes_duct[s_key] = duct
        pressure_changes_fixed[s_key] = fixed

    return pressure_changes_duct, pressure_changes_fixed


def process_pressure_changes(
    file_path: str,
) -> Tuple[
    Dict[int, Dict[Tuple[str, str], float]], Dict[int, Dict[Tuple[str, str], float]]
]:
    """Loads HDF5 file and extracts pressure change values for ducts and fixed parts."""
    with h5py.File(file_path, "r") as h5_file:
        return extract_pressure_change_data(h5_file)


def process_pressure_branch_and_paths(
    file_path: str,
) -> Tuple[Dict[int, Dict[str, Dict[str, float]]], Dict[str, List[str]]]:
    """
    Extracts pressure values along room paths and builds root-to-room paths from the graph.

    Returns:
        - pressure_branch: scenario → room → node → pressure
        - root_room_paths: room → node path
    """
    pressure_branch: Dict[int, Dict[str, Dict[str, float]]] = {}

    with h5py.File(file_path, "r") as h5_file:
        scenarios, E, V_port, pressure = extract_pressure_data(h5_file)

        V_source = [e[0] for e in E if e[0] in V_port]
        V_target = [e[1] for e in E if e[1] in V_port]

        graph = build_graph(E)
        root_room_paths = bfs_paths(graph, V_source[0], V_target)

        for scenario in scenarios:
            pressure_branch[scenario] = {
                room: {
                    v: pressure[scenario].get(v, np.nan) for v in root_room_paths[room]
                }
                for room in V_target
            }

    return pressure_branch, root_room_paths


def sum_pressure_loss_branch(
    root_room_paths: Dict[str, List[str]],
    pressure_changes: Dict[int, Dict[Tuple[str, str], float]],
    scenario: int,
) -> Dict[str, float]:
    """Sums the pressure loss along each root-to-room branch for a given scenario."""
    pressure_change_along_path = {room: 0.0 for room in root_room_paths.keys()}
    for room, path in root_room_paths.items():
        for V_in, V_out in zip(path[:-1], path[1:]):
            pressure_change_along_path[room] += pressure_changes[scenario].get(
                (V_in, V_out), 0.0
            )
    return pressure_change_along_path


def find_highest_pressure_loss_branch(
    root_room_paths: Dict[str, List[str]],
    pressure_changes: Dict[int, Dict[Tuple[str, str], float]],
) -> Tuple[Dict[int, str], Dict[int, float]]:
    """Returns the worst (highest-loss) room branch per scenario."""
    worst_branch, highest_dp = {}, {}
    for s in pressure_changes:
        dp_branch = sum_pressure_loss_branch(root_room_paths, pressure_changes, s)
        worst_branch[s] = max(dp_branch, key=dp_branch.get)
        highest_dp[s] = dp_branch[worst_branch[s]]
    return worst_branch, highest_dp


def sum_pressure_loss_branch_distribution(
    root_room_paths: Dict[str, List[str]],
    pressure_changes_duct: Dict[int, Dict[Tuple[str, str], float]],
    pressure_changes_fixed: Dict[int, Dict[Tuple[str, str], float]],
    scenario: int,
) -> Dict[str, Dict[str, float]]:
    """
    Separates pressure loss contributions from duct and fixed elements along each path.

    Returns:
        Dict[str, Dict[str, float]] with keys "duct" and "fixed", each mapping room → Δp.
    """
    pressure_change_along_path = {
        "duct": {room: 0.0 for room in root_room_paths},
        "fixed": {room: 0.0 for room in root_room_paths},
    }
    for room, path in root_room_paths.items():
        for V_in, V_out in zip(path[:-1], path[1:]):
            edge = (V_in, V_out)
            if edge in pressure_changes_duct[scenario]:
                pressure_change_along_path["duct"][room] += pressure_changes_duct[
                    scenario
                ][edge]
            elif edge in pressure_changes_fixed[scenario]:
                pressure_change_along_path["fixed"][room] += pressure_changes_fixed[
                    scenario
                ][edge]
    return pressure_change_along_path
