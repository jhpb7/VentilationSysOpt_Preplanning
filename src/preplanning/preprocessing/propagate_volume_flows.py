from typing import Dict, Tuple, List, Any


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
