# VentilationSysOpt_Preplanning
```mermaid
flowchart TD

    %% === Preprocessing stage ===
    subgraph Preprocessing
        n5["fan data.yml"] --> n9["create_fan_data.py"]
        n6["duct data.yml"] --> n11["create_duct_data.py"]
        n7["network data.yml"] --> n13["create_network_data.py"]
        n8["load case data.yml"] --> n12["create_scenario_data.py"]
    end

    %% Merge step
    n9 --> n14["merge_all_data.py"]
    n11 --> n14
    n13 --> n14
    n12 --> n14

    n14 --> n15["data1.yml (model input)"]

    %% === Modeling & Optimization stage ===
    subgraph Modeling_and_Optimization
        n15 --> n17["optimise_single.py"]
        n16["optimal_preplanning.py"] --> n17
        n17 --> n18["instantiate model (data1.yml)"]
        n18 --> n19["optimization in Gurobi"]
    end

    %% === Postprocessing stage ===
    subgraph Postprocessing
        n19 --> n20["postprocess"]
        n20 --> n21["results.hdf5"]
    end

