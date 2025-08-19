# VentilationSysOpt_Preplanning
```mermaid
flowchart TD
 subgraph Preprocessing["Preprocessing"]
        n9["create_fan_data.py"]
        n5["fan data.yml"]
        n11["create_duct_data.py"]
        n6["duct data.yml"]
        n13["create_network_data.py"]
        n7["network data.yml"]
        n12["create_scenario_data.py"]
        n8["load case data.yml"]
        n14["merge_all_data.py"]
  end
 subgraph Modeling_and_Optimization["Optimisation"]
        n17["optimise"]
        n18["instantiate model (data1.yml)"]
        n19["optimization in Gurobi"]
        n20["postprocess"]
        n21["results1.hdf5"]
  end
 subgraph s1["Postprocessing"]
        n23["analysis of influence of variation of #load cases"]
        n22["analysis of pareto front for multiple control strategies"]
        n24["..."]
  end
    n5 --> n9
    n6 --> n11
    n7 --> n13
    n8 --> n12
    n9 --> n14
    n11 --> n14
    n13 --> n14
    n12 --> n14
    n14 --> n15["data1.yml (model input)"]
    n15 --> n17
    n16["optimal_preplanning.py"] --> n17
    n17 --> n18
    n18 --> n19
    n19 --> n20
    n20 --> n21
    n21 --> n22 & n23 & n24

    n5@{ shape: tag-doc}
    n6@{ shape: doc}
    n7@{ shape: doc}
    n8@{ shape: doc}
    n18@{ shape: rounded}
    n19@{ shape: rounded}
    n20@{ shape: rounded}
    n21@{ shape: doc}
    n15@{ shape: doc}
    n16@{ shape: doc}




