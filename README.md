# VentilationSysOpt_Preplanning

Code utilised for publication "", Breuer et al.
The code can be used to optimise ventilation systems by duct sizing, fan selection, placement and operation as well as volume flow controller placement and operation.
Further, variations in control strategy, duct constrarints (maximal/minimal dimensions, maximal velocity) and fan data can be performed with the framework by utilising three packages:
- underestimating-hyperplanes (used for approximations of fan characteristic curves and duct pressure losses)
- vensys-clustering used for ventilation system scenario reduction, yielding load cases and their frequencies
- pyomo2h5 (used for working with ruamel.yaml and hdf5 files)

## Features


### Input
- fan data
- network data
- load case data (volume flow demand in rooms)

### Preprocess input
- fan characteristic curve approximation (using code from xy)
- duct pressure loss approximation (using code from xy)
- load case reduction (using code from xy)
- network data

### Optimisation
- contains the full optimisation problem from Breuer et al. modelled in Pyomo
- optimise code by merging all input data and instantiating an optimisation problem
- solving uses Gurobi (can be changed with minor adaptions)
- solve Pareto Front
- vary control strategy
- solve with duct constraints, e.g. variations in duct height/width
- postprocess results to create (i) Pareto Fronts, (ii) ...

## Examples

An example workflow is as follows:

```mermaid
flowchart TD
    A[data.csv] --> B[preprocess.py]
    B --> C[train.py]
    C --> D[evaluate.py]
    D --> E[plot_results.py]
