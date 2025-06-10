# TwoEnzymeSim

A Julia package for simulating and analyzing two-enzyme reaction systems.

## Installation

```julia
using Pkg
Pkg.add("https://github.com/yourusername/TwoEnzymeSim.jl")
```

## Project Structure

- `src/`: Source code directory
  - `TwoEnzymeSim.jl`: Main module file
  - `parameters.jl`: System parameters and initial conditions
  - `simulation.jl`: Reaction network and simulation functions
  - `analysis.jl`: Steady-state analysis and parameter sensitivity
  - `visualization.jl`: Plotting functions
- `examples/`: Example scripts
  - `basic_simulation.jl`: Basic usage example
- `test/`: Test files

## Usage

```julia
using TwoEnzymeSim

# Run simulation
sol = simulate_system()

# Calculate fluxes
v1_time, v2_time = calculate_fluxes(sol, params)

# Analyze steady state
steady_sol = analyze_steady_state()

# Plot results
p1, p2, p3 = plot_results(sol, v1_time, v2_time, steady_sol)
```

## Features

- Two-enzyme reaction system simulation
- Steady-state analysis
- Parameter sensitivity analysis
- Equilibrium constant analysis
- Visualization tools

## Dependencies

- Catalyst.jl
- DifferentialEquations.jl
- Plots.jl
- NonlinearSolve.jl

## License

MIT License 