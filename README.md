# TwoEnzymeSim

A Julia package for simulating and analyzing two-enzyme systems with detailed reaction kinetics.

## Installation

1. First, make sure you have Julia installed (version 1.6 or higher recommended).
2. Clone this repository:

```bash
git clone https://github.com/yourusername/TwoEnzymeSim.git
cd TwoEnzymeSim
```

3. Install required packages:

```julia
julia --project
julia> using Pkg
julia> Pkg.add([
    "Catalyst",
    "DifferentialEquations",
    "Plots",
    "PlotlyBase"
    "GR"
])
```

## Usage

The package provides several examples in the `examples` directory. To run the basic simulation:

```julia
julia --project examples/basic_simulation.jl
```

This will:

1. Simulate the two-enzyme system
2. Calculate reaction fluxes
3. Generate three plots:
   - `concentration_and_fluxes.png`: Shows the concentration profiles and reaction fluxes over time
   - `phase_portrait.html`: An interactive 3D plot showing the system's phase space trajectory

### Plot Explanations

#### Concentration and Fluxes Plot

- **Top Panel**: Shows the concentration profiles of species A, B, and C over time
- **Bottom Panel**: Displays the reaction fluxes for all six reactions:
  - A→B: Forward reaction
  - B→A: Reverse reaction
  - B→C: Second enzyme forward reaction
  - C→B: Second enzyme reverse reaction
  - A→C: Direct conversion
  - C→A: Direct reverse conversion

#### Phase Portrait (3D Interactive Plot)

- Shows the system's trajectory in the 3D space of concentrations [A], [B], and [C]
- Green marker indicates the starting point
- Red marker indicates the ending point
- The trajectory shows how the system evolves from initial conditions to steady state
- You can rotate and zoom the plot in your web browser to better understand the system's dynamics

## Project Structure

```
TwoEnzymeSim/
├── src/
│   ├── simulation.jl    # Core simulation functions
│   ├── parameters.jl    # System parameters and initial conditions
│   ├── analysis.jl      # Analysis functions
│   └── visualization.jl # Plotting functions
├── examples/
│   └── basic_simulation.jl  # Basic usage example
└── test/
    └── runtests.jl     # Test suite
```

## Features

- Detailed reaction kinetics simulation
- Multiple reaction pathways
- Interactive 3D phase space visualization
- Flux analysis
- Parameter sensitivity analysis
