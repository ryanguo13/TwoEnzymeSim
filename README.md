# TwoEnzymeSim

A Julia package for simulating and analyzing two-enzyme systems with detailed reaction kinetics.

## Installation

1. First, make sure you have Julia installed (version 1.6 or higher recommended).
2. Clone this repository:

```bash
git clone https://github.com/ryanguo13/TwoEnzymeSim.git
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
