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
    "ProgressMeter"
])
```

## Usage

The package provides several examples in the `examples` directory. To run the basic simulation:

```julia
julia --project examples/basic_simulation.jl
```


## Example
```
Total parameter combinations: 92280601600000000
✅ CUDA GPU is available
GPU device: NVIDIA GeForce RTX 3070 Ti
GPU memory: 8.0 GB
CUDA memory pool initialized
Starting CUDA GPU-accelerated parameter scan...
Auto-optimized batch size: 1638 (based on GPU memory)
Running 100000 simulations using CUDA GPU acceleration
Batch size: 1638
CUDA memory usage: 6.93 GB allocated, 8.0 GB free
Processing batch 1/62 (1638 simulations)
Using CUDA GPU acceleration for batch of 1638 simulations
```