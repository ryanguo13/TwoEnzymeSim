#!/usr/bin/env julia

# Minimal parameter scan test
println("=== Minimal Parameter Scan Test ===")

using Metal
using Plots

# Include the simulation module
include("../src/simulation.jl")
include("../src/visualization.jl")

# Test with a very small parameter set
println("Testing with minimal parameter set...")

# Create a small parameter grid
k1f_range = 0.1:1.0:2.0  # 2 points
k1r_range = 0.1:1.0:2.0  # 2 points
k2f_range = 0.1:1.0:2.0  # 2 points
k2r_range = 0.1:1.0:2.0  # 2 points
k3f_range = 0.1:1.0:2.0  # 2 points
k3r_range = 0.1:1.0:2.0  # 2 points
k4f_range = 0.1:1.0:2.0  # 2 points
k4r_range = 0.1:1.0:2.0  # 2 points

# Create a small grid (only 256 combinations)
param_grid = Iterators.product(
    k1f_range, k1r_range, k2f_range, k2r_range, k3f_range, k3r_range,
    k4f_range, k4r_range)

# Fixed initial conditions
fixed_initial_conditions = Dict(
    Symbol("A") => 5.0,
    Symbol("B") => 0.0,
    Symbol("C") => 0.0,
    Symbol("E1") => 20.0,
    Symbol("E2") => 15.0,
    Symbol("AE1") => 0.0,
    Symbol("BE2") => 0.0
)

# Preprocess function
function preprocess_solution(sol)
    try
        vals = [sol[Symbol("A")][end], sol[Symbol("B")][end], sol[Symbol("C")][end], sol[Symbol("E1")][end], sol[Symbol("E2")][end]]
        if any(x -> isnan(x) || isinf(x) || abs(x) > 1e6, vals)
            return nothing
        end
        return vals
    catch
        return nothing
    end
end

# Simple simulation function
function simulate_reaction_simple(params, tspan)
    k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = params
    
    if any(x -> isnan(x) || isinf(x) || abs(x) > 1e6, params)
        return nothing
    end
    
    p = Dict(:k1f=>k1f, :k1r=>k1r, :k2f=>k2f, :k2r=>k2r, :k3f=>k3f, :k3r=>k3r, :k4f=>k4f, :k4r=>k4r)
    
    try
        sol = simulate_system(p, fixed_initial_conditions, tspan, saveat=0.1)
        return preprocess_solution(sol)
    catch e
        println("Simulation error: $e")
        return nothing
    end
end

# Run a few simulations
println("Running 10 test simulations...")
results = []
param_array = collect(Iterators.take(param_grid, 10))

for (i, params) in enumerate(param_array)
    println("Simulation $i/10")
    res = simulate_reaction_simple(params, (0.0, 5.0))
    if res !== nothing
        push!(results, (params, res))
        println("  Success: A=$(res[1]), B=$(res[2]), C=$(res[3])")
    else
        println("  Failed")
    end
end

println("\nResults summary:")
println("Total simulations: 10")
println("Successful: $(length(results))")
println("Failed: $(10 - length(results))")

# Test visualization if we have results
if length(results) > 0
    println("\nTesting visualization...")
    try
        p1 = plot_multi_species_heatmap(results)
        if p1 !== nothing
            savefig(p1, "test_minimal_heatmap.png")
            println("✅ Visualization test passed")
        else
            println("⚠️ Visualization returned nothing")
        end
    catch e
        println("❌ Visualization failed: $e")
    end
else
    println("No results to visualize")
end

println("\n=== Minimal Test Complete ===") 