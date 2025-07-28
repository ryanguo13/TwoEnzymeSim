#!/usr/bin/env julia

# Simple parameter scan test
println("=== Simple Parameter Scan Test ===")

using Plots

# Include the visualization module
include("../src/visualization.jl")

# Create synthetic parameter scan results
println("Creating synthetic parameter scan results...")

# Generate synthetic data that mimics real parameter scan results
results = []

# Create a small parameter grid
k1f_vals = [0.1, 0.5, 1.0, 1.5, 2.0]
k1r_vals = [0.1, 0.5, 1.0, 1.5, 2.0]

# Generate synthetic results
for k1f in k1f_vals
    for k1r in k1r_vals
        # Create realistic concentration values based on parameters
        k2f = 1.0
        k2r = 0.5
        k3f = 1.5
        k3r = 0.8
        k4f = 2.0
        k4r = 1.2
        
        # Calculate synthetic concentrations
        # A decreases with k1f, increases with k1r
        A_final = 5.0 * exp(-k1f/2.0) * (1.0 + k1r/10.0)
        # B increases with k1f, decreases with k1r
        B_final = 2.0 * (1.0 - exp(-k1f/1.5)) * (1.0 - k1r/15.0)
        # C follows B but with some delay
        C_final = B_final * 0.8 * (1.0 + k3f/10.0)
        # E1 and E2 remain relatively constant
        E1_final = 20.0 * (1.0 - 0.1 * k1f/2.0)
        E2_final = 15.0 * (1.0 - 0.1 * k3f/3.0)
        
        # Add some noise to make it realistic
        A_final += randn() * 0.1
        B_final += randn() * 0.1
        C_final += randn() * 0.1
        E1_final += randn() * 0.5
        E2_final += randn() * 0.5
        
        # Ensure positive values
        A_final = max(A_final, 0.1)
        B_final = max(B_final, 0.1)
        C_final = max(C_final, 0.1)
        E1_final = max(E1_final, 10.0)
        E2_final = max(E2_final, 8.0)
        
        params = (k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r)
        concentrations = [A_final, B_final, C_final, E1_final, E2_final]
        
        push!(results, (params, concentrations))
    end
end

println("Generated $(length(results)) synthetic results")

# Test visualization
println("\nTesting visualization functions...")

try
    # Test multi-species heatmap
    p1 = plot_multi_species_heatmap(results)
    if p1 !== nothing
        savefig(p1, "test_simple_heatmap.png")
        println("✅ Multi-species heatmap created successfully")
    else
        println("⚠️ Multi-species heatmap returned nothing")
    end
    
    # Test parameter sensitivity analysis
    p2 = plot_parameter_sensitivity_analysis(results)
    if p2 !== nothing
        savefig(p2, "test_simple_sensitivity.png")
        println("✅ Parameter sensitivity analysis created successfully")
    else
        println("⚠️ Parameter sensitivity analysis returned nothing")
    end
    
    # Test concentration distributions
    p3 = plot_concentration_distributions(results)
    if p3 !== nothing
        savefig(p3, "test_simple_distributions.png")
        println("✅ Concentration distributions created successfully")
    else
        println("⚠️ Concentration distributions returned nothing")
    end
    
    # Test 3D parameter space
    p4 = plot_3d_parameter_space(results)
    if p4 !== nothing
        savefig(p4, "test_simple_3d.png")
        println("✅ 3D parameter space plot created successfully")
    else
        println("⚠️ 3D parameter space plot returned nothing")
    end
    
    println("\n=== All visualization tests passed! ===")
    
catch e
    println("❌ Visualization test failed: $e")
    println("Error details: $(typeof(e))")
end

# Print summary statistics
if length(results) > 0
    a_vals = [res[1] for (params, res) in results]
    b_vals = [res[2] for (params, res) in results]
    c_vals = [res[3] for (params, res) in results]
    
    println("\n=== Synthetic Data Summary ===")
    println("Total results: $(length(results))")
    println("A concentration range: $(minimum(a_vals)) to $(maximum(a_vals))")
    println("B concentration range: $(minimum(b_vals)) to $(maximum(b_vals))")
    println("C concentration range: $(minimum(c_vals)) to $(maximum(c_vals))")
end

println("\n=== Simple Parameter Scan Test Complete ===") 