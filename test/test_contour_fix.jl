#!/usr/bin/env julia

# Test the contour plot fix
println("=== Testing Contour Plot Fix ===")

using Plots

# Create test data that mimics the actual problem
println("Creating test data with parameter tuples...")

# Create test results with various data structures
test_results = [
    # Normal case - all scalars
    ((1.0, 0.5, 2.0, 1.0, 1.5, 0.8, 2.5, 1.2), [5.0, 2.1, 1.8, 18.5, 14.2]),
    
    # Case with vectors
    ((0.8, 1.2, 1.8, 0.9, 1.2, 1.0, 2.0, 1.5), [4.8, [2.3, 2.4], 1.9, 18.0, 14.0]),
    
    # Case with tuples
    ((1.2, 0.8, 2.2, 1.1, 1.8, 0.7, 2.8, 1.1), [(5.2, 5.3), 1.9, 2.1, 18.8, 14.5]),
    
    # Mixed case
    ((1.5, 1.0, 2.5, 1.2, 2.0, 0.9, 3.0, 1.3), [5.5, (2.5, 2.6), [2.2, 2.3], 19.0, 15.0]),
    
    # Short data
    ((1.8, 1.2, 2.8, 1.4, 2.3, 1.1, 3.3, 1.5), [5.8, 2.8]),
]

println("Testing with $(length(test_results)) results")

# Test the contour plot creation logic
println("\n--- Testing Contour Plot Logic ---")

try
    # Extract data with safe indexing (same as in param_scan_metal.jl)
    x_vals = [params[1] for (params, res) in test_results]  # k1f values
    y_vals = [params[2] for (params, res) in test_results]  # k1r values
    
    # Extract A concentration values for contour plot
    a_concentrations = [res[1] isa Vector ? res[1][end] : (res[1] isa Tuple ? res[1][end] : res[1]) for (params, res) in test_results]
    
    println("x_vals (k1f): $x_vals")
    println("y_vals (k1r): $y_vals")
    println("a_concentrations: $a_concentrations")
    
    # Create contour plot
    k1f_unique = sort(unique(x_vals))
    k1r_unique = sort(unique(y_vals))
    
    println("k1f_unique: $k1f_unique")
    println("k1r_unique: $k1r_unique")
    
    # Create grid
    z_grid = zeros(length(k1f_unique), length(k1r_unique))
    
    for i in eachindex(k1f_unique)
        for j in eachindex(k1r_unique)
            # Find closest k1f and k1r combination
            distances = [(x_vals[k] - k1f_unique[i])^2 + (y_vals[k] - k1r_unique[j])^2 for k in eachindex(x_vals)]
            closest_idx = argmin(distances)
            z_grid[i, j] = a_concentrations[closest_idx]
        end
    end
    
    println("z_grid:")
    for i in 1:size(z_grid, 1)
        println("  Row $i: $(z_grid[i, :])")
    end
    
    # Create contour plot
    p = contour(k1f_unique, k1r_unique, z_grid', 
                xlabel="k1f", ylabel="k1r", 
                title="Test Contour Plot (A concentration)",
                colorbar_title="[A] final")
    
    savefig(p, "test_contour_fix.png")
    println("✅ Contour plot created successfully")
    
    # Test statistical summary
    println("\n--- Testing Statistical Summary ---")
    
    a_vals = [res[1] isa Vector ? res[1][end] : (res[1] isa Tuple ? res[1][end] : res[1]) for (params, res) in test_results]
    b_vals = [length(res) >= 2 ? (res[2] isa Vector ? res[2][end] : (res[2] isa Tuple ? res[2][end] : res[2])) : 0.0 for (params, res) in test_results] 
    c_vals = [length(res) >= 3 ? (res[3] isa Vector ? res[3][end] : (res[3] isa Tuple ? res[3][end] : res[3])) : 0.0 for (params, res) in test_results]
    product_ratio = [(b + c)/max(a, 1e-6) for (a,b,c) in zip(a_vals, b_vals, c_vals)]
    
    println("A concentrations: $a_vals")
    println("B concentrations: $b_vals")
    println("C concentrations: $c_vals")
    println("Product ratios: $product_ratio")
    
    println("✅ Statistical summary calculated successfully")
    
catch e
    println("❌ Test failed: $e")
    println("Error details: $(typeof(e))")
end

println("\n=== Contour Plot Fix Test Complete ===") 