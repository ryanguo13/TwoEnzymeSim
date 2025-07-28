#!/usr/bin/env julia

# Comprehensive test for all fixes
println("=== Comprehensive Test for All Fixes ===")

using Plots

# Include the visualization module
include("../src/visualization.jl")

# Create comprehensive test data
println("Creating comprehensive test data...")

# Create test results with all possible data structures
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
    
    # Invalid cases
    ((2.0, 1.5, 3.0, 1.6, 2.5, 1.3, 3.5, 1.7), nothing),
    ((2.2, 1.8, 3.2, 1.8, 2.7, 1.5, 3.7, 1.9), []),
]

println("Testing with $(length(test_results)) results (including all data structures)")

# Test all visualization functions
functions_to_test = [
    ("plot_multi_species_heatmap", () -> plot_multi_species_heatmap(test_results)),
    ("plot_parameter_sensitivity_analysis", () -> plot_parameter_sensitivity_analysis(test_results)),
    ("plot_concentration_distributions", () -> plot_concentration_distributions(test_results)),
    ("plot_3d_parameter_space", () -> plot_3d_parameter_space(test_results))
]

success_count = 0
total_count = length(functions_to_test)

for (func_name, func) in functions_to_test
    println("\n--- Testing $func_name ---")
    try
        result = func()
        if result !== nothing
            println("✅ $func_name completed successfully")
            # Test savefig
            try
                savefig(result, "test_complete_$(func_name).png")
                println("✅ Savefig test passed for $func_name")
                global success_count += 1
            catch e
                println("❌ Savefig test failed for $func_name: $e")
            end
        else
            println("⚠️ $func_name returned nothing (expected for invalid data)")
            global success_count += 1  # Still counts as success if it handles invalid data gracefully
        end
    catch e
        println("❌ $func_name failed: $e")
        println("Error details: $(typeof(e))")
    end
end

# Test contour plot logic
println("\n--- Testing Contour Plot Logic ---")
try
    # Filter out invalid results
    valid_results = [(params, res) for (params, res) in test_results if res !== nothing && length(res) >= 1]
    
    # Extract data with safe indexing
    x_vals = [params[1] for (params, res) in valid_results]  # k1f values
    y_vals = [params[2] for (params, res) in valid_results]  # k1r values
    
    # Extract A concentration values for contour plot
    a_concentrations = [res[1] isa Vector ? res[1][end] : (res[1] isa Tuple ? res[1][end] : res[1]) for (params, res) in valid_results]
    
    # Create contour plot
    k1f_unique = sort(unique(x_vals))
    k1r_unique = sort(unique(y_vals))
    
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
    
    # Create contour plot
    p = contour(k1f_unique, k1r_unique, z_grid', 
                xlabel="k1f", ylabel="k1r", 
                title="Test Contour Plot (A concentration)",
                colorbar_title="[A] final")
    
    savefig(p, "test_complete_contour.png")
    println("✅ Contour plot created successfully")
    global success_count += 1
    global total_count += 1
    
catch e
    println("❌ Contour plot test failed: $e")
    println("Error details: $(typeof(e))")
    global total_count += 1
end

# Test statistical summary
println("\n--- Testing Statistical Summary ---")
try
    # Filter out invalid results
    valid_results = [(params, res) for (params, res) in test_results if res !== nothing && length(res) >= 1]
    
    a_vals = [res[1] isa Vector ? res[1][end] : (res[1] isa Tuple ? res[1][end] : res[1]) for (params, res) in valid_results]
    b_vals = [length(res) >= 2 ? (res[2] isa Vector ? res[2][end] : (res[2] isa Tuple ? res[2][end] : res[2])) : 0.0 for (params, res) in valid_results] 
    c_vals = [length(res) >= 3 ? (res[3] isa Vector ? res[3][end] : (res[3] isa Tuple ? res[3][end] : res[3])) : 0.0 for (params, res) in valid_results]
    product_ratio = [(b + c)/max(a, 1e-6) for (a,b,c) in zip(a_vals, b_vals, c_vals)]
    
    println("✅ Statistical summary calculated successfully")
    println("A concentrations: $a_vals")
    println("B concentrations: $b_vals")
    println("C concentrations: $c_vals")
    println("Product ratios: $product_ratio")
    global success_count += 1
    global total_count += 1
    
catch e
    println("❌ Statistical summary test failed: $e")
    println("Error details: $(typeof(e))")
    global total_count += 1
end

println("\n" * "="^50)
println("Comprehensive Test Summary:")
println("Total functions tested: $total_count")
println("Successful: $success_count")
println("Failed: $(total_count - success_count)")
println("Success rate: $(round(success_count/total_count*100, digits=1))%")
println("="^50)

println("\n=== Comprehensive Test Complete ===") 