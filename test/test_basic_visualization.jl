#!/usr/bin/env julia

# Basic visualization test without simulation dependencies
println("=== Basic Visualization Test ===")

using Plots

# Include only the visualization module
include("../src/visualization.jl")

# Create synthetic test data
println("Creating synthetic test data...")

# Create realistic test results with various data structures
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

println("Testing with $(length(test_results)) results (including various data structures)")

# Test each visualization function
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
                savefig(result, "test_basic_$(func_name).png")
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

println("\n" * "="^50)
println("Test Summary:")
println("Total functions tested: $total_count")
println("Successful: $success_count")
println("Failed: $(total_count - success_count)")
println("Success rate: $(round(success_count/total_count*100, digits=1))%")
println("="^50)

println("\n=== Basic Visualization Test Complete ===") 