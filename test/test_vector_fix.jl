using Metal

# Test the vector handling fix
println("=== Testing Vector Handling Fix ===")

# Include the visualization module
include("../src/visualization.jl")

# Create test data with vector elements (simulating the actual problem)
test_results = [
    ((1.0, 0.5, 2.0, 1.0, 1.5, 0.8, 2.5, 1.2), [5.0, [2.1, 2.2], 1.8, 18.5, 14.2]),  # B is a vector
    ((0.8, 1.2, 1.8, 0.9, 1.2, 1.0, 2.0, 1.5), [4.8, 2.3]),  # Only A and B (scalars)
    ((1.2, 0.8, 2.2, 1.1, 1.8, 0.7, 2.8, 1.1), [5.2, 1.9, [2.1, 2.3]]),  # C is a vector
    ((1.5, 1.0, 2.5, 1.2, 2.0, 0.9, 3.0, 1.3), [5.5, [2.5, 2.6], [2.2, 2.4], 19.0]),  # B and C are vectors
    ((1.8, 1.2, 2.8, 1.4, 2.3, 1.1, 3.3, 1.5), [5.8, 2.8, 2.5, 19.5, 15.5]),  # All scalars
    ((2.0, 1.5, 3.0, 1.6, 2.5, 1.3, 3.5, 1.7), nothing),  # No data
    ((2.2, 1.8, 3.2, 1.8, 2.7, 1.5, 3.7, 1.9), []),  # Empty array
]

println("Testing with $(length(test_results)) results (including vector elements)")

# Test each visualization function
functions_to_test = [
    ("plot_multi_species_heatmap", () -> plot_multi_species_heatmap(test_results)),
    ("plot_parameter_sensitivity_analysis", () -> plot_parameter_sensitivity_analysis(test_results)),
    ("plot_concentration_distributions", () -> plot_concentration_distributions(test_results)),
    ("plot_3d_parameter_space", () -> plot_3d_parameter_space(test_results))
]

for (func_name, func) in functions_to_test
    println("\n--- Testing $func_name ---")
    try
        result = func()
        if result !== nothing
            println("✅ $func_name completed successfully")
            # Test savefig
            try
                savefig(result, "test_vector_$(func_name).png")
                println("✅ Savefig test passed for $func_name")
            catch e
                println("❌ Savefig test failed for $func_name: $e")
            end
        else
            println("⚠️ $func_name returned nothing (expected for invalid data)")
        end
    catch e
        println("❌ $func_name failed: $e")
        println("Error details: $(typeof(e))")
    end
end

println("\n=== Vector Fix Test Complete ===") 