using Metal

# Test the fixed visualization functions
println("=== Testing Fixed Visualization Functions ===")

# Include the visualization module
include("../src/visualization.jl")

# Create test data with some invalid entries
test_results = [
    ((1.0, 0.5, 2.0, 1.0, 1.5, 0.8, 2.5, 1.2), [5.0, 2.1, 1.8, 18.5, 14.2]),  # Valid
    ((0.8, 1.2, 1.8, 0.9, 1.2, 1.0, 2.0, 1.5), [4.8, 2.3, 1.9, 18.2, 14.5]),  # Valid
    ((1.2, 0.8, 2.2, 1.1, 1.8, 0.7, 2.8, 1.1), [5.2, 1.9]),  # Invalid - too short
    ((1.5, 1.0, 2.5, 1.2, 2.0, 0.9, 3.0, 1.3), [5.5, 2.5, 2.2, 19.0, 15.0]),  # Valid
    ((1.8, 1.2, 2.8, 1.4, 2.3, 1.1, 3.3, 1.5), [])  # Invalid - empty
]

println("Testing with $(length(test_results)) results (including invalid ones)")

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
        else
            println("⚠️ $func_name returned nothing (expected for invalid data)")
        end
    catch e
        println("❌ $func_name failed: $e")
    end
end

println("\n=== Test Complete ===") 