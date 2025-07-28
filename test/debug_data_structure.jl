using Metal

# Create a simple test to check data structure
println("=== Debug Data Structure ===")

# Simulate the data structure that would be returned
test_results = [
    ((1.0, 0.5, 2.0, 1.0, 1.5, 0.8, 2.5, 1.2), [5.0, 2.1, 1.8, 18.5, 14.2]),
    ((0.8, 1.2, 1.8, 0.9, 1.2, 1.0, 2.0, 1.5), [4.8, 2.3, 1.9, 18.2, 14.5]),
    ((1.2, 0.8, 2.2, 1.1, 1.8, 0.7, 2.8, 1.1), [5.2, 1.9, 2.1, 18.8, 13.9])
]

println("Test results type: $(typeof(test_results))")
println("Number of results: $(length(test_results))")
println("First result type: $(typeof(test_results[1]))")
println("First result: $(test_results[1])")

# Test the data extraction that's causing the error
println("\n=== Testing Data Extraction ===")

try
    x_vals = [params[1] for (params, res) in test_results]
    y_vals = [params[2] for (params, res) in test_results]
    a_vals = [res[1] for (params, res) in test_results]
    
    println("x_vals: $x_vals")
    println("y_vals: $y_vals")
    println("a_vals: $a_vals")
    println("✅ Data extraction works correctly")
    
catch e
    println("❌ Error in data extraction: $e")
    println("Stacktrace: $(stacktrace())")
end

# Test the visualization function
println("\n=== Testing Visualization ===")

try
    # Include the visualization module
    include("../src/visualization.jl")
    
    # Test the plot function
    p = plot_multi_species_heatmap(test_results)
    println("✅ Visualization function works correctly")
    
catch e
    println("❌ Error in visualization: $e")
    println("Stacktrace: $(stacktrace())")
end

println("\n=== Debug Complete ===") 