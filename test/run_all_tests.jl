#!/usr/bin/env julia

# Run all tests in the test folder
println("=== Running All Tests ===")

# List of test files to run
test_files = [
    "simple_test.jl",
    "test_final_fix.jl", 
    "test_vector_fix.jl",
    "test_tuple_fix.jl",
    "test_visualization_fix.jl",
    "test_gpu_optimization.jl",
    "debug_data_structure.jl"
]

# Run each test
for test_file in test_files
    println("\n" * "="^50)
    println("Running: $test_file")
    println("="^50)
    
    try
        include(test_file)
        println("✅ $test_file completed successfully")
    catch e
        println("❌ $test_file failed: $e")
    end
end

println("\n" * "="^50)
println("All tests completed!")
println("="^50) 