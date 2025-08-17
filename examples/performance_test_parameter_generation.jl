using Distributed
using Plots
using IterTools
using Statistics
using CUDA  
using LinearAlgebra
using DifferentialEquations
using DiffEqGPU
using Printf
using BenchmarkTools
using Threads

# Include the main script to get access to the parameter generation functions
include("param_scan_thermodynamic_CUDA.jl")

# Performance comparison function
function compare_parameter_generation_methods()
    println("=== Parameter Generation Performance Comparison ===")
    println("Testing different parameter generation methods...")
    println()
    
    # Define smaller parameter ranges for testing
    test_k1f_range = 0.1:1.0:5.0  # 5 points
    test_k1r_range = 0.1:1.0:5.0  # 5 points
    test_k2f_range = 0.1:1.0:5.0  # 5 points
    test_k2r_range = 0.1:1.0:5.0  # 5 points
    test_k3f_range = 0.1:1.0:5.0  # 5 points
    test_k3r_range = 0.1:1.0:5.0  # 5 points
    test_k4f_range = 0.1:1.0:5.0  # 5 points
    test_k4r_range = 0.1:1.0:5.0  # 5 points
    test_A_range = 5.0:2.0:15.0   # 6 points
    test_B_range = 0.0:1.0:3.0    # 4 points
    test_C_range = 0.0:1.0:3.0    # 4 points
    test_E1_range = 5.0:2.0:15.0  # 6 points
    test_E2_range = 5.0:2.0:15.0  # 6 points
    
    # Calculate total combinations for this test
    total_combinations = length(test_k1f_range) * length(test_k1r_range) * length(test_k2f_range) * length(test_k2r_range) * 
                        length(test_k3f_range) * length(test_k3r_range) * length(test_k4f_range) * length(test_k4r_range) *
                        length(test_A_range) * length(test_B_range) * length(test_C_range) * length(test_E1_range) * length(test_E2_range)
    
    println("Test parameter space size: $total_combinations combinations")
    println("System capabilities:")
    println("  CPU threads: $(Threads.nthreads())")
    println("  CUDA available: $(CUDA.functional())")
    println()
    
    # Temporarily replace the global parameter ranges with test ranges
    global k1f_range, k1r_range, k2f_range, k2r_range, k3f_range, k3r_range, k4f_range, k4r_range
    global A_range, B_range, C_range, E1_range, E2_range
    
    original_k1f_range = k1f_range
    original_k1r_range = k1r_range
    original_k2f_range = k2f_range
    original_k2r_range = k2r_range
    original_k3f_range = k3f_range
    original_k3r_range = k3r_range
    original_k4f_range = k4f_range
    original_k4r_range = k4r_range
    original_A_range = A_range
    original_B_range = B_range
    original_C_range = C_range
    original_E1_range = E1_range
    original_E2_range = E2_range
    
    # Set test ranges
    k1f_range = test_k1f_range
    k1r_range = test_k1r_range
    k2f_range = test_k2f_range
    k2r_range = test_k2r_range
    k3f_range = test_k3f_range
    k3r_range = test_k3r_range
    k4f_range = test_k4f_range
    k4r_range = test_k4r_range
    A_range = test_A_range
    B_range = test_B_range
    C_range = test_C_range
    E1_range = test_E1_range
    E2_range = test_E2_range
    
    results = Dict()
    
    # Test 1: Original slow method
    println("1. Testing original method...")
    try
        @time original_params = generate_thermodynamic_parameters_original()
        results["Original"] = (length(original_params), @elapsed generate_thermodynamic_parameters_original())
        println("   Original method: $(length(original_params)) parameters in $(round(results["Original"][2], digits=3))s")
    catch e
        println("   Original method failed: $e")
        results["Original"] = (0, Inf)
    end
    
    # Test 2: Optimized method
    println("2. Testing optimized method...")
    try
        @time optimized_params = generate_thermodynamic_parameters_optimized()
        results["Optimized"] = (length(optimized_params), @elapsed generate_thermodynamic_parameters_optimized())
        println("   Optimized method: $(length(optimized_params)) parameters in $(round(results["Optimized"][2], digits=3))s")
    catch e
        println("   Optimized method failed: $e")
        results["Optimized"] = (0, Inf)
    end
    
    # Test 3: Vectorized method
    println("3. Testing vectorized method...")
    try
        @time vectorized_params = generate_thermodynamic_parameters_vectorized()
        results["Vectorized"] = (length(vectorized_params), @elapsed generate_thermodynamic_parameters_vectorized())
        println("   Vectorized method: $(length(vectorized_params)) parameters in $(round(results["Vectorized"][2], digits=3))s")
    catch e
        println("   Vectorized method failed: $e")
        results["Vectorized"] = (0, Inf)
    end
    
    # Test 4: Parallel method (if multiple threads available)
    if Threads.nthreads() > 1
        println("4. Testing parallel method...")
        try
            @time parallel_params = generate_thermodynamic_parameters_parallel()
            results["Parallel"] = (length(parallel_params), @elapsed generate_thermodynamic_parameters_parallel())
            println("   Parallel method: $(length(parallel_params)) parameters in $(round(results["Parallel"][2], digits=3))s")
        catch e
            println("   Parallel method failed: $e")
            results["Parallel"] = (0, Inf)
        end
    else
        println("4. Skipping parallel method (single thread)")
        results["Parallel"] = (0, Inf)
    end
    
    # Test 5: GPU method (if CUDA available)
    if CUDA.functional()
        println("5. Testing GPU method...")
        try
            @time gpu_params = generate_thermodynamic_parameters_gpu()
            results["GPU"] = (length(gpu_params), @elapsed generate_thermodynamic_parameters_gpu())
            println("   GPU method: $(length(gpu_params)) parameters in $(round(results["GPU"][2], digits=3))s")
        catch e
            println("   GPU method failed: $e")
            results["GPU"] = (0, Inf)
        end
    else
        println("5. Skipping GPU method (CUDA not available)")
        results["GPU"] = (0, Inf)
    end
    
    # Test 6: Streaming method
    println("6. Testing streaming method...")
    try
        @time streaming_params = generate_thermodynamic_parameters_streaming()
        results["Streaming"] = (length(streaming_params), @elapsed generate_thermodynamic_parameters_streaming())
        println("   Streaming method: $(length(streaming_params)) parameters in $(round(results["Streaming"][2], digits=3))s")
    catch e
        println("   Streaming method failed: $e")
        results["Streaming"] = (0, Inf)
    end
    
    # Restore original parameter ranges
    k1f_range = original_k1f_range
    k1r_range = original_k1r_range
    k2f_range = original_k2f_range
    k2r_range = original_k2r_range
    k3f_range = original_k3f_range
    k3r_range = original_k3r_range
    k4f_range = original_k4f_range
    k4r_range = original_k4r_range
    A_range = original_A_range
    B_range = original_B_range
    C_range = original_C_range
    E1_range = original_E1_range
    E2_range = original_E2_range
    
    # Print performance summary
    println("\n=== Performance Summary ===")
    println("Method                | Parameters | Time (s) | Speedup")
    println("----------------------|------------|----------|---------")
    
    baseline_time = results["Original"][2]
    for (method, (params, time)) in results
        if time < Inf
            speedup = baseline_time / time
            @printf("%-20s | %10d | %8.3f | %6.2fx\n", method, params, time, speedup)
        else
            @printf("%-20s | %10d | %8s | %6s\n", method, params, "FAILED", "N/A")
        end
    end
    
    # Find the fastest method
    valid_results = [(method, time) for (method, (params, time)) in results if time < Inf]
    if !isempty(valid_results)
        fastest_method, fastest_time = minimum(valid_results, by=x->x[2])
        println("\nFastest method: $fastest_method ($(round(fastest_time, digits=3))s)")
        println("Speedup over original: $(round(baseline_time / fastest_time, digits=2))x")
    end
    
    return results
end

# Function to test the original slow method (for comparison)
function generate_thermodynamic_parameters_original()
    params = []
    
    # Generate all combinations of parameters
    base_combinations = Iterators.product(
        k1f_range, k1r_range, k2f_range, k2r_range, k3f_range, k3r_range,
        k4f_range, k4r_range, A_range, B_range, C_range, E1_range, E2_range
    )
    
    # Calculate total combinations for progress tracking
    total_combinations = length(k1f_range) * length(k1r_range) * length(k2f_range) * length(k2r_range) * 
                        length(k3f_range) * length(k3r_range) * length(k4f_range) * length(k4r_range) *
                        length(A_range) * length(B_range) * length(C_range) * length(E1_range) * length(E2_range)
    
    println("Generating parameters with thermodynamic constraints...")
    println("Total possible combinations: $total_combinations")
    
    counter = 0
    for (k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, A0, B0, C0, E1_0, E2_0) in base_combinations
        counter += 1
        
        # Show progress every 10000 combinations
        if counter % 10000 == 0 || counter == total_combinations
            print_progress_bar(counter, total_combinations, 40, "Parameter Generation")
        end
        
        # Check thermodynamic constraints
        # Keq1 = (k1f * k2f) / (k1r * k2r) should be in reasonable range
        Keq1 = (k1f * k2f) / (k1r * k2r)
        
        # Keq2 = (k3f * k4f) / (k3r * k4r) should be in reasonable range
        Keq2 = (k3f * k4f) / (k3r * k4r)
        
        # Apply thermodynamic constraints
        # Keq1 and Keq2 should be in reasonable ranges (e.g., 0.1 to 10.0)
        if 0.1 <= Keq1 <= 10.0 && 0.1 <= Keq2 <= 10.0
            push!(params, (k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, A0, B0, C0, E1_0, E2_0))
        end
    end
    
    # Final progress bar update
    println()  # New line after progress bar
    
    return params
end

# Run the performance comparison
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting parameter generation performance test...")
    results = compare_parameter_generation_methods()
    println("\nPerformance test completed!")
end 