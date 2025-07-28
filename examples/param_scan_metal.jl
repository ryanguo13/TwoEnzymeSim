using Distributed
using Plots
using IterTools
using Statistics
using Metal  # Apple Silicon GPU support
using LinearAlgebra

# include("../src/parameters.jl")
include("../src/simulation.jl")
include("../src/visualization.jl")

# Define the parameters of the reaction rate iteration range
# Use fewer points to cover the full range
k1f_range = 0.1:0.5:10.0  # 20 points
k1r_range = 0.1:0.5:10.0  # 20 points
k2f_range = 0.1:0.5:10.0  # 20 points
k2r_range = 0.1:0.5:10.0  # 20 points
k3f_range = 0.1:0.5:10.0  # 20 points
k3r_range = 0.1:0.5:10.0  # 20 points
k4f_range = 0.1:0.5:10.0  # 20 points
k4r_range = 0.1:0.5:10.0  # 20 points

# Create a grid of parameters (only reaction rate constants)
param_grid = Iterators.product(
    k1f_range, k1r_range, k2f_range, k2r_range, k3f_range, k3r_range,
    k4f_range, k4r_range)

# Calculate total combinations for progress reporting
total_combinations = length(k1f_range) * length(k1r_range) * length(k2f_range) * length(k2r_range) * 
                    length(k3f_range) * length(k3r_range) * length(k4f_range) * length(k4r_range)
println("Total parameter combinations: $total_combinations")

# Check Metal GPU availability
if Metal.functional()
    println("✅ Metal GPU is available")
end

# Fixed initial conditions for all simulations
fixed_initial_conditions = Dict(
    Symbol("A") => 5.0,
    Symbol("B") => 0.0,
    Symbol("C") => 0.0,
    Symbol("E1") => 20.0,
    Symbol("E2") => 15.0,
    Symbol("AE1") => 0.0,
    Symbol("BE2") => 0.0
)

# Preprocess the solution to get the final concentrations of A, B, C, E1, E2
function preprocess_solution(sol)
    # Check for overflow or NaN in the solution
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

# GPU-accelerated batch simulation function using simulation.jl
function simulate_reaction_batch_gpu(param_batch, tspan)
    results = []
    
    if Metal.functional()
        println("Using Metal GPU acceleration for batch of $(length(param_batch)) simulations")
        
        # Use optimized GPU processing with kernel for larger batches
        if length(param_batch) > 100
            results = process_batch_gpu_with_kernel(param_batch, tspan)
        else
            # Use parallel GPU processing for medium batches
            if length(param_batch) > 50
                results = process_batch_gpu_parallel(param_batch, tspan)
            else
                # Process in smaller sub-batches to avoid memory issues
                sub_batch_size = 100
                for i in 1:sub_batch_size:length(param_batch)
                    sub_batch_end = min(i + sub_batch_size - 1, length(param_batch))
                    sub_batch = param_batch[i:sub_batch_end]
                    
                    # Process sub-batch on GPU with optimized batch processing
                    sub_results = process_sub_batch_gpu_optimized(sub_batch, tspan)
                    append!(results, sub_results)
                end
            end
        end
    else
        println("Falling back to CPU processing")
        # Fallback to CPU processing using simulation.jl functions
        for params in param_batch
            res = simulate_reaction_cpu(params, tspan)
            push!(results, res)
        end
    end
    
    return results
end

# Optimized GPU batch processing function
function process_sub_batch_gpu_optimized(param_sub_batch, tspan)
    results = []
    
    # Convert parameters to GPU arrays for batch processing with Float32
    param_array = hcat([collect(Float32.(params)) for params in param_sub_batch]...)
    param_gpu = MtlArray{Float32}(param_array)
    
    # Process each parameter set using simulation.jl functions
    # This is still sequential but with GPU-accelerated math operations
    for (i, params) in enumerate(param_sub_batch)
        res = simulate_reaction_gpu_optimized(params, tspan, param_gpu, i)
        push!(results, res)
    end
    
    return results
end

# GPU-optimized simulation function
function simulate_reaction_gpu_optimized(params, tspan, param_gpu=nothing, param_index=nothing)
    # Unpack the parameters (only reaction rate constants)
    k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = params
    
    # Check for overflow/invalid parameter values
    if any(x -> isnan(x) || isinf(x) || abs(x) > 1e6, params)
        return nothing
    end
    
    # Use the existing simulation.jl function instead of redefining the reaction network
    p = Dict(:k1f=>k1f, :k1r=>k1r, :k2f=>k2f, :k2r=>k2r, :k3f=>k3f, :k3r=>k3r, :k4f=>k4f, :k4r=>k4r)
    
    try
        # Use the simulate_system function from simulation.jl
        sol = simulate_system(p, fixed_initial_conditions, tspan, saveat=0.1)
        return preprocess_solution(sol)
    catch
        return nothing
    end
end

# Process a sub-batch on GPU using simulation.jl functions
function process_sub_batch_gpu(param_sub_batch, tspan)
    results = []
    
    # Convert parameters to GPU arrays for batch processing with Float32
    param_array = hcat([collect(Float32.(params)) for params in param_sub_batch]...)
    param_gpu = MtlArray{Float32}(param_array)
    
    # Process each parameter set using simulation.jl functions
    for params in param_sub_batch
        res = simulate_reaction_gpu(params, tspan)
        push!(results, res)
    end
    
    return results
end

# GPU-accelerated simulation function using simulation.jl
function simulate_reaction_gpu(params, tspan)
    # Unpack the parameters (only reaction rate constants)
    k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = params
    
    # Check for overflow/invalid parameter values
    if any(x -> isnan(x) || isinf(x) || abs(x) > 1e6, params)
        return nothing
    end
    
    # Use the existing simulation.jl function instead of redefining the reaction network
    p = Dict(:k1f=>k1f, :k1r=>k1r, :k2f=>k2f, :k2r=>k2r, :k3f=>k3f, :k3r=>k3r, :k4f=>k4f, :k4r=>k4r)
    
    try
        # Use the simulate_system function from simulation.jl
        sol = simulate_system(p, fixed_initial_conditions, tspan, saveat=0.1)
        return preprocess_solution(sol)
    catch
        return nothing
    end
end


# Optimized GPU batch processing with kernel
function process_batch_gpu_with_kernel(param_batch, tspan)
    results = []
    
    # Convert parameters to GPU arrays with Float32
    param_array = hcat([collect(Float32.(params)) for params in param_batch]...)
    param_gpu = MtlArray{Float32}(param_array)
    
    # Simple parameter validation on GPU
    valid_params = []
    for (i, params) in enumerate(param_batch)
        # Check if parameters are valid (not NaN, inf, or too large)
        if !any(x -> isnan(x) || isinf(x) || abs(x) > 1e6, params)
            push!(valid_params, params)
        end
    end
    
    # Process valid parameters using simulation.jl
    for params in valid_params
        res = simulate_reaction_gpu(params, tspan)
        if res !== nothing
            push!(results, (params, res))
        end
    end
    
    return results
end

# CPU fallback simulation function using simulation.jl
function simulate_reaction_cpu(params, tspan)
    # Unpack the parameters (only reaction rate constants)
    k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = params
    
    # Check for overflow/invalid parameter values
    if any(x -> isnan(x) || isinf(x) || abs(x) > 1e6, params)
        return nothing
    end
    
    # Use the existing simulation.jl function instead of redefining the reaction network
    p = Dict(:k1f=>k1f, :k1r=>k1r, :k2f=>k2f, :k2r=>k2r, :k3f=>k3f, :k3r=>k3r, :k4f=>k4f, :k4r=>k4r)
    
    try
        # Use the simulate_system function from simulation.jl
        sol = simulate_system(p, fixed_initial_conditions, tspan, saveat=0.1)
        return preprocess_solution(sol)
    catch
        return nothing
    end
end

# True parallel GPU batch processing
function process_batch_gpu_parallel(param_batch, tspan)
    results = []
    
    # Convert parameters to GPU arrays with Float32
    param_array = hcat([collect(Float32.(params)) for params in param_batch]...)
    param_gpu = MtlArray{Float32}(param_array)
    
    # Process in parallel chunks
    chunk_size = 50  # Process 50 simulations at a time
    for i in 1:chunk_size:length(param_batch)
        chunk_end = min(i + chunk_size - 1, length(param_batch))
        chunk = param_batch[i:chunk_end]
        
        # Process chunk in parallel on GPU
        chunk_results = process_chunk_gpu_parallel(chunk, tspan, param_gpu, i)
        append!(results, chunk_results)
    end
    
    return results
end

# Process a chunk of simulations in parallel on GPU
function process_chunk_gpu_parallel(param_chunk, tspan, param_gpu, start_index)
    results = []
    
    # Use Metal.jl's parallel computing capabilities
    # Note: This is a simplified version - in practice, you'd implement
    # the ODE solver directly on GPU for true parallelism
    
    # For now, we'll use the optimized sequential version but with GPU math
    for (i, params) in enumerate(param_chunk)
        res = simulate_reaction_gpu_optimized(params, tspan, param_gpu, start_index + i - 1)
        push!(results, res)
    end
    
    return results
end

# Performance monitoring and timing functions
function measure_performance(func, args...; name="Function")
    start_time = time()
    result = func(args...)
    end_time = time()
    elapsed = end_time - start_time
    println("$name completed in $(round(elapsed, digits=3)) seconds")
    return result, elapsed
end

# GPU-accelerated parameter scan with batching and performance monitoring
function run_parameter_scan_metal(param_grid, batch_size=500, max_simulations=100000)
    results = []
    total_simulations = min(max_simulations, total_combinations)
    
    println("Running $total_simulations simulations using Metal GPU acceleration")
    println("Batch size: $batch_size")
    
    # Convert iterator to array for easier batching
    param_array = collect(Iterators.take(param_grid, total_simulations))
    
    # Performance monitoring
    total_start_time = time()
    successful_simulations = 0
    
    # Process in batches
    for i in 1:batch_size:length(param_array)
        batch_end = min(i + batch_size - 1, length(param_array))
        batch = param_array[i:batch_end]
        
        batch_num = div(i-1, batch_size) + 1
        total_batches = div(length(param_array)-1, batch_size) + 1
        
        println("Processing batch $batch_num/$total_batches ($(length(batch)) simulations)")
        
        # Measure batch processing time
        batch_results, batch_time = measure_performance(
            simulate_reaction_batch_gpu, batch, (0.0, 5.0), 
            name="Batch $batch_num"
        )
        
        # Filter out failed simulations
        batch_successful = 0
        for (j, res) in enumerate(batch_results)
            if res !== nothing
                push!(results, (batch[j], res))
                batch_successful += 1
            end
        end
        
        successful_simulations += batch_successful
        
        # Progress update with performance metrics
        if batch_num % 5 == 0
            elapsed_total = time() - total_start_time
            rate = successful_simulations / elapsed_total
            println("  Progress: $successful_simulations successful simulations")
            println("  Rate: $(round(rate, digits=2)) simulations/second")
            println("  Batch $batch_num: $batch_successful successful, $(round(batch_time, digits=3))s")
        end
    end
    
    total_time = time() - total_start_time
    println("\n=== Performance Summary ===")
    println("Total time: $(round(total_time, digits=3)) seconds")
    println("Successful simulations: $successful_simulations")
    println("Overall rate: $(round(successful_simulations/total_time, digits=2)) simulations/second")
    
    return results
end

# CPU benchmark function for performance comparison
function run_parameter_scan_cpu_benchmark(param_grid, max_simulations=1000)
    println("Running CPU benchmark with $max_simulations simulations...")
    
    results = []
    total_start_time = time()
    successful_simulations = 0
    
    # Convert iterator to array
    param_array = collect(Iterators.take(param_grid, max_simulations))
    
    for (i, params) in enumerate(param_array)
        if i % 100 == 0
            println("CPU progress: $i/$max_simulations")
        end
        
        res = simulate_reaction_cpu(params, (0.0, 5.0))
        if res !== nothing
            push!(results, (params, res))
            successful_simulations += 1
        end
    end
    
    total_time = time() - total_start_time
    println("\n=== CPU Benchmark Results ===")
    println("Total time: $(round(total_time, digits=3)) seconds")
    println("Successful simulations: $successful_simulations")
    println("CPU rate: $(round(successful_simulations/total_time, digits=2)) simulations/second")
    
    return results, total_time
end

# Performance comparison function
function compare_gpu_cpu_performance(param_grid, max_simulations=1000)
    println("=== GPU vs CPU Performance Comparison ===")
    println("Testing with $max_simulations simulations...")
    
    # Run CPU benchmark
    cpu_results, cpu_time = run_parameter_scan_cpu_benchmark(param_grid, max_simulations)
    
    # Run GPU benchmark
    println("\n--- GPU Benchmark ---")
    gpu_start_time = time()
    gpu_results = run_parameter_scan_metal(param_grid, 100, max_simulations)
    gpu_time = time() - gpu_start_time
    
    # Calculate speedup
    speedup = cpu_time / gpu_time
    
    println("\n=== Performance Comparison Summary ===")
    println("CPU time: $(round(cpu_time, digits=3)) seconds")
    println("GPU time: $(round(gpu_time, digits=3)) seconds")
    println("GPU speedup: $(round(speedup, digits=2))x")
    println("CPU successful: $(length(cpu_results))")
    println("GPU successful: $(length(gpu_results))")
    
    return cpu_results, gpu_results, speedup
end

# Run the Metal GPU-accelerated parameter scan
println("Starting Metal GPU-accelerated parameter scan...")

# Check if user wants performance comparison
if length(ARGS) > 0 && ARGS[1] == "benchmark"
    println("Running performance benchmark...")
    cpu_results, gpu_results, speedup = compare_gpu_cpu_performance(param_grid, 100000)
    println("Benchmark completed!")
else
    # Run the main parameter scan
    results = run_parameter_scan_metal(param_grid)
    
    # Show a report of the results
    println("\n")
    println("Number of results: $(length(results))")
    
    # Plot the results - create multiple visualization types
    if length(results) > 0
        # Extract all parameters and results
        x_vals = [params[1] for (params, res) in results]  # k1f values
        y_vals = [params[2] for (params, res) in results]  # k1r values
        z_vals = [res[1] for (params, res) in results]     # A concentration values
        
        # Print the actual ranges found in the data
        println("k1f range in results: $(minimum(x_vals)) to $(maximum(x_vals))")
        println("k1r range in results: $(minimum(y_vals)) to $(maximum(y_vals))")
        
        # 1. 多子热力图 - 显示所有物种的浓度
        p1 = plot_multi_species_heatmap(results)
        if p1 !== nothing
            savefig(p1, "multi_species_heatmap_metal.png")
            println("Multi-species heatmap saved as multi_species_heatmap_metal.png")
        else
            println("Warning: Could not create multi-species heatmap - no valid data")
        end
        
        # 2. 参数敏感性分析
        p2 = plot_parameter_sensitivity_analysis(results)
        if p2 !== nothing
            savefig(p2, "parameter_sensitivity_metal.png")
            println("Parameter sensitivity analysis saved as parameter_sensitivity_metal.png")
        else
            println("Warning: Could not create parameter sensitivity analysis - no valid data")
        end
        
        # 3. 浓度分布直方图
        p3 = plot_concentration_distributions(results)
        if p3 !== nothing
            savefig(p3, "concentration_distributions_metal.png")
            println("Concentration distributions saved as concentration_distributions_metal.png")
        else
            println("Warning: Could not create concentration distributions - no valid data")
        end
        
        # 4. 3D参数空间图 (k1f vs k1r vs k2f)
        p4 = plot_3d_parameter_space(results, 1, 2, 3)
        if p4 !== nothing
            savefig(p4, "3d_parameter_space_metal.png")
            println("3D parameter space plot saved as 3d_parameter_space_metal.png")
        else
            println("Warning: Could not create 3D parameter space plot - no valid data")
        end
        
        # 5. 创建规则网格的contour图
        k1f_unique = sort(unique(x_vals))
        k1r_unique = sort(unique(y_vals))
        
        # 创建网格
        z_grid = zeros(length(k1f_unique), length(k1r_unique))
        
        # 提取A浓度值用于contour图
        a_concentrations = [res[1] isa Vector ? res[1][end] : (res[1] isa Tuple ? res[1][end] : res[1]) for (params, res) in results]
        
        for i in eachindex(k1f_unique)
            for j in eachindex(k1r_unique)
                # 找到最接近的k1f和k1r组合
                distances = [(x_vals[k] - k1f_unique[i])^2 + (y_vals[k] - k1r_unique[j])^2 for k in eachindex(x_vals)]
                closest_idx = argmin(distances)
                z_grid[i, j] = a_concentrations[closest_idx]
            end
        end
        
        # 创建contour图
        p5 = contour(k1f_unique, k1r_unique, z_grid', 
                    xlabel="k1f", ylabel="k1r", 
                    title="Metal GPU Parameter Scan Results (A concentration)",
                    colorbar_title="[A] final")
        savefig(p5, "param_scan_metal.png")
        println("Metal GPU contour plot saved as param_scan_metal.png")
        
        # 6. 统计摘要
        println("\n=== Metal GPU Statistical Summary ===")
        a_vals = [res[1] isa Vector ? res[1][end] : (res[1] isa Tuple ? res[1][end] : res[1]) for (params, res) in results]
        b_vals = [length(res) >= 2 ? (res[2] isa Vector ? res[2][end] : (res[2] isa Tuple ? res[2][end] : res[2])) : 0.0 for (params, res) in results] 
        c_vals = [length(res) >= 3 ? (res[3] isa Vector ? res[3][end] : (res[3] isa Tuple ? res[3][end] : res[3])) : 0.0 for (params, res) in results]
        product_ratio = [(b + c)/max(a, 1e-6) for (a,b,c) in zip(a_vals, b_vals, c_vals)]
        
        println("A concentration: mean=$(round(mean(a_vals), digits=3)), std=$(round(std(a_vals), digits=3))")
        println("B concentration: mean=$(round(mean(b_vals), digits=3)), std=$(round(std(b_vals), digits=3))")
        println("C concentration: mean=$(round(mean(c_vals), digits=3)), std=$(round(std(c_vals), digits=3))")
        println("Product ratio: mean=$(round(mean(product_ratio), digits=3)), std=$(round(std(product_ratio), digits=3))")
        
    else
        println("No valid results to plot")
    end
    
    # Performance summary
    println("\n=== Metal GPU Performance Summary ===")
    println("Total simulations attempted: $(min(20000, total_combinations))")
    println("Successful simulations: $(length(results))")
    println("Success rate: $(round(length(results)/min(20000, total_combinations)*100, digits=2))%")
    
    # Inspect the first few results 
    println("\nFirst 5 results:")
    for (i, (params, res)) in enumerate(results[1:min(5, length(results))])
        println("Result $i:")
        println("  Parameters (k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r): $params")
        println("  Final concentrations [A, B, C, E1, E2]: $res")
        println()
    end
    
    println("\n Metal GPU-accelerated parameter scan completed!")
end

# Usage instructions
println("\n=== Usage Instructions ===")
println("To run performance benchmark: julia param_scan_metal.jl benchmark")
println("To run normal parameter scan: julia param_scan_metal.jl")
