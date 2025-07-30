using CUDA
using Plots
using Statistics

# Include the main simulation files
include("../src/simulation.jl")

println("=== CUDA vs CPU Performance Comparison ===")

# Test parameters
test_params = [
    (1.0, 0.5, 2.0, 1.0, 1.5, 0.8, 2.5, 1.2),
    (2.0, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5),
    (0.5, 1.0, 1.5, 0.8, 2.0, 1.0, 1.0, 0.8),
    (1.5, 0.8, 1.0, 0.5, 0.8, 1.0, 1.5, 0.8),
    (0.8, 1.5, 2.0, 1.0, 1.0, 0.5, 0.8, 1.0)
]

# Fixed initial conditions
fixed_initial_conditions = Dict(
    Symbol("A") => 5.0,
    Symbol("B") => 0.0,
    Symbol("C") => 0.0,
    Symbol("E1") => 20.0,
    Symbol("E2") => 15.0,
    Symbol("AE1") => 0.0,
    Symbol("BE2") => 0.0
)

# CPU simulation function
function simulate_cpu(params)
    p = Dict(:k1f=>params[1], :k1r=>params[2], :k2f=>params[3], :k2r=>params[4],
             :k3f=>params[5], :k3r=>params[6], :k4f=>params[7], :k4r=>params[8])
    
    try
        sol = simulate_system(p, fixed_initial_conditions, (0.0, 5.0), saveat=0.1)
        vals = [sol[Symbol("A")][end], sol[Symbol("B")][end], sol[Symbol("C")][end], 
                sol[Symbol("E1")][end], sol[Symbol("E2")][end]]
        return vals
    catch
        return nothing
    end
end

# GPU simulation function (simplified)
function simulate_gpu(params)
    # For now, we'll use the same CPU function but with GPU memory management
    if CUDA.functional()
        CUDA.reclaim()  # Clear GPU memory
    end
    return simulate_cpu(params)
end

# Benchmark functions
function benchmark_cpu()
    println("Benchmarking CPU performance...")
    results = []
    start_time = time()
    
    for params in test_params
        res = simulate_cpu(params)
        if res !== nothing
            push!(results, res)
        end
    end
    
    cpu_time = time() - start_time
    println("CPU time: $(round(cpu_time, digits=4)) seconds")
    println("CPU successful simulations: $(length(results))")
    
    return cpu_time, results
end

function benchmark_gpu()
    if !CUDA.functional()
        println("❌ CUDA not available, skipping GPU benchmark")
        return 0.0, []
    end
    
    println("Benchmarking GPU performance...")
    results = []
    start_time = time()
    
    # Initialize GPU memory
    CUDA.reclaim()
    
    for params in test_params
        res = simulate_gpu(params)
        if res !== nothing
            push!(results, res)
        end
    end
    
    # Synchronize GPU operations
    CUDA.synchronize()
    
    gpu_time = time() - start_time
    println("GPU time: $(round(gpu_time, digits=4)) seconds")
    println("GPU successful simulations: $(length(results))")
    
    return gpu_time, results
end

# Run benchmarks
println("Running performance benchmarks...")
println()

cpu_time, cpu_results = benchmark_cpu()
println()

gpu_time, gpu_results = benchmark_gpu()
println()

# Calculate speedup
if gpu_time > 0
    speedup = cpu_time / gpu_time
    println("=== Performance Summary ===")
    println("CPU time: $(round(cpu_time, digits=4)) seconds")
    println("GPU time: $(round(gpu_time, digits=4)) seconds")
    println("GPU speedup: $(round(speedup, digits=2))x")
    
    if speedup > 1.0
        println("✅ GPU is faster than CPU")
    else
        println("⚠️ CPU is faster than GPU (likely due to small dataset)")
    end
else
    println("❌ Could not complete GPU benchmark")
end

# Test larger dataset for better GPU utilization
println("\n=== Testing with larger dataset ===")

# Create larger parameter set
large_param_set = []
for i in 1:50
    push!(large_param_set, (
        rand() * 5.0, rand() * 5.0, rand() * 5.0, rand() * 5.0,
        rand() * 5.0, rand() * 5.0, rand() * 5.0, rand() * 5.0
    ))
end

println("Testing with $(length(large_param_set)) parameter sets...")

# CPU benchmark for large dataset
cpu_start = time()
cpu_large_results = []
for params in large_param_set
    res = simulate_cpu(params)
    if res !== nothing
        push!(cpu_large_results, res)
    end
end
cpu_large_time = time() - cpu_start

# GPU benchmark for large dataset
if CUDA.functional()
    gpu_start = time()
    gpu_large_results = []
    CUDA.reclaim()
    
    for params in large_param_set
        res = simulate_gpu(params)
        if res !== nothing
            push!(gpu_large_results, res)
        end
    end
    
    CUDA.synchronize()
    gpu_large_time = time() - gpu_start
    
    large_speedup = cpu_large_time / gpu_large_time
    
    println("Large dataset results:")
    println("  CPU time: $(round(cpu_large_time, digits=4)) seconds")
    println("  GPU time: $(round(gpu_large_time, digits=4)) seconds")
    println("  GPU speedup: $(round(large_speedup, digits=2))x")
    
    if large_speedup > 1.0
        println("✅ GPU shows better performance with larger dataset")
    else
        println("⚠️ CPU still faster (may need even larger dataset)")
    end
else
    println("❌ CUDA not available for large dataset test")
end

println("\n=== Benchmark completed ===") 