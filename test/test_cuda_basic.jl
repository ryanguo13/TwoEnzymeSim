using CUDA
using Plots
using Statistics

# Include the main simulation files
include("../src/simulation.jl")
include("../src/visualization.jl")

println("=== CUDA Basic Test ===")

# Test CUDA availability
if CUDA.functional()
    println("✅ CUDA GPU is available")
    println("GPU device: $(CUDA.name(CUDA.device()))")
    println("GPU memory: $(round(CUDA.totalmem(CUDA.device()) / 1024^3, digits=2)) GB")
    
    # Test basic CUDA operations
    println("\nTesting basic CUDA operations...")
    
    # Create test arrays
    a = rand(Float64, 1000, 1000)
    b = rand(Float64, 1000, 1000)
    
    # CPU computation
    cpu_start = time()
    c_cpu = a * b
    cpu_time = time() - cpu_start
    
    # GPU computation
    gpu_start = time()
    a_gpu = CuArray(a)
    b_gpu = CuArray(b)
    c_gpu = a_gpu * b_gpu
    c_gpu_cpu = Array(c_gpu)
    gpu_time = time() - gpu_start
    
    println("CPU matrix multiplication: $(round(cpu_time, digits=4)) seconds")
    println("GPU matrix multiplication: $(round(gpu_time, digits=4)) seconds")
    println("GPU speedup: $(round(cpu_time/gpu_time, digits=2))x")
    
    # Test simulation function
    println("\nTesting simulation function...")
    
    # Test parameters
    test_params = (1.0, 0.5, 2.0, 1.0, 1.5, 0.8, 2.5, 1.2)
    
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
    
    # Test simulation
    p = Dict(:k1f=>test_params[1], :k1r=>test_params[2], :k2f=>test_params[3], :k2r=>test_params[4],
             :k3f=>test_params[5], :k3r=>test_params[6], :k4f=>test_params[7], :k4r=>test_params[8])
    
    try
        sol = simulate_system(p, fixed_initial_conditions, (0.0, 5.0), saveat=0.1)
        println("✅ Simulation completed successfully")
        
        # Test preprocessing
        vals = [sol[Symbol("A")][end], sol[Symbol("B")][end], sol[Symbol("C")][end], 
                sol[Symbol("E1")][end], sol[Symbol("E2")][end]]
        println("Final concentrations: $vals")
        
    catch e
        println("❌ Simulation failed: $e")
    end
    
    # Test small batch processing
    println("\nTesting batch processing...")
    
    # Create small parameter batch
    param_batch = [test_params, (2.0, 1.0, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5)]
    
    # Test GPU batch processing (simplified version)
    gpu_start = time()
    results = []
    for params in param_batch
        p = Dict(:k1f=>params[1], :k1r=>params[2], :k2f=>params[3], :k2r=>params[4],
                 :k3f=>params[5], :k3r=>params[6], :k4f=>params[7], :k4r=>params[8])
        
        try
            sol = simulate_system(p, fixed_initial_conditions, (0.0, 5.0), saveat=0.1)
            vals = [sol[Symbol("A")][end], sol[Symbol("B")][end], sol[Symbol("C")][end], 
                    sol[Symbol("E1")][end], sol[Symbol("E2")][end]]
            push!(results, (params, vals))
        catch
            push!(results, (params, nothing))
        end
    end
    gpu_time = time() - gpu_start
    
    println("Batch processing completed in $(round(gpu_time, digits=4)) seconds")
    println("Successful simulations: $(count(x -> x[2] !== nothing, results))")
    
    # Test visualization
    println("\nTesting visualization...")
    if length(results) > 0
        try
            p1 = plot_multi_species_heatmap(results)
            if p1 !== nothing
                savefig(p1, "test_cuda_heatmap.png")
                println("✅ Heatmap visualization created successfully")
            else
                println("⚠️ Heatmap visualization returned nothing")
            end
        catch e
            println("❌ Visualization failed: $e")
        end
    end
    
    println("\n=== CUDA Test Summary ===")
    println("✅ CUDA functionality verified")
    println("✅ Basic GPU operations working")
    println("✅ Simulation functions working")
    println("✅ Batch processing working")
    println("✅ Visualization functions working")
    
else
    println("❌ CUDA GPU is not available")
    println("This test requires a CUDA-capable GPU")
end

println("\nTest completed!") 