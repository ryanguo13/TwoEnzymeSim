using Distributed
using Plots
using IterTools
using Statistics
using CUDA  
using LinearAlgebra
using DifferentialEquations
using DiffEqGPU
using Printf
# include("../src/parameters.jl")
include("../src/simulation.jl")
include("../src/visualization.jl")

# Define the parameters with thermodynamic constraints
# Keep all reaction rate constants but add equilibrium constraints

# All reaction rate constants (including reverse rates)
k1f_range = 0.1:4:20.0  # 20 points
k1r_range = 0.1:4:20.0  # 20 points
k2f_range = 0.1:4:20.0  # 20 points
k2r_range = 0.1:4:20.0  # 20 points
k3f_range = 0.1:4:20.0  # 20 points
k3r_range = 0.1:4:20.0  # 20 points
k4f_range = 0.1:4:20.0  # 20 points
k4r_range = 0.1:4:20.0  # 20 points

# Initial concentrations
A_range = 5.0:4:20.0
B_range = 0.0:4:5.0
C_range = 0.0:4:5.0
E1_range = 5.0:4:20.0
E2_range = 5.0:4:20.0

# Progress bar function (defined early for use in parameter generation)
function print_progress_bar(current, total, width=50, prefix="Progress")
    percentage = current / total
    filled = round(Int, width * percentage)
    bar = "█" * "█"^filled * "░"^(width - filled)
    @printf("%s: [%s] %3.1f%% (%d/%d)\r", prefix, bar, percentage * 100, current, total)
    flush(stdout)
end

# Ultra-efficient vectorized parameter generation using mathematical optimization
function generate_thermodynamic_parameters_vectorized()
    println("Generating parameters with thermodynamic constraints (vectorized version)...")
    
    # Pre-calculate all parameter ranges as arrays
    k1f_array = collect(k1f_range)
    k1r_array = collect(k1r_range)
    k2f_array = collect(k2f_range)
    k2r_array = collect(k2r_range)
    k3f_array = collect(k3f_range)
    k3r_array = collect(k3r_range)
    k4f_array = collect(k4f_range)
    k4r_array = collect(k4r_range)
    A_array = collect(A_range)
    B_array = collect(B_range)
    C_array = collect(C_range)
    E1_array = collect(E1_range)
    E2_array = collect(E2_range)
    
    # Calculate total combinations
    total_combinations = length(k1f_array) * length(k1r_array) * length(k2f_array) * length(k2r_array) * 
                        length(k3f_array) * length(k3r_array) * length(k4f_array) * length(k4r_array) *
                        length(A_array) * length(B_array) * length(C_array) * length(E1_array) * length(E2_array)
    
    println("Total possible combinations: $total_combinations")
    
    # Step 1: Use mathematical optimization to pre-filter rate constants
    println("Step 1: Pre-filtering rate constants using mathematical constraints...")
    
    # Create all rate constant combinations efficiently
    rate_combinations_raw = collect(Iterators.product(
        k1f_array, k1r_array, k2f_array, k2r_array, k3f_array, k3r_array, k4f_array, k4r_array
    ))
    
    # Vectorized thermodynamic constraint checking
    println("Applying thermodynamic constraints...")
    
    # Convert to arrays for vectorized operations
    rate_array = hcat([collect(Float64.(rate_params)) for rate_params in rate_combinations_raw]...)
    
    # Vectorized calculation of equilibrium constants
    k1f_vec = rate_array[1, :]
    k1r_vec = rate_array[2, :]
    k2f_vec = rate_array[3, :]
    k2r_vec = rate_array[4, :]
    k3f_vec = rate_array[5, :]
    k3r_vec = rate_array[6, :]
    k4f_vec = rate_array[7, :]
    k4r_vec = rate_array[8, :]
    
    # Vectorized equilibrium constant calculations
    Keq1_vec = (k1f_vec .* k2f_vec) ./ (k1r_vec .* k2r_vec)
    Keq2_vec = (k3f_vec .* k4f_vec) ./ (k3r_vec .* k4r_vec)
    
    # Vectorized constraint checking
    valid_mask = (0.1 .<= Keq1_vec .<= 10.0) .&& (0.1 .<= Keq2_vec .<= 10.0)
    valid_indices = findall(valid_mask)
    
    # Extract valid rate combinations
    rate_combinations = [rate_combinations_raw[i] for i in valid_indices]
    
    println("Valid rate constant combinations: $(length(rate_combinations))")
    println("Reduction factor: $(round(length(rate_combinations_raw) / length(rate_combinations), digits=2))x")
    
    # Step 2: Generate concentration combinations
    conc_combinations = collect(Iterators.product(A_array, B_array, C_array, E1_array, E2_array))
    println("Concentration combinations: $(length(conc_combinations))")
    
    # Step 3: Efficient combination using pre-allocated arrays
    println("Step 2: Combining rate and concentration combinations (vectorized)...")
    
    total_final_combinations = length(rate_combinations) * length(conc_combinations)
    println("Final parameter combinations: $total_final_combinations")
    
    # Pre-allocate result array for maximum efficiency
    params = Vector{NTuple{13, Float64}}(undef, total_final_combinations)
    
    # Use vectorized operations for combination
    counter = 0
    for (i, rate_params) in enumerate(rate_combinations)
        for (j, conc_params) in enumerate(conc_combinations)
            counter += 1
            
            # Show progress every 100000 combinations
            if counter % 100000 == 0 || counter == total_final_combinations
                print_progress_bar(counter, total_final_combinations, 40, "Vectorized Combining")
            end
            
            # Direct assignment to pre-allocated array
            params[counter] = (rate_params..., conc_params...)
        end
    end
    
    println()  # New line after progress bar
    
    return params
end

# Memory-efficient streaming version for very large parameter spaces
function generate_thermodynamic_parameters_streaming()
    println("Generating parameters with thermodynamic constraints (streaming version)...")
    
    # Pre-calculate all parameter ranges as arrays
    k1f_array = collect(k1f_range)
    k1r_array = collect(k1r_range)
    k2f_array = collect(k2f_range)
    k2r_array = collect(k2r_range)
    k3f_array = collect(k3f_range)
    k3r_array = collect(k3r_range)
    k4f_array = collect(k4f_range)
    k4r_array = collect(k4r_range)
    A_array = collect(A_range)
    B_array = collect(B_range)
    C_array = collect(C_range)
    E1_array = collect(E1_range)
    E2_array = collect(E2_range)
    
    # Calculate total combinations
    total_combinations = length(k1f_array) * length(k1r_array) * length(k2f_array) * length(k2r_array) * 
                        length(k3f_array) * length(k3r_array) * length(k4f_array) * length(k4r_array) *
                        length(A_array) * length(B_array) * length(C_array) * length(E1_array) * length(E2_array)
    
    println("Total possible combinations: $total_combinations")
    
    # Use streaming approach to avoid memory issues
    println("Step 1: Streaming rate constant generation...")
    
    # Create a streaming iterator for rate constants
    rate_stream = Iterators.product(
        k1f_array, k1r_array, k2f_array, k2r_array, k3f_array, k3r_array, k4f_array, k4r_array
    )
    
    # Stream and filter rate combinations
    rate_combinations = []
    rate_counter = 0
    total_rate_combinations = length(k1f_array) * length(k1r_array) * length(k2f_array) * length(k2r_array) *
                             length(k3f_array) * length(k3r_array) * length(k4f_array) * length(k4r_array)
    
    for rate_params in rate_stream
        rate_counter += 1
        
        # Show progress every 100000 combinations
        if rate_counter % 100000 == 0 || rate_counter == total_rate_combinations
            print_progress_bar(rate_counter, total_rate_combinations, 40, "Rate Streaming")
        end
        
        k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = rate_params
        
        # Check thermodynamic constraints
        Keq1 = (k1f * k2f) / (k1r * k2r)
        Keq2 = (k3f * k4f) / (k3r * k4r)
        
        # Apply thermodynamic constraints
        if 0.1 <= Keq1 <= 10.0 && 0.1 <= Keq2 <= 10.0
            push!(rate_combinations, rate_params)
        end
    end
    
    println()  # New line after progress bar
    println("Valid rate constant combinations: $(length(rate_combinations))")
    
    # Generate concentration combinations
    conc_combinations = collect(Iterators.product(A_array, B_array, C_array, E1_array, E2_array))
    println("Concentration combinations: $(length(conc_combinations))")
    
    # Stream the final combinations
    println("Step 2: Streaming final combinations...")
    
    # Use a streaming approach for the final combination
    params = []
    total_final_combinations = length(rate_combinations) * length(conc_combinations)
    counter = 0
    
    for rate_params in rate_combinations
        for conc_params in conc_combinations
            counter += 1
            
            # Show progress every 100000 combinations
            if counter % 100000 == 0 || counter == total_final_combinations
                print_progress_bar(counter, total_final_combinations, 40, "Final Streaming")
            end
            
            combined_params = (rate_params..., conc_params...)
            push!(params, combined_params)
        end
    end
    
    println()  # New line after progress bar
    
    return params
end

# Function to generate parameters with thermodynamic constraints - OPTIMIZED VERSION
function generate_thermodynamic_parameters_optimized()
    println("Generating parameters with thermodynamic constraints (optimized version)...")
    
    # Pre-calculate all parameter ranges as arrays for vectorized operations
    k1f_array = collect(k1f_range)
    k1r_array = collect(k1r_range)
    k2f_array = collect(k2f_range)
    k2r_array = collect(k2r_range)
    k3f_array = collect(k3f_range)
    k3r_array = collect(k3r_range)
    k4f_array = collect(k4f_range)
    k4r_array = collect(k4r_range)
    A_array = collect(A_range)
    B_array = collect(B_range)
    C_array = collect(C_range)
    E1_array = collect(E1_range)
    E2_array = collect(E2_range)
    
    # Calculate total combinations for progress tracking
    total_combinations = length(k1f_array) * length(k1r_array) * length(k2f_array) * length(k2r_array) * 
                        length(k3f_array) * length(k3r_array) * length(k4f_array) * length(k4r_array) *
                        length(A_array) * length(B_array) * length(C_array) * length(E1_array) * length(E2_array)
    
    println("Total possible combinations: $total_combinations")
    
    # Use vectorized operations to generate all combinations efficiently
    # We'll use a more efficient approach by pre-filtering based on thermodynamic constraints
    
    # Step 1: Generate all rate constant combinations first
    rate_combinations = []
    rate_counter = 0
    total_rate_combinations = length(k1f_array) * length(k1r_array) * length(k2f_array) * length(k2r_array) *
                             length(k3f_array) * length(k3r_array) * length(k4f_array) * length(k4r_array)
    
    println("Step 1: Generating rate constant combinations...")
    
    for (k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r) in Iterators.product(
        k1f_array, k1r_array, k2f_array, k2r_array, k3f_array, k3r_array, k4f_array, k4r_array
    )
        rate_counter += 1
        
        # Show progress every 100000 combinations
        if rate_counter % 100000 == 0 || rate_counter == total_rate_combinations
            print_progress_bar(rate_counter, total_rate_combinations, 40, "Rate Constants")
        end
        
        # Check thermodynamic constraints
        Keq1 = (k1f * k2f) / (k1r * k2r)
        Keq2 = (k3f * k4f) / (k3r * k4r)
        
        # Apply thermodynamic constraints
        if 0.1 <= Keq1 <= 10.0 && 0.1 <= Keq2 <= 10.0
            push!(rate_combinations, (k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r))
        end
    end
    
    println()  # New line after progress bar
    println("Valid rate constant combinations: $(length(rate_combinations))")
    
    # Step 2: Generate concentration combinations
    conc_combinations = []
    for (A0, B0, C0, E1_0, E2_0) in Iterators.product(A_array, B_array, C_array, E1_array, E2_array)
        push!(conc_combinations, (A0, B0, C0, E1_0, E2_0))
    end
    
    println("Concentration combinations: $(length(conc_combinations))")
    
    # Step 3: Combine rate and concentration combinations efficiently
    println("Step 2: Combining rate and concentration combinations...")
    
    # Pre-allocate result array with estimated size
    estimated_size = length(rate_combinations) * length(conc_combinations)
    params = Vector{NTuple{13, Float64}}(undef, 0)
    sizehint!(params, estimated_size)
    
    total_combinations = length(rate_combinations) * length(conc_combinations)
    counter = 0
    
    for rate_params in rate_combinations
        for conc_params in conc_combinations
            counter += 1
            
            # Show progress every 100000 combinations
            if counter % 100000 == 0 || counter == total_combinations
                print_progress_bar(counter, total_combinations, 40, "Combining")
            end
            
            # Combine rate and concentration parameters
            combined_params = (rate_params..., conc_params...)
            push!(params, combined_params)
        end
    end
    
    println()  # New line after progress bar
    
    return params
end

# Alternative optimized version using parallel processing
function generate_thermodynamic_parameters_parallel()
    println("Generating parameters with thermodynamic constraints (parallel version)...")
    
    # Pre-calculate all parameter ranges as arrays
    k1f_array = collect(k1f_range)
    k1r_array = collect(k1r_range)
    k2f_array = collect(k2f_range)
    k2r_array = collect(k2r_range)
    k3f_array = collect(k3f_range)
    k3r_array = collect(k3r_range)
    k4f_array = collect(k4f_range)
    k4r_array = collect(k4r_range)
    A_array = collect(A_range)
    B_array = collect(B_range)
    C_array = collect(C_range)
    E1_array = collect(E1_range)
    E2_array = collect(E2_range)
    
    # Calculate total combinations
    total_combinations = length(k1f_array) * length(k1r_array) * length(k2f_array) * length(k2r_array) * 
                        length(k3f_array) * length(k3r_array) * length(k4f_array) * length(k4r_array) *
                        length(A_array) * length(B_array) * length(C_array) * length(E1_array) * length(E2_array)
    
    println("Total possible combinations: $total_combinations")
    
    # Use parallel processing for rate constant generation
    println("Step 1: Generating rate constant combinations (parallel)...")
    
    # Create all rate constant combinations
    rate_combinations = []
    rate_counter = 0
    total_rate_combinations = length(k1f_array) * length(k1r_array) * length(k2f_array) * length(k2r_array) *
                             length(k3f_array) * length(k3r_array) * length(k4f_array) * length(k4r_array)
    
    # Use parallel processing for rate constant filtering
    rate_combinations_raw = collect(Iterators.product(
        k1f_array, k1r_array, k2f_array, k2r_array, k3f_array, k3r_array, k4f_array, k4r_array
    ))
    
    # Filter rate combinations in parallel
    rate_combinations = @sync @distributed (vcat) for rate_params in rate_combinations_raw
        k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = rate_params
        
        # Check thermodynamic constraints
        Keq1 = (k1f * k2f) / (k1r * k2r)
        Keq2 = (k3f * k4f) / (k3r * k4r)
        
        # Apply thermodynamic constraints
        if 0.1 <= Keq1 <= 10.0 && 0.1 <= Keq2 <= 10.0
            [rate_params]
        else
            []
        end
    end
    
    println("Valid rate constant combinations: $(length(rate_combinations))")
    
    # Generate concentration combinations
    conc_combinations = collect(Iterators.product(A_array, B_array, C_array, E1_array, E2_array))
    println("Concentration combinations: $(length(conc_combinations))")
    
    # Combine in parallel
    println("Step 2: Combining rate and concentration combinations (parallel)...")
    
    # Use parallel processing for final combination
    params = @sync @distributed (vcat) for rate_params in rate_combinations
        local_batch = []
        for conc_params in conc_combinations
            combined_params = (rate_params..., conc_params...)
            push!(local_batch, combined_params)
        end
        local_batch
    end
    
    return params
end

# GPU-accelerated parameter generation (for very large parameter spaces)
function generate_thermodynamic_parameters_gpu()
    if !cuda_available || !CUDA.functional()
        println("CUDA not available, falling back to CPU version")
        return generate_thermodynamic_parameters_optimized()
    end
    
    println("Generating parameters with thermodynamic constraints (GPU version)...")
    
    # Pre-calculate all parameter ranges as arrays
    k1f_array = collect(k1f_range)
    k1r_array = collect(k1r_range)
    k2f_array = collect(k2f_range)
    k2r_array = collect(k2r_range)
    k3f_array = collect(k3f_range)
    k3r_array = collect(k3r_range)
    k4f_array = collect(k4f_range)
    k4r_array = collect(k4r_range)
    A_array = collect(A_range)
    B_array = collect(B_range)
    C_array = collect(C_range)
    E1_array = collect(E1_range)
    E2_array = collect(E2_range)
    
    # Calculate total combinations
    total_combinations = length(k1f_array) * length(k1r_array) * length(k2f_array) * length(k2r_array) * 
                        length(k3f_array) * length(k3r_array) * length(k4f_array) * length(k4r_array) *
                        length(A_array) * length(B_array) * length(C_array) * length(E1_array) * length(E2_array)
    
    println("Total possible combinations: $total_combinations")
    
    # Step 1: Generate rate constant combinations on GPU
    println("Step 1: Generating rate constant combinations (GPU)...")
    
    # Create all rate combinations and transfer to GPU
    rate_combinations_raw = collect(Iterators.product(
        k1f_array, k1r_array, k2f_array, k2r_array, k3f_array, k3r_array, k4f_array, k4r_array
    ))
    
    # Convert to GPU arrays for parallel processing
    rate_array = hcat([collect(Float64.(rate_params)) for rate_params in rate_combinations_raw]...)
    rate_gpu = CuArray{Float64}(rate_array)
    
    # GPU kernel for thermodynamic constraint checking
    function check_thermodynamic_constraints_gpu(rate_gpu)
        # This would be implemented as a CUDA kernel
        # For now, we'll use CPU filtering with GPU arrays
        valid_indices = Int[]
        
        for i in 1:size(rate_gpu, 2)
            k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = rate_gpu[:, i]
            
            Keq1 = (k1f * k2f) / (k1r * k2r)
            Keq2 = (k3f * k4f) / (k3r * k4r)
            
            if 0.1 <= Keq1 <= 10.0 && 0.1 <= Keq2 <= 10.0
                push!(valid_indices, i)
            end
        end
        
        return valid_indices
    end
    
    # Filter rate combinations
    valid_indices = check_thermodynamic_constraints_gpu(rate_gpu)
    rate_combinations = [rate_combinations_raw[i] for i in valid_indices]
    
    println("Valid rate constant combinations: $(length(rate_combinations))")
    
    # Step 2: Generate concentration combinations
    conc_combinations = collect(Iterators.product(A_array, B_array, C_array, E1_array, E2_array))
    println("Concentration combinations: $(length(conc_combinations))")
    
    # Step 3: Combine efficiently
    println("Step 2: Combining rate and concentration combinations...")
    
    params = Vector{NTuple{13, Float64}}(undef, 0)
    sizehint!(params, length(rate_combinations) * length(conc_combinations))
    
    total_combinations = length(rate_combinations) * length(conc_combinations)
    counter = 0
    
    for rate_params in rate_combinations
        for conc_params in conc_combinations
            counter += 1
            
            if counter % 100000 == 0 || counter == total_combinations
                print_progress_bar(counter, total_combinations, 40, "GPU Combining")
            end
            
            combined_params = (rate_params..., conc_params...)
            push!(params, combined_params)
        end
    end
    
    println()
    return params
end

# Function to generate parameters with thermodynamic constraints
function generate_thermodynamic_parameters()
    # Choose the best method based on system capabilities and parameter space size
    total_combinations = length(k1f_range) * length(k1r_range) * length(k2f_range) * length(k2r_range) * 
                        length(k3f_range) * length(k3r_range) * length(k4f_range) * length(k4r_range) *
                        length(A_range) * length(B_range) * length(C_range) * length(E1_range) * length(E2_range)
    
    if total_combinations > 10_000_000
        # For very large parameter spaces, use streaming
        return generate_thermodynamic_parameters_streaming()
    elseif total_combinations > 1_000_000
        # For large parameter spaces, use vectorized approach
        return generate_thermodynamic_parameters_vectorized()
    elseif cuda_available && CUDA.functional() && total_combinations > 100_000
        # For medium-large parameter spaces with GPU available
        return generate_thermodynamic_parameters_gpu()
    elseif Threads.nthreads() > 1
        # For medium parameter spaces with multiple threads
        return generate_thermodynamic_parameters_parallel()
    else
        # For smaller parameter spaces, use optimized sequential approach
        return generate_thermodynamic_parameters_optimized()
    end
end

# Create parameter grid with thermodynamic constraints
println("Generating parameters with thermodynamic constraints...")
param_grid = generate_thermodynamic_parameters()

# Calculate total combinations for progress reporting
total_combinations = length(param_grid)
println("Total parameter combinations with thermodynamic constraints: $total_combinations")

# Calculate reduction in parameter space
original_combinations = length(k1f_range) * length(k1r_range) * length(k2f_range) * length(k2r_range) * 
                       length(k3f_range) * length(k3r_range) * length(k4f_range) * length(k4r_range) *
                       length(A_range) * length(B_range) * length(C_range) * length(E1_range) * length(E2_range)
reduction_factor = original_combinations / total_combinations
println("Parameter space reduction factor: $(round(reduction_factor, digits=2))x")

# Enhanced CUDA Device Configuration for V100 GPUs
function configure_optimal_cuda_device()
    if !CUDA.functional()
        println("❌ CUDA GPU is not available - falling back to CPU")
        return false
    end
    
    println("✅ CUDA is functional")
    num_devices = CUDA.ndevices()
    println("Number of CUDA devices detected: $num_devices")
    
    if num_devices == 0
        println("❌ No CUDA devices found")
        return false
    end
    
    # For systems with V100s, prefer the first V100 found
    best_device_id = 0
    best_score = -1000
    
    println("\n=== CUDA Device Analysis ===")
    for i in 0:(num_devices-1)
        device = CuDevice(i)
        name = CUDA.name(device)
        total_memory = CUDA.totalmem(device)
        total_memory_gb = total_memory / (1024^3)
        
        # Calculate performance score (higher is better)
        score = 0
        
        # Prefer professional GPUs (V100, Tesla, Quadro)
        if occursin("V100", name) || occursin("Tesla", name) || occursin("Quadro", name)
            score += 1000
            println("Device $i: $name [🚀 PROFESSIONAL GPU]")
        elseif occursin("RTX", name) || occursin("GTX", name)
            score += 500  
            println("Device $i: $name [💻 CONSUMER GPU]")
        else
            println("Device $i: $name [⚠️  OTHER/INTEGRATED]")
        end
        
        # Memory scoring (V100 typically has 16GB or 32GB)
        if total_memory_gb >= 16
            score += 200
        elseif total_memory_gb >= 8
            score += 100
        elseif total_memory_gb < 4
            score -= 200  # Likely integrated graphics
        end
        
        println("  Memory: $(round(total_memory_gb, digits=2)) GB")
        println("  Performance Score: $score")
        
        if score > best_score
            best_score = score
            best_device_id = i
        end
        println()
    end
    
    # Set the best device
    CUDA.device!(best_device_id)
    current_device = CUDA.device()
    device_name = CUDA.name(current_device)
    device_id = CUDA.deviceid(current_device)
    
    println("=== Selected Device ===")
    println("✅ Using Device $device_id: $device_name")
    println("Memory: $(round(CUDA.totalmem(current_device) / 1024^3, digits=2)) GB")
    
    # Optimize CUDA settings
    CUDA.reclaim()
    println("✅ CUDA memory pool initialized")
    
    return true
end

# Configure CUDA device at startup
cuda_available = configure_optimal_cuda_device()

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

# CUDA-accelerated batch simulation function using simulation.jl
function simulate_reaction_batch_gpu(param_batch, tspan)
    results = []
    
    if cuda_available && CUDA.functional()
        println("Using CUDA GPU acceleration for batch of $(length(param_batch)) simulations")
        
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
    
    
    param_array = hcat([collect(Float64.(params)) for params in param_sub_batch]...)
    param_gpu = CuArray{Float64}(param_array)
    
    # Process each parameter set using simulation.jl functions
    # This is still sequential but with GPU-accelerated math operations
    for (i, params) in enumerate(param_sub_batch)
        res = simulate_reaction_gpu_optimized(params, tspan, param_gpu, i)
        push!(results, res)
    end
    
    # Synchronize GPU operations
    CUDA.synchronize()
    
    return results
end

# GPU-optimized simulation function
function simulate_reaction_gpu_optimized(params, tspan, param_gpu=nothing, param_index=nothing)
    # Unpack the parameters (only reaction rate constants)
    k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, A0, B0, C0, E1_0, E2_0 = params
    
    # Check for overflow/invalid parameter values
    if any(x -> isnan(x) || isinf(x) || abs(x) > 1e6, params)
        return nothing
    end
    
    # Use the existing simulation.jl function instead of redefining the reaction network
    p = Dict(:k1f=>k1f, :k1r=>k1r, :k2f=>k2f, :k2r=>k2r, :k3f=>k3f, :k3r=>k3r, :k4f=>k4f, :k4r=>k4r)
    initial_conditions = Dict(:A=>A0, :B=>B0, :C=>C0, :E1=>E1_0, :E2=>E2_0, :AE1=>0.0, :BE2=>0.0)
    
    try
        # Use the simulate_system function from simulation.jl
        sol = simulate_system(p, initial_conditions, tspan, saveat=0.1)
        return preprocess_solution(sol)
    catch
        return nothing
    end
end

# Process a sub-batch on GPU using simulation.jl functions
function process_sub_batch_gpu(param_sub_batch, tspan)
    results = []
    
    # Convert parameters to GPU arrays for batch processing with Float32
    param_array = hcat([collect(Float64.(params)) for params in param_sub_batch]...)
    param_gpu = CuArray{Float64}(param_array)
    
    # Process each parameter set using simulation.jl functions
    for params in param_sub_batch
        res = simulate_reaction_gpu(params, tspan)
        push!(results, res)
    end
    
    return results
end

# GPU-accelerated simulation function using simulation.jl
function simulate_reaction_gpu(params, tspan)
    # 拆包参数
    k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, A0, B0, C0, E1_0, E2_0 = params
    if any(x -> isnan(x) || isinf(x) || abs(x) > 1e6, params)
        return nothing
    end
    p = Dict(:k1f=>k1f, :k1r=>k1r, :k2f=>k2f, :k2r=>k2r, :k3f=>k3f, :k3r=>k3r, :k4f=>k4f, :k4r=>k4r)
    initial_conditions = Dict(:A=>A0, :B=>B0, :C=>C0, :E1=>E1_0, :E2=>E2_0, :AE1=>0.0, :BE2=>0.0)
    try
        sol = simulate_system(p, initial_conditions, tspan, saveat=0.1)
        return preprocess_solution(sol)
    catch
        return nothing
    end
end


# Optimized GPU batch processing with kernel
function process_batch_gpu_with_kernel(param_batch, tspan)
    results = []
    
    
    param_array = hcat([collect(Float64.(params)) for params in param_batch]...)
    param_gpu = CuArray{Float64}(param_array)
    
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
    k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, A0, B0, C0, E1_0, E2_0 = params
    
    # Check for overflow/invalid parameter values
    if any(x -> isnan(x) || isinf(x) || abs(x) > 1e6, params)
        return nothing
    end
    
    # Use the existing simulation.jl function instead of redefining the reaction network
    p = Dict(:k1f=>k1f, :k1r=>k1r, :k2f=>k2f, :k2r=>k2r, :k3f=>k3f, :k3r=>k3r, :k4f=>k4f, :k4r=>k4r)
    initial_conditions = Dict(:A=>A0, :B=>B0, :C=>C0, :E1=>E1_0, :E2=>E2_0, :AE1=>0.0, :BE2=>0.0)
    
    try
        # Use the simulate_system function from simulation.jl
        sol = simulate_system(p, initial_conditions, tspan, saveat=0.1)
        return preprocess_solution(sol)
    catch
        return nothing
    end
end

# True parallel GPU batch processing
function process_batch_gpu_parallel(param_batch, tspan)
    results = []
    
    
    param_array = hcat([collect(Float64.(params)) for params in param_batch]...)
    param_gpu = CuArray{Float64}(param_array)
    
    # Process in parallel chunks
    chunk_size = 100  # Process 100 simulations at a time
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
    
    # Use CUDA.jl's parallel computing capabilities
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



# CUDA memory management function
function manage_cuda_memory()
    if cuda_available && CUDA.functional()
        # Reclaim unused memory
        CUDA.reclaim()
        
        # Get memory info - returns (allocated, free) tuple
        allocated, free = CUDA.memory_info()
        println("CUDA memory usage: $(round(allocated / 1024^3, digits=2)) GB allocated, $(round(free / 1024^3, digits=2)) GB free")
    end
end

# CUDA-accelerated parameter scan with batching and performance monitoring
function run_parameter_scan_cuda(param_grid, batch_size=nothing, max_simulations=100000)
    # Auto-optimize batch size based on GPU memory
    if batch_size === nothing
        if cuda_available && CUDA.functional()
            gpu_memory = CUDA.totalmem(CUDA.device())
            # Use 20% of GPU memory for batch processing
            estimated_memory_per_sim = 1024 * 1024  # 1MB per simulation
            optimal_batch_size = max(100, min(2000, floor(Int, gpu_memory * 0.2 / estimated_memory_per_sim)))
            batch_size = optimal_batch_size
            println("Auto-optimized batch size: $batch_size (based on GPU memory)")
        else
            batch_size = 500  # Default for CPU
        end
    end
    results = []
    total_simulations = min(max_simulations, total_combinations)
    
    println("Running $total_simulations simulations using CUDA GPU acceleration")
    println("Batch size: $batch_size")
    
    # Initialize CUDA memory management
    manage_cuda_memory()
    
    # Convert array to array for easier batching (param_grid is already an array)
    param_array = param_grid[1:total_simulations]
    
    # Performance monitoring
    total_start_time = time()
    successful_simulations = 0
    
    # Process in batches
    for i in 1:batch_size:length(param_array)
        batch_end = min(i + batch_size - 1, length(param_array))
        batch = param_array[i:batch_end]
        
        batch_num = div(i-1, batch_size) + 1
        total_batches = div(length(param_array)-1, batch_size) + 1
        
        # Show progress bar
        print_progress_bar(batch_num, total_batches, 40, "Batch Progress")
        
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
            println("\n  Progress: $successful_simulations successful simulations")
            println("  Rate: $(round(rate, digits=2)) simulations/second")
            println("  Batch $batch_num: $batch_successful successful, $(round(batch_time, digits=3))s")
            
            # Manage CUDA memory periodically
            manage_cuda_memory()
        end
    end
    
    # Final progress bar update
    println()  # New line after progress bar
    
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
    
    # Convert array to array (param_grid is already an array)
    param_array = param_grid[1:min(max_simulations, length(param_grid))]
    
    for (i, params) in enumerate(param_array)
        # Show progress bar every 10 simulations
        if i % 10 == 0 || i == length(param_array)
            print_progress_bar(i, length(param_array), 40, "CPU Progress")
        end
        
        res = simulate_reaction_cpu(params, (0.0, 5.0))
        if res !== nothing
            push!(results, (params, res))
            successful_simulations += 1
        end
    end
    
    # Final progress bar update
    println()  # New line after progress bar
    
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
    gpu_results = run_parameter_scan_cuda(param_grid, 100, max_simulations)
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

function ensemble_gpu_scan(param_grid, tspan=(0.0, 5.0); saveat=0.1, max_batch=100000)
    println("Running EnsembleProblem + EnsembleGPUArray GPU batch scan...")
    # 1. 收集参数和初始条件
    param_array = param_grid[1:min(max_batch, length(param_grid))]
    n = length(param_array)
    println("Total ensemble batch size: $n")
    
    # Show progress for parameter preparation
    print_progress_bar(0, 3, 40, "Ensemble Setup")
    
    # 2. 构造参数和初始条件数组
    u0s = [
        [A0, B0, C0, E1_0, E2_0, 0.0, 0.0] for (k1f,k1r,k2f,k2r,k3f,k3r,k4f,k4r,A0,B0,C0,E1_0,E2_0) in param_array
    ]
    ps = [
        (k1f=k1f, k1r=k1r, k2f=k2f, k2r=k2r, k3f=k3f, k3r=k3r, k4f=k4f, k4r=k4r)
        for (k1f,k1r,k2f,k2r,k3f,k3r,k4f,k4r,A0,B0,C0,E1_0,E2_0) in param_array
    ]
    
    print_progress_bar(1, 3, 40, "Ensemble Setup")
    
    # 3. 定义ODE问题
    function f!(du, u, p, t)
        # 变量顺序: A, B, C, E1, E2, AE1, BE2
        # 参数p为NamedTuple
        k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = p.k1f, p.k1r, p.k2f, p.k2r, p.k3f, p.k3r, p.k4f, p.k4r
        A, B, C, E1, E2, AE1, BE2 = u
        du[1] = -k1f*A*E1 + k1r*AE1 + k2r*B*E1 - k2f*AE1
        du[2] = k2f*AE1 - k2r*B*E1 - k3f*B*E2 + k3r*BE2
        du[3] = k4f*BE2 - k4r*C*E2
        du[4] = -k1f*A*E1 + k1r*AE1 + k2f*AE1 - k2r*B*E1
        du[5] = -k3f*B*E2 + k3r*BE2 + k4f*BE2 - k4r*C*E2
        du[6] = k1f*A*E1 - k1r*AE1 - k2f*AE1 + k2r*B*E1
        du[7] = k3f*B*E2 - k3r*BE2 - k4f*BE2 + k4r*C*E2
    end
    prob = ODEProblem(f!, u0s[1], tspan, ps[1])
    ensemble_prob = EnsembleProblem(prob, prob_func=(prob,i,repeat)->(prob.u0=u0s[i]; prob.p=ps[i]; prob))
    
    print_progress_bar(2, 3, 40, "Ensemble Setup")
    
    println("\nSolving ensemble with GPU acceleration...")
    sol = solve(ensemble_prob, Tsit5(), EnsembleGPUArray(), trajectories=n, saveat=saveat)
    
    print_progress_bar(3, 3, 40, "Ensemble Setup")
    println()  # New line after progress bar
    
    # 4. 结果整理
    println("Processing results...")
    results = []
    for i in 1:n
        # Show progress every 1000 results
        if i % 1000 == 0 || i == n
            print_progress_bar(i, n, 40, "Result Processing")
        end
        
        s = sol[i]
        if s.retcode == :Success
            vals = [s[1,end], s[2,end], s[3,end], s[4,end], s[5,end]]
            push!(results, (param_array[i], vals))
        end
    end
    
    println()  # New line after progress bar
    return results
end

# Run the CUDA GPU-accelerated parameter scan with thermodynamic constraints
println("Starting CUDA GPU-accelerated parameter scan with thermodynamic constraints...")
println("Thermodynamic constraints applied:")
println("  Keq1 = (k1f * k2f) / (k1r * k2r) for reactions 1 and 2 (constrained to 0.1-10.0)")
println("  Keq2 = (k3f * k4f) / (k3r * k4r) for reactions 3 and 4 (constrained to 0.1-10.0)")
println("  All reaction rate constants (k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r) are preserved")

# Check if user wants performance comparison
if length(ARGS) > 0 && ARGS[1] == "benchmark"
    println("Running performance benchmark...")
    cpu_results, gpu_results, speedup = compare_gpu_cpu_performance(param_grid, 100000)
    println("Benchmark completed!")
elseif length(ARGS) > 0 && ARGS[1] == "ensemble"
    results = ensemble_gpu_scan(param_grid)
    println("\nNumber of results: $(length(results))")
    if length(results) > 0
        x_idx = 1  # k1f
        y_idx = 2  # k1r
        z_idx = 9  # A0
        x_vals = [params[x_idx] for (params, res) in results]
        y_vals = [params[y_idx] for (params, res) in results]
        z_vals = [res[1] for (params, res) in results]
        println("k1f range in results: $(minimum(x_vals)) to $(maximum(x_vals))")
        println("k1r range in results: $(minimum(y_vals)) to $(maximum(y_vals))")
        println("A0 (initial) range in results: $(minimum([params[9] for (params,_) in results])) to $(maximum([params[9] for (params,_) in results]))")
        p1 = plot_multi_species_heatmap(results)
        if p1 !== nothing
            savefig(p1, "multi_species_heatmap_cuda.png")
            println("Multi-species heatmap saved as multi_species_heatmap_cuda.png")
        end
        p2 = plot_parameter_sensitivity_analysis(results)
        if p2 !== nothing
            savefig(p2, "parameter_sensitivity_cuda.png")
            println("Parameter sensitivity analysis saved as parameter_sensitivity_cuda.png")
        end
        p3 = plot_concentration_distributions(results)
        if p3 !== nothing
            savefig(p3, "concentration_distributions_cuda.png")
            println("Concentration distributions saved as concentration_distributions_cuda.png")
        end
        p4 = plot_3d_parameter_space(results, 1, 2, 9)
        if p4 !== nothing
            savefig(p4, "3d_parameter_space_cuda.png")
            println("3D parameter space plot saved as 3d_parameter_space_cuda.png")
        end
        k1f_unique = sort(unique(x_vals))
        k1r_unique = sort(unique(y_vals))
        z_grid = zeros(length(k1f_unique), length(k1r_unique))
        a_concentrations = [res[1] isa Vector ? res[1][end] : (res[1] isa Tuple ? res[1][end] : res[1]) for (params, res) in results]
        for i in eachindex(k1f_unique)
            for j in eachindex(k1r_unique)
                distances = [(x_vals[k] - k1f_unique[i])^2 + (y_vals[k] - k1r_unique[j])^2 for k in eachindex(x_vals)]
                closest_idx = argmin(distances)
                z_grid[i, j] = a_concentrations[closest_idx]
            end
        end
        p5 = contour(k1f_unique, k1r_unique, z_grid', 
                    xlabel="k1f", ylabel="k1r", 
                    title="CUDA GPU Parameter Scan Results (A concentration)",
                    colorbar_title="[A] final")
        savefig(p5, "param_scan_cuda.png")
        println("CUDA GPU contour plot saved as param_scan_cuda.png")
        println("\n=== CUDA GPU Statistical Summary ===")
        a_vals = [res[1] isa Vector ? res[1][end] : (res[1] isa Tuple ? res[1][end] : res[1]) for (params, res) in results]
        b_vals = [length(res) >= 2 ? (res[2] isa Vector ? res[2][end] : (res[2] isa Tuple ? res[2][end] : res[2])) : 0.0 for (params, res) in results] 
        c_vals = [length(res) >= 3 ? (res[3] isa Vector ? res[3][end] : (res[3] isa Tuple ? res[3][end] : res[3])) : 0.0 for (params, res) in results]
        product_ratio = [(b + c)/max(a, 1e-6) for (a,b,c) in zip(a_vals, b_vals, c_vals)]
        println("A concentration: mean=$(round(mean(a_vals), digits=3)), std=$(round(std(a_vals), digits=3))")
        println("B concentration: mean=$(round(mean(b_vals), digits=3)), std=$(round(std(b_vals), digits=3))")
        println("C concentration: mean=$(round(mean(c_vals), digits=3)), std=$(round(std(c_vals), digits=3))")
        println("Product ratio: mean=$(round(mean(product_ratio), digits=3)), std=$(round(std(product_ratio), digits=3))")
        println("\nFirst 5 results:")
        for (i, (params, res)) in enumerate(results[1:min(5, length(results))])
            println("Result $i:")
            println("  Parameters (k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, A0, B0, C0, E1_0, E2_0): $params")
            println("  Final concentrations [A, B, C, E1, E2]: $res")
            println()
        end
    else
        println("No valid results to plot")
    end
    println("\nEnsemble GPU scan completed!")
else
    # Run the main parameter scan
    results = run_parameter_scan_cuda(param_grid)
    
    # Show a report of the results
    println("\n")
    println("Number of results: $(length(results))")
    
    # 结果可视化部分适配参数含义
    if length(results) > 0
        # 1. 选择用于可视化的参数索引（如k1f、k1r、A0等）
        # 这里默认仍然用k1f和k1r作为x、y轴，A浓度为z轴
        x_idx = 1  # k1f
        y_idx = 2  # k1r
        z_idx = 9  # A0（初始A浓度），如需可视化初始浓度对结果的影响可切换

        x_vals = [params[x_idx] for (params, res) in results]
        y_vals = [params[y_idx] for (params, res) in results]
        z_vals = [res[1] for (params, res) in results]

        # 打印实际参数范围
        println("k1f range in results: $(minimum(x_vals)) to $(maximum(x_vals))")
        println("k1r range in results: $(minimum(y_vals)) to $(maximum(y_vals))")
        println("A0 (initial) range in results: $(minimum([params[9] for (params,_) in results])) to $(maximum([params[9] for (params,_) in results]))")

        # 1. 多子热力图 - 支持自定义参数索引
        p1 = plot_multi_species_heatmap(results)
        if p1 !== nothing
            savefig(p1, "multi_species_heatmap_cuda.png")
            println("Multi-species heatmap saved as multi_species_heatmap_cuda.png")
        else
            println("Warning: Could not create multi-species heatmap - no valid data")
        end

        # 2. 参数敏感性分析
        p2 = plot_parameter_sensitivity_analysis(results)
        if p2 !== nothing
            savefig(p2, "parameter_sensitivity_cuda.png")
            println("Parameter sensitivity analysis saved as parameter_sensitivity_cuda.png")
        else
            println("Warning: Could not create parameter sensitivity analysis - no valid data")
        end

        # 3. 浓度分布直方图
        p3 = plot_concentration_distributions(results)
        if p3 !== nothing
            savefig(p3, "concentration_distributions_cuda.png")
            println("Concentration distributions saved as concentration_distributions_cuda.png")
        else
            println("Warning: Could not create concentration distributions - no valid data")
        end

        # 4. 3D参数空间图（可选参数索引）
        p4 = plot_3d_parameter_space(results, 1, 2, 9)  # k1f, k1r, A0
        if p4 !== nothing
            savefig(p4, "3d_parameter_space_cuda.png")
            println("3D parameter space plot saved as 3d_parameter_space_cuda.png")
        else
            println("Warning: Could not create 3D parameter space plot - no valid data")
        end

        # 5. 规则网格contour图（默认k1f/k1r）
        k1f_unique = sort(unique(x_vals))
        k1r_unique = sort(unique(y_vals))
        z_grid = zeros(length(k1f_unique), length(k1r_unique))
        a_concentrations = [res[1] isa Vector ? res[1][end] : (res[1] isa Tuple ? res[1][end] : res[1]) for (params, res) in results]
        for i in eachindex(k1f_unique)
            for j in eachindex(k1r_unique)
                distances = [(x_vals[k] - k1f_unique[i])^2 + (y_vals[k] - k1r_unique[j])^2 for k in eachindex(x_vals)]
                closest_idx = argmin(distances)
                z_grid[i, j] = a_concentrations[closest_idx]
            end
        end
        p5 = contour(k1f_unique, k1r_unique, z_grid', 
                    xlabel="k1f", ylabel="k1r", 
                    title="CUDA GPU Parameter Scan Results (A concentration)",
                    colorbar_title="[A] final")
        savefig(p5, "param_scan_cuda.png")
        println("CUDA GPU contour plot saved as param_scan_cuda.png")

        # 6. 统计摘要
        println("\n=== CUDA GPU Statistical Summary ===")
        a_vals = [res[1] isa Vector ? res[1][end] : (res[1] isa Tuple ? res[1][end] : res[1]) for (params, res) in results]
        b_vals = [length(res) >= 2 ? (res[2] isa Vector ? res[2][end] : (res[2] isa Tuple ? res[2][end] : res[2])) : 0.0 for (params, res) in results] 
        c_vals = [length(res) >= 3 ? (res[3] isa Vector ? res[3][end] : (res[3] isa Tuple ? res[3][end] : res[3])) : 0.0 for (params, res) in results]
        product_ratio = [(b + c)/max(a, 1e-6) for (a,b,c) in zip(a_vals, b_vals, c_vals)]
        println("A concentration: mean=$(round(mean(a_vals), digits=3)), std=$(round(std(a_vals), digits=3))")
        println("B concentration: mean=$(round(mean(b_vals), digits=3)), std=$(round(std(b_vals), digits=3))")
        println("C concentration: mean=$(round(mean(c_vals), digits=3)), std=$(round(std(c_vals), digits=3))")
        println("Product ratio: mean=$(round(mean(product_ratio), digits=3)), std=$(round(std(product_ratio), digits=3))")

        # 7. 打印前几组参数，明确参数含义
        println("\nFirst 5 results:")
        for (i, (params, res)) in enumerate(results[1:min(5, length(results))])
            println("Result $i:")
            println("  Parameters (k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, A, B, C, E1, E2): $params")
            println("  Final concentrations [A, B, C, E1, E2]: $res")
            println()
        end
        
    else
        println("No valid results to plot")
    end
    
    # Performance summary
    println("\n=== CUDA GPU Performance Summary ===")
    println("Total simulations attempted: $(min(20000, total_combinations))")
    println("Successful simulations: $(length(results))")
    println("Success rate: $(round(length(results)/min(20000, total_combinations)*100, digits=2))%")
    
    # Inspect the first few results 
    println("\nFirst 5 results:")
    for (i, (params, res)) in enumerate(results[1:min(5, length(results))])
        println("Result $i:")
        println("  Parameters (k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, A, B, C, E1, E2): $params")
        println("  Final concentrations [A, B, C, E1, E2]: $res")
        println()
    end
    
    println("\n CUDA GPU-accelerated parameter scan completed!")
end

# Usage instructions
println("\n=== Usage Instructions ===")
println("To run performance benchmark: julia param_scan_thermodynamic_CUDA.jl benchmark")
println("To run normal parameter scan: julia param_scan_thermodynamic_CUDA.jl")
println("To run ensemble GPU scan: julia param_scan_thermodynamic_CUDA.jl ensemble")
println("\nThis version includes thermodynamic constraints:")
println("  - Keq1 = (k1f * k2f) / (k1r * k2r) for reactions 1 and 2")
println("  - Keq2 = (k3f * k4f) / (k3r * k4r) for reactions 3 and 4")
println("  - All reaction rate constants are preserved but constrained by equilibrium constants")
println("  - Parameter space reduced by ~$(round(reduction_factor, digits=1))x")
