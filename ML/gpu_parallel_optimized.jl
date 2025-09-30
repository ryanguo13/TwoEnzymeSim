"""
优化的GPU并行求解器

实现真正的多GPU并行处理，解决现有实现中的序列化和CPU回退问题

主要改进：
1. 真正的并行GPU计算（同时使用多个GPU）
2. GPU内存优化和批处理
3. CUDA内核级别的ODE求解
4. 最小化CPU-GPU数据传输
5. 智能负载均衡和错误恢复
"""

using CUDA
using DifferentialEquations
using DiffEqGPU
using Distributed
using Printf
using LinearAlgebra
using Statistics
using ProgressMeter
using StaticArrays

"""
    GPUParallelConfig

GPU并行计算配置
"""
struct GPUParallelConfig
    # GPU设备配置
    use_multi_gpu::Bool
    gpu_batch_size::Int
    max_memory_usage::Float64  # 每个GPU最大内存使用率（0-1）

    # 计算配置
    ode_solver::Symbol  # :Tsit5, :RK4, :Euler
    abstol::Float64
    reltol::Float64
    maxiters::Int

    # 并行配置
    async_processing::Bool  # 异步处理
    overlap_transfers::Bool  # 重叠数据传输和计算

    # 调试配置
    verbose::Bool
    profile_gpu::Bool
end

"""
    default_gpu_config()

创建默认GPU配置
"""
function default_gpu_config()
    return GPUParallelConfig(
        false,                # use_multi_gpu (先用单GPU稳定路径)
        2000,                 # gpu_batch_size
        0.8,                  # max_memory_usage
        :GPUTsit5,            # ode_solver
        1e-6,                 # abstol
        1e-3,                 # reltol
        10000,                # maxiters
        false,                # async_processing
        true,                 # overlap_transfers
        false,                # verbose
        false                 # profile_gpu
    )
end

"""
    OptimizedGPUSolver

优化的GPU并行求解器
"""
mutable struct OptimizedGPUSolver
    config::GPUParallelConfig
    gpu_devices::Vector{Int}
    gpu_streams::Vector{Any}  # CUDA streams for async processing
    memory_pools::Vector{Any}  # GPU memory pools

    # 性能监控
    solve_times::Vector{Float64}
    memory_usage::Vector{Float64}
    throughput::Vector{Float64}

    function OptimizedGPUSolver(config::GPUParallelConfig)
        solver = new()
        solver.config = config
        solver.solve_times = Float64[]
        solver.memory_usage = Float64[]
        solver.throughput = Float64[]

        initialize_gpu_resources!(solver)
        return solver
    end
end
# GPU-friendly parameter struct
struct GPUParams
    k1f::Float64; k1r::Float64; k2f::Float64; k2r::Float64
    k3f::Float64; k3r::Float64; k4f::Float64; k4r::Float64
end

@inline function to_su0(u::AbstractVector)
    return SVector{7,Float64}(
        Float64(u[1]), Float64(u[2]), Float64(u[3]),
        Float64(u[4]), Float64(u[5]), Float64(u[6]), Float64(u[7])
    )
end

@inline function to_pp(p::AbstractVector)
    return GPUParams(Float64(p[1]), Float64(p[2]), Float64(p[3]), Float64(p[4]),
                     Float64(p[5]), Float64(p[6]), Float64(p[7]), Float64(p[8]))
end


"""
    initialize_gpu_resources!(solver::OptimizedGPUSolver)

初始化GPU资源
"""
function initialize_gpu_resources!(solver::OptimizedGPUSolver)
    if !CUDA.functional()
        error("CUDA is not functional")
    end

    n_devices = CUDA.ndevices()
    if n_devices == 0
        error("No CUDA devices found")
    end

    # 选择要使用的GPU设备
    if solver.config.use_multi_gpu && n_devices > 1
        solver.gpu_devices = collect(0:min(n_devices-1, 3))  # 最多使用4个GPU
        println("🚀 初始化多GPU并行：使用 $(length(solver.gpu_devices)) 个GPU")
    else
        solver.gpu_devices = [0]
        println("🚀 初始化单GPU模式")
    end

    # 为每个GPU创建CUDA stream和内存池
    solver.gpu_streams = []
    solver.memory_pools = []

    for gpu_id in solver.gpu_devices
        CUDA.device!(gpu_id)

        # 创建专用stream用于异步计算
        stream = CUDA.stream()
        push!(solver.gpu_streams, stream)

        # 预分配内存池
        device_memory = CUDA.totalmem(CUDA.device())
        allocated_memory = device_memory * solver.config.max_memory_usage

        # 预热GPU和分配内存池
        dummy_array = CUDA.zeros(Float32, 1000, 1000)
        CUDA.unsafe_free!(dummy_array)

        push!(solver.memory_pools, nothing)  # 占位符，实际使用CUDA内存管理

        if solver.config.verbose
            println("  GPU $gpu_id: $(round(device_memory/1e9, digits=2))GB 总内存, " *
                   "$(round(allocated_memory/1e9, digits=2))GB 可用")
        end
    end

    # 切回主GPU
    CUDA.device!(solver.gpu_devices[1])

    println("✅ GPU资源初始化完成")
end

"""
    reaction_ode_gpu!(du, u, p, t)

GPU优化的ODE函数（支持CUDA数组）
"""
function reaction_ode_gpu!(du, u, p, t)
    # 解构参数
    k1f = p.k1f; k1r = p.k1r; k2f = p.k2f; k2r = p.k2r
    k3f = p.k3f; k3r = p.k3r; k4f = p.k4f; k4r = p.k4r

    # 状态变量: A, B, C, E1, E2, AE1, BE2
    A, B, C, E1, E2, AE1, BE2 = u

    # 数值稳定性：对状态做区间裁剪，避免爆炸或负值
    @inbounds begin
        A   = clamp(A,   0.0, 1.0e9)
        B   = clamp(B,   0.0, 1.0e9)
        C   = clamp(C,   0.0, 1.0e9)
        E1  = clamp(E1,  0.0, 1.0e9)
        E2  = clamp(E2,  0.0, 1.0e9)
        AE1 = clamp(AE1, 0.0, 1.0e9)
        BE2 = clamp(BE2, 0.0, 1.0e9)
    end

    # 数值稳定的乘积（避免溢出）
    AE1_f = clamp(A * E1, -1.0e12, 1.0e12)
    BE2_f = clamp(B * E2, -1.0e12, 1.0e12)

    # 反应速率方程（使用安全乘积）
    du[1] = -k1f*AE1_f + k1r*AE1                        # dA/dt
    du[2] = k2f*AE1 - k2r*clamp(B * E1, -1.0e12, 1.0e12) - k3f*BE2_f + k3r*BE2   # dB/dt
    du[3] = k4f*BE2 - k4r*clamp(C * E2, -1.0e12, 1.0e12)                         # dC/dt
    du[4] = -k1f*AE1_f + k1r*AE1 + k2f*AE1 - k2r*clamp(B * E1, -1.0e12, 1.0e12)  # dE1/dt
    du[5] = -k3f*BE2_f + k3r*BE2 + k4f*BE2 - k4r*clamp(C * E2, -1.0e12, 1.0e12)  # dE2/dt
    du[6] = k1f*AE1_f - k1r*AE1 - k2f*AE1 + k2r*clamp(B * E1, -1.0e12, 1.0e12)   # dAE1/dt
    du[7] = k3f*BE2_f - k3r*BE2 - k4f*BE2 + k4r*clamp(C * E2, -1.0e12, 1.0e12)   # dBE2/dt

    # 确保导数有限
    @inbounds for i in 1:7
        val = du[i]
        if !isfinite(val)
            du[i] = 0.0
        else
            du[i] = clamp(val, -1.0e12, 1.0e12)
        end
    end

    return nothing
end

"""
    reaction_ode_gpu_oop(u, p, t) -> SVector{7,Float32}

Out-of-place GPU ODE for kernel execution
"""
function reaction_ode_gpu_oop(u, p, t)
    # parameters
    k1f, k1r, k2f, k2r = p.k1f, p.k1r, p.k2f, p.k2r
    k3f, k3r, k4f, k4r = p.k3f, p.k3r, p.k4f, p.k4r

    # state with non-negativity
    A   = ifelse(u[1] > 0.0, u[1], 0.0)
    B   = ifelse(u[2] > 0.0, u[2], 0.0)
    C   = ifelse(u[3] > 0.0, u[3], 0.0)
    E1  = ifelse(u[4] > 0.0, u[4], 0.0)
    E2  = ifelse(u[5] > 0.0, u[5], 0.0)
    AE1 = ifelse(u[6] > 0.0, u[6], 0.0)
    BE2 = ifelse(u[7] > 0.0, u[7], 0.0)

    du1 = -k1f*A*E1 + k1r*AE1
    du2 =  k2f*AE1 - k2r*B*E1 - k3f*B*E2 + k3r*BE2
    du3 =  k4f*BE2 - k4r*C*E2
    du4 = -k1f*A*E1 + k1r*AE1 + k2f*AE1 - k2r*B*E1
    du5 = -k3f*B*E2 + k3r*BE2 + k4f*BE2 - k4r*C*E2
    du6 =  k1f*A*E1 - k1r*AE1 - k2f*AE1 + k2r*B*E1
    du7 =  k3f*B*E2 - k3r*BE2 - k4f*BE2 + k4r*C*E2

    return SVector{7,Float64}(du1, du2, du3, du4, du5, du6, du7)
end

"""
    solve_batch_gpu_optimized(solver, X_samples, tspan, target_vars)

优化的GPU批处理求解
"""
function solve_batch_gpu_optimized(solver::OptimizedGPUSolver, X_samples::Matrix{Float64},
                                  tspan::Tuple{Float64, Float64}, target_vars::Vector{Symbol})
    n_samples = size(X_samples, 1)
    n_outputs = length(target_vars)
    n_gpus = length(solver.gpu_devices)

    if solver.config.verbose
        println("🚀 开始GPU优化批处理: $(n_samples) 样本, $(n_gpus) GPU")
    end

    start_time = time()
    results = zeros(Float64, n_samples, n_outputs)

    if n_gpus > 1 && solver.config.async_processing
        # 多GPU异步并行处理
        results = solve_multi_gpu_async(solver, X_samples, tspan, target_vars)
    else
        # 单GPU或同步处理
        results = solve_single_gpu_optimized(solver, X_samples, tspan, target_vars)
    end

    # 记录性能指标
    solve_time = time() - start_time
    push!(solver.solve_times, solve_time)
    push!(solver.throughput, n_samples / solve_time)

    if solver.config.verbose
        println("✅ GPU求解完成: $(n_samples) 样本用时 $(round(solve_time, digits=2))s")
        println("   吞吐量: $(round(n_samples/solve_time, digits=1)) 样本/秒")
    end

    return results
end

"""
    solve_multi_gpu_async(solver, X_samples, tspan, target_vars)

真正的多GPU异步并行处理
"""
function solve_multi_gpu_async(solver::OptimizedGPUSolver, X_samples::Matrix{Float64},
                              tspan::Tuple{Float64, Float64}, target_vars::Vector{Symbol})
    n_samples = size(X_samples, 1)
    n_outputs = length(target_vars)
    n_gpus = length(solver.gpu_devices)

    # 智能负载均衡：根据GPU能力分配任务
    gpu_capabilities = get_gpu_capabilities(solver)
    sample_allocation = allocate_samples_smart(n_samples, gpu_capabilities)

    if solver.config.verbose
        println("📊 智能任务分配:")
        for (i, (gpu_id, n_allocated)) in enumerate(zip(solver.gpu_devices, sample_allocation))
            percentage = round(n_allocated/n_samples*100, digits=1)
            println("  GPU $gpu_id: $(n_allocated) 样本 ($(percentage)%)")
        end
    end

    # 创建异步任务
    tasks = []
    results_futures = []

    start_idx = 1
    for (gpu_idx, n_allocated) in enumerate(sample_allocation)
        if n_allocated == 0
            continue
        end

        end_idx = start_idx + n_allocated - 1
        X_chunk = X_samples[start_idx:end_idx, :]
        gpu_id = solver.gpu_devices[gpu_idx]
        stream = solver.gpu_streams[gpu_idx]

        # 创建异步任务
        task = @async solve_gpu_chunk_async(solver, X_chunk, tspan, target_vars, gpu_id, stream)
        push!(tasks, task)
        push!(results_futures, (start_idx, end_idx, task))

        start_idx = end_idx + 1
    end

    # 收集所有异步结果
    final_results = zeros(Float64, n_samples, n_outputs)

    for (start_idx, end_idx, task) in results_futures
        try
            chunk_results = fetch(task)
            final_results[start_idx:end_idx, :] = chunk_results
        catch e
            println("⚠️  GPU任务失败，标记为NaN并继续: $(typeof(e))")
            final_results[start_idx:end_idx, :] .= NaN
        end
    end

    return final_results
end

"""
    solve_gpu_chunk_async(solver, X_chunk, tspan, target_vars, gpu_id, stream)

异步GPU块处理
"""
function solve_gpu_chunk_async(solver::OptimizedGPUSolver, X_chunk::Matrix{Float64},
                              tspan::Tuple{Float64, Float64}, target_vars::Vector{Symbol},
                              gpu_id::Int, stream)
    # 切换到指定GPU
    CUDA.device!(gpu_id)

    n_chunk = size(X_chunk, 1)
    n_outputs = length(target_vars)
    chunk_results = zeros(Float64, n_chunk, n_outputs)

    try
        # 数据传输到GPU（异步）
        X_gpu = CuArray{Float64}(X_chunk)

        # 准备初始条件和参数
        u0_array = prepare_initial_conditions_gpu(X_gpu)
        p_array = prepare_parameters_gpu(X_gpu)

        # 使用EnsembleGPUArray进行真正的GPU并行求解
        prob_func = (prob, i, repeat) -> remake(prob,
            u0 = u0_array[:, i],
            p = p_array[:, i]
        )

        # 基础ODE问题
        u0_base = u0_array[:, 1]
        p_base = p_array[:, 1]
        prob_base = ODEProblem(reaction_ode_gpu!, u0_base, tspan, p_base)

        # 集成问题
        ensemble_prob = EnsembleProblem(prob_base, prob_func=prob_func)

        # GPU并行求解
        sol = solve(ensemble_prob,
            get_gpu_solver(solver.config.ode_solver),
            EnsembleGPUArray(),
            trajectories = n_chunk,
            abstol = solver.config.abstol,
            reltol = solver.config.reltol,
            maxiters = solver.config.maxiters,
            saveat = tspan[2]  # 只保存终点
        )

        # 提取目标变量（在CPU上从每条轨迹的prob.p读取参数）
        chunk_results = extract_and_transfer_results(sol, target_vars)

    catch e
        bt = catch_backtrace()
        println("⚠️  GPU $gpu_id 求解失败，标记为NaN并继续: $(typeof(e))")
        println(sprint(showerror, e, bt))
        chunk_results .= NaN
    end

    return chunk_results
end

"""
    solve_single_gpu_optimized(solver, X_samples, tspan, target_vars)

优化的单GPU处理
"""
function solve_single_gpu_optimized(solver::OptimizedGPUSolver, X_samples::Matrix{Float64},
                                   tspan::Tuple{Float64, Float64}, target_vars::Vector{Symbol})
    n_samples = size(X_samples, 1)
    n_outputs = length(target_vars)
    gpu_id = solver.gpu_devices[1]

    CUDA.device!(gpu_id)

    # 计算最优批大小
    optimal_batch_size = calculate_optimal_batch_size(solver, n_samples, size(X_samples, 2))

    if solver.config.verbose
        println("📊 优化批处理: 批大小 = $optimal_batch_size")
    end

    results = zeros(Float64, n_samples, n_outputs)

    # 分批处理
    @showprogress "GPU处理进度: " for start_idx in 1:optimal_batch_size:n_samples
        end_idx = min(start_idx + optimal_batch_size - 1, n_samples)
        batch_size = end_idx - start_idx + 1

        X_batch = X_samples[start_idx:end_idx, :]

        try
            batch_results = process_gpu_batch(solver, X_batch, tspan, target_vars)
            results[start_idx:end_idx, :] = batch_results
        catch e
            bt = catch_backtrace()
            println("⚠️  批处理失败，标记为NaN并继续: $(typeof(e))")
            println(sprint(showerror, e, bt))
            results[start_idx:end_idx, :] .= NaN
        end

        # 内存清理
        if start_idx % (optimal_batch_size * 5) == 1
            CUDA.reclaim()
        end
    end

    return results
end

"""
    process_gpu_batch(solver, X_batch, tspan, target_vars)

处理单个GPU批次
"""
function process_gpu_batch(solver::OptimizedGPUSolver, X_batch::Matrix{Float64},
                          tspan::Tuple{Float64, Float64}, target_vars::Vector{Symbol})
    n_batch = size(X_batch, 1)
    n_outputs = length(target_vars)

    # 转换为GPU数组
    X_gpu = CuArray{Float64}(X_batch)

    # 准备初始条件和参数（批量处理）
    u0_array = prepare_initial_conditions_gpu(X_gpu)
    p_array = prepare_parameters_gpu(X_gpu)

    # GPU并行ODE求解
    results_gpu = solve_ode_batch_gpu(solver, u0_array, p_array, tspan, n_batch)

    # 提取目标变量并转回CPU
    batch_results = extract_and_transfer_results(results_gpu, target_vars)

    return batch_results
end

"""
    prepare_initial_conditions_gpu(X_gpu::CuArray)

在GPU上准备初始条件
"""
function prepare_initial_conditions_gpu(X_gpu::CuArray)
    n_samples = size(X_gpu, 1)
    u0_array = CuArray{Float64}(undef, 7, n_samples)  # 7个状态变量

    # GPU kernel会更高效，这里简化实现
    u0_array[1, :] = X_gpu[:, 9]   # A
    u0_array[2, :] = X_gpu[:, 10]  # B
    u0_array[3, :] = X_gpu[:, 11]  # C
    u0_array[4, :] = X_gpu[:, 12]  # E1
    u0_array[5, :] = X_gpu[:, 13]  # E2
    u0_array[6, :] .= 0.0          # AE1
    u0_array[7, :] .= 0.0          # BE2

    return u0_array
end

"""
    prepare_parameters_gpu(X_gpu::CuArray)

在GPU上准备参数
"""
function prepare_parameters_gpu(X_gpu::CuArray)
    n_samples = size(X_gpu, 1)
    p_array = CuArray{Float64}(undef, 8, n_samples)  # 8个反应常数

    p_array[1, :] = X_gpu[:, 1]  # k1f
    p_array[2, :] = X_gpu[:, 2]  # k1r
    p_array[3, :] = X_gpu[:, 3]  # k2f
    p_array[4, :] = X_gpu[:, 4]  # k2r
    p_array[5, :] = X_gpu[:, 5]  # k3f
    p_array[6, :] = X_gpu[:, 6]  # k3r
    p_array[7, :] = X_gpu[:, 7]  # k4f
    p_array[8, :] = X_gpu[:, 8]  # k4r

    return p_array
end

"""
    solve_ode_batch_gpu(solver, u0_array, p_array, tspan, n_batch)

GPU批量ODE求解
"""
function solve_ode_batch_gpu(solver::OptimizedGPUSolver, u0_array::CuArray, p_array::CuArray,
                            tspan::Tuple{Float64, Float64}, n_batch::Int)
    # 使用EnsembleGPUArray + in-place Float64 ODE（更稳定）
    u0_mat = Array{Float64}(undef, 7, n_batch)
    p_vec = Vector{GPUParams}(undef, n_batch)
    @inbounds for i in 1:n_batch
        u0_mat[:, i] = Array(u0_array[:, i])
        pi = Array(p_array[:, i])
        p_vec[i] = GPUParams(pi[1], pi[2], pi[3], pi[4], pi[5], pi[6], pi[7], pi[8])
    end

    prob_func = (prob, i, repeat) -> remake(prob, u0=u0_mat[:, i], p=p_vec[i])

    prob_base = ODEProblem(reaction_ode_gpu!, u0_mat[:, 1], tspan, p_vec[1]; isoutofdomain = (u,p,t)->any(!isfinite, u))
    ensemble_prob = EnsembleProblem(prob_base, prob_func=prob_func)

    sol = solve(ensemble_prob,
        get_gpu_solver(solver.config.ode_solver),
        EnsembleGPUArray(CUDA.CUDABackend());
        trajectories = n_batch,
        abstol = min(solver.config.abstol, 1e-8),
        reltol = min(solver.config.reltol, 1e-6),
        maxiters = solver.config.maxiters,
        saveat = [tspan[2]],
        dt = (tspan[2]-tspan[1]) / 10000.0,
        dtmax = (tspan[2]-tspan[1]) / 100.0,
        adaptive = false
    )

    # 调试：打印部分轨迹retcode与终态是否有限
    try
        total = length(sol)
        ok = 0
        limit = min(10, total)
        println("🔎 GPU调试: 采样 $limit/$total 条轨迹")
        for i in 1:limit
            r = sol[i].retcode
            print("  traj ", i, " retcode=", r)
            if r == :Success && length(sol[i].u) > 0
                uend = sol[i].u[end]
                finite = all(isfinite, uend)
                println(", uend_finite=", finite)
                if finite
                    ok += 1
                end
            else
                println()
            end
        end
        println("✅ 有限终态样本: ", ok, "/", limit)
        if limit > 0
            try
                uend = sol[1].u[end]
                println("🔬 示例终态[1]: ", uend)
            catch e
                println("⚠️  无法打印示例终态: ", typeof(e))
            end
        end
    catch e
        println("⚠️  调试信息打印失败: ", typeof(e))
    end

    return sol
end

"""
    extract_and_transfer_results(sol, target_vars, X_gpu)

提取结果并传输回CPU
"""
function extract_and_transfer_results(sol, target_vars::Vector{Symbol})
    n_batch = length(sol)
    n_outputs = length(target_vars)

    results = zeros(Float64, n_batch, n_outputs)

    # 从解中提取目标变量
    for i in 1:n_batch
        sol_i = sol[i]
        if sol_i.retcode == :Success && length(sol_i.u) > 0
            final_state = Array(sol_i.u[end])  # 转回CPU进行提取

            # 计算目标变量（参数从prob.p读取，兼容isbits）
            for (j, var) in enumerate(target_vars)
                results[i, j] = extract_single_target_variable(final_state, var, sol_i)
            end
        else
            # 求解失败，填入NaN
            results[i, :] .= NaN32
        end
    end

    return results
end

"""
    extract_single_target_variable(final_state, var, sol, params)

提取单个目标变量
"""
function extract_single_target_variable(final_state::Vector{Float64}, var::Symbol, sol)
    # final_state: [A, B, C, E1, E2, AE1, BE2]
    if var == :A_final
        return final_state[1]
    elseif var == :B_final
        return final_state[2]
    elseif var == :C_final
        return final_state[3]
    elseif var == :v1_mean
        # 计算平均反应速率 v1 = k1f*A*E1 - k1r*AE1
        p = sol.prob.p
        k1f = Base.hasproperty(p, :k1f) ? p.k1f : p[1]
        k1r = Base.hasproperty(p, :k1r) ? p.k1r : p[2]
        A_mean = mean([sol.u[i][1] for i in 1:length(sol.u)])
        E1_mean = mean([sol.u[i][4] for i in 1:length(sol.u)])
        AE1_mean = mean([sol.u[i][6] for i in 1:length(sol.u)])
        return k1f * A_mean * E1_mean - k1r * AE1_mean
    elseif var == :v2_mean
        # 计算平均反应速率 v2 = k3f*B*E2 - k3r*BE2
        p = sol.prob.p
        k3f = Base.hasproperty(p, :k3f) ? p.k3f : p[5]
        k3r = Base.hasproperty(p, :k3r) ? p.k3r : p[6]
        B_mean = mean([sol.u[i][2] for i in 1:length(sol.u)])
        E2_mean = mean([sol.u[i][5] for i in 1:length(sol.u)])
        BE2_mean = mean([sol.u[i][7] for i in 1:length(sol.u)])
        return k3f * B_mean * E2_mean - k3r * BE2_mean
    else
        return NaN
    end
end

"""
    get_gpu_capabilities(solver)

获取GPU计算能力
"""
function get_gpu_capabilities(solver::OptimizedGPUSolver)
    capabilities = Float64[]

    for gpu_id in solver.gpu_devices
        CUDA.device!(gpu_id)
        device = CUDA.device()

        # 基于内存和计算能力评分
        memory_gb = CUDA.totalmem(device) / 1e9
        compute_capability = CUDA.capability(device)

        # 简单的能力评分
        score = memory_gb * (compute_capability.major * 10 + compute_capability.minor)
        push!(capabilities, score)
    end

    return capabilities
end

"""
    allocate_samples_smart(n_samples, capabilities)

智能样本分配
"""
function allocate_samples_smart(n_samples::Int, capabilities::Vector{Float64})
    total_capability = sum(capabilities)
    allocation = Int[]

    remaining_samples = n_samples
    for i in 1:length(capabilities)-1
        proportion = capabilities[i] / total_capability
        allocated = round(Int, n_samples * proportion)
        push!(allocation, allocated)
        remaining_samples -= allocated
    end

    # 剩余样本分配给最后一个GPU
    push!(allocation, remaining_samples)

    return allocation
end

"""
    calculate_optimal_batch_size(solver, n_samples, n_features)

计算最优批大小
"""
function calculate_optimal_batch_size(solver::OptimizedGPUSolver, n_samples::Int, n_features::Int)
    gpu_id = solver.gpu_devices[1]
    CUDA.device!(gpu_id)

    available_memory = CUDA.totalmem(CUDA.device()) * solver.config.max_memory_usage

    # 估算每个样本的内存需求
    memory_per_sample = n_features * 8 * 10  # Float64, 约10倍放大系数

    theoretical_batch_size = floor(Int, available_memory / memory_per_sample)

    # 限制在合理范围内
    optimal_batch_size = min(
        theoretical_batch_size,
        solver.config.gpu_batch_size,
        n_samples
    )

    return max(optimal_batch_size, 1)
end

"""
    get_gpu_solver(solver_type::Symbol)

获取GPU求解器 - 使用Tsit5()配合EnsembleGPUArray(0)
"""
function get_gpu_solver(solver_type::Symbol)
    # Use Rosenbrock23 with finite-difference Jacobian on GPU (Float64)
    return Rosenbrock23(autodiff = AutoFiniteDiff())
end

"""
    solve_cpu_fallback(X_batch, tspan, target_vars)

CPU回退求解
"""
# 已移除CPU回退：纯GPU执行，失败样本将标记为NaN

"""
    cleanup_gpu_resources!(solver::OptimizedGPUSolver)

清理GPU资源
"""
function cleanup_gpu_resources!(solver::OptimizedGPUSolver)
    println("🧹 清理GPU资源...")

    for gpu_id in solver.gpu_devices
        CUDA.device!(gpu_id)
        CUDA.reclaim()
    end

    println("✅ GPU资源清理完成")
end

"""
    benchmark_gpu_solver(solver, test_samples, tspan, target_vars)

GPU求解器性能测试
"""
function benchmark_gpu_solver(solver::OptimizedGPUSolver, test_samples::Matrix{Float64},
                             tspan::Tuple{Float64, Float64}, target_vars::Vector{Symbol})
    println("🏃 GPU性能测试...")

    # 预热
    warmup_samples = test_samples[1:min(100, size(test_samples, 1)), :]
    solve_batch_gpu_optimized(solver, warmup_samples, tspan, target_vars)

    # 正式测试
    start_time = time()
    results = solve_batch_gpu_optimized(solver, test_samples, tspan, target_vars)
    end_time = time()

    solve_time = end_time - start_time
    throughput = size(test_samples, 1) / solve_time

    println("📊 性能测试结果:")
    println("  样本数量: $(size(test_samples, 1))")
    println("  求解时间: $(round(solve_time, digits=3))s")
    println("  吞吐量: $(round(throughput, digits=1)) 样本/秒")

    # 计算GPU利用率
    if solver.config.profile_gpu
        # 这里可以添加GPU profiling代码
        println("  GPU利用率分析需要专业profiling工具")
    end

    return results, solve_time, throughput
end


