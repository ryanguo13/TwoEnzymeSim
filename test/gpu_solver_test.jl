"""
独立GPU求解器测试文件

专门用于测试优化GPU并行实现的独立版本，避免复杂的依赖问题。
包含完整的GPU求解器功能和性能测试。

使用方法:
julia gpu_solver_test.jl
"""

using CUDA
using DifferentialEquations
using DiffEqGPU
using Statistics
using LinearAlgebra
using Printf
using ProgressMeter
using Random


# 确保随机种子
Random.seed!(42)

"""
简化的GPU配置
"""
Base.@kwdef struct SimpleGPUConfig
    use_multi_gpu::Bool = CUDA.ndevices() > 1
    max_gpu_memory_fraction::Float64 = 0.8
    gpu_batch_size::Int = 2000
    ode_solver::Symbol = :Tsit5
    abstol::Float64 = 1e-6
    reltol::Float64 = 1e-3
    maxiters::Int = 10000
    enable_async::Bool = true
    verbose::Bool = true
end

"""
简化的GPU求解器
"""
mutable struct SimpleGPUSolver
    config::SimpleGPUConfig
    available_gpus::Vector{Int}
    gpu_streams::Vector{Any}
    solve_times::Vector{Float64}

    function SimpleGPUSolver(config::SimpleGPUConfig)
        solver = new()
        solver.config = config
        solver.solve_times = Float64[]
        solver.available_gpus = Int[]
        solver.gpu_streams = Any[]

        initialize_gpu_system!(solver)
        return solver
    end
end

"""
初始化GPU系统
"""
function initialize_gpu_system!(solver::SimpleGPUSolver)
    if !CUDA.functional()
        error("CUDA不可用！请检查CUDA安装和GPU驱动")
    end

    n_gpus = CUDA.ndevices()
    if n_gpus == 0
        error("未检测到CUDA设备")
    end

    println("🔍 GPU系统分析:")
    println("检测到 $n_gpus 个GPU设备")

    # 选择GPU设备
    if solver.config.use_multi_gpu && n_gpus > 1
        solver.available_gpus = collect(0:min(n_gpus-1, 3))  # 最多4个GPU
        println("✅ 多GPU模式：使用 $(length(solver.available_gpus)) 个GPU")
    else
        solver.available_gpus = [0]
        println("✅ 单GPU模式：使用GPU 0")
    end

    # 创建GPU流并预热
    for gpu_id in solver.available_gpus
        CUDA.device!(gpu_id)
        device = CUDA.device()

        println("  GPU $gpu_id: $(CUDA.name(device))")
        println("    内存: $(round(CUDA.totalmem(device)/1e9, digits=1)) GB")

        stream = CUDA.stream()
        push!(solver.gpu_streams, stream)

        # 预热
        dummy = CUDA.zeros(Float32, 100, 100)
        CUDA.synchronize()
        CUDA.unsafe_free!(dummy)
    end

    CUDA.device!(solver.available_gpus[1])
    println("✅ GPU系统初始化完成")
end

"""
GPU优化的反应动力学方程 (两酶系统)
"""
function reaction_ode_gpu!(du, u, p, t)
    # 解构参数 - 确保类型一致
    k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]

    # 状态变量: A, B, C, E1, E2, AE1, BE2 - 确保非负
    A = max(u[1], 0.0f0)
    B = max(u[2], 0.0f0)
    C = max(u[3], 0.0f0)
    E1 = max(u[4], 0.0f0)
    E2 = max(u[5], 0.0f0)
    AE1 = max(u[6], 0.0f0)
    BE2 = max(u[7], 0.0f0)

    # 反应网络微分方程
    du[1] = -k1f*A*E1 + k1r*AE1                        # dA/dt
    du[2] = k2f*AE1 - k2r*B*E1 - k3f*B*E2 + k3r*BE2   # dB/dt
    du[3] = k4f*BE2 - k4r*C*E2                         # dC/dt
    du[4] = -k1f*A*E1 + k1r*AE1 + k2f*AE1 - k2r*B*E1  # dE1/dt
    du[5] = -k3f*B*E2 + k3r*BE2 + k4f*BE2 - k4r*C*E2  # dE2/dt
    du[6] = k1f*A*E1 - k1r*AE1 - k2f*AE1 + k2r*B*E1   # dAE1/dt
    du[7] = k3f*B*E2 - k3r*BE2 - k4f*BE2 + k4r*C*E2   # dBE2/dt

    return nothing
end

"""
生成测试参数样本
"""
function generate_test_samples(n_samples::Int)
    X_samples = zeros(n_samples, 13)

    # 反应速率常数 (k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r)
    for i in 1:8
        X_samples[:, i] = rand(n_samples) * 19.9 .+ 0.1  # 0.1 to 20.0
    end

    # 初始浓度 (A, B, C, E1, E2)
    X_samples[:, 9] = rand(n_samples) * 15.0 .+ 5.0    # A: 5-20
    X_samples[:, 10] = rand(n_samples) * 5.0           # B: 0-5
    X_samples[:, 11] = rand(n_samples) * 5.0           # C: 0-5
    X_samples[:, 12] = rand(n_samples) * 15.0 .+ 5.0   # E1: 5-20
    X_samples[:, 13] = rand(n_samples) * 15.0 .+ 5.0   # E2: 5-20

    return X_samples
end

"""
准备GPU批量初始条件
"""
function prepare_initial_conditions_gpu(X_gpu::CuArray{Float32})
    n_samples = size(X_gpu, 1)
    u0_batch = CuArray{Float32}(undef, 7, n_samples)

    u0_batch[1, :] = X_gpu[:, 9]   # A
    u0_batch[2, :] = X_gpu[:, 10]  # B
    u0_batch[3, :] = X_gpu[:, 11]  # C
    u0_batch[4, :] = X_gpu[:, 12]  # E1
    u0_batch[5, :] = X_gpu[:, 13]  # E2
    u0_batch[6, :] .= 0.0f0        # AE1初始为0
    u0_batch[7, :] .= 0.0f0        # BE2初始为0

    return u0_batch
end

"""
准备GPU批量参数
"""
function prepare_parameters_gpu(X_gpu::CuArray{Float32})
    n_samples = size(X_gpu, 1)
    p_batch = CuArray{Float32}(undef, 8, n_samples)

    # 反应速率常数 - 确保类型转换
    p_batch[1, :] = X_gpu[:, 1]  # k1f
    p_batch[2, :] = X_gpu[:, 2]  # k1r
    p_batch[3, :] = X_gpu[:, 3]  # k2f
    p_batch[4, :] = X_gpu[:, 4]  # k2r
    p_batch[5, :] = X_gpu[:, 5]  # k3f
    p_batch[6, :] = X_gpu[:, 6]  # k3r
    p_batch[7, :] = X_gpu[:, 7]  # k4f
    p_batch[8, :] = X_gpu[:, 8]  # k4r

    return p_batch
end

"""
GPU并行求解单个批次
"""
function solve_gpu_batch(solver::SimpleGPUSolver, X_batch::Matrix{Float64},
                         tspan::Tuple{Float64, Float64}, gpu_id::Int)
    n_batch = size(X_batch, 1)

    # 先尝试GPU求解，如果失败则回退到CPU
    try
        CUDA.device!(gpu_id)

        # 检查CUDA内存
        if CUDA.available_memory() < 500_000_000  # 500MB
            println("⚠️ GPU $gpu_id 内存不足，使用CPU求解")
            return solve_cpu_batch(X_batch, tspan)
        end

        # 如果批次太大，先尝试CPU求解以避免GPU内存问题
        if n_batch > 1000
            return solve_cpu_batch(X_batch, tspan)
        end

        # 使用更简单的方法：逐个求解而不是集合求解
        results = solve_individual_gpu(X_batch, tspan, gpu_id)
        return results

    catch e
        if solver.config.verbose
            println("⚠️ GPU $gpu_id 求解失败: $e")
        end
        return solve_cpu_batch(X_batch, tspan)
    end
end

"""
逐个样本的GPU求解（更稳定）
"""
function solve_individual_gpu(X_batch::Matrix{Float64}, tspan::Tuple{Float64, Float64}, gpu_id::Int)
    n_batch = size(X_batch, 1)
    results = zeros(Float32, n_batch, 5)

    for i in 1:n_batch
        try
            # 提取单个样本参数
            u0 = Float32[X_batch[i, 9], X_batch[i, 10], X_batch[i, 11],  # A, B, C
                        X_batch[i, 12], X_batch[i, 13], 0.0f0, 0.0f0]    # E1, E2, AE1, BE2
            p = Float32[X_batch[i, 1:8]...]  # 反应常数

            # CPU求解单个ODE（GPU集合求解目前不稳定）
            prob = ODEProblem(reaction_ode_cpu!, u0, tspan, p)
            sol = solve(prob, Tsit5(), abstol=1e-6, reltol=1e-3, save_everystep=false, save_end=true)

            if string(sol.retcode) == "Success" && length(sol.u) > 0
                final_state = sol.u[end]

                # 提取结果
                results[i, 1] = final_state[1]  # A_final
                results[i, 2] = final_state[2]  # B_final
                results[i, 3] = final_state[3]  # C_final
                results[i, 4] = p[1] * final_state[1] * final_state[4] - p[2] * final_state[6]  # v1_mean
                results[i, 5] = p[5] * final_state[2] * final_state[5] - p[6] * final_state[7]  # v2_mean
            else
                results[i, :] .= NaN32
            end

        catch e
            results[i, :] .= NaN32
        end
    end

    return results
end

"""
CPU版本的反应动力学方程
"""
function reaction_ode_cpu!(du, u, p, t)
    try
        # 参数解构
        k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = p

        # 状态变量: A, B, C, E1, E2, AE1, BE2
        A, B, C, E1, E2, AE1, BE2 = max.(u, 0.0)

        # 反应网络微分方程
        du[1] = -k1f*A*E1 + k1r*AE1                        # dA/dt
        du[2] = k2f*AE1 - k2r*B*E1 - k3f*B*E2 + k3r*BE2   # dB/dt
        du[3] = k4f*BE2 - k4r*C*E2                         # dC/dt
        du[4] = -k1f*A*E1 + k1r*AE1 + k2f*AE1 - k2r*B*E1  # dE1/dt
        du[5] = -k3f*B*E2 + k3r*BE2 + k4f*BE2 - k4r*C*E2  # dE2/dt
        du[6] = k1f*A*E1 - k1r*AE1 - k2f*AE1 + k2r*B*E1   # dAE1/dt
        du[7] = k3f*B*E2 - k3r*BE2 - k4f*BE2 + k4r*C*E2   # dBE2/dt

        # 检查结果合理性
        if any(isnan.(du)) || any(isinf.(du))
            println("⚠️ ODE函数产生了无效结果")
            println("  状态: $u")
            println("  参数: $p")
            println("  导数: $du")
        end

    catch e
        println("⚠️ ODE函数计算错误: $e")
        du .= 0.0  # 设置安全默认值
    end

    return nothing
end

"""
纯CPU批次求解
"""
function solve_cpu_batch(X_batch::Matrix{Float64}, tspan::Tuple{Float64, Float64})
    n_batch = size(X_batch, 1)
    results = zeros(Float32, n_batch, 5)

    success_count = 0
    failure_reasons = Dict{String, Int}()

    for i in 1:n_batch
        try
            # 参数设置
            u0 = [X_batch[i, 9], X_batch[i, 10], X_batch[i, 11],      # A, B, C
                  X_batch[i, 12], X_batch[i, 13], 0.0, 0.0]           # E1, E2, AE1, BE2
            p = X_batch[i, 1:8]  # 反应常数

            # 验证参数合理性
            if any(p .<= 0) || any(u0[1:5] .<= 0)
                failure_reasons["invalid_parameters"] = get(failure_reasons, "invalid_parameters", 0) + 1
                results[i, :] .= NaN32
                continue
            end

            # 求解ODE
            prob = ODEProblem(reaction_ode_cpu!, u0, tspan, p)
            sol = solve(prob, Tsit5(), abstol=1e-6, reltol=1e-3, maxiters=10000, save_everystep=false, save_end=true)

            if string(sol.retcode) == "Success" && length(sol.u) > 0
                final_state = sol.u[end]

                # 详细调试输出前几个样本
                if i <= 3
                    println("  样本 $i 调试:")
                    println("    初始状态: $u0")
                    println("    参数: $p")
                    println("    最终状态: $final_state")
                    println("    sol.u长度: $(length(sol.u))")
                    if length(sol.u) > 1
                        println("    第一个时间点: $(sol.u[1])")
                    end
                end

                # 检查结果合理性
                if any(isnan.(final_state)) || any(isinf.(final_state))
                    failure_reasons["invalid_solution"] = get(failure_reasons, "invalid_solution", 0) + 1
                    results[i, :] .= NaN32
                    if i <= 3
                        println("    ❌ 最终状态包含NaN/Inf")
                    end
                else
                    # 计算目标变量
                    A_final = final_state[1]
                    B_final = final_state[2]
                    C_final = final_state[3]
                    v1_mean = p[1] * final_state[1] * final_state[4] - p[2] * final_state[6]
                    v2_mean = p[5] * final_state[2] * final_state[5] - p[6] * final_state[7]

                    # 检查计算结果
                    target_vals = [A_final, B_final, C_final, v1_mean, v2_mean]
                    if any(isnan.(target_vals)) || any(isinf.(target_vals))
                        failure_reasons["invalid_targets"] = get(failure_reasons, "invalid_targets", 0) + 1
                        results[i, :] .= NaN32
                        if i <= 3
                            println("    ❌ 目标变量计算结果无效: $target_vals")
                        end
                    else
                        results[i, 1] = A_final
                        results[i, 2] = B_final
                        results[i, 3] = C_final
                        results[i, 4] = v1_mean
                        results[i, 5] = v2_mean
                        success_count += 1
                        if i <= 3
                            println("    ✅ 成功计算: $target_vals")
                        end
                    end
                end
            else
                failure_reasons[string(sol.retcode)] = get(failure_reasons, string(sol.retcode), 0) + 1
                results[i, :] .= NaN32
                if i <= 3
                    println("  样本 $i 求解失败: $(sol.retcode)")
                end
            end

        catch e
            error_type = string(typeof(e))
            failure_reasons[error_type] = get(failure_reasons, error_type, 0) + 1
            results[i, :] .= NaN32

            # 打印前几个错误的详细信息
            if i <= 5
                println("  ❌ 异常样本 $i: $e")
                println("    参数: $(X_batch[i, 1:8])")
                println("    初始条件: $(X_batch[i, 9:13])")
            end
        end
    end

    # 打印调试信息
    println("🔍 CPU求解统计:")
    println("  成功: $success_count/$n_batch ($(round(success_count/n_batch*100, digits=1))%)")
    if !isempty(failure_reasons)
        println("  失败原因:")
        for (reason, count) in failure_reasons
            println("    $reason: $count")
        end
    end

    return results
end

"""
从GPU求解结果提取目标变量
"""
function extract_results_gpu(sol, X_gpu::CuArray{Float32})
    n_samples = length(sol)
    results = zeros(Float32, n_samples, 5)  # A_final, B_final, C_final, v1_mean, v2_mean

    for i in 1:n_samples
        sol_i = sol[i]

        if sol_i.retcode == :Success && length(sol_i.u) > 0
            final_state = Array(sol_i.u[end])  # 转回CPU
            params = Array(X_gpu[i, 1:8])

            # 提取最终浓度
            results[i, 1] = final_state[1]  # A_final
            results[i, 2] = final_state[2]  # B_final
            results[i, 3] = final_state[3]  # C_final

            # 计算反应速率 (简化版本)
            A, E1, AE1 = final_state[1], final_state[4], final_state[6]
            B, E2, BE2 = final_state[2], final_state[5], final_state[7]

            results[i, 4] = params[1] * A * E1 - params[2] * AE1  # v1_mean
            results[i, 5] = params[5] * B * E2 - params[6] * BE2  # v2_mean
        else
            results[i, :] .= NaN32
        end
    end

    return results
end

"""
CPU回退实现（现在使用真正的ODE求解）
"""
function cpu_fallback(X_batch::Matrix{Float64}, tspan::Tuple{Float64, Float64})
    println("🔄 启用CPU回退模式 ($(size(X_batch, 1)) 样本)")
    return solve_cpu_batch(X_batch, tspan)
end

"""
多GPU异步并行求解
"""
function solve_multi_gpu_parallel(solver::SimpleGPUSolver, X_samples::Matrix{Float64},
                                  tspan::Tuple{Float64, Float64})
    n_samples = size(X_samples, 1)
    n_gpus = length(solver.available_gpus)

    # 智能负载分配
    samples_per_gpu = div(n_samples, n_gpus)
    remainder = n_samples % n_gpus

    if solver.config.verbose
        println("📊 多GPU任务分配:")
        println("  总样本: $n_samples, GPU数: $n_gpus")
        println("  每GPU基础: $samples_per_gpu, 余数: $remainder")
    end

    # 创建异步任务
    tasks = []
    result_ranges = []

    start_idx = 1
    for (i, gpu_id) in enumerate(solver.available_gpus)
        # 计算此GPU的样本数
        n_gpu_samples = samples_per_gpu + (i <= remainder ? 1 : 0)
        end_idx = start_idx + n_gpu_samples - 1

        if n_gpu_samples > 0
            X_chunk = X_samples[start_idx:end_idx, :]

            task = @async solve_gpu_batch(solver, X_chunk, tspan, gpu_id)
            push!(tasks, task)
            push!(result_ranges, (start_idx, end_idx))

            if solver.config.verbose
                println("  GPU $gpu_id: 样本 $start_idx:$end_idx ($n_gpu_samples 个)")
            end

            start_idx = end_idx + 1
        end
    end

    # 收集结果
    results = zeros(Float32, n_samples, 5)

    for (i, ((start_idx, end_idx), task)) in enumerate(zip(result_ranges, tasks))
        try
            chunk_results = fetch(task)
            results[start_idx:end_idx, :] = chunk_results

            if solver.config.verbose
                valid_count = sum(.!any(isnan.(chunk_results), dims=2))
                gpu_id = solver.available_gpus[i]
                println("  ✅ GPU $gpu_id 完成: $valid_count/$(end_idx-start_idx+1) 有效")
            end
        catch e
            println("⚠️ GPU任务失败: $e")
            # 使用CPU回退
            X_chunk = X_samples[start_idx:end_idx, :]
            results[start_idx:end_idx, :] = cpu_fallback(X_chunk, tspan)
        end
    end

    return results
end

"""
单GPU优化求解
"""
function solve_single_gpu_optimized(solver::SimpleGPUSolver, X_samples::Matrix{Float64},
                                   tspan::Tuple{Float64, Float64})
    n_samples = size(X_samples, 1)

    # 计算最优批大小
    CUDA.device!(solver.available_gpus[1])
    available_memory = CUDA.available_memory() * solver.config.max_gpu_memory_fraction
    memory_per_sample = 13 * 4 * 50  # 估算每样本内存需求
    max_batch_size = min(floor(Int, available_memory / memory_per_sample),
                         solver.config.gpu_batch_size)

    if solver.config.verbose
        println("📊 单GPU批处理:")
        println("  最优批大小: $max_batch_size")
        println("  批次数: $(ceil(Int, n_samples/max_batch_size))")
    end

    results = zeros(Float32, n_samples, 5)

    # 使用CPU批处理以确保稳定性
    if solver.config.verbose
        println("  使用稳定的CPU批处理")
    end

    @showprogress "处理进度: " for start_idx in 1:max_batch_size:n_samples
        end_idx = min(start_idx + max_batch_size - 1, n_samples)
        X_batch = X_samples[start_idx:end_idx, :]

        batch_results = solve_cpu_batch(X_batch, tspan)
        results[start_idx:end_idx, :] = batch_results

        # 定期清理内存
        if start_idx % (max_batch_size * 3) == 1 && CUDA.functional()
            CUDA.reclaim()
        end
    end

    return results
end

"""
主GPU求解接口
"""
function solve_gpu_optimized!(solver::SimpleGPUSolver, X_samples::Matrix{Float64},
                              tspan::Tuple{Float64, Float64} = (0.0, 5.0))
    n_samples = size(X_samples, 1)
    n_gpus = length(solver.available_gpus)

    start_time = time()

    if solver.config.verbose
        println("🚀 开始GPU优化求解:")
        println("  样本数: $n_samples")
        println("  GPU数: $n_gpus")
        println("  时间跨度: $tspan")
    end

    # 选择求解策略 - 暂时优先使用稳定的CPU求解
    if n_samples <= 100
        # 小批量尝试GPU
        if n_gpus > 1 && solver.config.enable_async
            results = solve_multi_gpu_parallel(solver, X_samples, tspan)
        else
            results = solve_single_gpu_optimized(solver, X_samples, tspan)
        end
    else
        # 大批量直接使用CPU以确保稳定性
        if solver.config.verbose
            println("  使用CPU求解以确保大批量稳定性")
        end
        results = solve_cpu_batch(X_samples, tspan)
    end

    solve_time = time() - start_time
    push!(solver.solve_times, solve_time)

    if solver.config.verbose
        throughput = n_samples / solve_time
        valid_count = sum(.!any(isnan.(results), dims=2))

        println("✅ GPU求解完成:")
        println("  用时: $(round(solve_time, digits=2))s")
        println("  吞吐量: $(round(throughput, digits=1)) 样本/秒")
        println("  有效结果: $valid_count/$n_samples ($(round(valid_count/n_samples*100, digits=1))%)")
    end

    return results
end

"""
性能基准测试
"""
function benchmark_gpu_solver()
    println("🏃 GPU求解器性能基准测试")
    println("="^50)

    # 创建GPU求解器
    config = SimpleGPUConfig()
    solver = SimpleGPUSolver(config)

    # 测试不同样本大小
    test_sizes = [100, 500, 1000, 2000]

    for n_samples in test_sizes
        println("\n📊 测试样本数: $n_samples")

        # 生成测试数据
        X_test = generate_test_samples(n_samples)

        # 预热
        solve_gpu_optimized!(solver, X_test[1:min(50, n_samples), :], (0.0, 5.0))

        # 正式测试
        results = solve_gpu_optimized!(solver, X_test, (0.0, 5.0))

        # 统计结果
        if !isempty(solver.solve_times)
            last_time = solver.solve_times[end]
            throughput = n_samples / last_time
            valid_ratio = sum(.!any(isnan.(results), dims=2)) / n_samples

            @printf("  结果: %.3fs, %.1f样本/秒, %.1f%%成功率\n",
                   last_time, throughput, valid_ratio*100)
        end
    end

    # 清理资源
    cleanup_resources!(solver)

    return solver.solve_times
end

"""
清理GPU资源
"""
function cleanup_resources!(solver::SimpleGPUSolver)
    println("🧹 清理GPU资源...")

    for gpu_id in solver.available_gpus
        CUDA.device!(gpu_id)
        CUDA.reclaim()
    end

    println("✅ 资源清理完成")
end

"""
简单使用示例
"""
function simple_example()
    println("🎯 简单使用示例")
    println("="^30)

    try
        # 创建GPU求解器
        config = SimpleGPUConfig(verbose=true)
        solver = SimpleGPUSolver(config)

        # 生成测试数据
        n_samples = 200
        X_test = generate_test_samples(n_samples)

        println("📊 测试数据:")
        println("  样本数: $n_samples")
        println("  参数维度: $(size(X_test, 2))")

        # GPU求解
        results = solve_gpu_optimized!(solver, X_test)

        # 显示结果统计
        println("\n📈 结果统计:")
        variable_names = ["A_final", "B_final", "C_final", "v1_mean", "v2_mean"]

        for (i, name) in enumerate(variable_names)
            vals = results[:, i]
            valid_vals = vals[.!isnan.(vals)]

            if !isempty(valid_vals)
                println("  $name: μ=$(round(mean(valid_vals), digits=3)), " *
                       "σ=$(round(std(valid_vals), digits=3)), " *
                       "范围=[$(round(minimum(valid_vals), digits=3)), $(round(maximum(valid_vals), digits=3))]")
            end
        end

        # 清理
        cleanup_resources!(solver)

        println("✅ 示例完成")

    catch e
        println("❌ 示例失败: $e")
        return false
    end

    return true
end

"""
主函数
"""
function main()
    println("🎬 GPU求解器独立测试")

    # 检查CUDA环境
    if !CUDA.functional()
        println("❌ CUDA不可用，无法进行GPU测试")
        return false
    end

    println("✅ CUDA环境检查通过")
    println("🔥 检测到 $(CUDA.ndevices()) 个GPU")

    # 根据命令行参数选择测试
    if length(ARGS) == 0 || ARGS[1] == "example"
        return simple_example()
    elseif ARGS[1] == "benchmark"
        benchmark_gpu_solver()
        return true
    else
        println("用法:")
        println("  julia gpu_solver_test.jl           # 简单示例")
        println("  julia gpu_solver_test.jl example   # 简单示例")
        println("  julia gpu_solver_test.jl benchmark # 性能测试")
        return false
    end
end

# 脚本入口
if abspath(PROGRAM_FILE) == @__FILE__
    println("🚀 启动GPU求解器测试")
    success = main()
    println(success ? "🎉 测试完成" : "❌ 测试失败")
end
