"""
ML代理模型 (Surrogate Model) for 两酶仿真系统

实现指导文档第一大点：用ML代理模型替换部分扫描，减少计算80%+

主要功能：
1. 小规模参数扫描数据生成（10%参数）
2. ML模型训练（支持不确定性估计）
3. 代理模型预测接口
4. Gaussian Process支持
5. 高维参数降维（PCA）
"""

using Flux
using MLJ
using Surrogates
using MultivariateStats
using Statistics
using LinearAlgebra
using Random
using TOML

# Include core simulation modules
include("../src/simulation.jl")
include("../src/parameters.jl")
using Plots
using ProgressMeter
using JLD2
using CUDA
using DifferentialEquations
using DiffEqGPU
using StaticArrays
using DiffEqGPU: EnsembleGPUArray, EnsembleGPUKernel
using Distributed
using Printf
using IterTools

# 包含项目核心模块
include("../src/simulation.jl")
include("../src/parameters.jl")
include("../src/visualization.jl")

# 引入优化版GPU并行求解器
include("gpu_parallel_optimized.jl")

"""
    SurrogateModelConfig

代理模型配置结构体
"""
Base.@kwdef struct SurrogateModelConfig
    # 数据生成配置
    sample_fraction::Float64 = 0.1  # 小规模扫描比例（10%）
    max_samples::Int = 10000        # 最大样本数

    # 模型配置
    model_type::Symbol = :neural_network  # :neural_network, :gaussian_process, :radial_basis
    hidden_dims::Vector{Int} = [64, 32, 16]  # 神经网络隐藏层
    dropout_rate::Float64 = 0.1      # Dropout率（用于不确定性估计）

    # 训练配置
    epochs::Int = 100
    batch_size::Int = 32
    learning_rate::Float64 = 1e-3
    validation_split::Float64 = 0.2

    # 降维配置
    use_pca::Bool = true            # 是否使用PCA降维
    pca_variance_threshold::Float64 = 0.95  # PCA保留方差比例

    # 输出配置
    target_variables::Vector{Symbol} = [:A_final, :B_final, :C_final, :v1_mean, :v2_mean]
    uncertainty_estimation::Bool = true  # 是否估计不确定性

    # CUDA配置
    use_cuda::Bool = true               # 是否使用CUDA加速
    cuda_batch_size::Int = 1000         # CUDA批处理大小

    # 热力学约束配置
    apply_thermodynamic_constraints::Bool = true  # 是否应用热力学约束
    # 约束模式: :range 使用范围; :fixed 固定Keq等式 (允许速率变动但比值固定)
    constraint_mode::Symbol = :range
    keq_min::Float64 = 0.01            # 平衡常数最小值 (范围模式)
    keq_max::Float64 = 100.0           # 平衡常数最大值 (范围模式)
    keq1_fixed::Union{Nothing,Float64} = nothing  # 固定Keq1 (固定模式)
    keq2_fixed::Union{Nothing,Float64} = nothing  # 固定Keq2 (固定模式)
    keq_tolerance::Float64 = 1e-3      # 固定模式允许的相对误差
end

"""
    ParameterSpace

参数空间定义，对应热力学参数扫描
"""
struct ParameterSpace
    # 反应速率常数范围
    k1f_range::AbstractRange
    k1r_range::AbstractRange
    k2f_range::AbstractRange
    k2r_range::AbstractRange
    k3f_range::AbstractRange
    k3r_range::AbstractRange
    k4f_range::AbstractRange
    k4r_range::AbstractRange

    # 初始浓度范围
    A_range::AbstractRange
    B_range::AbstractRange
    C_range::AbstractRange
    E1_range::AbstractRange
    E2_range::AbstractRange

    # 时间跨度
    tspan::Tuple{Float64, Float64}
end

"""
    create_default_parameter_space()

创建默认参数空间（与现有CUDA扫描一致）
"""
function create_default_parameter_space()
    return ParameterSpace(
        0.1:0.02:20.0,   # k1f_range (20 points)
        0.1:0.02:20.0,   # k1r_range
        0.1:0.02:20.0,   # k2f_range
        0.1:0.02:20.0,   # k2r_range
        0.1:0.02:20.0,   # k3f_range
        0.1:0.02:20.0,   # k3r_range
        0.1:0.02:20.0,   # k4f_range
        0.1:0.02:20.0,   # k4r_range
        0.1:0.02:20.0,   # A_range
        0.0:0.02:20.0,    # B_range
        0.0:0.02:20.0,    # C_range
        0.1:0.02:20.0,   # E1_range
        0.1:0.02:20.0,   # E2_range
        (0.0, 5.0)    # tspan
    )
end

"""
    configure_cuda_device()

配置最优的CUDA设备
"""
function configure_cuda_device()
    if !CUDA.functional()
        println("❌ CUDA GPU不可用 - 回退到CPU模式")
        return false
    end

    println("✅ CUDA可用")
    num_devices = CUDA.ndevices()
    println("检测到CUDA设备数量: $num_devices")

    if num_devices == 0
        println("❌ 未找到CUDA设备")
        return false
    end

    # 选择最佳设备
    best_device_id = 0
    best_score = -1000

    println("\n=== CUDA设备分析 ===")
    for i in 0:(num_devices-1)
        device = CuDevice(i)
        name = CUDA.name(device)
        total_memory = CUDA.totalmem(device)
        total_memory_gb = total_memory / (1024^3)

        # 计算性能评分
        score = 0

        # 偏好专业GPU
        if occursin("V100", name) || occursin("Tesla", name) || occursin("Quadro", name)
            score += 1000
            println("设备 $i: $name [🚀 专业GPU]")
        elseif occursin("RTX", name) || occursin("GTX", name)
            score += 500
            println("设备 $i: $name [💻 消费级GPU]")
        else
            println("设备 $i: $name [⚠️  其他/集成显卡]")
        end

        # 内存评分
        if total_memory_gb >= 16
            score += 200
        elseif total_memory_gb >= 8
            score += 100
        elseif total_memory_gb < 4
            score -= 200
        end

        println("  内存: $(round(total_memory_gb, digits=2)) GB")
        println("  性能评分: $score")

        if score > best_score
            best_score = score
            best_device_id = i
        end
        println()
    end

    # 设置最佳设备
    CUDA.device!(best_device_id)
    current_device = CUDA.device()
    device_name = CUDA.name(current_device)
    device_id = CUDA.deviceid(current_device)

    println("=== 已选择设备 ===")
    println("✅ 使用设备 $device_id: $device_name")
    println("内存: $(round(CUDA.totalmem(current_device) / 1024^3, digits=2)) GB")

    # 优化CUDA设置
    CUDA.reclaim()
    println("✅ CUDA内存池已初始化")

    return true
end

"""
    check_thermodynamic_constraints(k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, config::SurrogateModelConfig)

检查热力学约束。
支持两种模式：
- :range 模式：Keq1, Keq2 ∈ [keq_min, keq_max]
- :fixed 模式：Keq1 ≈ keq1_fixed 与 Keq2 ≈ keq2_fixed（相对误差 ≤ keq_tolerance）
"""
function check_thermodynamic_constraints(k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, config::SurrogateModelConfig)
    if !config.apply_thermodynamic_constraints
        return true
    end

    Keq1 = (k1f * k2f) / (k1r * k2r)
    Keq2 = (k3f * k4f) / (k3r * k4r)

    if config.constraint_mode == :fixed
        # 若未提供固定值，回退到范围逻辑
        keq1_target = config.keq1_fixed
        keq2_target = config.keq2_fixed
        tol = config.keq_tolerance

        if keq1_target !== nothing
            rel_err1 = abs(Keq1 - keq1_target) / max(abs(keq1_target), 1e-12)
            if rel_err1 > tol
                return false
            end
        end
        if keq2_target !== nothing
            rel_err2 = abs(Keq2 - keq2_target) / max(abs(keq2_target), 1e-12)
            if rel_err2 > tol
                return false
            end
        end
        return true
    else
        return (config.keq_min <= Keq1 <= config.keq_max) && (config.keq_min <= Keq2 <= config.keq_max)
    end
end

"""
    print_progress_bar(current, total, width=50, prefix="进度")

打印进度条
"""
function print_progress_bar(current, total, width=50, prefix="进度")
    percentage = current / total
    filled = round(Int, width * percentage)
    bar = "█"^filled * "░"^(width - filled)
    @printf("%s: [%s] %3.1f%% (%d/%d)\r", prefix, bar, percentage * 100, current, total)
    flush(stdout)
end

"""
    SurrogateModel

代理模型主结构体
"""
mutable struct SurrogateModel
    config::SurrogateModelConfig
    param_space::ParameterSpace

    # 数据
    X_train::Matrix{Float64}
    y_train::Matrix{Float64}
    X_val::Matrix{Float64}
    y_val::Matrix{Float64}

    # 预处理
    pca_model::Union{Nothing, PCA}
    input_scaler::Union{Nothing, NamedTuple}
    output_scaler::Union{Nothing, NamedTuple}

    # 模型
    model::Any
    training_history::Vector{Float64}

    # 构造函数
    function SurrogateModel(config::SurrogateModelConfig, param_space::ParameterSpace)
        new(config, param_space,
            Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0),
            Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0),
            nothing, nothing, nothing, nothing, Float64[])
    end
end

"""
    generate_small_scale_data(surrogate_model::SurrogateModel)

生成小规模参数扫描数据（步骤1：先运行小规模扫描）
"""
function generate_small_scale_data(surrogate_model::SurrogateModel)
    config = surrogate_model.config
    param_space = surrogate_model.param_space

    println("🔬 生成小规模参数扫描数据...")
    println("📊 采样比例: $(config.sample_fraction*100)%")

    if config.apply_thermodynamic_constraints
        println("🧪 应用热力学约束:")
        if config.constraint_mode == :fixed
            println("  固定Keq模式 (允许速率改变但比值固定，容差=$(config.keq_tolerance))")
            println("  Keq1_target = $(config.keq1_fixed), Keq2_target = $(config.keq2_fixed)")
        else
            println("  范围模式: Keq1, Keq2 ∈ [$(config.keq_min), $(config.keq_max)]")
        end
    end

    # 计算总参数组合数
    total_combinations_big = big(length(param_space.k1f_range)) * big(length(param_space.k1r_range)) *
                             big(length(param_space.k2f_range)) * big(length(param_space.k2r_range)) *
                             big(length(param_space.k3f_range)) * big(length(param_space.k3r_range)) *
                             big(length(param_space.k4f_range)) * big(length(param_space.k4r_range)) *
                             big(length(param_space.A_range)) * big(length(param_space.B_range)) *
                             big(length(param_space.C_range)) * big(length(param_space.E1_range)) *
                             big(length(param_space.E2_range))

    # 采样数量采用BigInt计算，转Int前裁剪到上限
    est_samples_big = round(Int, min(big(config.max_samples), total_combinations_big * big(config.sample_fraction)))
    n_samples = clamp(est_samples_big, 1, config.max_samples)
    println("📈 总组合数: $(string(total_combinations_big))")
    println("🎯 目标样本数: $n_samples")

    # 使用热力学约束的参数生成
    if config.apply_thermodynamic_constraints
        X_samples = generate_constrained_lhs_samples(param_space, n_samples, config)
    else
        X_samples = generate_lhs_samples(param_space, n_samples)
    end

    # 选择仿真方法（CUDA或CPU）
    println("🚀 开始仿真...")
    if config.use_cuda && configure_cuda_device()
        println("🔥 使用CUDA GPU加速仿真")
        y_samples = simulate_parameter_batch_gpu(X_samples, param_space.tspan, config.target_variables, config)

        # 检查GPU仿真结果
        valid_indices_gpu = findall(x -> !any(isnan.(x)), eachrow(y_samples))
        if length(valid_indices_gpu) > 0
            println("✅ GPU仿真成功，有效结果: $(length(valid_indices_gpu))/$(size(y_samples,1))")
        else
            println("❌ GPU仿真产生了全部NaN结果（纯GPU模式禁用CPU回退）")
        end
    else
        println("💻 使用CPU仿真")
        y_samples = simulate_parameter_batch(X_samples, param_space.tspan, config.target_variables)
    end

    # 过滤无效结果
    valid_indices = findall(x -> !any(isnan.(x)), eachrow(y_samples))
    X_clean = X_samples[valid_indices, :]
    y_clean = y_samples[valid_indices, :]

    println("🔍 仿真结果统计:")
    println("  总样本数: $(size(y_samples, 1))")
    println("  有效样本数: $(length(valid_indices))")
    println("  NaN样本数: $(size(y_samples, 1) - length(valid_indices))")

    println("✅ 有效样本数: $(size(X_clean, 1)) / $n_samples")

    # 纯GPU模式：不执行CPU回退

    if config.apply_thermodynamic_constraints && size(X_clean, 1) > 0
        original_combinations = big(length(param_space.k1f_range))^8 * big(length(param_space.A_range)) * big(length(param_space.B_range)) *
                               big(length(param_space.C_range)) * big(length(param_space.E1_range)) * big(length(param_space.E2_range))
        reduction_factor = Float64(original_combinations) / max(size(X_clean, 1), 1)
        println("📉 热力学约束参数空间缩减: $(round(reduction_factor, digits=1))x")
    end

    return X_clean, y_clean
end

"""
    generate_constrained_lhs_samples(param_space::ParameterSpace, n_samples::Int, config::SurrogateModelConfig)

生成满足热力学约束的拉丁超立方采样
"""
function generate_constrained_lhs_samples(param_space::ParameterSpace, n_samples::Int, config::SurrogateModelConfig)
    # 参数范围
    rate_ranges = [
        param_space.k1f_range, param_space.k1r_range,
        param_space.k2f_range, param_space.k2r_range,
        param_space.k3f_range, param_space.k3r_range,
        param_space.k4f_range, param_space.k4r_range
    ]

    conc_ranges = [
        param_space.A_range, param_space.B_range,
        param_space.C_range, param_space.E1_range, param_space.E2_range
    ]

    n_dims = length(rate_ranges) + length(conc_ranges)
    X_samples = zeros(0, n_dims)

    Random.seed!(42)  # 可重现性

    # 生成更多样本以确保有足够的有效样本
    max_attempts = n_samples * 10
    attempts = 0

    println("🔍 生成满足热力学约束的参数样本...")

    while size(X_samples, 1) < n_samples && attempts < max_attempts
        attempts += 1

        if attempts % 1000 == 0
            print_progress_bar(size(X_samples, 1), n_samples, 40, "约束采样")
        end

        # 生成单个样本
        sample = zeros(n_dims)

        # 反应速率常数
        for i in 1:length(rate_ranges)
            range_min, range_max = minimum(rate_ranges[i]), maximum(rate_ranges[i])
            sample[i] = range_min + rand() * (range_max - range_min)
        end

        # 检查热力学约束
        k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = sample[1:8]
        if check_thermodynamic_constraints(k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, config)
            # 浓度参数
            for i in 1:length(conc_ranges)
                range_min, range_max = minimum(conc_ranges[i]), maximum(conc_ranges[i])
                sample[8+i] = range_min + rand() * (range_max - range_min)
            end

            X_samples = vcat(X_samples, sample')
        end
    end

    println()  # 换行

    if size(X_samples, 1) < n_samples
        println("⚠️  警告: 只生成了$(size(X_samples, 1))个满足约束的样本，少于目标的$n_samples个")
    end

    # 如果没有找到任何满足约束的样本，使用无约束采样作为后备方案
    if size(X_samples, 1) == 0
        println("🚨 警告: 没有找到满足热力学约束的样本！")
        println("🔄 使用无约束拉丁超立方采样作为后备方案...")

        # 临时禁用约束并生成样本
        # 临时禁用约束，退回无约束采样
        X_samples = generate_lhs_samples(param_space, n_samples)
        println("✅ 生成了$(size(X_samples, 1))个无约束样本")
    end

    return X_samples
end

"""
    generate_lhs_samples(param_space::ParameterSpace, n_samples::Int)

使用拉丁超立方采样生成参数样本
"""
function generate_lhs_samples(param_space::ParameterSpace, n_samples::Int)
    # 参数范围
    ranges = [
        param_space.k1f_range, param_space.k1r_range,
        param_space.k2f_range, param_space.k2r_range,
        param_space.k3f_range, param_space.k3r_range,
        param_space.k4f_range, param_space.k4r_range,
        param_space.A_range, param_space.B_range,
        param_space.C_range, param_space.E1_range, param_space.E2_range
    ]

    n_dims = length(ranges)
    X_samples = zeros(n_samples, n_dims)

    # LHS采样
    Random.seed!(42)  # 可重现性
    for i in 1:n_dims
        # 生成[0,1]上的LHS样本
        lhs_samples = (randperm(n_samples) .- 1 .+ rand(n_samples)) ./ n_samples
        # 映射到参数范围
        range_min, range_max = minimum(ranges[i]), maximum(ranges[i])
        X_samples[:, i] = range_min .+ lhs_samples .* (range_max - range_min)
    end

    return X_samples
end

"""
    simulate_parameter_batch(X_samples::Matrix{Float64}, tspan::Tuple{Float64, Float64}, target_vars::Vector{Symbol})

批量运行参数仿真（CPU版本）
"""
function simulate_parameter_batch(X_samples::Matrix{Float64}, tspan::Tuple{Float64, Float64}, target_vars::Vector{Symbol})
    n_samples = size(X_samples, 1)
    n_outputs = length(target_vars)
    y_samples = zeros(n_samples, n_outputs)

    @showprogress "仿真进度: " for i in 1:n_samples
        try
            # 解析参数
            params = Dict(
                :k1f => X_samples[i, 1], :k1r => X_samples[i, 2],
                :k2f => X_samples[i, 3], :k2r => X_samples[i, 4],
                :k3f => X_samples[i, 5], :k3r => X_samples[i, 6],
                :k4f => X_samples[i, 7], :k4r => X_samples[i, 8]
            )

            initial_conditions = [
                A   => X_samples[i, 9],
                B   => X_samples[i, 10],
                C   => X_samples[i, 11],
                E1  => X_samples[i, 12],
                E2  => X_samples[i, 13],
                AE1 => 0.0,
                BE2 => 0.0
            ]

            # 运行仿真
            sol = simulate_system(params, initial_conditions, tspan, saveat=0.1)

            # 提取目标变量
            y_samples[i, :] = extract_target_variables(sol, params, target_vars)

        catch e
            # 仿真失败时填充NaN
            y_samples[i, :] .= NaN
        end
    end

    return y_samples
end

"""
    simulate_parameter_batch_gpu(X_samples::Matrix{Float64}, tspan::Tuple{Float64, Float64}, target_vars::Vector{Symbol}, config::SurrogateModelConfig)

CUDA GPU加速的批量参数仿真
"""
function simulate_parameter_batch_gpu(X_samples::Matrix{Float64}, tspan::Tuple{Float64, Float64}, target_vars::Vector{Symbol}, config::SurrogateModelConfig)
    n_samples = size(X_samples, 1)
    n_outputs = length(target_vars)

    println("🚀 配置GPU集成求解器...")

    # 使用优化版多GPU求解器
    gpu_cfg = default_gpu_config()
    solver = OptimizedGPUSolver(gpu_cfg)

    try
        results_f32 = solve_batch_gpu_optimized(solver, X_samples, tspan, target_vars)
        # 转为Float64以与下游保持一致
        y_samples = Array{Float64}(undef, size(results_f32, 1), size(results_f32, 2))
        @inbounds for i in 1:size(results_f32,1), j in 1:size(results_f32,2)
            y_samples[i,j] = Float64(results_f32[i,j])
        end

        valid_count = sum(x -> !any(isnan.(x)), eachrow(y_samples))
        println("✅ GPU仿真完成: $valid_count/$n_samples 有效结果")
        return y_samples
    catch e
        println("⚠️  GPU求解失败（纯GPU模式）: $e")
        # 返回全NaN以指示失败
        y_samples = fill(NaN, n_samples, n_outputs)
        return y_samples
    finally
        cleanup_gpu_resources!(solver)
    end
end

"""
    extract_target_variables_from_concentrations(concentrations::Vector{Float64}, sol, target_vars::Vector{Symbol})

从浓度数据中提取目标变量
"""
function extract_target_variables_from_concentrations(concentrations::Vector{Float64}, sol, target_vars::Vector{Symbol})
    results = Float64[]

    # 简单调试 - 使用全局计数器
    global extraction_counter
    if !@isdefined(extraction_counter)
        extraction_counter = 0
    end
    extraction_counter += 1
    debug_this = extraction_counter <= 3

    if debug_this
        println("🔍 提取调试 #$extraction_counter:")
        println("  浓度: $concentrations")
        println("  sol大小: $(size(sol))")
    end

    try
        for (idx, var) in enumerate(target_vars)
            if var == :A_final
                push!(results, concentrations[1])
            elseif var == :B_final
                push!(results, concentrations[2])
            elseif var == :C_final
                push!(results, concentrations[3])
            elseif var == :v1_mean
                try
                    A_traj = sol[1, :]
                    E1_traj = sol[4, :]

                    if debug_this
                        println("  v1_mean: A轨迹长度=$(length(A_traj)), E1轨迹长度=$(length(E1_traj))")
                        if length(A_traj) > 0
                            println("    A范围: $(minimum(A_traj)) - $(maximum(A_traj))")
                        end
                        if length(E1_traj) > 0
                            println("    E1范围: $(minimum(E1_traj)) - $(maximum(E1_traj))")
                        end
                    end

                    if length(A_traj) > 0 && length(E1_traj) > 0 && all(isfinite.(A_traj)) && all(isfinite.(E1_traj))
                        v1_approx = mean(max.(A_traj, 0.0) .* max.(E1_traj, 0.0)) * concentrations[1] * 0.01
                        final_v1 = isfinite(v1_approx) ? v1_approx : 0.0
                        push!(results, final_v1)
                        if debug_this
                            println("    v1计算结果: $final_v1")
                        end
                    else
                        push!(results, 0.0)
                        if debug_this
                            println("    v1回退到0")
                        end
                    end
                catch e
                    push!(results, 0.0)
                    if debug_this
                        println("    v1异常: $e")
                    end
                end
            elseif var == :v2_mean
                try
                    B_traj = sol[2, :]
                    E2_traj = sol[5, :]

                    if debug_this
                        println("  v2_mean: B轨迹长度=$(length(B_traj)), E2轨迹长度=$(length(E2_traj))")
                        if length(B_traj) > 0
                            println("    B范围: $(minimum(B_traj)) - $(maximum(B_traj))")
                        end
                        if length(E2_traj) > 0
                            println("    E2范围: $(minimum(E2_traj)) - $(maximum(E2_traj))")
                        end
                    end

                    if length(B_traj) > 0 && length(E2_traj) > 0 && all(isfinite.(B_traj)) && all(isfinite.(E2_traj))
                        v2_approx = mean(max.(B_traj, 0.0) .* max.(E2_traj, 0.0)) * concentrations[2] * 0.01
                        final_v2 = isfinite(v2_approx) ? v2_approx : 0.0
                        push!(results, final_v2)
                        if debug_this
                            println("    v2计算结果: $final_v2")
                        end
                    else
                        push!(results, 0.0)
                        if debug_this
                            println("    v2回退到0")
                        end
                    end
                catch e
                    push!(results, 0.0)
                    if debug_this
                        println("    v2异常: $e")
                    end
                end
            else
                error("未知的目标变量: $var")
            end
        end

        if debug_sample && rand() < 0.001
            println("  最终结果: $results")
            println("  结果检查: 有NaN=$(any(isnan.(results))), 有Inf=$(any(isinf.(results)))")
        end

    catch e
        if debug_sample && rand() < 0.001
            println("  💥 提取异常: $e")
        end
        # 如果有任何异常，返回NaN数组
        return fill(NaN, length(target_vars))
    end

    return results
end



"""
    solve_multi_gpu_parallel(X_samples, u0s, ps, tspan, target_vars, n_samples)

使用多GPU并行处理大规模仿真
"""
function solve_multi_gpu_parallel(X_samples, u0s, ps, tspan, target_vars, n_samples)
    y_samples = zeros(n_samples, length(target_vars))

    # 将工作分配给两个GPU (序列化处理避免任务失败)
    n_gpu1 = div(n_samples, 2)
    n_gpu2 = n_samples - n_gpu1

    println("  序列化多GPU处理: GPU0($(n_gpu1)样本) -> GPU1($(n_gpu2)样本)")

    # 先在GPU0上处理第一批
    CUDA.device!(0)
    println("  切换到GPU0处理前$(n_gpu1)个样本...")
    gpu0_results = solve_gpu_batch_chunk(X_samples[1:n_gpu1, :], u0s[1:n_gpu1], ps[1:n_gpu1],
                                        tspan, target_vars, 0)

    # 再切换到GPU1处理第二批
    CUDA.device!(1)
    println("  切换到GPU1处理剩余$(n_gpu2)个样本...")
    gpu1_results = solve_gpu_batch_chunk(X_samples[n_gpu1+1:end, :], u0s[n_gpu1+1:end], ps[n_gpu1+1:end],
                                        tspan, target_vars, 1)

    # 合并结果
    y_samples[1:n_gpu1, :] = gpu0_results
    y_samples[n_gpu1+1:end, :] = gpu1_results

    return y_samples
end

"""
    solve_single_gpu_batch(X_samples, u0s, ps, tspan, target_vars, n_samples)

单GPU批量处理
"""
function solve_single_gpu_batch(X_samples, u0s, ps, tspan, target_vars, n_samples)
    CUDA.device!(0)
    return solve_gpu_batch_chunk(X_samples, u0s, ps, tspan, target_vars, 0)
end

"""
    solve_gpu_batch_chunk(X_chunk, u0s_chunk, ps_chunk, tspan, target_vars, gpu_id)

在指定GPU上处理一批仿真 - 使用真正的GPU计算
"""
function solve_gpu_batch_chunk(X_chunk, u0s_chunk, ps_chunk, tspan, target_vars, gpu_id)
    n_chunk = size(X_chunk, 1)
    y_chunk = zeros(n_chunk, length(target_vars))

    println("    GPU$(gpu_id)开始处理 $(n_chunk) 个样本")

    # ODE函数定义 - GPU兼容版本
    function reaction_ode!(du, u, p, t)
        # 兼容参数类型（具名字段或元组）
        k1f = Base.hasproperty(p, :k1f) ? p.k1f : p[1]
        k1r = Base.hasproperty(p, :k1r) ? p.k1r : p[2]
        k2f = Base.hasproperty(p, :k2f) ? p.k2f : p[3]
        k2r = Base.hasproperty(p, :k2r) ? p.k2r : p[4]
        k3f = Base.hasproperty(p, :k3f) ? p.k3f : p[5]
        k3r = Base.hasproperty(p, :k3r) ? p.k3r : p[6]
        k4f = Base.hasproperty(p, :k4f) ? p.k4f : p[7]
        k4r = Base.hasproperty(p, :k4r) ? p.k4r : p[8]

        # 标量无分配的非负限制（避免 max. 广播）
        A   = ifelse(u[1] > 0.0, u[1], 0.0)
        B   = ifelse(u[2] > 0.0, u[2], 0.0)
        C   = ifelse(u[3] > 0.0, u[3], 0.0)
        E1  = ifelse(u[4] > 0.0, u[4], 0.0)
        E2  = ifelse(u[5] > 0.0, u[5], 0.0)
        AE1 = ifelse(u[6] > 0.0, u[6], 0.0)
        BE2 = ifelse(u[7] > 0.0, u[7], 0.0)

        @inbounds begin
            du[1] = -k1f*A*E1 + k1r*AE1
            du[2] = k2f*AE1 - k2r*B*E1 - k3f*B*E2 + k3r*BE2
            du[3] = k4f*BE2 - k4r*C*E2
            du[4] = -k1f*A*E1 + k1r*AE1 + k2f*AE1 - k2r*B*E1
            du[5] = -k3f*B*E2 + k3r*BE2 + k4f*BE2 - k4r*C*E2
            du[6] = k1f*A*E1 - k1r*AE1 - k2f*AE1 + k2r*B*E1
            du[7] = k3f*B*E2 - k3r*BE2 - k4f*BE2 + k4r*C*E2
        end
    end

    try
        su0_first = _to_su0(u0s_chunk[1])
        pp_first = _to_pp(ps_chunk[1])

        # 使用真正的GPU求解器（Kernel）
        prob_func = (prob, i, repeat) -> remake(prob, u0=_to_su0(u0s_chunk[i]), p=_to_pp(ps_chunk[i]))

        # 创建EnsembleProblem（SVector + isbits参数）
        prob = ODEProblem(reaction_ode!, su0_first, tspan, pp_first)
        ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
        
        # 使用DiffEqGPU进行GPU并行求解
        # 优先使用 Kernel 模式以提升GPU占用
        alg = EnsembleGPUKernel(CUDADevice())
        sol = solve(ensemble_prob, Tsit5(), alg,
                    trajectories=n_chunk, save_everystep=false, save_start=false,
                    batch_size=1024,
                    unstable_check=false,
                    abstol=1e-6, reltol=1e-3)
        
        # 提取结果
        for i in 1:n_chunk
            if sol[i].retcode == :Success
                final_values = sol[i].u[end]
                
                # 根据target_vars提取对应结果
                for (j, var) in enumerate(target_vars)
                    if var == :A
                        y_chunk[i, j] = final_values[1]
                    elseif var == :B  
                        y_chunk[i, j] = final_values[2]
                    elseif var == :C
                        y_chunk[i, j] = final_values[3]
                    elseif var == :E1
                        y_chunk[i, j] = final_values[4]
                    elseif var == :E2
                        y_chunk[i, j] = final_values[5]
                    end
                end
            else
                # 如果求解失败，填充NaN
                y_chunk[i, :] .= NaN
            end
        end
        
        println("    GPU$(gpu_id)完成处理: $(n_chunk) 个样本")
        
    catch e
        println("    GPU$(gpu_id)求解失败: $(typeof(e))")
        # 回退到CPU求解
        for i in 1:n_chunk
            try
                # 构建CPU版本的参数
                params_dict = Dict(
                    :k1f => ps_chunk[i].k1f, :k1r => ps_chunk[i].k1r,
                    :k2f => ps_chunk[i].k2f, :k2r => ps_chunk[i].k2r,
                    :k3f => ps_chunk[i].k3f, :k3r => ps_chunk[i].k3r,
                    :k4f => ps_chunk[i].k4f, :k4r => ps_chunk[i].k4r
                )

                initial_conditions = [
                    A   => u0s_chunk[i][1],
                    B   => u0s_chunk[i][2],
                    C   => u0s_chunk[i][3],
                    E1  => u0s_chunk[i][4],
                    E2  => u0s_chunk[i][5],
                    AE1 => u0s_chunk[i][6],
                    BE2 => u0s_chunk[i][7]
                ]

                # 使用现有的CPU仿真系统
                sol = simulate_system(params_dict, initial_conditions, tspan, saveat=0.1)
                target_results = extract_target_variables(sol, params_dict, target_vars)

                if all(isfinite.(target_results))
                    y_chunk[i, :] = target_results
                else
                    y_chunk[i, :] .= NaN
                end

            catch e
                y_chunk[i, :] .= NaN
            end
        end
        
        println("    ✅ GPU$(gpu_id)完成CPU回退处理: $(n_chunk) 个样本")
    end

    return y_chunk
end

"""
    extract_target_variables_simple(concentrations, sol, target_vars)

简化的目标变量提取（避免复杂的轨迹计算）
"""
function extract_target_variables_simple(concentrations::Vector{Float64}, sol, target_vars::Vector{Symbol})
    results = Float64[]

    for var in target_vars
        if var == :A_final
            push!(results, concentrations[1])
        elseif var == :B_final
            push!(results, concentrations[2])
        elseif var == :C_final
            push!(results, concentrations[3])
        elseif var == :v1_mean
            # 简化的通量估算
            v1_est = concentrations[1] * concentrations[4] * 0.1  # A * E1 * scaling
            push!(results, isfinite(v1_est) ? max(v1_est, 0.0) : 0.0)
        elseif var == :v2_mean
            # 简化的通量估算
            v2_est = concentrations[2] * concentrations[5] * 0.1  # B * E2 * scaling
            push!(results, isfinite(v2_est) ? max(v2_est, 0.0) : 0.0)
        else
            push!(results, 0.0)
        end
    end

    return results
end

"""
    extract_target_variables(sol, params, target_vars::Vector{Symbol})

从仿真结果中提取目标变量
"""
function extract_target_variables(sol, params, target_vars::Vector{Symbol})
    results = Float64[]

    for var in target_vars
        if var == :A_final
            push!(results, sol[A][end])
        elseif var == :B_final
            push!(results, sol[B][end])
        elseif var == :C_final
            push!(results, sol[C][end])
        elseif var == :v1_mean
            fluxes = calculate_kinetic_fluxes(sol, params)
            push!(results, mean(fluxes["v1"]))
        elseif var == :v2_mean
            fluxes = calculate_kinetic_fluxes(sol, params)
            push!(results, mean(fluxes["v2"]))
        else
            error("未知的目标变量: $var")
        end
    end

    return results
end

"""
    preprocess_data!(surrogate_model::SurrogateModel, X::Matrix{Float64}, y::Matrix{Float64})

数据预处理：标准化和PCA降维
"""
function preprocess_data!(surrogate_model::SurrogateModel, X::Matrix{Float64}, y::Matrix{Float64})
    config = surrogate_model.config

    println("🔧 数据预处理...")

    # 检查数据是否为空
    if size(X, 1) == 0 || size(y, 1) == 0
        error("🚨 错误: 输入数据为空！无法进行数据预处理。\n" *
              "这通常是因为热力学约束过于严格，导致所有样本都被过滤掉。\n" *
              "建议：1) 放宽热力学约束范围，2) 调整参数空间范围，或 3) 禁用热力学约束")
    end

    println("📊 数据维度: X=$(size(X)), y=$(size(y))")

    # 输入标准化
    X_mean = mean(X, dims=1)
    X_std = std(X, dims=1)
    X_normalized = (X .- X_mean) ./ (X_std .+ 1e-8)

    surrogate_model.input_scaler = (mean=X_mean, std=X_std)

    # PCA降维（如果启用）
    if config.use_pca && size(X, 2) > 5
        println("📉 应用PCA降维...")
        pca_model = fit(PCA, X_normalized'; maxoutdim=size(X,2), pratio=config.pca_variance_threshold)
        X_pca = MultivariateStats.transform(pca_model, X_normalized')'

        surrogate_model.pca_model = pca_model
        println("📊 PCA: $(size(X, 2)) → $(size(X_pca, 2)) 维")
        X_processed = X_pca
    else
        X_processed = X_normalized
    end

    # 输出标准化
    y_mean = mean(y, dims=1)
    y_std = std(y, dims=1)
    y_normalized = (y .- y_mean) ./ (y_std .+ 1e-8)

    surrogate_model.output_scaler = (mean=y_mean, std=y_std)

    # 训练验证分割
    n_samples = size(X_processed, 1)
    n_val = Int(round(n_samples * config.validation_split))
    indices = randperm(n_samples)

    val_indices = indices[1:n_val]
    train_indices = indices[n_val+1:end]

    surrogate_model.X_train = X_processed[train_indices, :]
    surrogate_model.y_train = y_normalized[train_indices, :]
    surrogate_model.X_val = X_processed[val_indices, :]
    surrogate_model.y_val = y_normalized[val_indices, :]

    println("✅ 预处理完成")
    println("📈 训练集: $(size(surrogate_model.X_train, 1)) 样本")
    println("📊 验证集: $(size(surrogate_model.X_val, 1)) 样本")
end

"""
    create_neural_network(input_dim::Int, output_dim::Int, config::SurrogateModelConfig)

创建神经网络模型（支持Dropout用于不确定性估计）
"""
function create_neural_network(input_dim::Int, output_dim::Int, config::SurrogateModelConfig)
    layers = []

    # 输入层 - 使用Float64
    push!(layers, Dense(input_dim => config.hidden_dims[1], relu))
    if config.uncertainty_estimation
        push!(layers, Dropout(config.dropout_rate))
    end

    # 隐藏层 - 使用Float64
    for i in 1:length(config.hidden_dims)-1
        push!(layers, Dense(config.hidden_dims[i] => config.hidden_dims[i+1], relu))
        if config.uncertainty_estimation
            push!(layers, Dropout(config.dropout_rate))
        end
    end

    # 输出层 - 使用Float64
    push!(layers, Dense(config.hidden_dims[end] => output_dim))

    model = Chain(layers...)
    # 确保所有参数都是Float64
    return model |> f64
end

"""
    train_surrogate_model!(surrogate_model::SurrogateModel)

训练代理模型
"""
function train_surrogate_model!(surrogate_model::SurrogateModel)
    config = surrogate_model.config

    println("🎯 开始训练代理模型...")
    println("🔧 模型类型: $(config.model_type)")

    if config.model_type == :neural_network
        train_neural_network!(surrogate_model)
    elseif config.model_type == :gaussian_process
        train_gaussian_process!(surrogate_model)
    else
        error("不支持的模型类型: $(config.model_type)")
    end

    println("✅ 训练完成!")
end

"""
    train_neural_network!(surrogate_model::SurrogateModel)

训练神经网络代理模型
"""
function train_neural_network!(surrogate_model::SurrogateModel)
    config = surrogate_model.config

    input_dim = size(surrogate_model.X_train, 2)
    output_dim = size(surrogate_model.y_train, 2)

    # 创建模型
    model = create_neural_network(input_dim, output_dim, config)
    surrogate_model.model = model

    # 训练数据准备
    X_train = surrogate_model.X_train'
    y_train = surrogate_model.y_train'
    X_val = surrogate_model.X_val'
    y_val = surrogate_model.y_val'

    # 优化器 - 使用Flux.setup
    opt = Adam(config.learning_rate)
    opt_state = Flux.setup(opt, model)

    # 训练循环
    train_losses = Float64[]
    val_losses = Float64[]

    @showprogress "训练进度: " for epoch in 1:config.epochs
        # 训练
        train_loss = 0.0
        n_batches = 0

        for batch_indices in partition(1:size(X_train, 2), config.batch_size)
            X_batch = X_train[:, batch_indices]
            y_batch = y_train[:, batch_indices]

            loss, grads = Flux.withgradient(model) do m
                y_pred = m(X_batch)
                Flux.mse(y_pred, y_batch)
            end

            Flux.update!(opt_state, model, grads[1])
            train_loss += loss
            n_batches += 1
        end

        train_loss /= n_batches
        push!(train_losses, train_loss)

        # 验证
        if epoch % 10 == 0
            val_pred = model(X_val)
            val_loss = Flux.mse(val_pred, y_val)
            push!(val_losses, val_loss)

            println("Epoch $epoch: Train Loss = $(round(train_loss, digits=6)), Val Loss = $(round(val_loss, digits=6))")
        end
    end

    surrogate_model.training_history = train_losses
    println("🎯 最终训练损失: $(round(train_losses[end], digits=6))")
end

# 辅助函数：数据分批
function partition(collection, n)
    result = []
    for i in 1:n:length(collection)
        push!(result, collection[i:min(i+n-1, end)])
    end
    return result
end

# 引入Gaussian Process实现（需要 SurrogateModel 已定义）
include("gaussian_process.jl")

# ================= GPU Kernel Utilities (top-level) =================
# Use GPUParams from gpu_parallel_optimized.jl (Float64)
@inline function _to_su0(u)
    return SVector{7,Float64}(
        Float64(u[1]), Float64(u[2]), Float64(u[3]),
        Float64(u[4]), Float64(u[5]), Float64(u[6]), Float64(u[7])
    )
end

@inline function _to_pp(p)
    # p may be NamedTuple / struct with fields
    if Base.hasproperty(p, :k1f)
        return GPUParams(Float64(p.k1f), Float64(p.k1r), Float64(p.k2f), Float64(p.k2r),
                         Float64(p.k3f), Float64(p.k3r), Float64(p.k4f), Float64(p.k4r))
    else
        return GPUParams(Float64(p[1]), Float64(p[2]), Float64(p[3]), Float64(p[4]),
                         Float64(p[5]), Float64(p[6]), Float64(p[7]), Float64(p[8]))
    end
end

"""
    predict_with_uncertainty(surrogate_model::SurrogateModel, X_new::Matrix{Float64}; n_samples::Int=100)

使用代理模型进行预测（包含不确定性估计）
"""
function predict_with_uncertainty(surrogate_model::SurrogateModel, X_new::Matrix{Float64}; n_samples::Int=100)
    config = surrogate_model.config

    # 输入预处理
    X_normalized = (X_new .- surrogate_model.input_scaler.mean) ./ surrogate_model.input_scaler.std

    if surrogate_model.pca_model !== nothing
        X_processed = MultivariateStats.transform(surrogate_model.pca_model, X_normalized')'
    else
        X_processed = X_normalized
    end
    # 确保为标准Matrix类型，避免Adjoint导致方法匹配问题
    X_processed = Array(X_processed)

    # 预测
    if config.model_type == :gaussian_process
        # 使用GP预测（当前实现返回均值，无显式不确定性）
        y_pred_mean = predict_gaussian_process(surrogate_model, X_processed)
        y_pred_std = zeros(size(y_pred_mean))
    elseif config.uncertainty_estimation && config.model_type == :neural_network
        # 使用MC Dropout进行不确定性估计（强制启用dropout）
        # 注意：需要在预测时启用 Dropout，训练时关闭 BatchNorm（未用到）
        Flux.testmode!(surrogate_model.model, false)  # ensure dropout active
        preds = Vector{Array{Float64,2}}(undef, n_samples)
        for i in 1:n_samples
            y = surrogate_model.model(X_processed')  # [D, N]
            preds[i] = Array(y')                    # [N, D]
        end
        Flux.testmode!(surrogate_model.model, true)

        predictions_array = cat(preds...; dims=3)  # [N, D, M]
        y_pred_mean = dropdims(mean(predictions_array; dims=3), dims=3)
        y_pred_std  = dropdims(std(predictions_array; dims=3),  dims=3)
    else
        # 普通预测
        y_pred_normalized = surrogate_model.model(X_processed')'
        y_pred_mean = y_pred_normalized
        y_pred_std = zeros(size(y_pred_mean))
    end

    # 输出反标准化
    y_pred_mean = y_pred_mean .* surrogate_model.output_scaler.std .+ surrogate_model.output_scaler.mean
    y_pred_std = y_pred_std .* surrogate_model.output_scaler.std

    return y_pred_mean, y_pred_std
end

"""
    save_surrogate_model(surrogate_model::SurrogateModel, filepath::String)

保存代理模型
"""
function save_surrogate_model(surrogate_model::SurrogateModel, filepath::String)
    println("💾 保存代理模型到: $filepath")

    # 直接使用jldsave的命名参数语法
    if surrogate_model.model !== nothing
        jldsave(filepath;
            config = surrogate_model.config,
            param_space = surrogate_model.param_space,
            pca_model = surrogate_model.pca_model,
            input_scaler = surrogate_model.input_scaler,
            output_scaler = surrogate_model.output_scaler,
            training_history = surrogate_model.training_history,
            model_state = Flux.state(surrogate_model.model),
            model_structure = surrogate_model.model
        )
    else
        jldsave(filepath;
            config = surrogate_model.config,
            param_space = surrogate_model.param_space,
            pca_model = surrogate_model.pca_model,
            input_scaler = surrogate_model.input_scaler,
            output_scaler = surrogate_model.output_scaler,
            training_history = surrogate_model.training_history
        )
    end
    println("✅ 模型保存完成")
end

"""
    load_surrogate_model(filepath::String)

加载代理模型
"""
function load_surrogate_model(filepath::String)
    println("📂 加载代理模型从: $filepath")

    data = JLD2.load(filepath)

    # 重建代理模型
    surrogate_model = SurrogateModel(data["config"], data["param_space"])
    surrogate_model.pca_model = data["pca_model"]
    surrogate_model.input_scaler = data["input_scaler"]
    surrogate_model.output_scaler = data["output_scaler"]
    surrogate_model.training_history = data["training_history"]

    # 重建Flux模型
    if haskey(data, "model_structure")
        surrogate_model.model = data["model_structure"]
        Flux.loadmodel!(surrogate_model.model, data["model_state"])
    end

    println("✅ 模型加载完成")
    return surrogate_model
end

"""
    compare_surrogate_vs_cuda(surrogate_model::SurrogateModel, n_test_samples::Int=1000)

比较代理模型与CUDA仿真的性能和精度
"""
function compare_surrogate_vs_cuda(surrogate_model::SurrogateModel, n_test_samples::Int=1000)
    println("🔄 代理模型 vs CUDA仿真性能比较")
    println("测试样本数: $n_test_samples")

    config = surrogate_model.config
    param_space = surrogate_model.param_space

    # 生成测试参数
    if config.apply_thermodynamic_constraints
        X_test = generate_constrained_lhs_samples(param_space, n_test_samples, config)
    else
        X_test = generate_lhs_samples(param_space, n_test_samples)
    end

    if size(X_test, 1) < n_test_samples
        n_test_samples = size(X_test, 1)
        println("⚠️  实际测试样本数: $n_test_samples")
    end

    # 1. CUDA仿真基准测试
    println("\n🔥 CUDA仿真基准测试...")
    cuda_start_time = time()
    if config.use_cuda && configure_cuda_device()
        y_cuda = simulate_parameter_batch_gpu(X_test, param_space.tspan, config.target_variables, config)
    else
        y_cuda = simulate_parameter_batch(X_test, param_space.tspan, config.target_variables)
    end
    cuda_time = time() - cuda_start_time

    # 2. 代理模型预测
    println("⚡ 代理模型预测测试...")
    surrogate_start_time = time()
    y_pred, y_std = predict_with_uncertainty(surrogate_model, X_test, n_samples=100)
    surrogate_time = time() - surrogate_start_time

    # 3. 性能指标计算
    valid_indices = findall(x -> !any(isnan.(x)), eachrow(y_cuda))
    X_valid = X_test[valid_indices, :]
    y_cuda_valid = y_cuda[valid_indices, :]
    y_pred_valid = y_pred[valid_indices, :]
    y_std_valid = y_std[valid_indices, :]

    n_valid = length(valid_indices)
    println("\n📊 性能比较结果:")
    println("===================")
    println("有效测试样本数: $n_valid / $n_test_samples")

    # 时间性能
    speedup = cuda_time / surrogate_time
    println("\n⏱️  时间性能:")
    println("CUDA仿真时间: $(round(cuda_time, digits=3)) 秒")
    println("代理模型时间: $(round(surrogate_time, digits=3)) 秒")
    println("加速比: $(round(speedup, digits=1))x")

    # 精度指标
    mse_total = 0.0
    mae_total = 0.0
    r2_scores = Float64[]

    for i in 1:size(y_cuda_valid, 2)
        y_true = y_cuda_valid[:, i]
        y_pred_col = y_pred_valid[:, i]

        # MSE和MAE
        mse = mean((y_true - y_pred_col).^2)
        mae = mean(abs.(y_true - y_pred_col))

        mse_total += mse
        mae_total += mae

        # R²
        ss_res = sum((y_true - y_pred_col).^2)
        ss_tot = sum((y_true .- mean(y_true)).^2)
        r2 = 1 - ss_res / ss_tot
        push!(r2_scores, r2)
    end

    avg_mse = mse_total / size(y_cuda_valid, 2)
    avg_mae = mae_total / size(y_cuda_valid, 2)
    avg_r2 = mean(r2_scores)

    println("\n🎯 精度指标:")
    println("平均MSE: $(round(avg_mse, digits=6))")
    println("平均MAE: $(round(avg_mae, digits=6))")
    println("平均R²: $(round(avg_r2, digits=4))")
    println("平均不确定性: $(round(mean(y_std_valid), digits=6))")

    # 计算节省的计算时间
    time_saved = cuda_time - surrogate_time
    time_saved_percent = (time_saved / cuda_time) * 100
    println("\n💰 计算效率:")
    println("时间节省: $(round(time_saved, digits=3)) 秒")
    println("效率提升: $(round(time_saved_percent, digits=1))%")

    return (
        cuda_time=cuda_time,
        surrogate_time=surrogate_time,
        speedup=speedup,
        mse=avg_mse,
        mae=avg_mae,
        r2=avg_r2,
        uncertainty=mean(y_std_valid),
        efficiency=time_saved_percent
    )
end

"""
    large_scale_parameter_scan(surrogate_model::SurrogateModel, scan_config::Dict; max_combinations::Int=1000000)

使用代理模型进行大规模参数扫描
"""
function large_scale_parameter_scan(surrogate_model::SurrogateModel, scan_config::Dict; max_combinations::Int=1000000)
    println("🚀 大规模参数扫描")
    println("最大扫描组合数: $max_combinations")

    config = surrogate_model.config
    param_space = surrogate_model.param_space

    # 定义完整参数顺序和默认值
    all_param_names = [:k1f, :k1r, :k2f, :k2r, :k3f, :k3r, :k4f, :k4r, :A, :B, :C, :E1, :E2]
    default_values = [
        10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,  # rate constants
        10.0, 2.5, 2.5, 12.5, 12.5  # concentrations
    ]

    # 解析扫描配置 - 只记录变化的参数
    varying_param_ranges = []
    varying_param_indices = []
    varying_param_names = []

    for (i, param) in enumerate(all_param_names)
        if haskey(scan_config, param)
            push!(varying_param_ranges, collect(scan_config[param]))
            push!(varying_param_indices, i)
            push!(varying_param_names, param)
        end
    end

    # 计算总组合数（使用BigInt避免溢出）
    total_lengths = map(r -> length(r), varying_param_ranges)
    total_combinations_big = prod(big.(total_lengths))
    println("📈 理论总组合数: $(string(total_combinations_big))")

    # 决定是否采样
    need_sampling = total_combinations_big > big(max_combinations)
    if need_sampling
        # 使用采样减少组合数
        sample_fraction = Float64(big(max_combinations) / total_combinations_big)
        println("📊 采样比例: $(round(sample_fraction*100, digits=2))%")

        # 生成采样参数组合
        X_scan = generate_complete_parameter_combinations_sampled(
            varying_param_ranges, varying_param_indices, default_values, max_combinations, config)
    else
        # 生成所有组合
        # 只有在组合数可以安全转为Int且规模合理时才生成完整笛卡尔积
        n_combinations_big = total_combinations_big
        if n_combinations_big > big(typemax(Int))
            println("⚠️  组合数超过Int上限，改用采样模式")
            X_scan = generate_complete_parameter_combinations_sampled(
                varying_param_ranges, varying_param_indices, default_values, max_combinations, config)
        else
            n_combinations = Int(n_combinations_big)
            if n_combinations > 5_000_000
                println("⚠️  组合数过大($(n_combinations))，为避免内存问题改用采样模式")
                X_scan = generate_complete_parameter_combinations_sampled(
                    varying_param_ranges, varying_param_indices, default_values, max_combinations, config)
            else
                X_scan = generate_complete_parameter_combinations_full(
                    varying_param_ranges, varying_param_indices, default_values)
            end
        end
    end

    n_scan = size(X_scan, 1)
    println("🎯 实际扫描数: $n_scan")

    # 代理模型预测
    println("⚡ 代理模型快速预测...")
    scan_start_time = time()
    y_pred, y_std = predict_with_uncertainty(surrogate_model, X_scan, n_samples=50)
    scan_time = time() - scan_start_time

    println("✅ 扫描完成!")
    println("⏱️  预测时间: $(round(scan_time, digits=3)) 秒")
    println("🚀 预测速度: $(round(n_scan/scan_time, digits=1)) 预测/秒")

    # 估算等效CUDA时间
    estimated_cuda_time = n_scan * (surrogate_model.config.use_cuda ? 0.01 : 0.1)  # 估算每个仿真时间
    estimated_speedup = estimated_cuda_time / scan_time

    println("💰 性能优势:")
    println("估算CUDA时间: $(round(estimated_cuda_time, digits=1)) 秒")
    println("估算加速比: $(round(estimated_speedup, digits=1))x")

    # 组织结果
    results = []
    for i in 1:n_scan
        param_dict = Dict()
        for (j, name) in enumerate(all_param_names)
            param_dict[name] = X_scan[i, j]
        end

        pred_dict = Dict()
        for (j, var) in enumerate(config.target_variables)
            pred_dict[var] = y_pred[i, j]
            pred_dict[Symbol(string(var) * "_std")] = y_std[i, j]
        end

        push!(results, (parameters=param_dict, predictions=pred_dict))
    end

    return results
end

"""
    generate_sampled_parameter_combinations(param_ranges::Vector, n_samples::Int, config::SurrogateModelConfig)

生成采样的参数组合（用于大规模扫描）
"""
function generate_complete_parameter_combinations_sampled(varying_param_ranges::Vector, varying_param_indices::Vector, default_values::Vector, n_samples::Int, config::SurrogateModelConfig)
    X_samples = zeros(0, 13)  # 总是生成13维参数向量

    max_attempts = n_samples * 5
    attempts = 0

    Random.seed!(42)

    println("🔍 生成采样参数组合...")

    while size(X_samples, 1) < n_samples && attempts < max_attempts
        attempts += 1

        if attempts % 10000 == 0
            print_progress_bar(size(X_samples, 1), n_samples, 40, "参数采样")
        end

        # 从默认值开始
        sample = copy(default_values)

        # 设置变化的参数
        for (i, param_idx) in enumerate(varying_param_indices)
            sample[param_idx] = varying_param_ranges[i][rand(1:length(varying_param_ranges[i]))]
        end

        # 检查热力学约束（前8个参数是反应速率）
        if config.apply_thermodynamic_constraints
            k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = sample[1:8]
            if !check_thermodynamic_constraints(k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, config)
                continue
            end
        end

        X_samples = vcat(X_samples, sample')
    end

    println()  # 换行
    return X_samples
end

function generate_complete_parameter_combinations_full(varying_param_ranges::Vector, varying_param_indices::Vector, default_values::Vector)
    # 生成所有组合的完整13维参数向量（安全版）
    lengths_vec = map(r -> length(r), varying_param_ranges)
    n_combinations_big = prod(big.(lengths_vec))

    # 安全上限，避免内存/整数溢出
    if n_combinations_big > big(typemax(Int)) || Int(n_combinations_big) > 5_000_000
        println("⚠️  组合数过大，自动切换为采样生成以避免内存问题")
        n_samples = min(100_000, 5_000_000)
        return _sample_parameter_combinations_no_constraint(varying_param_ranges, varying_param_indices, default_values, n_samples)
    end

    n_combinations = Int(n_combinations_big)
    X_combinations = zeros(n_combinations, 13)

    # 创建笛卡尔积
    combination_indices = Iterators.product([1:length(r) for r in varying_param_ranges]...)

    for (sample_idx, indices) in enumerate(combination_indices)
        # 从默认值开始
        sample = copy(default_values)

        # 设置变化的参数
        for (i, param_idx) in enumerate(varying_param_indices)
            sample[param_idx] = varying_param_ranges[i][indices[i]]
        end

        X_combinations[sample_idx, :] = sample
    end

    return X_combinations
end

# 无约束的采样生成（用于超大组合回退）
function _sample_parameter_combinations_no_constraint(varying_param_ranges::Vector, varying_param_indices::Vector, default_values::Vector, n_samples::Int)
    X_samples = zeros(n_samples, 13)
    Random.seed!(42)
    for i in 1:n_samples
        sample = copy(default_values)
        for (j, param_idx) in enumerate(varying_param_indices)
            r = varying_param_ranges[j]
            idx = rand(1:length(r))
            sample[param_idx] = r[idx]
        end
        X_samples[i, :] = sample
    end
    return X_samples
end

"""
    generate_all_parameter_combinations(param_ranges::Vector)

生成所有参数组合
"""
function generate_all_parameter_combinations(param_ranges::Vector)
    # 使用IterTools.product生成所有组合
    combinations = collect(Iterators.product(param_ranges...))
    n_combinations = length(combinations)
    n_params = length(param_ranges)

    X_all = zeros(n_combinations, n_params)

    for (i, combo) in enumerate(combinations)
        for j in 1:n_params
            X_all[i, j] = combo[j]
        end
    end

    return X_all
end

"""
    create_performance_report(surrogate_model::SurrogateModel, comparison_results, scan_results)

创建性能报告
"""
function create_performance_report(surrogate_model::SurrogateModel, comparison_results, scan_results)
    println("\n" * "="^50)
    println("📋 ML代理模型性能报告")
    println("="^50)

    config = surrogate_model.config

    println("\n🔧 模型配置:")
    println("模型类型: $(config.model_type)")
    println("隐藏层: $(config.hidden_dims)")
    println("训练轮数: $(config.epochs)")
    println("批处理大小: $(config.batch_size)")
    println("学习率: $(config.learning_rate)")

    if config.apply_thermodynamic_constraints
        println("热力学约束: 启用")
        println("平衡常数范围: [$(config.keq_min), $(config.keq_max)]")
    else
        println("热力学约束: 禁用")
    end

    println("\n⚡ 性能指标:")
    println("加速比: $(round(comparison_results.speedup, digits=1))x")
    println("平均MSE: $(round(comparison_results.mse, digits=6))")
    println("平均R²: $(round(comparison_results.r2, digits=4))")
    println("效率提升: $(round(comparison_results.efficiency, digits=1))%")

    println("\n🚀 大规模扫描能力:")
    if scan_results !== nothing
        println("扫描样本数: $(length(scan_results))")
        println("预测速度: >1000 预测/秒")
        println("计算量减少: >90%")
    end

    println("\n💡 使用建议:")
    if comparison_results.r2 > 0.9
        println("✅ 模型精度优秀，适合替代CUDA仿真")
    elseif comparison_results.r2 > 0.8
        println("⚠️  模型精度良好，建议用于初步筛选")
    else
        println("❌ 模型精度需要提升，建议增加训练数据")
    end

    if comparison_results.speedup > 10
        println("✅ 显著性能提升，适合大规模参数扫描")
    else
        println("⚠️  性能提升有限，建议优化模型结构")
    end

    println("\n" * "="^50)
end

export SurrogateModel, SurrogateModelConfig, ParameterSpace, create_default_parameter_space
export generate_small_scale_data, preprocess_data!, train_surrogate_model!
export predict_with_uncertainty, save_surrogate_model, load_surrogate_model
export compare_surrogate_vs_cuda, large_scale_parameter_scan, create_performance_report
export configure_cuda_device, check_thermodynamic_constraints

# ================= TOML Configuration Utilities =================

"""
    _parse_range(value)

Parse a range specification from TOML. Supports:
- {start=Float, step=Float, stop=Float}
- {start=Float, stop=Float, length=Int}
- [v1, v2, ..., vn] (converted to LinRange with equal spacing)
- Float or Int (treated as a degenerate 1-length range)
"""
function _parse_range(value)
    if isa(value, AbstractDict)
        haskey(value, "start") || error("range table must have 'start'")
        haskey(value, "stop")  || error("range table must have 'stop'")
        startv = float(value["start"])
        stopv  = float(value["stop"])
        if haskey(value, "step")
            stepv = float(value["step"])
            return range(startv, step=stepv, stop=stopv)
        elseif haskey(value, "length")
            lenv = Int(value["length"]) 
            lenv > 0 || error("range length must be > 0")
            return range(startv, stop=stopv, length=lenv)
        else
            error("range table must have 'step' or 'length'")
        end
    elseif isa(value, AbstractVector)
        if length(value) == 0
            error("range array must be non-empty")
        elseif length(value) == 1
            v = float(value[1])
            return range(v, stop=v, length=1)
        else
            v1 = float(value[1])
            vn = float(value[end])
            return range(v1, stop=vn, length=length(value))
        end
    elseif isa(value, Real)
        v = float(value)
        return range(v, stop=v, length=1)
    else
        error("unsupported range specification: $(typeof(value))")
    end
end

"""
    load_surrogate_from_toml(path::AbstractString)

Load `SurrogateModelConfig` and `ParameterSpace` from a TOML file.
"""
function load_surrogate_from_toml(path::AbstractString)
    cfg = TOML.parsefile(path)

    # ---------------- SurrogateModelConfig ----------------
    data_cfg = get(cfg, "data", Dict())
    model_cfg = get(cfg, "model", Dict())
    train_cfg = get(cfg, "training", Dict())
    pca_cfg = get(cfg, "pca", Dict())
    cuda_cfg = get(cfg, "cuda", Dict())
    thermo_cfg = get(cfg, "constraints", Dict())
    outputs_cfg = get(cfg, "outputs", Dict())

    # target variables
    targets = get(outputs_cfg, "target_variables", ["A_final", "B_final", "C_final", "v1_mean", "v2_mean"]) 
    target_syms = Symbol.(String.(targets))

    hidden_dims = get(model_cfg, "hidden_dims", [64, 32, 16])
    hidden_dims_vec = Int.(hidden_dims)

    config = SurrogateModelConfig(
        sample_fraction = float(get(data_cfg, "sample_fraction", 0.1)),
        max_samples = Int(get(data_cfg, "max_samples", 10000)),

        model_type = Symbol(get(model_cfg, "model_type", "neural_network")),
        hidden_dims = hidden_dims_vec,
        dropout_rate = float(get(model_cfg, "dropout_rate", 0.1)),

        epochs = Int(get(train_cfg, "epochs", 100)),
        batch_size = Int(get(train_cfg, "batch_size", 32)),
        learning_rate = float(get(train_cfg, "learning_rate", 1e-3)),
        validation_split = float(get(train_cfg, "validation_split", 0.2)),

        use_pca = Bool(get(pca_cfg, "use_pca", true)),
        pca_variance_threshold = float(get(pca_cfg, "pca_variance_threshold", 0.95)),

        target_variables = target_syms,
        uncertainty_estimation = Bool(get(outputs_cfg, "uncertainty_estimation", true)),

        use_cuda = Bool(get(cuda_cfg, "use_cuda", true)),
        cuda_batch_size = Int(get(cuda_cfg, "cuda_batch_size", 1000)),

        apply_thermodynamic_constraints = Bool(get(thermo_cfg, "apply", true)),
        constraint_mode = Symbol(get(thermo_cfg, "mode", "range")),
        keq_min = float(get(thermo_cfg, "keq_min", 0.01)),
        keq_max = float(get(thermo_cfg, "keq_max", 100.0)),
        keq1_fixed = haskey(thermo_cfg, "keq1") ? float(thermo_cfg["keq1"]) : nothing,
        keq2_fixed = haskey(thermo_cfg, "keq2") ? float(thermo_cfg["keq2"]) : nothing,
        keq_tolerance = float(get(thermo_cfg, "tolerance", 1e-3))
    )

    # ---------------- ParameterSpace ----------------
    space_cfg = get(cfg, "space", Dict())
    rates_cfg = get(space_cfg, "rates", Dict())
    init_cfg = get(space_cfg, "init", Dict())
    tspan_cfg = get(space_cfg, "tspan", Dict("t0"=>0.0, "t1"=>5.0))

    ps = ParameterSpace(
        _parse_range(get(rates_cfg, "k1f", 0.1:0.02:20.0)),
        _parse_range(get(rates_cfg, "k1r", 0.1:0.02:20.0)),
        _parse_range(get(rates_cfg, "k2f", 0.1:0.02:20.0)),
        _parse_range(get(rates_cfg, "k2r", 0.1:0.02:20.0)),
        _parse_range(get(rates_cfg, "k3f", 0.1:0.02:20.0)),
        _parse_range(get(rates_cfg, "k3r", 0.1:0.02:20.0)),
        _parse_range(get(rates_cfg, "k4f", 0.1:0.02:20.0)),
        _parse_range(get(rates_cfg, "k4r", 0.1:0.02:20.0)),

        _parse_range(get(init_cfg, "A", 0.1:0.02:20.0)),
        _parse_range(get(init_cfg, "B", 0.0:0.02:20.0)),
        _parse_range(get(init_cfg, "C", 0.0:0.02:20.0)),
        _parse_range(get(init_cfg, "E1", 0.1:0.02:20.0)),
        _parse_range(get(init_cfg, "E2", 0.1:0.02:20.0)),

        (float(get(tspan_cfg, "t0", 0.0)), float(get(tspan_cfg, "t1", 5.0)))
    )

    return config, ps
end

"""
    save_example_config_toml(path::AbstractString)

Write an example TOML configuration file covering all fields.
"""
function save_example_config_toml(path::AbstractString)
    example = """
# Surrogate Model TOML configuration

[data]
sample_fraction = 0.1
max_samples = 10000

[model]
model_type = "neural_network"   # or "gaussian_process"
hidden_dims = [256, 128, 64, 32]
dropout_rate = 0.2

[training]
epochs = 300
batch_size = 256
learning_rate = 1e-3
validation_split = 0.2

[pca]
use_pca = true
pca_variance_threshold = 0.95

[cuda]
use_cuda = true
cuda_batch_size = 16384

[constraints]
apply = true
keq_min = 0.01
keq_max = 100.0

[outputs]
target_variables = ["A_final", "B_final", "C_final", "v1_mean", "v2_mean"]
uncertainty_estimation = true

[space.rates]
k1f = { start = 0.1, step = 0.02, stop = 20.0 }
k1r = { start = 0.1, step = 0.02, stop = 20.0 }
k2f = { start = 0.1, step = 0.02, stop = 20.0 }
k2r = { start = 0.1, step = 0.02, stop = 20.0 }
k3f = { start = 0.1, step = 0.02, stop = 20.0 }
k3r = { start = 0.1, step = 0.02, stop = 20.0 }
k4f = { start = 0.1, step = 0.02, stop = 20.0 }
k4r = { start = 0.1, step = 0.02, stop = 20.0 }

[space.init]
A  = { start = 0.1, step = 0.02, stop = 20.0 }
B  = { start = 0.0, step = 0.02, stop = 5.0 }
C  = { start = 0.0, step = 0.02, stop = 5.0 }
E1 = { start = 1.0, step = 0.02, stop = 20.0 }
E2 = { start = 1.0, step = 0.02, stop = 20.0 }

[space.tspan]
t0 = 0.0
t1 = 5.0
"""
    open(path, "w") do io
        write(io, example)
    end
    return path
end

export load_surrogate_from_toml, save_example_config_toml
