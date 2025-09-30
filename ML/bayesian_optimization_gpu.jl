"""
CUDA多GPU并行贝叶斯优化模块 (GPU-Accelerated Bayesian Optimization)

基于gpu_parallel_optimized.jl实现的高性能贝叶斯优化，解决单线程评估瓶颈

主要改进：
1. 批量GPU并行目标函数评估
2. 智能批次管理和内存优化
3. 多GPU异步处理支持
4. 保持原有贝叶斯优化算法不变
5. 兼容现有配置和接口

设计哲学：
- "好品味"：批量评估消除单点串行的特殊情况
- "Never break userspace"：完全兼容现有BayesianOptimizer接口
- 实用主义：GPU加速真正的计算瓶颈（ODE求解）
- 简洁执念：最小化CPU-GPU数据传输次数
"""

using CUDA
using Statistics
using LinearAlgebra
using Random
using Plots
using JLD2
using Printf
using TOML
using Distributions

# 引入依赖模块
include("bayesian_optimization.jl")
include("gpu_parallel_optimized.jl")
include("surrogate_model.jl")

# 在无显示环境下，强制GR走headless模式并避免字体问题
try
    ENV["GKSwstype"] = "100"
    default(fontfamily="sans")
catch
end

"""
    _row_key(x::AbstractVector{Float64})

为浮点向量生成稳定的去重键，避免浮点比较误差。
"""
function _row_key(x::AbstractVector{Float64})
    io = IOBuffer()
    for i in axes(x, 1)
        @printf(io, "%.12g,", x[i])
    end
    return String(take!(io))
end

"""
    robust_optimize_acquisition(base::BayesianOptimizer; n_starts=8)

多启动采集函数优化，失败则回退为 nothing。
"""
function robust_optimize_acquisition(base::BayesianOptimizer; n_starts::Int=8)
    best_x = nothing
    best_val = -Inf
    for _ in 1:n_starts
        x = nothing
        try
            x = optimize_acquisition_function(base)
        catch
            x = nothing
        end
        if x === nothing
            continue
        end
        # 粗略用采集值评估排序标准
        val = try
            evaluate_acquisition_function(base, x)
        catch
            -Inf
        end
        if isfinite(val) && val > best_val
            best_val = val
            best_x = x
        end
    end
    return best_x
end

"""
deduplicate_training_data!(base_optimizer::BayesianOptimizer)

移除完全重复的训练样本，避免Kriging因重复样本报错。
"""
function deduplicate_training_data!(base_optimizer::BayesianOptimizer)
    X = base_optimizer.X_evaluated
    y = base_optimizer.y_evaluated
    n = size(X, 1)
    if n <= 1
        return
    end
    seen = Set{String}()
    keep_indices = Int[]
    keep_indices_sizehint = sizehint!(keep_indices, n)
    for i in 1:n
        k = _row_key(view(X, i, :))
        if !(k in seen)
            push!(seen, k)
            push!(keep_indices, i)
        end
    end
    if length(keep_indices) < n
        base_optimizer.X_evaluated = X[keep_indices, :]
        base_optimizer.y_evaluated = y[keep_indices]
    end
end

 

"""
    _lhs_sample(n::Int, ranges) -> Matrix{Float64}

简易Latin Hypercube Sampling，范围按参数区间映射。
"""
function _lhs_sample(n::Int, ranges)
    d = length(ranges)
    X = zeros(Float64, n, d)
    for j in 1:d
        lo, hi = minimum(ranges[j]), maximum(ranges[j])
        # 均匀分桶并随机打乱
        edges = range(0.0, 1.0; length = n + 1)
        # 在每个桶内选择一个随机点
        vals = [rand(edges[i]:(edges[i+1])) for i in 1:n]
        shuffle!(vals)
        for i in 1:n
            X[i, j] = lo + vals[i] * (hi - lo)
        end
    end
    return X
end

"""
    GPUBayesianConfig

GPU加速贝叶斯优化配置
"""
Base.@kwdef struct GPUBayesianConfig
    # 继承基础贝叶斯配置
    base_config::BayesianOptimizationConfig
    
    # GPU并行配置
    gpu_config::GPUParallelConfig = default_gpu_config()
    
    # 批量评估配置
    batch_evaluation::Bool = true          # 启用批量评估
    min_batch_size::Int = 10              # 最小批次大小
    max_batch_size::Int = 100             # 最大批次大小
    adaptive_batching::Bool = true         # 自适应批次大小
    
    # 内存管理
    gpu_memory_threshold::Float64 = 0.8   # GPU内存使用阈值
    auto_memory_management::Bool = true    # 自动内存管理
    
    # 性能监控
    profile_gpu_performance::Bool = false  # GPU性能分析
    track_memory_usage::Bool = true       # 内存使用跟踪
    
    # 容错配置
    gpu_fallback_enabled::Bool = true     # GPU失败时CPU回退
    max_gpu_retries::Int = 3              # GPU重试次数
end

"""
    default_gpu_bayesian_config(base_config::BayesianOptimizationConfig)

创建默认GPU贝叶斯配置
"""
function default_gpu_bayesian_config(base_config::BayesianOptimizationConfig)
    gpu_config = default_gpu_config()
    # 针对贝叶斯优化调整GPU配置
    gpu_config = GPUParallelConfig(
        gpu_config.use_multi_gpu,
        min(2000, 500),  # 适中的批次大小
        0.7,             # 保守的内存使用
        :GPUTsit5,
        1e-6, 1e-3, 10000,
        false,           # 同步处理更稳定
        true,
        false,           # 减少调试输出
        false
    )
    
    return GPUBayesianConfig(
        base_config = base_config,
        gpu_config = gpu_config,
        batch_evaluation = true,
        min_batch_size = 10,
        max_batch_size = 100,
        adaptive_batching = true,
        gpu_memory_threshold = 0.7,
        auto_memory_management = true,
        profile_gpu_performance = false,
        track_memory_usage = true,
        gpu_fallback_enabled = true,
        max_gpu_retries = 3
    )
end

"""
    GPUBayesianOptimizer

GPU加速的贝叶斯优化器
"""
mutable struct GPUBayesianOptimizer
    # 继承基础优化器组件
    base_optimizer::BayesianOptimizer
    gpu_config::GPUBayesianConfig
    
    # GPU求解器
    gpu_solver::Union{Nothing, OptimizedGPUSolver}
    
    # 批量管理
    current_batch_size::Int
    batch_history::Vector{Int}
    
    # 性能监控
    gpu_evaluation_times::Vector{Float64}
    cpu_evaluation_times::Vector{Float64}
    memory_usage_history::Vector{Float64}
    
    # 错误处理
    gpu_failure_count::Int
    fallback_mode::Bool
    consecutive_invalid_batches::Int
    
    function GPUBayesianOptimizer(gpu_config::GPUBayesianConfig, param_space::ParameterSpace)
        # 创建基础优化器
        base_optimizer = BayesianOptimizer(gpu_config.base_config, param_space)
        
        optimizer = new()
        optimizer.base_optimizer = base_optimizer
        optimizer.gpu_config = gpu_config
        optimizer.gpu_solver = nothing
        optimizer.current_batch_size = gpu_config.min_batch_size
        optimizer.batch_history = Int[]
        optimizer.gpu_evaluation_times = Float64[]
        optimizer.cpu_evaluation_times = Float64[]
        optimizer.memory_usage_history = Float64[]
        optimizer.gpu_failure_count = 0
        optimizer.fallback_mode = false
        optimizer.consecutive_invalid_batches = 0
        
        # 初始化GPU求解器
        initialize_gpu_solver!(optimizer)
        
        return optimizer
    end
end

"""
    fit_gp_with_filtered_data!(opt::GPUBayesianOptimizer)

在拟合GP前：去重 + 过滤惩罚/非有限样本，仅用有效数据训练；拟合后恢复完整数据。
"""
function fit_gp_with_filtered_data!(opt::GPUBayesianOptimizer)
    base = opt.base_optimizer
    # 先去重
    deduplicate_training_data!(base)
    # 备份
    X_full = base.X_evaluated
    y_full = base.y_evaluated
    # 过滤有效样本
    penalty = opt.gpu_config.base_config.constraint_penalty
    valid_idx = findall(i -> isfinite(y_full[i]) && y_full[i] != penalty, 1:length(y_full))
    if !isempty(valid_idx)
        base.X_evaluated = X_full[valid_idx, :]
        y_valid = y_full[valid_idx]
        # 针对目标做稳健标准化，提升GP拟合稳定性
        μ = mean(y_valid)
        σ = std(y_valid)
        if !(isfinite(σ) && σ > 1e-8)
            σ = 1.0
        end
        y_std = (y_valid .- μ) ./ σ
        # 微小抖动，避免奇异
        y_std .+= (1e-6 .* randn(length(y_std)))
        base.y_evaluated = y_std
    else
        # 无有效样本，直接用全部数据拟合（可能为空由下游处理）
        base.X_evaluated = X_full
        base.y_evaluated = y_full
    end
    # 拟合
    gp_model = fit_gp_model(base)
    opt.base_optimizer.gp_model = gp_model
    # 恢复完整数据
    base.X_evaluated = X_full
    base.y_evaluated = y_full
end

"""
    initialize_gpu_solver!(optimizer::GPUBayesianOptimizer)

初始化GPU求解器
"""
function initialize_gpu_solver!(optimizer::GPUBayesianOptimizer)
    try
        if CUDA.functional()
            println("🚀 初始化GPU贝叶斯优化器...")
            optimizer.gpu_solver = OptimizedGPUSolver(optimizer.gpu_config.gpu_config)
            println("✅ GPU求解器初始化成功")
        else
            println("⚠️  CUDA不可用，启用CPU回退模式")
            optimizer.fallback_mode = true
        end
    catch e
        println("❌ GPU初始化失败: $e")
        println("🔄 启用CPU回退模式")
        optimizer.fallback_mode = true
        optimizer.gpu_failure_count += 1
    end
end

"""
    create_gpu_objective_function(optimizer::GPUBayesianOptimizer)

创建GPU加速的目标函数（支持批量评估）
"""
function create_gpu_objective_function(optimizer::GPUBayesianOptimizer)
    config = optimizer.gpu_config.base_config
    param_space = optimizer.base_optimizer.param_space
    
    # 单点评估函数（保持兼容性）
    function objective_single(x::Vector{Float64})
        return evaluate_batch_gpu([x], optimizer)[1]
    end
    
    # 批量评估函数（新增）
    function objective_batch(X::Matrix{Float64})
        return evaluate_batch_gpu(X, optimizer)
    end
    
    # 返回包含两种接口的函数对象
    return (single=objective_single, batch=objective_batch)
end

"""
    evaluate_batch_gpu(X::Matrix{Float64}, optimizer::GPUBayesianOptimizer)

GPU批量评估目标函数
"""
function evaluate_batch_gpu(X::Matrix{Float64}, optimizer::GPUBayesianOptimizer)
    if size(X, 1) == 0
        return Float64[]
    end
    
    config = optimizer.gpu_config.base_config
    param_space = optimizer.base_optimizer.param_space
    n_samples = size(X, 1)
    
    start_time = time()
    
    # 检查是否使用GPU
    if optimizer.fallback_mode || optimizer.gpu_solver === nothing
        return evaluate_batch_cpu(X, optimizer)
    end
    
    try
        # 自适应批次大小管理
        if optimizer.gpu_config.adaptive_batching
            optimizer.current_batch_size = calculate_optimal_gpu_batch_size(optimizer, n_samples)
        end
        
        # 构建扩展的参数矩阵（包含初始条件）
        X_extended = prepare_extended_parameter_matrix(X, config, param_space)
        
        # GPU批量求解
        target_vars = [config.target_variable]
        tspan = param_space.tspan
        
        if optimizer.gpu_config.track_memory_usage
            try
                initial_memory = CUDA.available_memory()
            catch
                initial_memory = 0
            end
        end
        
        # 使用GPU求解器进行批量计算
        results = solve_batch_gpu_optimized(
            optimizer.gpu_solver, 
            X_extended, 
            tspan, 
            target_vars
        )
        
        # 提取目标值
        objective_values = results[:, 1]  # 第一列是目标变量
        
        # 应用约束和优化方向
        objective_values = process_objective_values(objective_values, X, config, param_space)
        
        # 记录性能
        evaluation_time = time() - start_time
        push!(optimizer.gpu_evaluation_times, evaluation_time)
        push!(optimizer.batch_history, n_samples)
        
        if optimizer.gpu_config.track_memory_usage
            try
                final_memory = CUDA.available_memory()
                memory_used = (initial_memory - final_memory) / 1e9  # GB
                push!(optimizer.memory_usage_history, memory_used)
            catch
                # 忽略内存统计失败
            end
        end
        
        if optimizer.gpu_config.gpu_config.verbose
            throughput = n_samples / evaluation_time
            println("🚀 GPU批量评估: $(n_samples)样本, $(round(evaluation_time, digits=2))s, $(round(throughput, digits=1))样本/秒")
        end
        
        return objective_values
        
    catch e
        println("⚠️  GPU评估失败: $e")
        optimizer.gpu_failure_count += 1
        
        # 判断是否需要切换到CPU回退
        if optimizer.gpu_failure_count >= optimizer.gpu_config.max_gpu_retries
            println("🔄 GPU失败次数过多，切换到CPU模式")
            optimizer.fallback_mode = true
        end
        
        return evaluate_batch_cpu(X, optimizer)
    end
end

"""
    evaluate_batch_cpu(X::Matrix{Float64}, optimizer::GPUBayesianOptimizer)

CPU回退批量评估
"""
function evaluate_batch_cpu(X::Matrix{Float64}, optimizer::GPUBayesianOptimizer)
    config = optimizer.gpu_config.base_config
    param_space = optimizer.base_optimizer.param_space
    n_samples = size(X, 1)
    
    start_time = time()
    
    # 创建标准的单点目标函数
    standard_objective = create_objective_function(optimizer.base_optimizer)
    
    # 串行评估每个点
    objective_values = zeros(Float64, n_samples)
    for i in 1:n_samples
        objective_values[i] = standard_objective(X[i, :])
    end
    
    # 记录CPU性能
    evaluation_time = time() - start_time
    push!(optimizer.cpu_evaluation_times, evaluation_time)
    
    if optimizer.gpu_config.gpu_config.verbose
        throughput = n_samples / evaluation_time
        println("💻 CPU批量评估: $(n_samples)样本, $(round(evaluation_time, digits=2))s, $(round(throughput, digits=1))样本/秒")
    end
    
    return objective_values
end

"""
    prepare_extended_parameter_matrix(X::Matrix{Float64}, config, param_space)

准备扩展的参数矩阵（包含反应常数和初始条件）
"""
function prepare_extended_parameter_matrix(X::Matrix{Float64}, config, param_space)
    n_samples = size(X, 1)
    
    # 参数顺序：[k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, A, B, C, E1, E2]
    # X 应该包含这13个参数
    
    if size(X, 2) != 13
        error("参数矩阵X应该有13列: [k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, A, B, C, E1, E2]")
    end
    
    return X  # 直接返回，因为X已经包含所有必要参数
end

"""
    process_objective_values(values::Vector{Float64}, X::Matrix{Float64}, config, param_space)

处理目标值（应用约束、优化方向等）
"""
function process_objective_values(values::Vector{Float64}, X::Matrix{Float64}, config, param_space)
    n_samples = length(values)
    processed_values = copy(values)
    
    for i in 1:n_samples
        # 检查约束
        if config.apply_constraints
            if !check_parameter_constraints(X[i, :], param_space, config)
                processed_values[i] = config.constraint_penalty
                continue
            end
        end
        
        # 检查有效性
        if !isfinite(processed_values[i])
            processed_values[i] = config.constraint_penalty
            continue
        end
        
        # 应用优化方向
        if config.optimization_direction == :minimize
            processed_values[i] = -processed_values[i]
        end
    end
    
    return processed_values
end

"""
    calculate_optimal_gpu_batch_size(optimizer::GPUBayesianOptimizer, n_samples::Int)

计算最优GPU批次大小
"""
function calculate_optimal_gpu_batch_size(optimizer::GPUBayesianOptimizer, n_samples::Int)
    config = optimizer.gpu_config
    
    # 基于GPU内存状态调整
    if config.auto_memory_management && CUDA.functional()
        try
            free_mem = CUDA.available_memory()
            total_mem = CUDA.totalmem(CUDA.device())
            memory_ratio = free_mem / total_mem
            if memory_ratio < config.gpu_memory_threshold
                new_batch_size = max(config.min_batch_size, optimizer.current_batch_size ÷ 2)
            else
                new_batch_size = min(config.max_batch_size, optimizer.current_batch_size * 2)
            end
        catch
            new_batch_size = optimizer.current_batch_size
        end
    else
        new_batch_size = optimizer.current_batch_size
    end
    
    # 不超过待评估样本数
    return min(new_batch_size, n_samples, config.max_batch_size)
end

"""
    run_gpu_bayesian_optimization!(optimizer::GPUBayesianOptimizer)

运行GPU加速的贝叶斯优化
"""
function run_gpu_bayesian_optimization!(optimizer::GPUBayesianOptimizer)
    base_config = optimizer.gpu_config.base_config
    gpu_config = optimizer.gpu_config
    
    println("🚀 开始GPU加速贝叶斯优化...")
    println("🎯 优化目标: $(base_config.target_variable)")
    println("🔄 迭代次数: $(base_config.n_iterations)")
    println("💾 GPU模式: $(optimizer.fallback_mode ? "CPU回退" : "GPU加速")")
    
    # 初始化基础优化器
    initialize_optimizer!(optimizer.base_optimizer)
    
    # 创建GPU目标函数
    objective_functions = create_gpu_objective_function(optimizer)
    
    # 优化主循环
    for iter in 1:base_config.n_iterations
        println("\n--- GPU迭代 $iter/$(base_config.n_iterations) ---")
        
        # 1. 拟合前：去重 + 有效样本过滤，提高采集函数稳定性
        fit_gp_with_filtered_data!(optimizer)
        
        # 2. 优化采集函数（批量候选点生成）
        candidate_points = generate_candidate_batch(optimizer)
        
        if isempty(candidate_points)
            println("⚠️  候选点生成失败，跳过此次迭代")
            continue
        end
        
        # 3. 批量评估候选点
        batch_size = size(candidate_points, 1)
        println("📊 批量评估 $(batch_size) 个候选点...")
        
        candidate_values = evaluate_batch_gpu(candidate_points, optimizer)
        # 如果GPU批量全部为无效/惩罚值，尝试CPU整体回退
        if all(v -> !isfinite(v) || v == optimizer.gpu_config.base_config.constraint_penalty, candidate_values)
            if !optimizer.fallback_mode
                println("⚠️  GPU批量评估得到全无效值，临时使用CPU评估回退")
            end
            # 统计为一次GPU失败，并考虑切换到CPU模式
            optimizer.gpu_failure_count += 1
            optimizer.consecutive_invalid_batches += 1
            if optimizer.gpu_failure_count >= optimizer.gpu_config.max_gpu_retries
                optimizer.fallback_mode = true
            end
            # 连续多次无效时，缩小批次并回退CPU评估
            if optimizer.consecutive_invalid_batches >= 2
                optimizer.current_batch_size = max(optimizer.gpu_config.min_batch_size, optimizer.current_batch_size ÷ 2)
                optimizer.fallback_mode = true
            end
            candidate_values = evaluate_batch_cpu(candidate_points, optimizer)
        else
            optimizer.consecutive_invalid_batches = 0
            # 对无效子集混合重评：仅对惩罚/非有限元素用CPU重评
            penalty = optimizer.gpu_config.base_config.constraint_penalty
            for i in eachindex(candidate_values)
                if !isfinite(candidate_values[i]) || candidate_values[i] == penalty
                    candidate_values[i] = evaluate_batch_cpu(candidate_points[i:i, :], optimizer)[1]
                end
            end
        end
        
        # 4. 选择最佳候选点
        best_candidate_idx = argmax(candidate_values)
        next_x = candidate_points[best_candidate_idx, :]
        next_y = candidate_values[best_candidate_idx]
        
        # 5. 更新历史记录
        optimizer.base_optimizer.X_evaluated = vcat(optimizer.base_optimizer.X_evaluated, next_x')
        optimizer.base_optimizer.y_evaluated = vcat(optimizer.base_optimizer.y_evaluated, next_y)
        
        # 6. 更新最优结果
        if next_y > optimizer.base_optimizer.best_y
            optimizer.base_optimizer.best_x = next_x
            optimizer.base_optimizer.best_y = next_y
            optimizer.base_optimizer.best_params = vector_to_params_dict(next_x, optimizer.base_optimizer.param_space)
            println("🎉 发现新的最优点! GPU目标值: $(round(next_y, digits=4))")
        else
            println("📊 当前点GPU目标值: $(round(next_y, digits=4))")
        end
        
        # 7. 记录采集函数值
        acquisition_value = evaluate_acquisition_function(optimizer.base_optimizer, next_x)
        optimizer.base_optimizer.acquisition_history = vcat(optimizer.base_optimizer.acquisition_history, acquisition_value)
        
        # 8. 性能监控和调整
        if gpu_config.adaptive_batching
            update_batch_size_strategy(optimizer)
        end
        
        # 9. 中间结果保存
        if base_config.save_intermediate && iter % 10 == 0
            save_gpu_intermediate_results(optimizer, iter)
        end
        
        # 10. 内存清理
        if gpu_config.auto_memory_management && !optimizer.fallback_mode
            if iter % 5 == 0
                CUDA.reclaim()
            end
        end
    end
    
    println("\n🎉 GPU贝叶斯优化完成!")
    analyze_gpu_performance(optimizer)
    
    return optimizer
end

"""
    generate_candidate_batch(optimizer::GPUBayesianOptimizer)

生成候选点批次（改进的采集函数优化）
"""
function generate_candidate_batch(optimizer::GPUBayesianOptimizer)
    config = optimizer.gpu_config
    param_space = optimizer.base_optimizer.param_space
    
    # 期望批次大小（针对采集函数建议小批次，但要确保数量足够）
    target_batch = min(config.max_batch_size, 20)
    min_batch = max(10, config.min_batch_size)
    n_dims = length(get_parameter_ranges(param_space))
    
    # 收集候选（混合策略：采集函数 + 随机 + 最优点附近扰动）
    seen = Set{String}()
    collected = Vector{Vector{Float64}}()
    
    # 1) 采集函数尝试（限次，避免刷屏）
    acq_trials = min(5, target_batch)
    for _ in 1:acq_trials
        if length(collected) >= target_batch
            break
        end
        candidate = robust_optimize_acquisition(optimizer.base_optimizer; n_starts=6)
        if candidate === nothing
            continue
        end
        k = _row_key(candidate)
        if !(k in seen) && check_parameter_constraints(candidate, param_space, config.base_config)
            push!(seen, k)
            push!(collected, candidate)
        end
    end
    
    # 2) 最优点附近的局部扰动（探索-开发混合）
    if length(collected) < target_batch && optimizer.base_optimizer.best_x !== nothing
        best_x = optimizer.base_optimizer.best_x
        # 扰动尺度：按参数范围的 5%
        ranges = get_parameter_ranges(param_space)
        sigma = [0.05 * (maximum(r) - minimum(r)) for r in ranges]
        local_attempts = 0
        while length(collected) < target_batch && local_attempts < target_batch * 5
            local_attempts += 1
            x = similar(best_x)
            for j in 1:n_dims
                x[j] = best_x[j] + randn() * sigma[j]
                # 投影回边界
                lo, hi = minimum(ranges[j]), maximum(ranges[j])
                if x[j] < lo
                    x[j] = lo + abs(x[j] - lo)
                elseif x[j] > hi
                    x[j] = hi - abs(x[j] - hi)
                end
                x[j] = clamp(x[j], lo, hi)
            end
            if check_parameter_constraints(x, param_space, config.base_config)
                k = _row_key(x)
                if !(k in seen)
                    push!(seen, k)
                    push!(collected, x)
                end
            end
        end
    end

    # 2b) 前K个历史最优的邻域扰动（增强开发能力）
    if length(collected) < target_batch
        K = min(5, size(optimizer.base_optimizer.X_evaluated, 1))
        if K > 0
            # 选取前K大y对应的X
            y_hist = optimizer.base_optimizer.y_evaluated
            idx_sorted = sortperm(y_hist, rev=true)
            ranges = get_parameter_ranges(param_space)
            sigma2 = [0.08 * (maximum(r) - minimum(r)) for r in ranges]
            added = 0
            for kidx in idx_sorted[1:K]
                center = vec(optimizer.base_optimizer.X_evaluated[kidx, :])
                for _ in 1:3
                    x = copy(center)
                    for j in 1:n_dims
                        x[j] = center[j] + randn() * sigma2[j]
                        lo, hi = minimum(ranges[j]), maximum(ranges[j])
                        x[j] = clamp(x[j], lo, hi)
                    end
                    if check_parameter_constraints(x, param_space, config.base_config)
                        k = _row_key(x)
                        if !(k in seen)
                            push!(seen, k)
                            push!(collected, x)
                            added += 1
                            if length(collected) >= target_batch
                                break
                            end
                        end
                    end
                end
                if length(collected) >= target_batch
                    break
                end
            end
        end
    end
    
    # 3) 随机补齐（优先用LHS，保证覆盖度）
    if length(collected) < target_batch
        ranges = get_parameter_ranges(param_space)
        need = target_batch - length(collected)
        if need > 0
            Xlhs = _lhs_sample(need, ranges)
            for i in 1:size(Xlhs, 1)
                x = vec(Xlhs[i, :])
                if check_parameter_constraints(x, param_space, config.base_config)
                    k = _row_key(x)
                    if !(k in seen)
                        push!(seen, k)
                        push!(collected, x)
                    end
                end
            end
        end
        # 若仍不足，少量均匀随机补齐
        random_attempts = 0
        while length(collected) < target_batch && random_attempts < target_batch * 10
            random_attempts += 1
            x = zeros(Float64, n_dims)
            for j in 1:n_dims
                lo, hi = minimum(ranges[j]), maximum(ranges[j])
                x[j] = lo + rand() * (hi - lo)
            end
            if check_parameter_constraints(x, param_space, config.base_config)
                k = _row_key(x)
                if !(k in seen)
                    push!(seen, k)
                    push!(collected, x)
                end
            end
        end
        # 若采集优化未获得候选，静默回退为随机/LHS，不重复打印
    end
    
    if isempty(collected)
        return Matrix{Float64}(undef, 0, 0)
    end
    
    # 若仍不足最小批次，尽力补齐（再多尝试一些随机点）
    if length(collected) < min_batch
        ranges = get_parameter_ranges(param_space)
        need = min_batch - length(collected)
        Xlhs2 = _lhs_sample(need, ranges)
        for i in 1:size(Xlhs2, 1)
            x = vec(Xlhs2[i, :])
            if check_parameter_constraints(x, param_space, config.base_config)
                k = _row_key(x)
                if !(k in seen)
                    push!(seen, k)
                    push!(collected, x)
                end
            end
        end
    end
    
    # 输出矩阵
    candidate_matrix = zeros(Float64, length(collected), n_dims)
    for (i, x) in enumerate(collected)
        candidate_matrix[i, :] = x
    end
    return candidate_matrix
end

"""
    update_batch_size_strategy(optimizer::GPUBayesianOptimizer)

更新批次大小策略
"""
function update_batch_size_strategy(optimizer::GPUBayesianOptimizer)
    config = optimizer.gpu_config
    
    # 基于最近的性能调整批次大小
    if length(optimizer.gpu_evaluation_times) >= 3
        recent_times = optimizer.gpu_evaluation_times[end-2:end]
        recent_batches = optimizer.batch_history[end-2:end]
        
        # 计算平均吞吐量
        avg_throughput = mean(recent_batches ./ recent_times)
        
        # 动态调整策略
        if avg_throughput > 50  # 吞吐量较高，可以增大批次
            optimizer.current_batch_size = min(config.max_batch_size, 
                                              Int(round(optimizer.current_batch_size * 1.2)))
        elseif avg_throughput < 10  # 吞吐量较低，减小批次
            optimizer.current_batch_size = max(config.min_batch_size,
                                              Int(round(optimizer.current_batch_size * 0.8)))
        end
    end
end

"""
    analyze_gpu_performance(optimizer::GPUBayesianOptimizer)

分析GPU性能表现
"""
function analyze_gpu_performance(optimizer::GPUBayesianOptimizer)
    println("\n📊 GPU性能分析:")
    
    # GPU vs CPU性能对比
    if !isempty(optimizer.gpu_evaluation_times) && !isempty(optimizer.cpu_evaluation_times)
        gpu_avg_time = mean(optimizer.gpu_evaluation_times)
        cpu_avg_time = mean(optimizer.cpu_evaluation_times)
        speedup = cpu_avg_time / gpu_avg_time
        
        println("  GPU平均评估时间: $(round(gpu_avg_time, digits=3))s")
        println("  CPU平均评估时间: $(round(cpu_avg_time, digits=3))s")
        println("  GPU加速比: $(round(speedup, digits=1))x")
    elseif !isempty(optimizer.gpu_evaluation_times)
        gpu_avg_time = mean(optimizer.gpu_evaluation_times)
        total_evaluations = sum(optimizer.batch_history)
        total_time = sum(optimizer.gpu_evaluation_times)
        avg_throughput = total_evaluations / total_time
        
        println("  GPU总评估时间: $(round(total_time, digits=2))s")
        println("  GPU总评估样本: $total_evaluations")
        println("  GPU平均吞吐量: $(round(avg_throughput, digits=1)) 样本/秒")
    end
    
    # 批次大小演化
    if !isempty(optimizer.batch_history)
        println("  批次大小范围: $(minimum(optimizer.batch_history)) - $(maximum(optimizer.batch_history))")
        println("  平均批次大小: $(round(mean(optimizer.batch_history), digits=1))")
    end
    
    # 内存使用情况
    if !isempty(optimizer.memory_usage_history)
        println("  平均GPU内存使用: $(round(mean(optimizer.memory_usage_history), digits=2)) GB")
        println("  峰值GPU内存使用: $(round(maximum(optimizer.memory_usage_history), digits=2)) GB")
    end
    
    # 错误统计
    if optimizer.gpu_failure_count > 0
        println("  GPU失败次数: $(optimizer.gpu_failure_count)")
        println("  当前模式: $(optimizer.fallback_mode ? "CPU回退" : "GPU加速")")
    end
end

"""
    save_gpu_intermediate_results(optimizer::GPUBayesianOptimizer, iteration::Int)

保存GPU优化中间结果
"""
function save_gpu_intermediate_results(optimizer::GPUBayesianOptimizer, iteration::Int)
    results_dir = "/home/ryankwok/Documents/TwoEnzymeSim/ML/result"
    
    if !isdir(results_dir)
        mkpath(results_dir)
    end
    
    filename = joinpath(results_dir, "gpu_bayesian_opt_iter_$(iteration).jld2")
    
    try
        jldsave(filename;
                # 基础优化结果
                X_evaluated = optimizer.base_optimizer.X_evaluated,
                y_evaluated = optimizer.base_optimizer.y_evaluated,
                best_x = optimizer.base_optimizer.best_x,
                best_y = optimizer.base_optimizer.best_y,
                best_params = optimizer.base_optimizer.best_params,
                
                # GPU性能数据
                gpu_evaluation_times = optimizer.gpu_evaluation_times,
                cpu_evaluation_times = optimizer.cpu_evaluation_times,
                batch_history = optimizer.batch_history,
                memory_usage_history = optimizer.memory_usage_history,
                current_batch_size = optimizer.current_batch_size,
                gpu_failure_count = optimizer.gpu_failure_count,
                fallback_mode = optimizer.fallback_mode,
                
                # 配置信息
                gpu_config = optimizer.gpu_config,
                iteration = iteration)
        
        println("💾 GPU中间结果已保存: $filename")
    catch e
        println("⚠️  保存GPU中间结果失败: $e")
    end
end

"""
    create_gpu_bayesian_workflow(config_path::String="config/bayesian_optimization_config.toml", section::String="single_objective")

创建GPU加速贝叶斯优化工作流程
"""
function create_gpu_bayesian_workflow(config_path::String="config/bayesian_optimization_config.toml", section::String="single_objective")
    println("🚀 GPU加速贝叶斯优化工作流程")
    println("="^60)
    
    # 第1步：加载配置
    println("\n📋 第1步：加载配置")
    base_config = load_bayesian_config(config_path, section)
    param_space = load_parameter_space_from_config(config_path)
    
    # 创建GPU配置
    gpu_config = default_gpu_bayesian_config(base_config)
    
    println("✅ GPU贝叶斯优化配置完成")
    println("🎯 优化目标: $(base_config.target_variable)")
    println("🔍 初始点数: $(base_config.n_initial_points)")
    println("🔄 优化迭代: $(base_config.n_iterations)")
    println("💾 GPU模式: $(CUDA.functional() ? "CUDA可用" : "CPU回退")")
    println("📊 批量大小: $(gpu_config.min_batch_size)-$(gpu_config.max_batch_size)")
    
    # 第2步：创建GPU优化器
    println("\n🏗️  第2步：创建GPU贝叶斯优化器")
    optimizer = GPUBayesianOptimizer(gpu_config, param_space)
    
    # 第3步：运行GPU优化
    println("\n🎯 第3步：GPU智能参数优化")
    run_gpu_bayesian_optimization!(optimizer)
    
    # 第4步：结果分析
    println("\n📊 第4步：结果分析")
    analyze_optimization_results(optimizer.base_optimizer)
    analyze_gpu_performance(optimizer)
    
    # 第5步：可视化结果
    println("\n📈 第5步：生成可视化")
    if base_config.plot_convergence
        plot_gpu_optimization_convergence(optimizer)
    end
    
    # 第6步：保存最终结果
    println("\n💾 第6步：保存GPU优化结果")
    save_gpu_optimization_results(optimizer)
    
    # 第7步：性能对比
    println("\n⚡ 第7步：GPU效率分析")
    compare_gpu_efficiency(optimizer)
    
    println("\n🎉 GPU贝叶斯优化工作流程完成!")
    
    return optimizer
end

"""
    plot_gpu_optimization_convergence(optimizer::GPUBayesianOptimizer)

绘制GPU优化收敛曲线
"""
function plot_gpu_optimization_convergence(optimizer::GPUBayesianOptimizer)
    base_optimizer = optimizer.base_optimizer
    y_values = base_optimizer.y_evaluated
    n_points = length(y_values)
    
    # 计算累积最优值
    cumulative_best = zeros(n_points)
    cumulative_best[1] = y_values[1]
    
    for i in 2:n_points
        cumulative_best[i] = max(cumulative_best[i-1], y_values[i])
    end
    
    # 绘制收敛曲线
    p1 = plot(1:n_points, cumulative_best, 
              xlabel="评估次数", ylabel="最优目标值", 
              title="GPU贝叶斯优化收敛曲线", 
              lw=2, label="GPU累积最优值", color=:blue)
    
    # 添加初始探索阶段标记
    vline!([base_optimizer.config.n_initial_points], 
           label="初始探索结束", color=:red, ls=:dash)
    
    # 保存图片
    results_dir = "/home/ryankwok/Documents/TwoEnzymeSim/ML/result"
    if !isdir(results_dir)
        mkpath(results_dir)
    end
    
    savefig(p1, joinpath(results_dir, "gpu_bayesian_convergence.png"))
    println("📁 已保存GPU收敛曲线: $(joinpath(results_dir, "gpu_bayesian_convergence.png"))")
    
    # 绘制GPU性能曲线
    if !isempty(optimizer.gpu_evaluation_times)
        p2 = plot(1:length(optimizer.gpu_evaluation_times), optimizer.gpu_evaluation_times,
                   xlabel="GPU评估批次", ylabel="评估时间 (秒)",
                   title="GPU批量评估性能",
                   lw=2, label="GPU评估时间", color=:green)
        
        # 添加批次大小信息
        batch_sizes = optimizer.batch_history
        if length(batch_sizes) == length(optimizer.gpu_evaluation_times)
            throughput = batch_sizes ./ optimizer.gpu_evaluation_times
            
            p3 = plot(1:length(throughput), throughput,
                     xlabel="GPU评估批次", ylabel="吞吐量 (样本/秒)",
                     title="GPU评估吞吐量",
                     lw=2, label="GPU吞吐量", color=:purple)
            
            # 组合图
            p_combined = plot(p2, p3, layout=(2,1), size=(800, 600))
            savefig(p_combined, joinpath(results_dir, "gpu_performance_curves.png"))
            println("📁 已保存GPU性能曲线: $(joinpath(results_dir, "gpu_performance_curves.png"))")
        end
    end
end

"""
    save_gpu_optimization_results(optimizer::GPUBayesianOptimizer)

保存GPU优化结果
"""
function save_gpu_optimization_results(optimizer::GPUBayesianOptimizer)
    results_path = "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/gpu_bayesian_optimization_results.jld2"
    
    try
        jldsave(results_path;
                # 基础优化结果
                config = optimizer.base_optimizer.config,
                param_space = optimizer.base_optimizer.param_space,
                X_evaluated = optimizer.base_optimizer.X_evaluated,
                y_evaluated = optimizer.base_optimizer.y_evaluated,
                best_x = optimizer.base_optimizer.best_x,
                best_y = optimizer.base_optimizer.best_y,
                best_params = optimizer.base_optimizer.best_params,
                acquisition_history = optimizer.base_optimizer.acquisition_history,
                
                # GPU特定结果
                gpu_config = optimizer.gpu_config,
                gpu_evaluation_times = optimizer.gpu_evaluation_times,
                cpu_evaluation_times = optimizer.cpu_evaluation_times,
                batch_history = optimizer.batch_history,
                memory_usage_history = optimizer.memory_usage_history,
                final_batch_size = optimizer.current_batch_size,
                gpu_failure_count = optimizer.gpu_failure_count,
                final_mode = optimizer.fallback_mode ? "CPU" : "GPU")
        
        println("✅ GPU优化结果已保存: $results_path")
        
        # 保存文件大小信息
        file_size_mb = round(filesize(results_path) / 1024^2, digits=1)
        println("📊 结果文件大小: $(file_size_mb) MB")
        
    catch e
        println("❌ 保存GPU结果失败: $e")
    end
end

"""
    compare_gpu_efficiency(optimizer::GPUBayesianOptimizer)

GPU效率对比分析
"""
function compare_gpu_efficiency(optimizer::GPUBayesianOptimizer)
    n_evaluated = size(optimizer.base_optimizer.X_evaluated, 1)
    
    println("⚡ GPU贝叶斯优化效率分析:")
    println("📊 GPU总评估次数: $n_evaluated")
    
    # 估算传统方法的计算量
    param_space = optimizer.base_optimizer.param_space
    ranges = get_parameter_ranges(param_space)
    
    # 网格搜索估算
    grid_points_coarse = 10^length(ranges)
    grid_points_fine = 20^length(ranges)
    
    println("🔲 等效粗网格点数: $(grid_points_coarse)")
    println("🔳 等效细网格点数: $(grid_points_fine)")
    
    # GPU加速效果
    if !isempty(optimizer.gpu_evaluation_times) && !isempty(optimizer.cpu_evaluation_times)
        gpu_total_time = sum(optimizer.gpu_evaluation_times)
        cpu_total_time = sum(optimizer.cpu_evaluation_times)
        speedup = cpu_total_time / gpu_total_time
        
        println("\n🚀 GPU加速效果:")
        println("  GPU总计算时间: $(round(gpu_total_time, digits=2))s")
        println("  CPU总计算时间: $(round(cpu_total_time, digits=2))s")
        println("  GPU加速比: $(round(speedup, digits=1))x")
        
        # 估算网格搜索时间节省
        estimated_grid_time_hours = grid_points_coarse * (cpu_total_time / n_evaluated) / 3600
        gpu_time_hours = gpu_total_time / 3600
        time_saving_hours = estimated_grid_time_hours - gpu_time_hours
        
        println("\n⏰ 时间节省估算:")
        println("  估算网格搜索时间: $(round(estimated_grid_time_hours, digits=1)) 小时")
        println("  GPU贝叶斯优化时间: $(round(gpu_time_hours, digits=3)) 小时")
        println("  节省时间: $(round(time_saving_hours, digits=1)) 小时")
        
    elseif !isempty(optimizer.gpu_evaluation_times)
        gpu_total_time = sum(optimizer.gpu_evaluation_times)
        total_samples = sum(optimizer.batch_history)
        avg_throughput = total_samples / gpu_total_time
        
        println("\n🚀 GPU性能统计:")
        println("  GPU平均吞吐量: $(round(avg_throughput, digits=1)) 样本/秒")
        println("  GPU总计算时间: $(round(gpu_total_time, digits=2))s")
        
        # 估算网格搜索时间
        estimated_grid_time = grid_points_coarse / avg_throughput / 3600  # 小时
        gpu_time_hours = gpu_total_time / 3600
        
        println("  估算网格搜索时间: $(round(estimated_grid_time, digits=1)) 小时")
        println("  节省时间: $(round(estimated_grid_time - gpu_time_hours, digits=1)) 小时")
    end
    
    # 内存效率
    if !isempty(optimizer.memory_usage_history)
        avg_memory = mean(optimizer.memory_usage_history)
        max_memory = maximum(optimizer.memory_usage_history)
        
        println("\n💾 GPU内存效率:")
        println("  平均内存使用: $(round(avg_memory, digits=2)) GB")
        println("  峰值内存使用: $(round(max_memory, digits=2)) GB")
        
        if CUDA.functional()
            total_memory = CUDA.totalmem(CUDA.device()) / 1e9
            memory_efficiency = (avg_memory / total_memory) * 100
            println("  内存利用率: $(round(memory_efficiency, digits=1))%")
        end
    end
    
    println("\n✅ GPU贝叶斯优化成功实现智能参数探索加速!")
end

# 导出主要函数
export GPUBayesianConfig, GPUBayesianOptimizer
export create_gpu_bayesian_workflow, run_gpu_bayesian_optimization!
export default_gpu_bayesian_config
export analyze_gpu_performance, save_gpu_optimization_results

# 主程序入口
if abspath(PROGRAM_FILE) == @__FILE__
    println("🎬 执行GPU加速贝叶斯优化工作流程...")
    
    # 配置文件路径
    config_path = "config/bayesian_optimization_config.toml"
    
    # 根据命令行参数选择工作流程
    if length(ARGS) > 0
        if ARGS[1] == "--single" || ARGS[1] == "-s"
            section = length(ARGS) > 1 ? ARGS[2] : "single_objective"
            optimizer = create_gpu_bayesian_workflow(config_path, section)
            
        elseif ARGS[1] == "--config" || ARGS[1] == "-c"
            # 指定配置文件
            if length(ARGS) > 1
                config_path = ARGS[2]
            end
            section = length(ARGS) > 2 ? ARGS[3] : "single_objective"
            optimizer = create_gpu_bayesian_workflow(config_path, section)
            
        elseif ARGS[1] == "--help" || ARGS[1] == "-h"
            println("📚 GPU贝叶斯优化使用说明:")
            println("  julia bayesian_optimization_gpu.jl                    # 默认GPU优化")
            println("  julia bayesian_optimization_gpu.jl --single           # GPU单目标优化")
            println("  julia bayesian_optimization_gpu.jl --config <path>    # 指定配置文件")
            println("  julia bayesian_optimization_gpu.jl --help             # 显示帮助")
            println("\n🚀 GPU特性:")
            println("  - 批量GPU并行评估")
            println("  - 智能内存管理")
            println("  - 多GPU支持")
            println("  - 自动CPU回退")
            println("  - 性能监控和优化")
            
        else
            # 默认GPU优化
            optimizer = create_gpu_bayesian_workflow(config_path, "single_objective")
        end
    else
        # 默认GPU优化
        optimizer = create_gpu_bayesian_workflow(config_path, "single_objective")
    end
    
    println("\n🎉 GPU贝叶斯优化演示完成！")
    println("💡 现在可以用GPU加速的智能算法替代网格扫描，获得更快的参数优化！")
end