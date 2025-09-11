"""
TwoEnzymeSim GPU性能比较脚本

比较三种实现的性能：
1. 原始伪并行GPU实现（存在问题）
2. 优化GPU并行实现（真正并行）
3. CPU基准实现

测试指标：
- 计算时间和吞吐量
- GPU内存利用率
- 结果准确性
- 扩展性分析
"""

using Pkg
using CUDA
using Plots
using Printf
using Statistics
using LinearAlgebra
using ProgressMeter
using BenchmarkTools
using JLD2

# 导入项目文件
include("../../src/simulation.jl")
include("../../src/parameters.jl")

# 检查并导入优化实现
if isfile("cuda_integrated_example_optimized.jl")
    include("cuda_integrated_example_optimized.jl")
    const OPTIMIZED_AVAILABLE = true
else
    const OPTIMIZED_AVAILABLE = false
    println("⚠️  优化实现不可用，将跳过相关测试")
end

# 检查原始实现
if isfile("cuda_integrated_example.jl") && isfile("surrogate_model.jl")
    include("surrogate_model.jl")
    const ORIGINAL_AVAILABLE = true
else
    const ORIGINAL_AVAILABLE = false
    println("⚠️  原始实现不可用，将跳过相关测试")
end

"""
性能测试配置
"""
struct PerformanceTestConfig
    sample_sizes::Vector{Int}
    target_variables::Vector{Symbol}
    tspan::Tuple{Float64, Float64}
    num_runs::Int
    warmup_runs::Int
    timeout_seconds::Float64

    function PerformanceTestConfig()
        new(
            [100, 500, 1000, 2000, 5000],  # 测试样本量
            [:A_final, :B_final, :C_final, :v1_mean, :v2_mean],  # 目标变量
            (0.0, 5.0),  # 时间范围
            3,           # 每个测试运行3次
            1,           # 预热1次
            300.0        # 超时5分钟
        )
    end
end

"""
性能测试结果
"""
struct PerformanceResult
    name::String
    sample_size::Int
    times::Vector{Float64}
    mean_time::Float64
    std_time::Float64
    throughput::Float64
    memory_used::Float64
    gpu_utilization::Float64
    success_rate::Float64
    accuracy_score::Float64
end

"""
生成测试参数样本
"""
function generate_test_samples(n_samples::Int)
    # 13维参数：8个反应常数 + 5个初始浓度
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
CPU基准测试实现
"""
function cpu_benchmark_solve(X_samples::Matrix{Float64}, tspan::Tuple{Float64, Float64},
                           target_vars::Vector{Symbol})
    n_samples = size(X_samples, 1)
    n_outputs = length(target_vars)
    results = zeros(Float32, n_samples, n_outputs)

    success_count = 0

    for i in 1:n_samples
        try
            # 构建参数
            params_dict = Dict(
                :k1f => X_samples[i, 1], :k1r => X_samples[i, 2],
                :k2f => X_samples[i, 3], :k2r => X_samples[i, 4],
                :k3f => X_samples[i, 5], :k3r => X_samples[i, 6],
                :k4f => X_samples[i, 7], :k4r => X_samples[i, 8]
            )

            # 初始条件
            initial_conditions = [
                A => X_samples[i, 9], B => X_samples[i, 10], C => X_samples[i, 11],
                E1 => X_samples[i, 12], E2 => X_samples[i, 13],
                AE1 => 0.0, BE2 => 0.0
            ]

            # 仿真
            sol = simulate_system(params_dict, initial_conditions, tspan, saveat=0.1)

            if sol.retcode == :Success
                # 提取目标变量
                for (j, var) in enumerate(target_vars)
                    if var == :A_final
                        results[i, j] = sol.u[end][1]
                    elseif var == :B_final
                        results[i, j] = sol.u[end][2]
                    elseif var == :C_final
                        results[i, j] = sol.u[end][3]
                    elseif var == :v1_mean
                        # 简化：使用最终状态计算
                        A_f, E1_f, AE1_f = sol.u[end][1], sol.u[end][4], sol.u[end][6]
                        results[i, j] = params_dict[:k1f] * A_f * E1_f - params_dict[:k1r] * AE1_f
                    elseif var == :v2_mean
                        B_f, E2_f, BE2_f = sol.u[end][2], sol.u[end][5], sol.u[end][7]
                        results[i, j] = params_dict[:k3f] * B_f * E2_f - params_dict[:k3r] * BE2_f
                    else
                        results[i, j] = NaN32
                    end
                end
                success_count += 1
            else
                results[i, :] .= NaN32
            end

        catch e
            results[i, :] .= NaN32
        end
    end

    return results, success_count / n_samples
end

"""
原始GPU实现测试
"""
function original_gpu_test(X_samples::Matrix{Float64}, tspan::Tuple{Float64, Float64},
                          target_vars::Vector{Symbol})
    if !ORIGINAL_AVAILABLE
        return nothing, 0.0
    end

    try
        # 创建配置（模拟原始实现）
        config = SurrogateModelConfig(
            use_cuda = true,
            cuda_batch_size = size(X_samples, 1),
            target_variables = target_vars
        )

        # 调用原始GPU函数
        results = simulate_parameter_batch_gpu(X_samples, tspan, target_vars, config)

        # 计算成功率
        valid_mask = .!any(isnan.(results), dims=2)[:, 1]
        success_rate = sum(valid_mask) / length(valid_mask)

        return results, success_rate

    catch e
        println("原始GPU实现失败: $e")
        return nothing, 0.0
    end
end

"""
优化GPU实现测试
"""
function optimized_gpu_test(X_samples::Matrix{Float64}, tspan::Tuple{Float64, Float64},
                           target_vars::Vector{Symbol})
    if !OPTIMIZED_AVAILABLE
        return nothing, 0.0
    end

    try
        # 创建优化GPU求解器
        gpu_config = create_optimized_gpu_config()
        gpu_config.verbose = false  # 关闭详细输出以提高测试速度
        gpu_solver = OptimizedGPUSolver(gpu_config)

        # 求解
        results = solve_batch_gpu_optimized!(gpu_solver, X_samples, tspan, target_vars)

        # 计算成功率
        valid_mask = .!any(isnan.(results), dims=2)[:, 1]
        success_rate = sum(valid_mask) / length(valid_mask)

        # 清理资源
        cleanup_gpu_resources!(gpu_solver)

        return results, success_rate

    catch e
        println("优化GPU实现失败: $e")
        return nothing, 0.0
    end
end

"""
执行单个性能测试
"""
function run_single_test(test_func::Function, X_samples::Matrix{Float64},
                        tspan::Tuple{Float64, Float64}, target_vars::Vector{Symbol},
                        num_runs::Int, warmup_runs::Int, timeout::Float64)

    times = Float64[]
    success_rates = Float64[]
    memory_before = 0.0
    memory_after = 0.0

    # 预热运行
    for _ in 1:warmup_runs
        try
            test_func(X_samples, tspan, target_vars)
        catch
            # 预热失败不影响正式测试
        end
    end

    # 正式测试
    for run in 1:num_runs
        try
            # 记录内存使用
            if CUDA.functional() && CUDA.ndevices() > 0
                CUDA.reclaim()  # 清理GPU内存
                memory_before = CUDA.totalmem() - CUDA.available_memory()
            end
            GC.gc()  # CPU垃圾回收

            # 计时测试
            start_time = time()
            results, success_rate = test_func(X_samples, tspan, target_vars)
            elapsed = time() - start_time

            # 检查超时
            if elapsed > timeout
                println("⚠️  测试超时 ($(elapsed)s > $(timeout)s)")
                break
            end

            # 记录结果
            if results !== nothing
                push!(times, elapsed)
                push!(success_rates, success_rate)
            end

            # 记录内存使用
            if CUDA.functional() && CUDA.ndevices() > 0
                memory_after = CUDA.totalmem() - CUDA.available_memory()
            end

        catch e
            println("测试运行失败: $e")
            continue
        end
    end

    if isempty(times)
        return nothing
    end

    mean_time = mean(times)
    std_time = std(times)
    mean_success_rate = mean(success_rates)
    memory_used = memory_after - memory_before

    return (
        times = times,
        mean_time = mean_time,
        std_time = std_time,
        success_rate = mean_success_rate,
        memory_used = memory_used
    )
end

"""
运行完整性能比较
"""
function run_performance_comparison()
    config = PerformanceTestConfig()

    println("🏃 启动TwoEnzymeSim GPU性能比较测试")
    println("="^60)

    # 检查可用的实现
    available_methods = []
    if CUDA.functional()
        push!(available_methods, ("CPU基准", cpu_benchmark_solve))
        if ORIGINAL_AVAILABLE
            push!(available_methods, ("原始GPU", original_gpu_test))
        end
        if OPTIMIZED_AVAILABLE
            push!(available_methods, ("优化GPU", optimized_gpu_test))
        end
    else
        push!(available_methods, ("CPU基准", cpu_benchmark_solve))
        println("⚠️  CUDA不可用，仅测试CPU性能")
    end

    println("📊 测试配置:")
    println("  样本大小: $(config.sample_sizes)")
    println("  目标变量: $(config.target_variables)")
    println("  运行次数: $(config.num_runs)")
    println("  可用方法: $([name for (name, _) in available_methods])")

    # 存储所有结果
    all_results = Dict()

    # 对每个样本大小进行测试
    for n_samples in config.sample_sizes
        println("\n🔬 测试样本数: $n_samples")
        println("-"^40)

        # 生成测试数据
        X_samples = generate_test_samples(n_samples)

        sample_results = Dict()

        # 测试每种方法
        for (method_name, test_func) in available_methods
            print("  $method_name: ")

            result = run_single_test(
                test_func, X_samples, config.tspan, config.target_variables,
                config.num_runs, config.warmup_runs, config.timeout_seconds
            )

            if result !== nothing
                throughput = n_samples / result.mean_time

                sample_results[method_name] = PerformanceResult(
                    method_name, n_samples, result.times, result.mean_time,
                    result.std_time, throughput, result.memory_used, 0.0,
                    result.success_rate, 0.0
                )

                @printf("%.3fs (±%.3fs), %.1f 样本/秒, %.1f%% 成功率\n",
                       result.mean_time, result.std_time, throughput,
                       result.success_rate * 100)
            else
                println("失败")
                sample_results[method_name] = nothing
            end
        end

        all_results[n_samples] = sample_results
    end

    return all_results
end

"""
生成性能报告
"""
function generate_performance_report(results::Dict)
    println("\n📊 性能对比报告")
    println("="^80)

    # 表头
    @printf("%-10s %-12s %-12s %-12s %-15s %-10s\n",
            "样本数", "CPU时间(s)", "原始GPU(s)", "优化GPU(s)", "最佳加速比", "成功率")
    println("-"^80)

    # 数据行
    for n_samples in sort(collect(keys(results)))
        sample_results = results[n_samples]

        cpu_time = sample_results["CPU基准"] !== nothing ?
                  sample_results["CPU基准"].mean_time : NaN

        original_time = haskey(sample_results, "原始GPU") &&
                       sample_results["原始GPU"] !== nothing ?
                       sample_results["原始GPU"].mean_time : NaN

        optimized_time = haskey(sample_results, "优化GPU") &&
                        sample_results["优化GPU"] !== nothing ?
                        sample_results["优化GPU"].mean_time : NaN

        # 计算最佳加速比
        best_gpu_time = min(original_time, optimized_time)
        speedup = isnan(best_gpu_time) ? NaN : cpu_time / best_gpu_time

        # 平均成功率
        success_rates = [r.success_rate for r in values(sample_results) if r !== nothing]
        avg_success = isempty(success_rates) ? 0.0 : mean(success_rates)

        @printf("%-10d %-12.3f %-12.3f %-12.3f %-15.2fx %-10.1f%%\n",
                n_samples, cpu_time, original_time, optimized_time,
                speedup, avg_success * 100)
    end

    println("\n🎯 性能总结:")

    # 计算总体统计
    cpu_throughputs = Float64[]
    gpu_throughputs = Float64[]

    for (_, sample_results) in results
        if sample_results["CPU基准"] !== nothing
            push!(cpu_throughputs, sample_results["CPU基准"].throughput)
        end

        if haskey(sample_results, "优化GPU") && sample_results["优化GPU"] !== nothing
            push!(gpu_throughputs, sample_results["优化GPU"].throughput)
        end
    end

    if !isempty(cpu_throughputs)
        println("  平均CPU吞吐量: $(round(mean(cpu_throughputs), digits=1)) 样本/秒")
    end

    if !isempty(gpu_throughputs)
        println("  平均优化GPU吞吐量: $(round(mean(gpu_throughputs), digits=1)) 样本/秒")

        if !isempty(cpu_throughputs)
            overall_speedup = mean(gpu_throughputs) / mean(cpu_throughputs)
            println("  总体加速比: $(round(overall_speedup, digits=2))x")
        end
    end
end

"""
可视化性能结果
"""
function plot_performance_results(results::Dict)
    if !haskey(Main, :Plots)
        println("⚠️  Plots包不可用，跳过可视化")
        return
    end

    sample_sizes = sort(collect(keys(results)))

    # 提取数据
    cpu_times = Float64[]
    original_times = Float64[]
    optimized_times = Float64[]

    for n_samples in sample_sizes
        sample_results = results[n_samples]

        push!(cpu_times, sample_results["CPU基准"] !== nothing ?
              sample_results["CPU基准"].mean_time : NaN)

        push!(original_times, haskey(sample_results, "原始GPU") &&
              sample_results["原始GPU"] !== nothing ?
              sample_results["原始GPU"].mean_time : NaN)

        push!(optimized_times, haskey(sample_results, "优化GPU") &&
              sample_results["优化GPU"] !== nothing ?
              sample_results["优化GPU"].mean_time : NaN)
    end

    # 创建性能对比图
    p1 = plot(sample_sizes, cpu_times, label="CPU基准", marker=:circle, linewidth=2)
    plot!(p1, sample_sizes, original_times, label="原始GPU", marker=:square, linewidth=2)
    plot!(p1, sample_sizes, optimized_times, label="优化GPU", marker=:diamond, linewidth=2)

    xlabel!(p1, "样本数量")
    ylabel!(p1, "计算时间 (秒)")
    title!(p1, "TwoEnzymeSim GPU性能对比 - 计算时间")

    # 创建吞吐量对比图
    cpu_throughput = sample_sizes ./ cpu_times
    original_throughput = sample_sizes ./ original_times
    optimized_throughput = sample_sizes ./ optimized_times

    p2 = plot(sample_sizes, cpu_throughput, label="CPU基准", marker=:circle, linewidth=2)
    plot!(p2, sample_sizes, original_throughput, label="原始GPU", marker=:square, linewidth=2)
    plot!(p2, sample_sizes, optimized_throughput, label="优化GPU", marker=:diamond, linewidth=2)

    xlabel!(p2, "样本数量")
    ylabel!(p2, "吞吐量 (样本/秒)")
    title!(p2, "TwoEnzymeSim GPU性能对比 - 吞吐量")

    # 创建加速比图
    original_speedup = cpu_times ./ original_times
    optimized_speedup = cpu_times ./ optimized_times

    p3 = plot(sample_sizes, original_speedup, label="原始GPU加速比", marker=:square, linewidth=2)
    plot!(p3, sample_sizes, optimized_speedup, label="优化GPU加速比", marker=:diamond, linewidth=2)
    plot!(p3, sample_sizes, ones(length(sample_sizes)), label="CPU基准线", linestyle=:dash, color=:gray)

    xlabel!(p3, "样本数量")
    ylabel!(p3, "加速比")
    title!(p3, "TwoEnzymeSim GPU加速效果")

    # 组合图表
    final_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 1000))

    # 保存图表
    savefig(final_plot, "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/gpu_performance_comparison.png")
    println("📊 性能对比图已保存: gpu_performance_comparison.png")

    return final_plot
end

"""
保存结果到文件
"""
function save_results(results::Dict)
    output_path = "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/performance_comparison_results.jld2"

    try
        jldsave(output_path;
               results = results,
               timestamp = now(),
               julia_version = VERSION,
               cuda_version = CUDA.version(),
               gpu_info = CUDA.functional() ?
                         [(i, CUDA.name(CuDevice(i))) for i in 0:CUDA.ndevices()-1] :
                         nothing)

        println("💾 结果已保存到: $output_path")
    catch e
        println("⚠️  保存结果失败: $e")
    end
end

"""
主函数
"""
function main()
    println("🎬 TwoEnzymeSim GPU性能比较测试")

    # 系统信息
    println("\n🔧 系统信息:")
    println("  Julia版本: $(VERSION)")
    println("  CUDA可用性: $(CUDA.functional())")

    if CUDA.functional()
        println("  GPU数量: $(CUDA.ndevices())")
        for i in 0:CUDA.ndevices()-1
            device = CuDevice(i)
            memory_gb = CUDA.totalmem(device) / 1e9
            println("    GPU $i: $(CUDA.name(device)) ($(round(memory_gb, digits=1)) GB)")
        end
    end

    # 运行性能比较
    results = run_performance_comparison()

    # 生成报告
    generate_performance_report(results)

    # 可视化结果
    try
        plot_performance_results(results)
    catch e
        println("⚠️  可视化失败: $e")
    end

    # 保存结果
    save_results(results)

    println("\n🎉 性能比较测试完成！")

    return results
end

# 脚本入口点
if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end
