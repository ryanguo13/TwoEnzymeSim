"""
CPU 版代理模型综合示例（安全回退）

目标：
- 完全禁用 CUDA，确保任何范围配置都能稳定运行
- 与现有 ML 代码保持一致的 API/输出（模型、比较结果、扫描结果、报告、分析）
- 大规模参数扫描使用安全采样，避免组合数溢出与内存错误
"""

include("surrogate_model.jl")

"""
    cpu_fallback_workflow()

执行 CPU-only 的完整工作流程。
"""
function cpu_fallback_workflow()
    println("🧠 CPU 回退工作流程")
    println("="^60)

    # ===== 第1步：基础配置（CPU-only） =====
    println("\n🔧 第1步：系统初始化（CPU-only）")

    config = SurrogateModelConfig(
        # 数据生成配置
        sample_fraction = 0.15,
        max_samples = 100000,

        # 模型配置
        model_type = :neural_network,
        hidden_dims = [128, 64, 32],
        dropout_rate = 0.15,

        # 训练配置
        epochs = 250,
        batch_size = 128,
        learning_rate = 1e-3,
        validation_split = 0.2,

        # 降维配置
        use_pca = true,
        pca_variance_threshold = 0.95,

        # CUDA配置（强制禁用）
        use_cuda = false,
        cuda_batch_size = 2000,

        # 热力学约束配置
        apply_thermodynamic_constraints = true,
        keq_min = 0.01,
        keq_max = 100.0,

        # 输出配置
        target_variables = [:A_final, :B_final, :C_final, :v1_mean, :v2_mean],
        uncertainty_estimation = true,
    )

    # 参数空间与 GPU 示例保持一致（可手动改范围，不会崩）
    param_space = ParameterSpace(
        0.1:0.02:20.0,   # k1f_range
        0.1:0.02:20.0,   # k1r_range
        0.1:0.02:20.0,   # k2f_range
        0.1:0.02:20.0,   # k2r_range
        0.1:0.02:20.0,   # k3f_range
        0.1:0.02:20.0,   # k3r_range
        0.1:0.02:20.0,   # k4f_range
        0.1:0.02:20.0,   # k4r_range

        5.0:0.02:20.0,   # A_range
        0.0:0.02:5.0,    # B_range
        0.0:0.02:5.0,    # C_range
        5.0:0.02:20.0,   # E1_range
        5.0:0.02:20.0,   # E2_range

        (0.0, 5.0),
    )

    println("✅ 配置完成 (CPU-only)")
    println("📊 参数空间: 13维参数")
    println("🧪 热力学约束: 启用")

    # ===== 第2步：数据生成与训练 =====
    println("\n🎯 第2步：训练 CPU 代理模型")
    surrogate_model = SurrogateModel(config, param_space)

    println("📊 生成训练数据...")
    X_data, y_data = generate_small_scale_data(surrogate_model)
    println("📊 数据维度: X=$(size(X_data)), y=$(size(y_data))")

    println("🔧 数据预处理...")
    preprocess_data!(surrogate_model, X_data, y_data)

    println("🚀 开始训练 (CPU)...")
    train_surrogate_model!(surrogate_model)

    # ===== 第3步：性能对比（CPU仿真） =====
    println("\n📊 第3步：性能对比测试（CPU 仿真 vs 代理模型）")
    # 复用现有对比接口：接口内部会根据 config.use_cuda=false 走 CPU 路径
    comparison_results = compare_surrogate_vs_cuda(surrogate_model, 500)

    # ===== 第4步：大规模参数扫描（安全版） =====
    println("\n🚀 第4步：大规模参数扫描（安全采样，避免组合溢出）")
    scan_config = Dict(
        :k1f => 0.1:0.5:20.0,
        :k1r => 0.1:0.5:20.0,
        :k2f => 0.1:0.5:20.0,
        :k2r => 0.1:0.5:20.0,
        :k3f => 0.1:0.5:20.0,
        :k3r => 0.1:0.5:20.0,
        :k4f => 0.1:0.5:20.0,
        :k4r => 0.1:0.5:20.0,
        :A => 5.0:0.5:25.0,
        :B => 0.0:0.5:5.0,
        :C => 0.0:0.5:5.0,
        :E1 => 5.0:0.5:25.0,
        :E2 => 5.0:0.5:25.0,
    )

    println("🎯 扫描范围（点数）:")
    for (p, r) in scan_config
        println("  $p: $(length(collect(r))) points")
    end

    scan_results = safe_large_scale_parameter_scan(surrogate_model, scan_config; max_combinations=100000)

    # ===== 第5步：结果分析与保存 =====
    println("\n📈 第5步：结果分析与保存")
    create_performance_report(surrogate_model, comparison_results, scan_results)
    analyze_scan_results_cpu(scan_results, surrogate_model.config.target_variables)

    model_path = "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/cpu_fallback_surrogate.jld2"
    save_surrogate_model(surrogate_model, model_path)

    results_path = "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/cpu_fallback_scan_results.jld2"
    jldsave(results_path; scan_results=scan_results, scan_config=scan_config, comparison_results=comparison_results)

    println("✅ 结果保存完成")
    println("📁 模型文件: $model_path")
    println("📁 扫描结果: $results_path")

    println("\n🎉 CPU 回退流程完成！")
    return surrogate_model, comparison_results, scan_results
end

"""
    safe_large_scale_parameter_scan(surrogate_model, scan_config; max_combinations=100_000)

安全采样版的大规模扫描：
- 始终采样（LHS 或均匀随机），避免构造完整笛卡尔积
- 返回值结构与现有分析代码兼容：Vector{NamedTuple{(:parameters,:predictions), ...}}
"""
function safe_large_scale_parameter_scan(surrogate_model, scan_config::Dict{Symbol,<:AbstractRange}; max_combinations::Int=100_000)
    # 参数顺序与模型输入一致
    param_order = [:k1f, :k1r, :k2f, :k2r, :k3f, :k3r, :k4f, :k4r, :A, :B, :C, :E1, :E2]
    ranges = Dict{Symbol,AbstractRange}()
    for name in param_order
        @assert haskey(scan_config, name) "scan_config 缺少参数: $(name)"
        ranges[name] = scan_config[name]
    end

    N = max_combinations
    X = _sample_from_ranges(ranges, param_order, N)

    # 使用代理模型预测（带不确定性，如果可用）
    if surrogate_model.config.uncertainty_estimation
        y_pred, _ = predict_with_uncertainty(surrogate_model, X, n_samples=20)
    else
        y_pred = predict_surrogate(surrogate_model, X)
    end

    target_vars = surrogate_model.config.target_variables

    results = Vector{NamedTuple{(:parameters, :predictions), Tuple{Dict{Symbol,Float64}, Dict{Symbol,Float64}}}}(undef, N)
    for i in 1:N
        params_dict = Dict{Symbol,Float64}()
        for (j, name) in enumerate(param_order)
            params_dict[name] = X[i, j]
        end

        preds_dict = Dict{Symbol,Float64}()
        for (j, var) in enumerate(target_vars)
            preds_dict[var] = y_pred[i, j]
        end

        results[i] = (parameters=params_dict, predictions=preds_dict)
    end

    println("✅ 安全扫描完成: $N 样本（代理模型预测）")
    return results
end

# 简单的均匀随机采样（可替换为更严格的 LHS）
function _sample_from_ranges(ranges::Dict{Symbol,<:AbstractRange}, order::Vector{Symbol}, N::Int)
    X = zeros(N, length(order))
    for (j, name) in enumerate(order)
        r = ranges[name]
        rmin = float(minimum(r))
        rmax = float(maximum(r))
        @inbounds @simd for i in 1:N
            X[i, j] = rmin + rand() * (rmax - rmin)
        end
    end
    return X
end

"""
    analyze_scan_results_cpu(scan_results, target_variables)

轻量版分析（与 GPU 示例输出兼容）
"""
function analyze_scan_results_cpu(scan_results, target_variables)
    println("🔍 扫描结果分析 (CPU 回退):")
    n_results = length(scan_results)
    println("📊 总扫描结果数: $n_results")

    predictions = Dict{Symbol,Vector{Float64}}()
    for var in target_variables
        predictions[var] = [result.predictions[var] for result in scan_results]
    end

    println("\n📈 目标变量统计:")
    for var in target_variables
        vals = predictions[var]
        println("  $var:")
        println("    均值: $(round(mean(vals), digits=4))")
        println("    标准差: $(round(std(vals), digits=4))")
        println("    范围: [$(round(minimum(vals), digits=4)), $(round(maximum(vals), digits=4))]")
    end
end

"""
    main()

CLI 入口
"""
function main()
    if length(ARGS) == 0
        cpu_fallback_workflow()
    elseif ARGS[1] == "scan-only"
        # 只运行安全扫描（假设已有模型）
        println("🚀 仅运行安全扫描 (加载或快速训练模型)...")

        # 尝试加载；失败则快速训练一个
        model_path = "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/cpu_fallback_surrogate.jld2"
        surrogate_model = try
            load_surrogate_model(model_path)
        catch
            println("⚠️ 未找到现有模型，执行快速训练...")
            config = SurrogateModelConfig(
                sample_fraction = 0.05,
                max_samples = 5000,
                epochs = 50,
                use_cuda = false,
                apply_thermodynamic_constraints = true,
                keq_min = 0.01,
                keq_max = 100.0,
            )
            param_space = create_default_parameter_space()
            surrogate_model = SurrogateModel(config, param_space)
            X, y = generate_small_scale_data(surrogate_model)
            preprocess_data!(surrogate_model, X, y)
            train_surrogate_model!(surrogate_model)
            save_surrogate_model(surrogate_model, model_path)
            surrogate_model
        end

        scan_config = Dict(
            :k1f => 0.1:0.5:20.0,
            :k1r => 0.1:0.5:20.0,
            :k2f => 0.1:0.5:20.0,
            :k2r => 0.1:0.5:20.0,
            :k3f => 0.1:0.5:20.0,
            :k3r => 0.1:0.5:20.0,
            :k4f => 0.1:0.5:20.0,
            :k4r => 0.1:0.5:20.0,
            :A => 5.0:0.5:25.0,
            :B => 0.0:0.5:5.0,
            :C => 0.0:0.5:5.0,
            :E1 => 5.0:0.5:25.0,
            :E2 => 5.0:0.5:25.0,
        )

        scan_results = safe_large_scale_parameter_scan(surrogate_model, scan_config; max_combinations=100000)
        results_path = "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/cpu_fallback_scan_results.jld2"
        jldsave(results_path; scan_results=scan_results, scan_config=scan_config)
        println("✅ 扫描结果保存: $results_path")
    else
        println("❌ 未知参数: $(ARGS[1])")
        println("使用方法:")
        println("  julia cpu_fallback_example.jl          # 完整 CPU-only 工作流程")
        println("  julia cpu_fallback_example.jl scan-only # 仅执行安全扫描")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("🎬 CPU 回退演示")
    println("🔧 系统检查 (仅必要包)...")
    required_packages = ["Flux", "MultivariateStats", "JLD2"]
    for pkg in required_packages
        try
            eval(Meta.parse("using $pkg"))
            println("✅ $pkg")
        catch
            println("❌ $pkg - 请运行: Pkg.add(\"$pkg\")")
        end
    end
    println("\n开始执行...")
    main()
    println("\n🎉 演示完成！")
    println("📚 查看生成的文件了解详细结果")
end


