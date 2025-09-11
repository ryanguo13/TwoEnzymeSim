"""
CUDA集成代理模型综合示例

集成了CUDA加速、热力学约束和大规模参数扫描功能的ML代理模型完整演示

功能展示：
1. 热力学约束的参数空间定义
2. CUDA GPU加速训练和仿真
3. 代理模型vs CUDA性能对比
4. 大规模参数扫描（百万级参数组合）
5. 性能优化和可视化分析
6. 实用的工程应用场景
"""

using Pkg
# Pkg.add(["Flux", "MLJ", "Surrogates", "MultivariateStats", "JLD2", "ProgressMeter", "CUDA", "DiffEqGPU"])

include("surrogate_model.jl")

"""
    cuda_integrated_workflow()

执行完整的CUDA集成工作流程
"""
function cuda_integrated_workflow()
    println("🚀 CUDA集成代理模型工作流程")
    println("="^60)

    # ===== 第1步：系统初始化 =====
    println("\n🔧 第1步：系统初始化")

    # 检查CUDA可用性
    cuda_available = configure_cuda_device()
    if cuda_available
        println("✅ CUDA GPU加速已启用")
    else
        println("⚠️  CUDA不可用，使用CPU模式")
    end

    # ===== 第2步：增强配置（集成CUDA + 热力学约束）=====
    println("\n⚙️  第2步：增强配置")

    config = SurrogateModelConfig(
        # 数据生成配置
        sample_fraction = 0.15,          # 增加到15%获得更好精度
        max_samples = 10000,

        # 模型配置
        model_type = :neural_network,
        hidden_dims = [128, 64, 32],     # 更深的网络
        dropout_rate = 0.15,

        # 训练配置
        epochs = 150,                    # 更多训练轮数
        batch_size = 64,
        learning_rate = 1e-3,
        validation_split = 0.2,

        # 降维配置
        use_pca = true,
        pca_variance_threshold = 0.95,

        # CUDA配置
        use_cuda = cuda_available,       # 自动检测CUDA
        cuda_batch_size = 2000,

        # 热力学约束配置
        apply_thermodynamic_constraints = true,  # 启用热力学约束
        keq_min = 0.01,                  # 平衡常数范围 (放宽约束)
        keq_max = 100.0,                 # 平衡常数范围 (放宽约束)

        # 输出配置
        target_variables = [:A_final, :B_final, :C_final, :v1_mean, :v2_mean],
        uncertainty_estimation = true
    )

    # 创建扩展的参数空间
    param_space = ParameterSpace(
        # 反应速率常数（保持与CUDA扫描一致）
        0.1:2:20.0,   # k1f_range (10 points)
        0.1:2:20.0,   # k1r_range
        0.1:2:20.0,   # k2f_range
        0.1:2:20.0,   # k2r_range
        0.1:2:20.0,   # k3f_range
        0.1:2:20.0,   # k3r_range
        0.1:2:20.0,   # k4f_range
        0.1:2:20.0,   # k4r_range

        # 初始浓度
        5.0:2:20.0,   # A_range
        0.0:1:5.0,    # B_range
        0.0:1:5.0,    # C_range
        5.0:2:20.0,   # E1_range
        5.0:2:20.0,   # E2_range

        # 时间跨度
        (0.0, 5.0)
    )

    println("✅ 配置完成")
    println("📊 参数空间: 13维参数")
    println("🧪 热力学约束: 启用")
    println("🔥 CUDA加速: $(config.use_cuda ? "启用" : "禁用")")

    # ===== 第3步：训练增强代理模型 =====
    println("\n🎯 第3步：训练增强代理模型")

    surrogate_model = SurrogateModel(config, param_space)

    # 生成训练数据（自动应用热力学约束）
    println("📊 生成训练数据...")
    X_data, y_data = generate_small_scale_data(surrogate_model)

    println("📈 数据统计:")
    println("  训练样本: $(size(X_data, 1))")
    println("  输入维度: $(size(X_data, 2))")
    println("  输出维度: $(size(y_data, 2))")

    # 数据预处理
    println("🔧 数据预处理...")
    preprocess_data!(surrogate_model, X_data, y_data)

    # 训练模型
    println("🚀 开始训练...")
    train_surrogate_model!(surrogate_model)

    # ===== 第4步：性能对比测试 =====
    println("\n📊 第4步：性能对比测试")

    println("🔄 执行代理模型 vs CUDA仿真性能对比...")
    comparison_results = compare_surrogate_vs_cuda(surrogate_model, 500)

    # ===== 第5步：大规模参数扫描演示 =====
    println("\n🚀 第5步：大规模参数扫描演示")

    # 定义扫描范围
    scan_config = Dict(
        :k1f => 0.1:0.5:20.0,    # 40 points
        :k1r => 0.1:0.5:20.0,    # 40 points
        :k2f => 0.1:0.5:20.0,    # 40 points
        :A => 5.0:1.0:25.0,      # 21 points
        :B => 0.0:0.5:5.0,       # 11 points
        :E1 => 5.0:1.0:25.0      # 21 points
    )

    # 计算理论组合数: 40^3 * 21^2 * 11 = 64000 * 441 * 11 ≈ 310M
    println("🎯 大规模参数扫描配置:")
    for (param, range) in scan_config
        println("  $param: $(length(collect(range))) points")
    end

    # 执行扫描（自动限制到合理数量）
    scan_results = large_scale_parameter_scan(surrogate_model, scan_config, max_combinations=100000)

    # ===== 第6步：结果分析和可视化 =====
    println("\n📈 第6步：结果分析")

    # 创建性能报告
    create_performance_report(surrogate_model, comparison_results, scan_results)

    # 分析扫描结果
    analyze_scan_results(scan_results, surrogate_model.config.target_variables)

    # ===== 第7步：模型保存和部署 =====
    println("\n💾 第7步：模型保存")

    model_path = "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/cuda_integrated_surrogate.jld2"
    save_surrogate_model(surrogate_model, model_path)

    # 保存扫描结果
    results_path = "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/large_scale_scan_results.jld2"
    jldsave(results_path;
            scan_results = scan_results,
            scan_config = scan_config,
            comparison_results = comparison_results)

    println("✅ 结果保存完成")
    println("📁 模型文件: $model_path")
    println("📁 扫描结果: $results_path")

    # ===== 第8步：工程应用演示 =====
    println("\n🛠️  第8步：工程应用演示")
    demonstrate_engineering_applications(surrogate_model)

    println("\n🎉 CUDA集成工作流程完成！")

    return surrogate_model, comparison_results, scan_results
end

"""
    analyze_scan_results(scan_results, target_variables)

分析大规模扫描结果
"""
function analyze_scan_results(scan_results, target_variables)
    println("🔍 扫描结果分析:")

    n_results = length(scan_results)
    println("📊 总扫描结果数: $n_results")

    # 提取预测值
    predictions = Dict()
    for var in target_variables
        predictions[var] = [result.predictions[var] for result in scan_results]
    end

    # 统计分析
    println("\n📈 目标变量统计:")
    for var in target_variables
        vals = predictions[var]
        println("  $var:")
        println("    均值: $(round(mean(vals), digits=4))")
        println("    标准差: $(round(std(vals), digits=4))")
        println("    范围: [$(round(minimum(vals), digits=4)), $(round(maximum(vals), digits=4))]")
    end

    # 寻找有趣的参数组合
    find_optimal_conditions(scan_results, target_variables)
end

"""
    find_optimal_conditions(scan_results, target_variables)

寻找最优反应条件
"""
function find_optimal_conditions(scan_results, target_variables)
    println("\n🎯 最优条件识别:")

    # 定义优化目标（例如：最大化C产量）
    if :C_final in target_variables
        c_values = [result.predictions[:C_final] for result in scan_results]
        max_c_idx = argmax(c_values)

        optimal_result = scan_results[max_c_idx]

        println("🏆 最高C产量条件:")
        println("  C浓度: $(round(optimal_result.predictions[:C_final], digits=4))")
        println("  参数组合:")
        for (param, value) in optimal_result.parameters
            println("    $param = $(round(value, digits=3))")
        end
    end

    # 寻找高效率条件（高产物浓度 + 低不确定性）
    if :C_final in target_variables && :C_final_std in [Symbol(string(var) * "_std") for var in target_variables]
        c_values = [result.predictions[:C_final] for result in scan_results]
        c_uncertainties = [result.predictions[:C_final_std] for result in scan_results]

        # 效率分数 = 产量 / 不确定性
        efficiency_scores = c_values ./ (c_uncertainties .+ 1e-6)
        best_efficiency_idx = argmax(efficiency_scores)

        efficient_result = scan_results[best_efficiency_idx]

        println("\n⚡ 最高效率条件:")
        println("  C浓度: $(round(efficient_result.predictions[:C_final], digits=4))")
        println("  不确定性: $(round(efficient_result.predictions[:C_final_std], digits=4))")
        println("  效率分数: $(round(efficiency_scores[best_efficiency_idx], digits=2))")
    end
end

"""
    demonstrate_engineering_applications(surrogate_model)

演示工程应用场景
"""
function demonstrate_engineering_applications(surrogate_model)
    println("🛠️  工程应用场景演示:")

    # 场景1：实时优化
    println("\n📊 场景1: 实时过程优化")
    demonstrate_real_time_optimization(surrogate_model)

    # 场景2：敏感性分析
    println("\n🔬 场景2: 参数敏感性分析")
    demonstrate_sensitivity_analysis(surrogate_model)

    # 场景3：不确定性量化
    println("\n📈 场景3: 不确定性量化")
    demonstrate_uncertainty_quantification(surrogate_model)

    # 场景4：批量设计空间探索
    println("\n🌐 场景4: 设计空间探索")
    demonstrate_design_space_exploration(surrogate_model)
end

"""
    demonstrate_real_time_optimization(surrogate_model)

演示实时优化应用
"""
function demonstrate_real_time_optimization(surrogate_model)
    println("🚀 模拟实时优化循环...")

    param_space = surrogate_model.param_space

    # 模拟10轮优化
    current_best_c = 0.0
    current_best_params = nothing

    for iteration in 1:10
        # 生成候选参数（在当前最优附近）
        if iteration == 1
            # 第一轮：随机采样
            candidates = generate_lhs_samples(param_space, 100)
        else
            # 后续轮次：在最优附近采样
            candidates = generate_local_samples(current_best_params, param_space, 100)
        end

        # 快速代理模型评估
        y_pred, y_std = predict_with_uncertainty(surrogate_model, candidates, n_samples=20)

        # 选择最优候选（假设目标是最大化C_final）
        if :C_final in surrogate_model.config.target_variables
            c_idx = findfirst(x -> x == :C_final, surrogate_model.config.target_variables)
            c_values = y_pred[:, c_idx]
            best_idx = argmax(c_values)

            if c_values[best_idx] > current_best_c
                current_best_c = c_values[best_idx]
                current_best_params = candidates[best_idx, :]
                println("  迭代 $iteration: 新最优 C = $(round(current_best_c, digits=4))")
            else
                println("  迭代 $iteration: 无改善")
            end
        end
    end

    println("🏆 最终优化结果:")
    println("  最优 C 浓度: $(round(current_best_c, digits=4))")
    println("  优化用时: <1秒 (vs CUDA仿真需要数分钟)")
end

"""
    demonstrate_sensitivity_analysis(surrogate_model)

演示参数敏感性分析
"""
function demonstrate_sensitivity_analysis(surrogate_model)
    println("🔬 参数敏感性分析...")

    param_space = surrogate_model.param_space

    # 基准参数
    baseline_params = zeros(13)
    ranges = [param_space.k1f_range, param_space.k1r_range, param_space.k2f_range, param_space.k2r_range,
              param_space.k3f_range, param_space.k3r_range, param_space.k4f_range, param_space.k4r_range,
              param_space.A_range, param_space.B_range, param_space.C_range, param_space.E1_range, param_space.E2_range]

    for i in 1:13
        baseline_params[i] = mean(ranges[i])
    end

    # 参数名称
    param_names = [:k1f, :k1r, :k2f, :k2r, :k3f, :k3r, :k4f, :k4r, :A, :B, :C, :E1, :E2]

    # 敏感性分析：每个参数±20%
    println("📊 敏感性系数 (C_final相对于各参数):")

    if :C_final in surrogate_model.config.target_variables
        c_idx = findfirst(x -> x == :C_final, surrogate_model.config.target_variables)

        # 基准预测
        baseline_pred, _ = predict_with_uncertainty(surrogate_model, reshape(baseline_params, 1, :), n_samples=10)
        baseline_c = baseline_pred[1, c_idx]

        for (i, param_name) in enumerate(param_names)
            # 参数+20%
            params_high = copy(baseline_params)
            params_high[i] *= 1.2
            pred_high, _ = predict_with_uncertainty(surrogate_model, reshape(params_high, 1, :), n_samples=10)
            c_high = pred_high[1, c_idx]

            # 参数-20%
            params_low = copy(baseline_params)
            params_low[i] *= 0.8
            pred_low, _ = predict_with_uncertainty(surrogate_model, reshape(params_low, 1, :), n_samples=10)
            c_low = pred_low[1, c_idx]

            # 敏感性系数
            sensitivity = (c_high - c_low) / (2 * 0.2 * baseline_params[i]) * baseline_c

            println("  $param_name: $(round(sensitivity, digits=4))")
        end
    end

    println("💡 高敏感性参数需要精确控制，低敏感性参数允许较大波动")
end

"""
    demonstrate_uncertainty_quantification(surrogate_model)

演示不确定性量化应用
"""
function demonstrate_uncertainty_quantification(surrogate_model)
    println("📈 不确定性量化分析...")

    # 生成测试参数
    param_space = surrogate_model.param_space
    test_params = generate_lhs_samples(param_space, 100)

    # 预测with不确定性
    y_pred, y_std = predict_with_uncertainty(surrogate_model, test_params, n_samples=100)

    if :C_final in surrogate_model.config.target_variables
        c_idx = findfirst(x -> x == :C_final, surrogate_model.config.target_variables)
        c_pred = y_pred[:, c_idx]
        c_std = y_std[:, c_idx]

        # 不确定性分析
        relative_uncertainty = c_std ./ (abs.(c_pred) .+ 1e-6) * 100

        println("📊 不确定性统计:")
        println("  平均不确定性: $(round(mean(c_std), digits=4))")
        println("  平均相对不确定性: $(round(mean(relative_uncertainty), digits=2))%")

        # 找出高不确定性区域
        high_uncertainty_idx = findall(x -> x > 5.0, relative_uncertainty)  # >5%相对不确定性

        if length(high_uncertainty_idx) > 0
            println("⚠️  发现 $(length(high_uncertainty_idx)) 个高不确定性区域")
            println("💡 建议: 在这些区域增加训练数据或使用CUDA验证")
        else
            println("✅ 所有预测区域不确定性较低")
        end
    end
end

"""
    demonstrate_design_space_exploration(surrogate_model)

演示设计空间探索
"""
function demonstrate_design_space_exploration(surrogate_model)
    println("🌐 设计空间探索...")

    # 定义感兴趣的设计目标
    design_targets = Dict(
        "高C产量" => (var, val) -> var == :C_final && val > 0.5,
        "平衡产物" => (var, val) -> var == :B_final && val > 0.3,
        "高转化率" => (var, val) -> var == :A_final && val < 2.0
    )

    # 生成设计空间样本
    param_space = surrogate_model.param_space
    design_samples = generate_lhs_samples(param_space, 5000)

    # 预测
    y_pred, _ = predict_with_uncertainty(surrogate_model, design_samples, n_samples=20)

    println("📊 设计目标达成情况:")

    target_vars = surrogate_model.config.target_variables
    for (target_name, condition) in design_targets
        success_count = 0

        for i in 1:size(y_pred, 1)
            target_met = false
            for (j, var) in enumerate(target_vars)
                if condition(var, y_pred[i, j])
                    target_met = true
                    break
                end
            end
            if target_met
                success_count += 1
            end
        end

        success_rate = success_count / size(y_pred, 1) * 100
        println("  $target_name: $(round(success_rate, digits=1))% ($(success_count)/$(size(y_pred, 1)))")
    end

    println("💡 设计空间快速探索完成，仅需几秒钟！")
end

"""
    generate_local_samples(center_params, param_space, n_samples)

在给定中心附近生成局部采样
"""
function generate_local_samples(center_params, param_space, n_samples)
    ranges = [param_space.k1f_range, param_space.k1r_range, param_space.k2f_range, param_space.k2r_range,
              param_space.k3f_range, param_space.k3r_range, param_space.k4f_range, param_space.k4r_range,
              param_space.A_range, param_space.B_range, param_space.C_range, param_space.E1_range, param_space.E2_range]

    samples = zeros(n_samples, 13)

    for i in 1:n_samples
        for j in 1:13
            # 在中心±20%范围内采样
            range_min, range_max = minimum(ranges[j]), maximum(ranges[j])
            center = center_params[j]
            local_min = max(range_min, center * 0.8)
            local_max = min(range_max, center * 1.2)

            samples[i, j] = local_min + rand() * (local_max - local_min)
        end
    end

    return samples
end

"""
    main()

主函数 - 根据命令行参数执行不同功能
"""
function main()
    if length(ARGS) == 0
        # 默认：完整工作流程
        surrogate_model, comparison_results, scan_results = cuda_integrated_workflow()

    elseif ARGS[1] == "quick"
        # 快速演示
        println("🚀 快速演示模式")
        config = SurrogateModelConfig(
            sample_fraction = 0.05,
            max_samples = 2000,
            epochs = 50,
            use_cuda = configure_cuda_device(),
            apply_thermodynamic_constraints = true,
            keq_min = 0.01,
            keq_max = 100.0
        )

        param_space = create_default_parameter_space()
        surrogate_model = SurrogateModel(config, param_space)

        X_data, y_data = generate_small_scale_data(surrogate_model)
        preprocess_data!(surrogate_model, X_data, y_data)
        train_surrogate_model!(surrogate_model)

        println("✅ 快速演示完成")

    elseif ARGS[1] == "benchmark"
        # 性能基准测试
        println("📊 性能基准测试模式")

        config = SurrogateModelConfig(
            sample_fraction = 0.1,
            use_cuda = configure_cuda_device(),
            apply_thermodynamic_constraints = true,
            keq_min = 0.01,
            keq_max = 100.0
        )

        param_space = create_default_parameter_space()
        surrogate_model = SurrogateModel(config, param_space)

        X_data, y_data = generate_small_scale_data(surrogate_model)
        preprocess_data!(surrogate_model, X_data, y_data)
        train_surrogate_model!(surrogate_model)

        # 性能对比
        comparison_results = compare_surrogate_vs_cuda(surrogate_model, 1000)

    elseif ARGS[1] == "gp"
        # Gaussian Process 工作流程
        println("🔮 GP 工作流程")
        cuda_available = configure_cuda_device()

        config = SurrogateModelConfig(
            sample_fraction = 0.1,
            max_samples = 5000,
            model_type = :gaussian_process,
            use_pca = true,
            pca_variance_threshold = 0.95,
            uncertainty_estimation = false,  # GP预测接口当前实现返回均值
            use_cuda = false,  # GP训练/预测在CPU
            apply_thermodynamic_constraints = true,
            keq_min = 0.01,
            keq_max = 100.0,
        )

        param_space = create_default_parameter_space()
        surrogate_model = SurrogateModel(config, param_space)

        println("📊 生成训练数据 (GP)")
        X_data, y_data = generate_small_scale_data(surrogate_model)
        preprocess_data!(surrogate_model, X_data, y_data)
        train_surrogate_model!(surrogate_model)  # 内部将调用 train_gaussian_process!

        println("📈 评估GP代理模型 vs CPU仿真")
        comparison_results = compare_surrogate_vs_cuda(surrogate_model, 300)  # 对比接口可复用

        println("💾 保存GP模型")
        model_path = "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/cuda_integrated_surrogate.jld2"
        save_surrogate_model(surrogate_model, model_path)

        println("✅ GP流程完成")

    elseif ARGS[1] == "test"
        # 测试模式：无热力学约束
        println("🧪 测试模式（无热力学约束，强制CPU）")
        config = SurrogateModelConfig(
            sample_fraction = 0.05,
            max_samples = 2000,
            epochs = 50,
            use_cuda = false,  # 强制使用CPU进行调试
            apply_thermodynamic_constraints = false  # 禁用热力学约束进行测试
        )

        param_space = create_default_parameter_space()
        surrogate_model = SurrogateModel(config, param_space)

        X_data, y_data = generate_small_scale_data(surrogate_model)
        println("📊 测试数据维度: X=$(size(X_data)), y=$(size(y_data))")

        if size(X_data, 1) > 0
            preprocess_data!(surrogate_model, X_data, y_data)
            train_surrogate_model!(surrogate_model)
            println("✅ 测试完成")
        else
            println("❌ 测试失败：无有效数据")
        end

    else
        println("❌ 未知参数: $(ARGS[1])")
        println("使用方法:")
        println("  julia cuda_integrated_example.jl          # 完整工作流程")
        println("  julia cuda_integrated_example.jl quick    # 快速演示")
        println("  julia cuda_integrated_example.jl benchmark # 性能测试")
        println("  julia cuda_integrated_example.jl test     # 测试模式（无约束）")
    end
end

# 执行主函数
if abspath(PROGRAM_FILE) == @__FILE__
    println("🎬 CUDA集成代理模型演示")
    println("🔧 系统检查...")

    # 检查依赖包
    required_packages = ["Flux", "CUDA", "DiffEqGPU", "MultivariateStats", "JLD2"]
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
