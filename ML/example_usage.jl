"""
代理模型使用示例

展示如何使用ML代理模型替换CUDA参数扫描，实现80%+计算量减少
"""

using Pkg
# 确保必要的包已安装
# Pkg.add(["Flux", "MLJ", "Surrogates", "MultivariateStats", "JLD2", "ProgressMeter"])

include("surrogate_model.jl")

"""
    main_surrogate_workflow()

主要代理模型工作流程
"""
function main_surrogate_workflow()
    println("🚀 启动ML代理模型工作流程")
    println(repeat("=", 50))

    # 步骤1：配置代理模型
    println("\n📋 步骤1: 配置代理模型")
    config = SurrogateModelConfig(
        sample_fraction = 0.1,          # 使用10%的参数进行训练
        max_samples = 5000,             # 最大5000个样本
        model_type = :neural_network,   # 使用神经网络
        hidden_dims = [64, 32, 16],     # 网络结构
        dropout_rate = 0.1,             # Dropout用于不确定性估计
        epochs = 100,                   # 训练轮数
        batch_size = 32,
        learning_rate = 1e-3,
        use_pca = true,                 # 启用PCA降维
        uncertainty_estimation = true    # 启用不确定性估计
    )

    # 创建参数空间（与现有CUDA扫描一致）
    param_space = create_default_parameter_space()

    println("✅ 配置完成")
    println("📊 参数空间维度: 13维 (8个反应速率 + 5个初始浓度)")

    # 步骤2：创建代理模型
    println("\n🏗️  步骤2: 创建代理模型")
    surrogate_model = SurrogateModel(config, param_space)

    # 步骤3：生成小规模训练数据
    println("\n📊 步骤3: 生成小规模训练数据")
    X_data, y_data = generate_small_scale_data(surrogate_model)

    println("✅ 数据生成完成")
    println("📈 训练样本数: $(size(X_data, 1))")
    println("📉 输入维度: $(size(X_data, 2))")
    println("📊 输出维度: $(size(y_data, 2))")

    # 步骤4：数据预处理
    println("\n🔧 步骤4: 数据预处理和降维")
    preprocess_data!(surrogate_model, X_data, y_data)

    # 步骤5：训练代理模型
    println("\n🎯 步骤5: 训练代理模型")
    train_surrogate_model!(surrogate_model)

    # 步骤6：模型验证
    println("\n✅ 步骤6: 模型验证")
    validate_surrogate_model(surrogate_model)

    # 步骤7：保存模型
    println("\n💾 步骤7: 保存训练好的模型")
    model_path = "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/trained_surrogate.jld2"
    save_surrogate_model(surrogate_model, model_path)

    # 步骤8：演示快速预测
    println("\n🚀 步骤8: 演示快速预测")
    demonstrate_fast_prediction(surrogate_model, param_space)

    println("\n🎉 代理模型工作流程完成!")
    println("💡 现在可以使用代理模型进行快速参数扫描，避免昂贵的CUDA计算")

    return surrogate_model
end

"""
    validate_surrogate_model(surrogate_model::SurrogateModel)

验证代理模型性能
"""
function validate_surrogate_model(surrogate_model::SurrogateModel)
    println("🔍 验证代理模型性能...")

    # 使用验证集进行预测
    if size(surrogate_model.X_val, 1) > 0
        y_pred_mean, y_pred_std = predict_with_uncertainty(surrogate_model,
                                    surrogate_model.X_val, n_samples=50)

        # 反标准化真实值进行比较
        y_true = surrogate_model.y_val .* surrogate_model.output_scaler.std .+ surrogate_model.output_scaler.mean

        # 计算各种误差指标
        mse = mean((y_pred_mean - y_true).^2)
        mae = mean(abs.(y_pred_mean - y_true))

        # 计算R²
        ss_res = sum((y_true - y_pred_mean).^2)
        ss_tot = sum((y_true .- mean(y_true, dims=1)).^2)
        r2 = 1 - ss_res / ss_tot

        println("📊 验证结果:")
        println("   MSE: $(round(mse, digits=6))")
        println("   MAE: $(round(mae, digits=6))")
        println("   R²:  $(round(mean(r2), digits=4))")

        # 不确定性统计
        if surrogate_model.config.uncertainty_estimation
            mean_uncertainty = mean(y_pred_std)
            println("   平均不确定性: $(round(mean_uncertainty, digits=6))")
        end
    end

    println("✅ 模型验证完成")
end

"""
    demonstrate_fast_prediction(surrogate_model::SurrogateModel, param_space::ParameterSpace)

演示快速预测能力
"""
function demonstrate_fast_prediction(surrogate_model::SurrogateModel, param_space::ParameterSpace)
    println("⚡ 演示快速预测能力...")

    # 生成测试参数
    n_test = 1000
    X_test = generate_lhs_samples(param_space, n_test)

    println("🧪 测试参数数量: $n_test")

    # 代理模型预测（快速）
    println("🚀 代理模型预测中...")
    t_surrogate = @elapsed begin
        y_pred_mean, y_pred_std = predict_with_uncertainty(surrogate_model, X_test, n_samples=50)
    end

    println("⚡ 代理模型预测时间: $(round(t_surrogate, digits=3))秒")
    println("📊 预测速度: $(round(n_test/t_surrogate, digits=1)) 预测/秒")

    # 估算CUDA仿真时间（基于经验）
    estimated_cuda_time = n_test * 0.1  # 假设每个仿真0.1秒
    speedup = estimated_cuda_time / t_surrogate

    println("🐌 估算CUDA仿真时间: $(round(estimated_cuda_time, digits=1))秒")
    println("🚀 加速比: $(round(speedup, digits=1))x")
    println("💰 计算量减少: $(round((1 - 1/speedup)*100, digits=1))%")

    # 显示一些预测结果
    println("\n📋 示例预测结果 (前5个):")
    target_vars = surrogate_model.config.target_variables
    for i in 1:min(5, size(y_pred_mean, 1))
        println("  样本 $i:")
        for (j, var) in enumerate(target_vars)
            mean_val = y_pred_mean[i, j]
            std_val = y_pred_std[i, j]
            println("    $var: $(round(mean_val, digits=3)) ± $(round(std_val, digits=3))")
        end
    end
end

"""
    compare_with_original_simulation()

与原始仿真结果比较（小规模测试）
"""
function compare_with_original_simulation(surrogate_model::SurrogateModel; n_compare::Int=10)
    println("\n🔬 与原始仿真结果比较...")

    param_space = surrogate_model.param_space
    X_test = generate_lhs_samples(param_space, n_compare)

    # 代理模型预测
    y_pred_mean, y_pred_std = predict_with_uncertainty(surrogate_model, X_test, n_samples=50)

    # 原始仿真
    println("🧮 运行原始仿真进行比较...")
    y_true = simulate_parameter_batch(X_test, param_space.tspan, surrogate_model.config.target_variables)

    # 过滤有效结果
    valid_indices = findall(x -> !any(isnan.(x)), eachrow(y_true))
    if length(valid_indices) < n_compare
        println("⚠️  只有 $(length(valid_indices))/$n_compare 个有效仿真结果")
    end

    # 计算误差
    y_pred_valid = y_pred_mean[valid_indices, :]
    y_true_valid = y_true[valid_indices, :]

    errors = abs.(y_pred_valid - y_true_valid)
    relative_errors = errors ./ (abs.(y_true_valid) .+ 1e-8)

    println("📊 比较结果:")
    target_vars = surrogate_model.config.target_variables
    for (j, var) in enumerate(target_vars)
        mean_error = mean(errors[:, j])
        mean_rel_error = mean(relative_errors[:, j]) * 100
        println("  $var:")
        println("    平均绝对误差: $(round(mean_error, digits=4))")
        println("    平均相对误差: $(round(mean_rel_error, digits=2))%")
    end

    println("✅ 比较完成")
end

"""
    create_high_density_predictions()

创建高密度预测结果（替代全网格扫描）
"""
function create_high_density_predictions(surrogate_model::SurrogateModel; n_predictions::Int=50000)
    println("\n🎯 创建高密度预测结果...")
    println("📊 目标预测数量: $n_predictions")

    param_space = surrogate_model.param_space

    # 生成高密度参数网格
    X_dense = generate_lhs_samples(param_space, n_predictions)

    # 批量预测
    println("🚀 批量预测中...")
    batch_size = 1000
    n_batches = ceil(Int, n_predictions / batch_size)

    y_pred_all = Matrix{Float64}(undef, n_predictions, length(surrogate_model.config.target_variables))
    y_std_all = Matrix{Float64}(undef, n_predictions, length(surrogate_model.config.target_variables))

    @showprogress "预测进度: " for i in 1:n_batches
        start_idx = (i-1) * batch_size + 1
        end_idx = min(i * batch_size, n_predictions)

        X_batch = X_dense[start_idx:end_idx, :]
        y_pred_batch, y_std_batch = predict_with_uncertainty(surrogate_model, X_batch, n_samples=20)

        y_pred_all[start_idx:end_idx, :] = y_pred_batch
        y_std_all[start_idx:end_idx, :] = y_std_batch
    end

    println("✅ 高密度预测完成")

    # 保存结果
    results_path = "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/high_density_predictions.jld2"
    jldsave(results_path;
            X_parameters=X_dense,
            y_predictions=y_pred_all,
            y_uncertainties=y_std_all,
            target_variables=surrogate_model.config.target_variables)

    println("💾 结果保存到: $results_path")

    return X_dense, y_pred_all, y_std_all
end

# 如果直接运行此文件，执行主工作流程
if abspath(PROGRAM_FILE) == @__FILE__
    println("🎬 执行代理模型工作流程...")
    surrogate_model = main_surrogate_workflow()

    # 可选：创建高密度预测
    if length(ARGS) > 0 && ARGS[1] == "--high-density"
        create_high_density_predictions(surrogate_model, n_predictions=50000)
    end

    # 可选：与原始仿真比较
    if length(ARGS) > 0 && ARGS[1] == "--compare"
        compare_with_original_simulation(surrogate_model, n_compare=20)
    end
end

export main_surrogate_workflow, validate_surrogate_model, demonstrate_fast_prediction
export compare_with_original_simulation, create_high_density_predictions
