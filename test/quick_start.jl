"""
快速启动脚本 - ML代理模型

一键启动代理模型训练和使用，替换昂贵的CUDA参数扫描
"""

# 安装必要的包（如果尚未安装）
using Pkg

required_packages = [
    "Flux", "MLJ", "Surrogates", "MultivariateStats", 
    "JLD2", "ProgressMeter", "Plots", "Statistics"
]

println("📦 检查并安装必要的包...")
for pkg in required_packages
    try
        @eval using $(Symbol(pkg))
    catch
        println("🔧 安装 $pkg...")
        Pkg.add(pkg)
    end
end

# 导入模块
include("surrogate_model.jl")
include("gaussian_process.jl")
include("example_usage.jl")

"""
    quick_train_surrogate(; model_type::Symbol=:neural_network, sample_fraction::Float64=0.1)

快速训练代理模型
"""
function quick_train_surrogate(; 
                              model_type::Symbol=:neural_network,
                              sample_fraction::Float64=0.1,
                              max_samples::Int=5000,
                              epochs::Int=100)
    
    println("🚀 快速代理模型训练")
    println(repeat("=", 50))
    
    # 创建配置
    config = SurrogateModelConfig(
        sample_fraction = sample_fraction,
        max_samples = max_samples,
        model_type = model_type,
        hidden_dims = [64, 32, 16],
        dropout_rate = 0.1,
        epochs = epochs,
        batch_size = 32,
        learning_rate = 1e-3,
        use_pca = true,
        uncertainty_estimation = true
    )
    
    # 创建参数空间
    param_space = create_default_parameter_space()
    
    # 训练模型
    println("🔧 开始训练...")
    surrogate_model = SurrogateModel(config, param_space)
    
    # 生成数据
    X_data, y_data = generate_small_scale_data(surrogate_model)
    preprocess_data!(surrogate_model, X_data, y_data)
    
    # 训练
    if model_type == :gaussian_process
        train_gaussian_process!(surrogate_model)
    else
        train_surrogate_model!(surrogate_model)
    end
    
    # 验证
    validate_surrogate_model(surrogate_model)
    
    # 保存
    model_path = "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/quick_trained_surrogate.jld2"
    save_surrogate_model(surrogate_model, model_path)
    
    println("✅ 快速训练完成!")
    println("💾 模型保存在: $model_path")
    
    return surrogate_model
end

"""
    quick_predict(parameters::Dict; model_path::String="")

使用训练好的模型进行快速预测
"""
function quick_predict(parameters::Dict; model_path::String="/home/ryankwok/Documents/TwoEnzymeSim/ML/model/quick_trained_surrogate.jld2")
    
    if !isfile(model_path)
        println("❌ 模型文件不存在: $model_path")
        println("💡 请先运行 quick_train_surrogate() 训练模型")
        return nothing
    end
    
    # 加载模型
    surrogate_model = load_surrogate_model(model_path)
    
    # 准备输入参数
    param_names = [:k1f, :k1r, :k2f, :k2r, :k3f, :k3r, :k4f, :k4r, :A, :B, :C, :E1, :E2]
    X_input = zeros(1, 13)
    
    for (i, param) in enumerate(param_names)
        if haskey(parameters, param)
            X_input[1, i] = parameters[param]
        else
            # 使用默认值
            default_values = [2.0, 1.5, 1.8, 1.0, 1.2, 1.0, 1.6, 0.8, 5.0, 0.0, 0.0, 20.0, 15.0]
            X_input[1, i] = default_values[i]
            println("⚠️  使用默认值 $param = $(default_values[i])")
        end
    end
    
    # 预测
    if surrogate_model.config.model_type == :gaussian_process
        y_pred = predict_gaussian_process(surrogate_model, X_input)
        y_std = zeros(size(y_pred))
    else
        y_pred, y_std = predict_with_uncertainty(surrogate_model, X_input, n_samples=50)
    end
    
    # 显示结果
    target_vars = surrogate_model.config.target_variables
    println("\n📊 预测结果:")
    for (i, var) in enumerate(target_vars)
        mean_val = y_pred[1, i]
        std_val = y_std[1, i]
        println("  $var: $(round(mean_val, digits=4)) ± $(round(std_val, digits=4))")
    end
    
    return y_pred, y_std
end

"""
    quick_parameter_scan(param_ranges::Dict; n_samples::Int=1000, model_path::String="")

使用代理模型进行快速参数扫描
"""
function quick_parameter_scan(param_ranges::Dict; 
                             n_samples::Int=1000,
                             model_path::String="/home/ryankwok/Documents/TwoEnzymeSim/ML/model/quick_trained_surrogate.jld2")
    
    if !isfile(model_path)
        println("❌ 模型文件不存在: $model_path")
        println("💡 请先运行 quick_train_surrogate() 训练模型")
        return nothing
    end
    
    println("🔍 快速参数扫描 ($n_samples 个样本)")
    
    # 加载模型
    surrogate_model = load_surrogate_model(model_path)
    
    # 生成扫描参数
    param_space = surrogate_model.param_space
    X_scan = generate_lhs_samples(param_space, n_samples)
    
    # 应用用户指定的参数范围
    param_names = [:k1f, :k1r, :k2f, :k2r, :k3f, :k3r, :k4f, :k4r, :A, :B, :C, :E1, :E2]
    for (param, range) in param_ranges
        if param in param_names
            param_idx = findfirst(x -> x == param, param_names)
            X_scan[:, param_idx] .= rand(range, n_samples)
            println("🎯 设置 $param 范围: $(minimum(range)) - $(maximum(range))")
        end
    end
    
    # 批量预测
    println("🚀 批量预测中...")
    if surrogate_model.config.model_type == :gaussian_process
        y_pred = predict_gaussian_process(surrogate_model, X_scan)
        y_std = zeros(size(y_pred))
    else
        y_pred, y_std = predict_with_uncertainty(surrogate_model, X_scan, n_samples=20)
    end
    
    # 保存结果
    results_path = "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/quick_scan_results.jld2"
    jldsave(results_path; 
            X_parameters=X_scan, 
            y_predictions=y_pred, 
            y_uncertainties=y_std,
            target_variables=surrogate_model.config.target_variables,
            param_ranges=param_ranges)
    
    println("💾 扫描结果保存到: $results_path")
    
    # 简单统计
    target_vars = surrogate_model.config.target_variables
    println("\n📊 扫描结果统计:")
    for (i, var) in enumerate(target_vars)
        mean_val = mean(y_pred[:, i])
        std_val = std(y_pred[:, i])
        min_val = minimum(y_pred[:, i])
        max_val = maximum(y_pred[:, i])
        
        println("  $var:")
        println("    均值: $(round(mean_val, digits=4))")
        println("    标准差: $(round(std_val, digits=4))")
        println("    范围: $(round(min_val, digits=4)) - $(round(max_val, digits=4))")
    end
    
    return X_scan, y_pred, y_std
end

"""
    quick_visualization(results_path::String=""; show_uncertainty::Bool=true)

快速可视化扫描结果
"""
function quick_visualization(results_path::String="/home/ryankwok/Documents/TwoEnzymeSim/ML/model/quick_scan_results.jld2"; 
                           show_uncertainty::Bool=true)
    
    if !isfile(results_path)
        println("❌ 结果文件不存在: $results_path")
        return nothing
    end
    
    println("📊 加载和可视化结果...")
    
    # 加载数据
    data = load(results_path)
    X_params = data["X_parameters"]
    y_pred = data["y_predictions"]
    y_std = data["y_uncertainties"]
    target_vars = data["target_variables"]
    
    plots_array = []
    
    # 为每个目标变量创建图表
    for (i, var) in enumerate(target_vars)
        # 直方图
        p1 = histogram(y_pred[:, i], 
                      title="$var 分布", 
                      xlabel="值", ylabel="频次",
                      alpha=0.7, bins=50)
        
        # 如果有不确定性信息，添加误差条
        if show_uncertainty && any(y_std[:, i] .> 0)
            # 选择部分样本显示不确定性
            n_show = min(100, size(y_pred, 1))
            indices = rand(1:size(y_pred, 1), n_show)
            
            p2 = scatter(indices, y_pred[indices, i], 
                        yerror=y_std[indices, i],
                        title="$var 不确定性", 
                        xlabel="样本索引", ylabel="值",
                        alpha=0.6, markersize=2)
        else
            # 简单的值分布
            p2 = plot(y_pred[:, i], 
                     title="$var 值序列", 
                     xlabel="样本索引", ylabel="值",
                     linewidth=1, alpha=0.7)
        end
        
        push!(plots_array, p1)
        push!(plots_array, p2)
    end
    
    # 组合图表
    n_vars = length(target_vars)
    combined_plot = plot(plots_array..., 
                        layout=(n_vars, 2), 
                        size=(1200, 300*n_vars))
    
    # 保存图表
    plot_path = "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/quick_visualization.png"
    savefig(combined_plot, plot_path)
    println("💾 可视化结果保存到: $plot_path")
    
    display(combined_plot)
    return combined_plot
end

"""
    compare_with_cuda(n_compare::Int=50)

与CUDA仿真结果比较
"""
function compare_with_cuda(n_compare::Int=50; 
                          model_path::String="/home/ryankwok/Documents/TwoEnzymeSim/ML/model/quick_trained_surrogate.jld2")
    
    if !isfile(model_path)
        println("❌ 模型文件不存在: $model_path")
        return nothing
    end
    
    println("🔬 与CUDA仿真结果比较 ($n_compare 个样本)")
    
    # 加载代理模型
    surrogate_model = load_surrogate_model(model_path)
    param_space = surrogate_model.param_space
    
    # 生成测试参数
    X_test = generate_lhs_samples(param_space, n_compare)
    
    println("⚡ 代理模型预测...")
    t_surrogate = @elapsed begin
        if surrogate_model.config.model_type == :gaussian_process
            y_surrogate = predict_gaussian_process(surrogate_model, X_test)
            y_std = zeros(size(y_surrogate))
        else
            y_surrogate, y_std = predict_with_uncertainty(surrogate_model, X_test, n_samples=20)
        end
    end
    
    println("🐌 CUDA仿真...")
    t_cuda = @elapsed begin
        y_cuda = simulate_parameter_batch(X_test, param_space.tspan, surrogate_model.config.target_variables)
    end
    
    # 过滤有效结果
    valid_indices = findall(x -> !any(isnan.(x)), eachrow(y_cuda))
    n_valid = length(valid_indices)
    
    if n_valid == 0
        println("❌ 没有有效的CUDA仿真结果")
        return nothing
    end
    
    println("✅ 有效比较样本: $n_valid/$n_compare")
    
    # 计算误差
    y_surrogate_valid = y_surrogate[valid_indices, :]
    y_cuda_valid = y_cuda[valid_indices, :]
    
    errors = abs.(y_surrogate_valid - y_cuda_valid)
    relative_errors = errors ./ (abs.(y_cuda_valid) .+ 1e-8)
    
    # 性能对比
    speedup = t_cuda / t_surrogate
    
    println("\n📊 比较结果:")
    println("⚡ 代理模型时间: $(round(t_surrogate, digits=3))秒")
    println("🐌 CUDA仿真时间: $(round(t_cuda, digits=3))秒")
    println("🚀 加速比: $(round(speedup, digits=1))x")
    println("💰 计算量减少: $(round((1 - 1/speedup)*100, digits=1))%")
    
    println("\n🎯 精度分析:")
    target_vars = surrogate_model.config.target_variables
    for (i, var) in enumerate(target_vars)
        mae = mean(errors[:, i])
        mre = mean(relative_errors[:, i]) * 100
        r2 = cor(y_cuda_valid[:, i], y_surrogate_valid[:, i])^2
        
        println("  $var:")
        println("    平均绝对误差: $(round(mae, digits=4))")
        println("    平均相对误差: $(round(mre, digits=2))%")
        println("    R²: $(round(r2, digits=4))")
    end
    
    return y_surrogate_valid, y_cuda_valid, errors
end

"""
    interactive_menu()

交互式菜单
"""
function interactive_menu()
    println("\n🎯 ML代理模型 - 交互式菜单")
    println(repeat("=", 40))
    println("1. 快速训练代理模型")
    println("2. 单点预测")
    println("3. 参数扫描")
    println("4. 可视化结果")
    println("5. 与CUDA比较")
    println("6. 退出")
    println(repeat("=", 40))
    
    while true
        print("请选择操作 (1-6): ")
        choice = readline()
        
        try
            if choice == "1"
                println("\n选择模型类型:")
                println("1. 神经网络 (推荐)")
                println("2. Gaussian Process")
                print("模型类型 (1-2): ")
                model_choice = readline()
                
                model_type = model_choice == "2" ? :gaussian_process : :neural_network
                
                print("采样比例 (0.05-0.2, 默认0.1): ")
                sample_input = readline()
                sample_fraction = isempty(sample_input) ? 0.1 : parse(Float64, sample_input)
                
                quick_train_surrogate(model_type=model_type, sample_fraction=sample_fraction)
                
            elseif choice == "2"
                println("\n输入参数值 (回车使用默认值):")
                params = Dict{Symbol, Float64}()
                
                param_names = [:k1f, :k1r, :k2f, :k2r, :k3f, :k3r, :k4f, :k4r, :A, :B, :C, :E1, :E2]
                default_values = [2.0, 1.5, 1.8, 1.0, 1.2, 1.0, 1.6, 0.8, 5.0, 0.0, 0.0, 20.0, 15.0]
                
                for (i, param) in enumerate(param_names)
                    print("$param (默认$(default_values[i])): ")
                    input = readline()
                    if !isempty(input)
                        params[param] = parse(Float64, input)
                    end
                end
                
                quick_predict(params)
                
            elseif choice == "3"
                println("\n参数扫描设置:")
                print("扫描样本数 (默认1000): ")
                n_input = readline()
                n_samples = isempty(n_input) ? 1000 : parse(Int, n_input)
                
                # 简化：使用默认参数范围
                param_ranges = Dict(
                    :k1f => 0.1:0.1:20.0,
                    :A => 1.0:1.0:25.0
                )
                
                quick_parameter_scan(param_ranges, n_samples=n_samples)
                
            elseif choice == "4"
                quick_visualization()
                
            elseif choice == "5"
                print("比较样本数 (默认50): ")
                n_input = readline()
                n_compare = isempty(n_input) ? 50 : parse(Int, n_input)
                
                compare_with_cuda(n_compare)
                
            elseif choice == "6"
                println("👋 再见!")
                break
                
            else
                println("❌ 无效选择，请输入1-6")
            end
            
        catch e
            println("❌ 错误: $e")
        end
        
        println("\n按回车继续...")
        readline()
    end
end

# 主函数
function main()
    println("🚀 ML代理模型快速启动")
    println("🎯 目标: 用ML代理模型替换CUDA参数扫描，减少计算80%+")
    
    if length(ARGS) == 0
        # 交互式模式
        interactive_menu()
    else
        # 命令行模式
        if ARGS[1] == "train"
            model_type = length(ARGS) > 1 ? Symbol(ARGS[2]) : :neural_network
            quick_train_surrogate(model_type=model_type)
        elseif ARGS[1] == "scan"
            n_samples = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 1000
            param_ranges = Dict(:k1f => 0.1:0.1:20.0, :A => 1.0:1.0:25.0)
            quick_parameter_scan(param_ranges, n_samples=n_samples)
        elseif ARGS[1] == "compare"
            n_compare = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 50
            compare_with_cuda(n_compare)
        else
            println("❌ 未知命令: $(ARGS[1])")
            println("💡 可用命令: train, scan, compare")
        end
    end
end

# 如果直接运行此文件
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

export quick_train_surrogate, quick_predict, quick_parameter_scan
export quick_visualization, compare_with_cuda, interactive_menu
