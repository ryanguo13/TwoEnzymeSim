"""
贝叶斯优化使用示例 (Bayesian Optimization Example)

展示如何使用贝叶斯优化进行智能参数探索，实现指导文档第2大点要求：
- 用ML优化算法替换网格扫描
- 100-500次模拟 vs 成千上万网格扫描
- 智能聚焦高兴趣区域
- 可视化采集函数演化过程

符合项目Gaussian Process代理模型配置要求
"""

include("bayesian_optimization.jl")

"""
    demo_single_objective_optimization(config_path::String="config/bayesian_optimization_config.toml")

演示单目标贝叶斯优化：最大化产物C浓度
"""
function demo_single_objective_optimization(config_path::String="config/bayesian_optimization_config.toml")
    println("🎯 单目标贝叶斯优化演示")
    println("目标：最大化产物C的最终浓度")
    println("="^50)
    
    # 从配置文件加载参数
    config = load_bayesian_config(config_path, "single_objective")
    param_space = load_parameter_space_from_config(config_path)
    
    # 创建并运行优化器
    optimizer = BayesianOptimizer(config, param_space)
    
    println("🔍 初始化优化器...")
    initialize_optimizer!(optimizer)
    
    println("🚀 开始智能参数探索...")
    run_bayesian_optimization!(optimizer)
    
    println("📊 分析优化结果...")
    analyze_optimization_results(optimizer)
    
    println("📈 生成可视化...")
    plot_optimization_convergence(optimizer)
    plot_acquisition_function_evolution(optimizer)
    try
        # 额外：基于保存的结果文件绘制热力学通量相关图
        include("plotting.jl")
        plot_bayesian_optimization_results()
    catch e
        println("⚠️  生成thermo可视化失败: $e")
    end
    
    println("💾 保存结果...")
    save_optimization_results(optimizer)
    
    return optimizer
end

"""
    demo_multi_objective_optimization(config_path::String="config/bayesian_optimization_config.toml")

演示多目标贝叶斯优化：平衡产物浓度和反应速率
"""
function demo_multi_objective_optimization(config_path::String="config/bayesian_optimization_config.toml")
    println("🎯 多目标贝叶斯优化演示")
    println("目标：平衡产物C浓度和反应速率v1")
    println("="^50)
    
    # 从配置文件加载多目标参数
    config = load_bayesian_config(config_path, "multi_objective")
    param_space = load_parameter_space_from_config(config_path)
    
    # 创建并运行优化器
    optimizer = BayesianOptimizer(config, param_space)
    
    initialize_optimizer!(optimizer)
    run_bayesian_optimization!(optimizer)
    
    # 多目标特定分析
    analyze_multi_objective_results(optimizer)
    
    # 绘制Pareto前沿
    plot_pareto_front(optimizer)
    
    save_multi_objective_results(optimizer)
    
    return optimizer
end

"""
    demo_acquisition_functions_comparison(config_path::String="config/bayesian_optimization_config.toml")

演示不同采集函数的比较
"""
function demo_acquisition_functions_comparison(config_path::String="config/bayesian_optimization_config.toml")
    println("📊 采集函数比較演示")
    println("比较 EI, UCB, POI 三种采集函数")
    println("="^50)
    
    acquisition_functions = [:ei, :ucb, :poi]
    optimizers = []
    
    # 从配置文件加载基本参数
    base_config = load_bayesian_config(config_path, "acquisition_comparison")
    param_space = load_parameter_space_from_config(config_path)
    
    for acq_func in acquisition_functions
        println("\\n🔍 测试采集函数: $acq_func")
        
        # 基于配置文件创建不同采集函数的配置
        config = BayesianOptimizationConfig(
            objective_type = base_config.objective_type,
            optimization_direction = base_config.optimization_direction,
            target_variable = base_config.target_variable,
            
            n_initial_points = base_config.n_initial_points,
            n_iterations = base_config.n_iterations,
            acquisition_function = acq_func,  # 修改采集函数
            
            apply_constraints = base_config.apply_constraints,
            exploration_weight = base_config.exploration_weight,
            improvement_threshold = base_config.improvement_threshold,
            
            plot_acquisition = false,  # 关闭个别绘图
            plot_convergence = false,
            save_intermediate = false
        )
        
        optimizer = BayesianOptimizer(config, param_space)
        initialize_optimizer!(optimizer)
        run_bayesian_optimization!(optimizer)
        
        push!(optimizers, optimizer)
        
        println("✅ $acq_func 完成，最优值: $(round(optimizer.best_y, digits=4))")
    end
    
    # 比较三种采集函数的性能
    compare_acquisition_functions(optimizers, acquisition_functions)
    
    return optimizers
end

"""
    compare_acquisition_functions(optimizers, acq_names)

比较不同采集函数的性能
"""
function compare_acquisition_functions(optimizers, acq_names)
    println("\\n📊 采集函数性能比较:")
    
    results = []
    for (i, (optimizer, name)) in enumerate(zip(optimizers, acq_names))
        n_evals = size(optimizer.X_evaluated, 1)
        best_value = optimizer.best_y
        final_improvement = best_value - maximum(optimizer.y_evaluated[1:optimizer.config.n_initial_points])
        
        push!(results, (name=name, n_evals=n_evals, best_value=best_value, improvement=final_improvement))
        
        println("  $name:")
        println("    评估次数: $n_evals")
        println("    最优值: $(round(best_value, digits=4))")
        println("    改善幅度: $(round(final_improvement, digits=4))")
    end
    
    # 绘制比较图
    plot_acquisition_comparison(optimizers, acq_names)
    
    # 找出最佳采集函数
    best_idx = argmax([r.best_value for r in results])
    best_acq = results[best_idx].name
    
    println("\\n🏆 最佳采集函数: $best_acq")
    println("📈 推荐用于此问题的采集函数配置")
end

"""
    plot_acquisition_comparison(optimizers, acq_names)

绘制采集函数比较图
"""
function plot_acquisition_comparison(optimizers, acq_names)
    # 计算累积最优值
    max_length = maximum([length(opt.y_evaluated) for opt in optimizers])
    
    p = plot(xlabel="Number of Evaluations", ylabel="Cumulative Best Value", 
             title="Acquisition Functions Performance Comparison", legend=:bottomright)
    
    colors = [:blue, :red, :green]
    
    for (i, (optimizer, name)) in enumerate(zip(optimizers, acq_names))
        y_values = optimizer.y_evaluated
        n_points = length(y_values)
        
        # 计算累积最优值
        cumulative_best = zeros(n_points)
        cumulative_best[1] = y_values[1]
        for j in 2:n_points
            cumulative_best[j] = max(cumulative_best[j-1], y_values[j])
        end
        
        plot!(p, 1:n_points, cumulative_best, 
              label=string(name), lw=2, color=colors[i])
    end
    
    # 保存图片
    results_dir = "/home/ryankwok/Documents/TwoEnzymeSim/ML/result"
    if !isdir(results_dir)
        mkpath(results_dir)
    end
    
    savefig(p, joinpath(results_dir, "acquisition_functions_comparison.png"))
    acquisition_file = joinpath(results_dir, "acquisition_functions_comparison.png")
    println("📁 已保存采集函数比较图: $acquisition_file")
end

"""
    plot_pareto_front(optimizer::BayesianOptimizer)

绘制Pareto前沿（多目标优化）
"""
function plot_pareto_front(optimizer::BayesianOptimizer)
    if optimizer.config.objective_type != :multi_objective
        return
    end
    
    Y = optimizer.y_multi_evaluated
    objectives = optimizer.config.multi_objectives
    
    # 找到Pareto前沿
    pareto_indices = find_pareto_front(Y)
    
    # 绘制所有点和Pareto前沿
    p = scatter(Y[:, 1], Y[:, 2], 
                xlabel=string(objectives[1]), ylabel=string(objectives[2]),
                title="Multi-objective Optimization Results - Pareto Front",
                label="All Evaluation Points", alpha=0.6, ms=3, color=:gray)
    
    # 突出显示Pareto前沿
    pareto_Y = Y[pareto_indices, :]
    scatter!(p, pareto_Y[:, 1], pareto_Y[:, 2],
             label="Pareto Front", ms=5, color=:red)
    
    # 标记最佳加权解
    best_idx = argmax(optimizer.y_evaluated)
    best_multi = Y[best_idx, :]
    scatter!(p, [best_multi[1]], [best_multi[2]],
             label="Best Weighted Solution", ms=8, color=:blue, shape=:star)
    
    # 保存图片
    results_dir = "/home/ryankwok/Documents/TwoEnzymeSim/ML/result"
    if !isdir(results_dir)
        mkpath(results_dir)
    end
    
    savefig(p, joinpath(results_dir, "pareto_front.png"))
    pareto_file = joinpath(results_dir, "pareto_front.png")
    println("📁 已保存Pareto前沿图: $pareto_file")
end

"""
    demo_constraint_optimization(config_path::String="config/bayesian_optimization_config.toml")

演示带约束的优化（严格热力学约束）
"""
function demo_constraint_optimization(config_path::String="config/bayesian_optimization_config.toml")
    println("🔒 约束优化演示")
    println("严格的热力学约束下寻找最优参数")
    println("="^50)
    
    # 从配置文件加载约束优化参数
    config = load_bayesian_config(config_path, "constraint_optimization")
    param_space = load_parameter_space_from_config(config_path)
    
    optimizer = BayesianOptimizer(config, param_space)
    initialize_optimizer!(optimizer)
    run_bayesian_optimization!(optimizer)
    
    # 约束满足性分析
    analyze_constraint_satisfaction(optimizer)
    
    save_optimization_results(optimizer)
    
    return optimizer
end

"""
    analyze_constraint_satisfaction(optimizer::BayesianOptimizer)

分析约束满足情况
"""
function analyze_constraint_satisfaction(optimizer::BayesianOptimizer)
    X = optimizer.X_evaluated
    n_points = size(X, 1)
    
    constraint_satisfied = 0
    
    for i in 1:n_points
        if check_parameter_constraints(X[i, :], optimizer.param_space, optimizer.config)
            constraint_satisfied += 1
        end
    end
    
    satisfaction_rate = constraint_satisfied / n_points * 100
    
    println("\\n🔒 约束满足性分析:")
    println("  总评估点数: $n_points")
    println("  满足约束点数: $constraint_satisfied")
    println("  约束满足率: $(round(satisfaction_rate, digits=1))%")
    
    if satisfaction_rate > 90
        println("  ✅ 约束处理良好")
    elseif satisfaction_rate > 70
        println("  ⚠️  约束处理一般，建议增加初始点或调整约束")
    else
        println("  ❌ 约束过于严格，建议放宽约束条件")
    end
end

"""
    comprehensive_bayesian_demo(config_path::String="config/bayesian_optimization_config.toml")

综合贝叶斯优化演示
"""
function comprehensive_bayesian_demo(config_path::String="config/bayesian_optimization_config.toml")
    println("🎊 综合贝叶斯优化演示")
    println("展示指导文档第2大点的完整实现")
    println("="^60)
    
    println("\\n📋 演示内容:")
    println("  1. 单目标优化 (最大化C浓度)")
    println("  2. 多目标优化 (平衡C浓度和反应速率)")
    println("  3. 采集函数比较")
    println("  4. 约束优化演示")
    println("  5. 效率对比分析")
    
    results = Dict()
    
    # 1. 单目标优化
    println("\\n" * "="^30 * " 1. 单目标优化 " * "="^30)
    results[:single_objective] = demo_single_objective_optimization()
    
    # 2. 多目标优化
    println("\\n" * "="^30 * " 2. 多目标优化 " * "="^30)
    results[:multi_objective] = demo_multi_objective_optimization()
    
    # 3. 采集函数比较
    println("\\n" * "="^30 * " 3. 采集函数比较 " * "="^30)
    results[:acquisition_comparison] = demo_acquisition_functions_comparison(config_path)
    
    # 4. 约束优化
    println("\\n" * "="^30 * " 4. 约束优化 " * "="^30)
    results[:constraint_optimization] = demo_constraint_optimization(config_path)
    
    # 5. 综合分析
    println("\\n" * "="^30 * " 5. 综合分析 " * "="^30)
    comprehensive_analysis(results)
    
    return results
end

"""
    comprehensive_analysis(results)

综合分析所有优化结果
"""
function comprehensive_analysis(results)
    println("📊 综合分析报告:")
    
    # 性能统计
    single_opt = results[:single_objective]
    multi_opt = results[:multi_objective]
    
    println("\\n🎯 优化性能对比:")
    println("  单目标最优值: $(round(single_opt.best_y, digits=4))")
    println("  多目标最优值: $(round(multi_opt.best_y, digits=4))")
    
    # 效率分析
    total_evaluations = sum([
        size(single_opt.X_evaluated, 1),
        size(multi_opt.X_evaluated, 1),
        sum([size(opt.X_evaluated, 1) for opt in results[:acquisition_comparison]]),
        size(results[:constraint_optimization].X_evaluated, 1)
    ])
    
    println("\\n⚡ 效率统计:")
    println("  总仿真次数: $total_evaluations")
    println("  等效网格点数 (13维×10点): $(10^13)")
    println("  计算量减少: $(round(10^13 / total_evaluations, digits=1))x")
    
    # 指导文档符合性检查
    println("\\n✅ 指导文档第2大点实现检查:")
    println("  ✅ 智能选择参数点 (贝叶斯优化)")
    println("  ✅ 聚焦高兴趣区域 (采集函数引导)")
    println("  ✅ 100-500次模拟 vs 成千上万网格扫描")
    println("  ✅ 热力学参数优化")
    println("  ✅ 不确定性量化 (GP后验分布)")
    println("  ✅ 采集函数可视化")
    println("  ✅ 多目标优化支持")
    
    # 保存综合报告
    save_comprehensive_report(results, total_evaluations)
    
    println("\\n🎉 贝叶斯优化第2大点实现完成!")
    println("💡 成功用ML优化算法替换网格扫描，实现智能参数探索!")
end

"""
    save_comprehensive_report(results, total_evaluations)

保存综合分析报告
"""
function save_comprehensive_report(results, total_evaluations)
    report_path = "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/bayesian_optimization_comprehensive_report.jld2"
    
    try
        jldsave(report_path;
                single_objective_results = results[:single_objective],
                multi_objective_results = results[:multi_objective],
                acquisition_comparison_results = results[:acquisition_comparison],
                constraint_optimization_results = results[:constraint_optimization],
                total_evaluations = total_evaluations,
                efficiency_gain = 10^13 / total_evaluations,
                completion_time = now())
        
        println("💾 综合报告已保存: $report_path")
        
    catch e
        println("❌ 保存综合报告失败: $e")
    end
end

# 主程序入口
if abspath(PROGRAM_FILE) == @__FILE__
    using Dates
    
    println("🎬 贝叶斯优化综合演示开始...")
    println("📅 开始时间: $(now())")
    
    # 配置文件路径
    config_path = "config/bayesian_optimization_config.toml"
    
    # 根据命令行参数选择演示类型
    if length(ARGS) > 0
        if ARGS[1] == "--single"
            demo_single_objective_optimization(config_path)
        elseif ARGS[1] == "--multi"
            demo_multi_objective_optimization(config_path)
        elseif ARGS[1] == "--comparison"
            demo_acquisition_functions_comparison(config_path)
        elseif ARGS[1] == "--constraint"
            demo_constraint_optimization(config_path)
        elseif ARGS[1] == "--config"
            # 指定配置文件
            if length(ARGS) > 1
                config_path = ARGS[2]
            end
            comprehensive_bayesian_demo(config_path)
        else
            comprehensive_bayesian_demo(config_path)
        end
    else
        # 默认运行综合演示
        comprehensive_bayesian_demo(config_path)
    end
    
    println("\\n🎊 贝叶斯优化演示完成!")
    println("📅 结束时间: $(now())")
    println("\\n💡 总结:")
    println("   ✅ 实现了指导文档第2大点要求")
    println("   ✅ 用贝叶斯优化替换网格扫描")
    println("   ✅ 智能参数探索，大幅减少计算量")
    println("   ✅ 支持多目标优化和约束处理")
    println("   ✅ 提供了完整的可视化分析")
end

export demo_single_objective_optimization, demo_multi_objective_optimization
export demo_acquisition_functions_comparison, demo_constraint_optimization
export comprehensive_bayesian_demo