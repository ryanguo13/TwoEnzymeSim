"""
TOML集成演示 - 展示如何将TOML配置集成到Julia糖酵解模拟中

这个演示展示了：
1. 从TOML文件加载配置
2. 使用配置数据设置参数
3. 运行简化的糖酵解模拟
"""

using Catalyst
using DifferentialEquations
using TOML
using Plots

# 定义参数和变量
@parameters k1f k1r k2f k2r
@variables t Glucose(t) G6P(t) ATP(t) ADP(t)

# 创建简化的糖酵解反应网络（只包含前两步）
glycolysis_demo = @reaction_network begin
    k1f, Glucose + ATP --> G6P + ADP
    k1r, G6P + ADP --> Glucose + ATP
    k2f, G6P --> F6P
    k2r, F6P --> G6P
end

@variables t F6P(t)

# 重新定义网络以包含F6P
glycolysis_demo = @reaction_network begin
    k1f, Glucose + ATP --> G6P + ADP
    k1r, G6P + ADP --> Glucose + ATP
    k2f, G6P --> F6P
    k2r, F6P --> G6P
end

"""
    从TOML文件加载配置并创建参数
"""
function load_config_and_create_parameters()
    println("=== 加载TOML配置 ===")
    
    # 加载TOML配置
    config = TOML.parsefile("deltaG.toml")
    println("✅ TOML配置加载成功")
    
    # 显示配置内容
    println("\n📊 配置内容:")
    println("热力学数据节: ", keys(config["deltaG"]))
    println("初始条件节: ", keys(config["initial_conditions"]))
    println("动力学参数节: ", [k for k in keys(config) if startswith(k, "k")])
    
    # 创建初始条件（使用TOML数据）
    initial_conditions = [
        Glucose => Float64(config["initial_conditions"]["Glucose"]),
        G6P => Float64(config["initial_conditions"]["G6P"]),
        F6P => Float64(config["initial_conditions"]["F6P"]),
        ATP => Float64(config["initial_conditions"]["ATP"]),
        ADP => Float64(config["initial_conditions"]["ADP"])
    ]
    
    # 创建动力学参数（使用TOML数据）
    parameters = [
        k1f => Float64(config["k1"]["kf"]),
        k1r => Float64(config["k1"]["kr"]),
        k2f => Float64(config["k2"]["kf"]),
        k2r => Float64(config["k2"]["kr"])
    ]
    
    println("\n🔧 参数设置:")
    println("初始条件: ", initial_conditions)
    println("动力学参数: ", parameters)
    
    return initial_conditions, parameters, config
end

"""
    运行糖酵解模拟
"""
function run_glycolysis_simulation()
    println("\n=== 运行糖酵解模拟 ===")
    
    # 加载配置和参数
    u0, p, config = load_config_and_create_parameters()
    
    # 创建ODE问题
    ode_prob = ODEProblem(glycolysis_demo, u0, (0.0, 50.0), p)
    
    # 求解
    println("🔄 开始求解...")
    sol = solve(ode_prob, Tsit5(), saveat=0.5)
    println("✅ 求解完成！")
    
    # 显示结果
    println("\n📈 模拟结果:")
    println("最终葡萄糖浓度: $(round(sol[Glucose][end], digits=3)) mM")
    println("最终G6P浓度: $(round(sol[G6P][end], digits=3)) mM")
    println("最终F6P浓度: $(round(sol[F6P][end], digits=3)) mM")
    println("最终ATP浓度: $(round(sol[ATP][end], digits=3)) mM")
    println("最终ADP浓度: $(round(sol[ADP][end], digits=3)) mM")
    
    return sol
end

"""
    创建可视化图表
"""
function create_visualization(sol)
    println("\n=== 创建可视化图表 ===")
    
    # 创建浓度时间曲线
    p1 = plot(sol.t, sol[Glucose], label="Glucose", linewidth=2, color=:blue)
    plot!(p1, sol.t, sol[G6P], label="G6P", linewidth=2, color=:red)
    plot!(p1, sol.t, sol[F6P], label="F6P", linewidth=2, color=:green)
    plot!(p1, xlabel="Time (s)", ylabel="Concentration (mM)", title="糖酵解代谢物浓度变化")
    
    # ATP/ADP浓度变化
    p2 = plot(sol.t, sol[ATP], label="ATP", linewidth=2, color=:orange)
    plot!(p2, sol.t, sol[ADP], label="ADP", linewidth=2, color=:purple)
    plot!(p2, xlabel="Time (s)", ylabel="Concentration (mM)", title="ATP/ADP浓度变化")
    
    # 组合图
    combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
    
    # 保存图表
    savefig(combined_plot, "glycolysis_toml_demo.png")
    println("📊 图表已保存为 glycolysis_toml_demo.png")
    
    return combined_plot
end

"""
    演示TOML配置的优势
"""
function demonstrate_toml_advantages()
    println("\n=== TOML配置的优势 ===")
    
    # 加载配置
    config = TOML.parsefile("deltaG.toml")
    
    println("🎯 配置管理优势:")
    println("1. 所有参数集中在一个文件中")
    println("2. 易于修改和调整")
    println("3. 支持注释和文档")
    println("4. 版本控制友好")
    
    println("\n📋 当前配置摘要:")
    println("热力学步骤数: $(length(config["deltaG"]))")
    println("代谢物数量: $(length(config["initial_conditions"]))")
    println("动力学参数组数: $(length([k for k in keys(config) if startswith(k, "k")]))")
    
    println("\n🔧 配置示例:")
    println("初始葡萄糖浓度: $(config["initial_conditions"]["Glucose"]) mM")
    println("第一步ΔG: $(config["deltaG"]["step1"]) kJ/mol")
    println("k1正向速率: $(config["k1"]["kf"])")
end

# 主程序
println("🚀 TOML集成演示开始")
println("="^50)

# 运行演示
sol = run_glycolysis_simulation()
plot_result = create_visualization(sol)
demonstrate_toml_advantages()

println("\n" * "="^50)
println("✅ TOML集成演示完成！")
println("🎉 所有参数现在都从TOML配置文件加载")
println("📁 配置文件: deltaG.toml")
println("📊 结果图表: glycolysis_toml_demo.png")
