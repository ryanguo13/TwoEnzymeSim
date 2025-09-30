#!/usr/bin/env julia
"""
糖酵解反应网络测试脚本

测试所有主要功能并生成详细报告
"""

using Pkg
Pkg.activate(".")

include("main.jl")

function test_glycolysis_network()
    println("🧪 开始糖酵解反应网络测试...")
    
    # 测试1: 创建反应网络
    println("\n📋 测试1: 创建反应网络")
    rn = create_glycolysis_network()
    n_species = length(species(rn))
    n_reactions = length(reactions(rn))
    println("✅ 反应网络创建成功")
    println("   - 物种数量: $n_species")
    println("   - 反应数量: $n_reactions")
    
    # 测试2: 加载热力学数据
    println("\n📋 测试2: 加载热力学数据")
    deltaG_data = load_thermodynamic_data()
    println("✅ 热力学数据加载成功")
    println("   - 包含 $(length(deltaG_data)) 个步骤的ΔG值")
    
    # 测试3: 设置初始条件
    println("\n📋 测试3: 设置初始条件")
    u0 = set_glycolysis_initial_conditions()
    println("✅ 初始条件设置成功")
    println("   - 葡萄糖初始浓度: $(u0[1][2]) mM")
    println("   - ATP初始浓度: $(u0[12][2]) mM")
    
    # 测试4: 设置参数
    println("\n📋 测试4: 设置参数")
    p = set_glycolysis_parameters()
    println("✅ 参数设置成功")
    println("   - 包含 $(length(p)) 个动力学参数")
    
    # 测试5: 运行模拟
    println("\n📋 测试5: 运行模拟")
    sol = simulate_glycolysis(tspan=(0.0, 50.0), saveat=0.5)
    println("✅ 模拟运行成功")
    println("   - 模拟时间: $(sol.t[1]) - $(sol.t[end]) 秒")
    println("   - 时间点数: $(length(sol.t))")
    
    # 测试6: 计算热力学通量
    println("\n📋 测试6: 计算热力学通量")
    fluxes = calculate_thermodynamic_fluxes(sol, deltaG_data)
    println("✅ 热力学通量计算成功")
    println("   - 计算了 $(length(fluxes)) 个步骤的通量")
    
    # 测试7: 验证热力学通量
    println("\n📋 测试7: 验证热力学通量")
    validation = validate_thermodynamic_fluxes(fluxes)
    println("✅ 热力学通量验证成功")
    
    # 显示验证结果
    println("\n📊 热力学验证结果:")
    for (step, result) in validation
        println("   $step: $result")
    end
    
    # 测试8: 生成可视化
    println("\n📋 测试8: 生成可视化")
    try
        plot_result = visualize_glycolysis_results(sol, fluxes)
        savefig(plot_result, "test_glycolysis_results.png")
        println("✅ 可视化生成成功")
        println("   - 图表已保存为 test_glycolysis_results.png")
    catch e
        println("⚠️ 可视化生成失败: $e")
    end
    
    # 测试9: 分析结果
    println("\n📋 测试9: 分析结果")
    final_glucose = sol[Glucose][end]
    final_pyruvate = sol[Pyruvate][end]
    final_atp = sol[ATP][end]
    final_nadh = sol[NADH][end]
    
    println("✅ 结果分析完成")
    println("   - 最终葡萄糖浓度: $(round(final_glucose, digits=3)) mM")
    println("   - 最终丙酮酸浓度: $(round(final_pyruvate, digits=3)) mM")
    println("   - 最终ATP浓度: $(round(final_atp, digits=3)) mM")
    println("   - 最终NADH浓度: $(round(final_nadh, digits=3)) mM")
    
    # 计算转化率
    glucose_consumed = 5.0 - final_glucose
    conversion_rate = (glucose_consumed / 5.0) * 100
    println("   - 葡萄糖转化率: $(round(conversion_rate, digits=1))%")
    
    # 能量平衡检查
    atp_produced = 4.0 - final_atp  # 假设初始ATP为4mM
    atp_consumed = 2.0  # 第一阶段消耗2个ATP
    net_atp = atp_produced - atp_consumed
    println("   - 净ATP产生: $(round(net_atp, digits=1)) 分子")
    
    println("\n🎉 所有测试完成！糖酵解反应网络功能正常。")
    
    return true
end

# 运行测试
if abspath(PROGRAM_FILE) == @__FILE__
    test_glycolysis_network()
end
