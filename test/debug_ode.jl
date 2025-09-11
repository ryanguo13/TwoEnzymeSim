"""
简化的ODE调试文件

用于定位两酶系统ODE求解问题的根本原因
"""

using DifferentialEquations
using Random
using Printf

# 设置随机种子
Random.seed!(42)

"""
简化的两酶反应系统ODE
"""
function reaction_system!(du, u, p, t)
    # 参数: k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r
    k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = p

    # 状态变量: A, B, C, E1, E2, AE1, BE2
    A, B, C, E1, E2, AE1, BE2 = u

    # 确保非负
    A, B, C, E1, E2, AE1, BE2 = max.(u, 0.0)

    # 两酶反应网络:
    # E1 + A <-> AE1 -> B + E1
    # E2 + B <-> BE2 -> C + E2

    du[1] = -k1f*A*E1 + k1r*AE1                        # dA/dt
    du[2] = k2f*AE1 - k2r*B*E1 - k3f*B*E2 + k3r*BE2   # dB/dt
    du[3] = k4f*BE2 - k4r*C*E2                         # dC/dt
    du[4] = -k1f*A*E1 + k1r*AE1 + k2f*AE1 - k2r*B*E1  # dE1/dt
    du[5] = -k3f*B*E2 + k3r*BE2 + k4f*BE2 - k4r*C*E2  # dE2/dt
    du[6] = k1f*A*E1 - k1r*AE1 - k2f*AE1 + k2r*B*E1   # dAE1/dt
    du[7] = k3f*B*E2 - k3r*BE2 - k4f*BE2 + k4r*C*E2   # dBE2/dt

    return nothing
end

"""
测试单个ODE求解
"""
function test_single_ode()
    println("🔍 测试单个ODE求解")

    # 合理的参数设置
    p = [1.0, 0.5, 2.0, 0.1, 1.5, 0.2, 1.8, 0.3]  # k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r
    u0 = [10.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0]      # A, B, C, E1, E2, AE1, BE2
    tspan = (0.0, 5.0)

    println("初始参数:")
    println("  反应常数: $p")
    println("  初始状态: $u0")
    println("  时间跨度: $tspan")

    # 测试ODE函数
    du_test = zeros(7)
    reaction_system!(du_test, u0, p, 0.0)
    println("  初始导数: $du_test")

    if any(isnan.(du_test)) || any(isinf.(du_test))
        println("❌ ODE函数产生了无效导数")
        return false
    end

    # 求解ODE
    try
        prob = ODEProblem(reaction_system!, u0, tspan, p)
        sol = solve(prob, Tsit5(), abstol=1e-6, reltol=1e-3)

        println("求解结果:")
        println("  返回码: $(sol.retcode)")
        println("  时间点数: $(length(sol.t))")
        println("  状态数: $(length(sol.u))")

        println("  详细检查:")
        println("    sol.retcode: $(sol.retcode)")
        println("    sol.retcode类型: $(typeof(sol.retcode))")
        println("    sol.retcode == :Success? $(sol.retcode == :Success)")
        println("    sol.retcode == SciMLBase.Success? $(sol.retcode == SciMLBase.Success)")
        println("    string(sol.retcode): $(string(sol.retcode))")
        println("    length(sol.u) > 0? $(length(sol.u) > 0)")

        if string(sol.retcode) == "Success" && length(sol.u) > 0
            final_state = sol.u[end]
            println("  最终状态: $final_state")
            println("  最终状态类型: $(typeof(final_state))")
            println("  包含NaN? $(any(isnan.(final_state)))")
            println("  包含Inf? $(any(isinf.(final_state)))")

            # 检查质量守恒
            initial_total = u0[1] + u0[2] + u0[3] + u0[6] + u0[7]  # A + B + C + AE1 + BE2
            final_total = final_state[1] + final_state[2] + final_state[3] + final_state[6] + final_state[7]

            println("  质量守恒检查:")
            println("    初始总量: $initial_total")
            println("    最终总量: $final_total")
            println("    相对误差: $(abs(final_total - initial_total) / initial_total * 100)%")

            if any(isnan.(final_state)) || any(isinf.(final_state))
                println("❌ 最终状态包含无效值")
                return false
            else
                println("✅ 求解成功")
                return true
            end
        else
            println("❌ 求解失败或无解")
            return false
        end

    catch e
        println("❌ 求解异常: $e")
        return false
    end
end

"""
测试批量参数
"""
function test_batch_parameters()
    println("\n🔍 测试批量随机参数")

    n_tests = 10
    success_count = 0

    for i in 1:n_tests
        # 生成随机参数
        p = rand(8) * 10.0 .+ 0.1  # k值: 0.1-10.1
        u0 = [
            rand() * 15.0 + 5.0,   # A: 5-20
            rand() * 5.0,          # B: 0-5
            rand() * 5.0,          # C: 0-5
            rand() * 15.0 + 5.0,   # E1: 5-20
            rand() * 15.0 + 5.0,   # E2: 5-20
            0.0,                   # AE1: 0
            0.0                    # BE2: 0
        ]
        tspan = (0.0, 5.0)

        try
            prob = ODEProblem(reaction_system!, u0, tspan, p)
            sol = solve(prob, Tsit5(), abstol=1e-6, reltol=1e-3, maxiters=10000)

            if string(sol.retcode) == "Success" && length(sol.u) > 0
                final_state = sol.u[end]
                if !any(isnan.(final_state)) && !any(isinf.(final_state))
                    success_count += 1
                    if i <= 3
                        println("  测试 $i ✅: 最终状态 $(round.(final_state, digits=4))")
                    end
                else
                    if i <= 3
                        println("  测试 $i ❌: 最终状态包含NaN/Inf: $final_state")
                    end
                end
            else
                if i <= 3
                    println("  测试 $i ❌: 求解失败 $(sol.retcode), 解长度: $(length(sol.u))")
                end
            end

        catch e
            if i <= 3
                println("  测试 $i ❌: 异常 $e")
            end
        end
    end

    success_rate = success_count / n_tests * 100
    println("批量测试结果: $success_count/$n_tests 成功 ($(round(success_rate, digits=1))%)")

    return success_count > 0
end

"""
测试问题参数识别
"""
function test_problematic_parameters()
    println("\n🔍 测试问题参数识别")

    # 测试一些可能有问题的参数组合
    test_cases = [
        # 案例1: 极大的k值
        ([100.0, 0.1, 100.0, 0.1, 100.0, 0.1, 100.0, 0.1], "极大k值"),
        # 案例2: 极小的k值
        ([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], "极小k值"),
        # 案例3: 不平衡的k值
        ([10.0, 0.001, 10.0, 0.001, 10.0, 0.001, 10.0, 0.001], "不平衡k值"),
        # 案例4: 高浓度
        ([1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5], "高浓度"),
    ]

    for (i, (p_test, description)) in enumerate(test_cases)
        println("  案例 $i - $description:")

        # 正常浓度
        u0_normal = [10.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0]
        # 高浓度（用于案例4）
        u0_high = [100.0, 0.0, 0.0, 50.0, 50.0, 0.0, 0.0]

        u0 = i == 4 ? u0_high : u0_normal
        tspan = (0.0, 5.0)

        try
            prob = ODEProblem(reaction_system!, u0, tspan, p_test)
            sol = solve(prob, Tsit5(), abstol=1e-8, reltol=1e-6, maxiters=50000)

            if string(sol.retcode) == "Success" && length(sol.u) > 0
                final_state = sol.u[end]
                if !any(isnan.(final_state)) && !any(isinf.(final_state))
                    println("    ✅ 成功: $(round.(final_state, digits=4))")
                else
                    println("    ❌ 结果包含NaN/Inf: $final_state")
                end
            else
                println("    ❌ 求解失败: $(sol.retcode), 解长度: $(length(sol.u))")
            end

        catch e
            println("    ❌ 异常: $e")
        end
    end
end

"""
测试求解器选项
"""
function test_solver_options()
    println("\n🔍 测试不同求解器选项")

    # 标准参数
    p = [1.0, 0.5, 2.0, 0.1, 1.5, 0.2, 1.8, 0.3]
    u0 = [10.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0]
    tspan = (0.0, 5.0)

    solvers = [
        (Tsit5(), "Tsit5"),
        (Rosenbrock23(), "Rosenbrock23"),
        (Rodas4(), "Rodas4")
    ]

    for (solver, name) in solvers
        try
            prob = ODEProblem(reaction_system!, u0, tspan, p)
            sol = solve(prob, solver, abstol=1e-6, reltol=1e-3)

            if string(sol.retcode) == "Success" && length(sol.u) > 0
                final_state = sol.u[end]
                if !any(isnan.(final_state)) && !any(isinf.(final_state))
                    println("  $name ✅: 最终状态 $(round.(final_state, digits=4))")
                else
                    println("  $name ❌: 结果包含NaN/Inf: $final_state")
                end
            else
                println("  $name ❌: $(sol.retcode), 解长度: $(length(sol.u))")
            end

        catch e
            println("  $name ❌: $e")
        end
    end
end

"""
主测试函数
"""
function main()
    println("🧪 Two-Enzyme System ODE 调试测试")
    println("="^50)

    # 测试1: 单个ODE求解
    test1_success = test_single_ode()

    # 测试2: 批量参数
    test2_success = test_batch_parameters()

    # 测试3: 问题参数
    test_problematic_parameters()

    # 测试4: 求解器选项
    test_solver_options()

    # 总结
    println("\n📊 调试总结:")
    println("  单个ODE测试: $(test1_success ? "✅" : "❌")")
    println("  批量参数测试: $(test2_success ? "✅" : "❌")")

    if test1_success && test2_success
        println("🎉 ODE求解系统基本正常")
        println("💡 问题可能在于:")
        println("   - 参数生成范围")
        println("   - 数据类型转换")
        println("   - 批处理逻辑")
    else
        println("❌ ODE求解系统存在基础问题")
        println("💡 需要检查:")
        println("   - 反应网络方程")
        println("   - 参数合理性")
        println("   - 求解器设置")
    end
end

# 运行测试
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
