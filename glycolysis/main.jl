"""
糖酵解反应网络模拟 - 使用Catalyst.jl构建完整的10步糖酵解反应网络

糖酵解是细胞在细胞质基质中将葡萄糖分解为丙酮酸的过程，共包含10步酶催化反应，
分为两个阶段：活化吸能阶段（前5步）和产能阶段（后5步）。

网络结构：
- 第一阶段：活化吸能阶段（消耗2分子ATP）
- 第二阶段：产能阶段（生成4分子ATP和2分子NADH）
"""

using Plots
using ProgressMeter
using Catalyst
using DifferentialEquations
using Unitful
using TOML
# using GraphMakie, NetworkLayout
# using CairoMakie


# include("../src/simulation.jl")
# include("../src/analysis.jl")


@parameters k1f k1r k2f k2r k3f k3r k4f k4r k5f k5r k6f k6r k7f k7r k8f k8r k9f k9r k10f k10r katp_in kadp_out knad_in knadh_out kpi_in kadp_in katp_out kh_out

@variables t Glucose(t) G6P(t) F6P(t) F16BP(t) DHAP(t) GAP(t) BPG13(t) PG3(t) PG2(t) PEP(t) Pyruvate(t) ATP(t) ADP(t) NAD(t) NADH(t) Pi(t) H2O(t) H(t)


glycolysis_network = @reaction_network begin
    # 第一阶段：活化吸能阶段
    # 1. 葡萄糖磷酸化 (己糖激酶)
    k1f, Glucose + ATP --> G6P + ADP
    k1r, G6P + ADP --> Glucose + ATP
    
    # 2. 6-磷酸葡萄糖异构化 (磷酸己糖异构酶)
    k2f, G6P --> F6P
    k2r, F6P --> G6P
    
    # 3. 6-磷酸果糖磷酸化 (磷酸果糖激酶-1)
    k3f, F6P + ATP --> F16BP + ADP
    k3r, F16BP + ADP --> F6P + ATP
    
    # 4. 1,6-二磷酸果糖裂解 (醛缩酶)
    k4f, F16BP --> DHAP + GAP
    k4r, DHAP + GAP --> F16BP
    
    # 5. 磷酸二羟丙酮异构化 (丙糖磷酸异构酶)
    k5f, DHAP --> GAP
    k5r, GAP --> DHAP
    
    # 第二阶段：产能阶段
    # 6. 3-磷酸甘油醛氧化 (3-磷酸甘油醛脱氢酶)
    k6f, GAP + NAD + Pi --> BPG13 + NADH + H
    k6r, BPG13 + NADH + H --> GAP + NAD + Pi
    
    # 7. 1,3-二磷酸甘油酸底物水平磷酸化 (磷酸甘油酸激酶)
    k7f, BPG13 + ADP --> PG3 + ATP
    k7r, PG3 + ATP --> BPG13 + ADP
    
    # 8. 3-磷酸甘油酸变位 (磷酸甘油酸变位酶)
    k8f, PG3 --> PG2
    k8r, PG2 --> PG3
    
    # 9. 2-磷酸甘油酸脱水 (烯醇化酶)
    k9f, PG2 --> PEP + H2O
    k9r, PEP + H2O --> PG2
    
    # 10. 磷酸烯醇式丙酮酸底物水平磷酸化 (丙酮酸激酶)
    k10f, PEP + ADP --> Pyruvate + ATP
    k10r, Pyruvate + ATP --> PEP + ADP

    # 外围供给/移除（开放体系，维持能量与氧化还原池）
    katp_in, 0 --> ATP
    kadp_out, ADP --> 0
    kadp_in, 0 --> ADP
    katp_out, ATP --> 0
    knad_in, 0 --> NAD
    knadh_out, NADH --> 0
    kpi_in, 0 --> Pi
    kh_out, H --> 0
end

"""
    构建糖酵解反应网络

返回完整的糖酵解反应网络，包含10步反应和所有代谢物
"""
function create_glycolysis_network()
    return glycolysis_network
end

"""
    加载TOML配置文件

从TOML文件加载所有糖酵解参数，包括热力学数据、动力学参数和初始条件
"""
function load_glycolysis_config(config_path="deltaG.toml")
    try
        resolved_path = isabspath(config_path) ? config_path : joinpath(@__DIR__, config_path)
        config = TOML.parsefile(resolved_path)
        required_sections = ["deltaG", "initial_conditions"]
        for section in required_sections
            if !haskey(config, section)
                error("Missing required section '$section' in config file")
            end
        end
        
        return config
    catch e
        error("Failed to load config file '$config_path': $e")
    end
end

"""
    获取糖酵解热力学数据

从TOML配置中读取各步骤的ΔG值，单位为kJ/mol
"""
function load_thermodynamic_data(config=nothing)
    if config === nothing
        config = load_glycolysis_config()
    end
    
    deltaG_section = config["deltaG"]
    deltaG_values = Dict{String, Float64}()
    
    for (key, value) in deltaG_section
        deltaG_values[key] = Float64(value)
    end
    
    return deltaG_values
end

"""
    计算热力学通量

基于ΔG值和浓度计算各步骤的热力学通量
"""
function calculate_thermodynamic_fluxes(sol, deltaG_data; T=298.15, R=8.314e-3)
    # R = 8.314 J/(mol·K) = 8.314e-3 kJ/(mol·K)
    
    fluxes = Dict()
    
    # 提取浓度数据
    Glucose_conc = sol[Glucose]
    G6P_conc = sol[G6P]
    F6P_conc = sol[F6P]
    F16BP_conc = sol[F16BP]
    DHAP_conc = sol[DHAP]
    GAP_conc = sol[GAP]
    BPG13_conc = sol[BPG13]
    PG3_conc = sol[PG3]
    PG2_conc = sol[PG2]
    PEP_conc = sol[PEP]
    Pyruvate_conc = sol[Pyruvate]
    ATP_conc = sol[ATP]
    ADP_conc = sol[ADP]
    NAD_conc = sol[NAD]
    NADH_conc = sol[NADH]
    Pi_conc = sol[Pi]
    H_conc = sol[H]

    
    # 定义反应商计算规则 - 消除所有特殊情况
    reaction_rules = Dict(
        "step1" => (products=[G6P_conc, ADP_conc], reactants=[Glucose_conc, ATP_conc]),
        "step2" => (products=[F6P_conc], reactants=[G6P_conc]),
        "step3" => (products=[F16BP_conc, ADP_conc], reactants=[F6P_conc, ATP_conc]),
        "step4" => (products=[DHAP_conc, GAP_conc], reactants=[F16BP_conc]),
        "step5" => (products=[GAP_conc], reactants=[DHAP_conc]),
        "step6" => (products=[BPG13_conc, NADH_conc, H_conc], reactants=[GAP_conc, NAD_conc, Pi_conc]),
        "step7" => (products=[PG3_conc, ATP_conc], reactants=[BPG13_conc, ADP_conc]),
        "step8" => (products=[PG2_conc], reactants=[PG3_conc]),
        "step9" => (products=[PEP_conc], reactants=[PG2_conc]),  # H2O浓度恒定
        "step10" => (products=[Pyruvate_conc, ATP_conc], reactants=[PEP_conc, ADP_conc])
    )
    
    
    function calculate_reaction_quotient(products, reactants)
        prod_term = ones(length(products[1]))
        for p in products
            prod_term .*= p
        end
        
        react_term = ones(length(reactants[1]))
        for r in reactants
            react_term .*= r
        end
        react_term .+= 1e-12
        
        Q = prod_term ./ react_term
        Q = max.(min.(Q, 1e12), 1e-12)
        return Q
    end
    
    
    for (step, deltaG_std) in deltaG_data
        rule = reaction_rules[step]
        Q = calculate_reaction_quotient(rule.products, rule.reactants)
        log_Q = log.(max.(min.(Q, 1e6), 1e-6))
        deltaG_actual = deltaG_std .+ R * T .* log_Q
        deltaG_actual = max.(min.(deltaG_actual, 1000.0), -1000.0)
        fluxes[step] = deltaG_actual
    end

    # 计算每步的热力学驱动力（ΔG）和反应商Q
    step_results = Dict{String, Any}()
    for (step, deltaG_std) in deltaG_data
        rule = reaction_rules[step]
        Q = calculate_reaction_quotient(rule.products, rule.reactants)     
        log_Q = log.(max.(min.(Q, 1e6), 1e-6))
        deltaG_actual = deltaG_std .+ R * T .* log_Q
        deltaG_actual = max.(min.(deltaG_actual, 1000.0), -1000.0)
        # Keq = exp(-ΔG°/RT) - 限制指数计算范围
        Keq_arg = max.(min.(-deltaG_std ./ (R * T), 50.0), -50.0)
        Keq = exp.(Keq_arg)
        
    
        Q_over_Keq = max.(min.(Q ./ Keq, 1e12), 1e-12)
        
        step_results[step] = Dict(
            "ΔG" => deltaG_actual,
            "Q" => Q,
            "Keq" => Keq,
            "Q/Keq" => Q_over_Keq
        )
    end

    # 返回逐步结果，包含 ΔG / Q / Keq / Q/Keq
    return step_results
end


"""
    验证热力学通量

检查各步骤的热力学可行性并把结果写成 toml
"""
function validate_thermodynamic_fluxes(step_results)
    validation_results = Dict{String, String}()
    
    for (step, metrics) in step_results
        deltaG_values = haskey(metrics, "ΔG") ? metrics["ΔG"] : nothing
        if deltaG_values !== nothing && isa(deltaG_values, Vector) && length(deltaG_values) > 0
            final_deltaG = deltaG_values[end]
            
            if final_deltaG < -5.0  # 强烈放能
                validation_results[step] = "✅ 强烈放能反应 (ΔG = $(round(final_deltaG, digits=2)) kJ/mol)"
                println("✅ 强烈放能反应 (ΔG = $(round(final_deltaG, digits=2)) kJ/mol)")
            elseif final_deltaG < 0.0  # 放能
                validation_results[step] = "✅ 放能反应 (ΔG = $(round(final_deltaG, digits=2)) kJ/mol)"
                println("✅ 放能反应 (ΔG = $(round(final_deltaG, digits=2)) kJ/mol)")
            elseif final_deltaG < 5.0  # 接近平衡
                validation_results[step] = "⚠️ 接近平衡 (ΔG = $(round(final_deltaG, digits=2)) kJ/mol)"
                println("⚠️ 接近平衡 (ΔG = $(round(final_deltaG, digits=2)) kJ/mol)")
            else  # 吸能
                validation_results[step] = "❌ 吸能反应 (ΔG = $(round(final_deltaG, digits=2)) kJ/mol)"
                println("❌ 吸能反应 (ΔG = $(round(final_deltaG, digits=2)) kJ/mol)")
            end
            # sort the validation_results by the step number
            # validation_results = sort(validation_results)
            # validation_results = sort(validation_results, by=x->findfirst(x->x==step, ["step1", "step2", "step3", "step4", "step5", "step6", "step7", "step8", "step9", "step10"]))
           
        else    
            validation_results[step] = "⚠️ 未提供 ΔG 序列"
        end
    end
    
    return validation_results
end

"""
    设置糖酵解初始条件

从TOML配置中读取初始浓度条件
"""
function set_glycolysis_initial_conditions(config=nothing)
    if config === nothing
        config = load_glycolysis_config()
    end
    
    initial_conditions_section = config["initial_conditions"]
    
    # 直接从TOML文件构建初始条件
    initial_conditions = [
        Glucose => Float64(initial_conditions_section["Glucose"]),
        G6P => Float64(initial_conditions_section["G6P"]),
        F6P => Float64(initial_conditions_section["F6P"]),
        F16BP => Float64(initial_conditions_section["F16BP"]),
        DHAP => Float64(initial_conditions_section["DHAP"]),
        GAP => Float64(initial_conditions_section["GAP"]),
        BPG13 => Float64(initial_conditions_section["BPG13"]),
        PG3 => Float64(initial_conditions_section["PG3"]),
        PG2 => Float64(initial_conditions_section["PG2"]),
        PEP => Float64(initial_conditions_section["PEP"]),
        Pyruvate => Float64(initial_conditions_section["Pyruvate"]),
        ATP => Float64(initial_conditions_section["ATP"]),
        ADP => Float64(initial_conditions_section["ADP"]),
        NAD => Float64(initial_conditions_section["NAD"]),
        NADH => Float64(initial_conditions_section["NADH"]),
        Pi => Float64(initial_conditions_section["Pi"]),
        H2O => Float64(initial_conditions_section["H2O"]),
        H => Float64(initial_conditions_section["H"])
    ]
    
    return initial_conditions
end

"""
    设置糖酵解反应参数

从TOML配置中读取动力学参数
"""
function set_glycolysis_parameters(config=nothing)
    if config === nothing
        config = load_glycolysis_config()
    end
    
    # 直接从TOML文件构建参数数组
    parameters = [
        k1f => Float64(config["k1"]["kf"]),
        k1r => Float64(config["k1"]["kr"]),
        k2f => Float64(config["k2"]["kf"]),
        k2r => Float64(config["k2"]["kr"]),
        k3f => Float64(config["k3"]["kf"]),
        k3r => Float64(config["k3"]["kr"]),
        k4f => Float64(config["k4"]["kf"]),
        k4r => Float64(config["k4"]["kr"]),
        k5f => Float64(config["k5"]["kf"]),
        k5r => Float64(config["k5"]["kr"]),
        k6f => Float64(config["k6"]["kf"]),
        k6r => Float64(config["k6"]["kr"]),
        k7f => Float64(config["k7"]["kf"]),
        k7r => Float64(config["k7"]["kr"]),
        k8f => Float64(config["k8"]["kf"]),
        k8r => Float64(config["k8"]["kr"]),
        k9f => Float64(config["k9"]["kf"]),
        k9r => Float64(config["k9"]["kr"]),
        k10f => Float64(config["k10"]["kf"]),
        k10r => Float64(config["k10"]["kr"]),
        # 外围供给参数（若缺省则置零，确保向后兼容）
        katp_in => Float64(get(get(config, "supply", Dict{String,Any}()), "atp_in", 0.0)),
        kadp_out => Float64(get(get(config, "supply", Dict{String,Any}()), "adp_out", 0.0)),
        kadp_in => Float64(get(get(config, "supply", Dict{String,Any}()), "adp_in", 0.0)),
        katp_out => Float64(get(get(config, "supply", Dict{String,Any}()), "atp_out", 0.0)),
        knad_in => Float64(get(get(config, "supply", Dict{String,Any}()), "nad_in", 0.0)),
        knadh_out => Float64(get(get(config, "supply", Dict{String,Any}()), "nadh_out", 0.0)),
        kpi_in => Float64(get(get(config, "supply", Dict{String,Any}()), "pi_in", 0.0)),
        kh_out => Float64(get(get(config, "supply", Dict{String,Any}()), "h_out", 0.0))
    ]
    
    return parameters
end

"""
    模拟糖酵解反应

运行完整的糖酵解反应模拟，支持从TOML配置文件加载参数
"""
function simulate_glycolysis(; tspan=(0.0, 100.0), saveat=0.1, config_path="deltaG.toml")
    config = load_glycolysis_config(config_path)
    rn = create_glycolysis_network()
    u0 = set_glycolysis_initial_conditions(config)
    p = set_glycolysis_parameters(config)
    ode_prob = ODEProblem(rn, u0, tspan, p)
    sol = solve(ode_prob, Tsit5(), saveat=saveat)
    return sol
end

"""
    可视化糖酵解反应结果

创建浓度时间曲线和热力学通量图
"""
function visualize_glycolysis_results(sol, step_results)
    # 创建浓度时间曲线
    p1 = plot(sol.t, sol[Glucose], label="Glucose", linewidth=2)
    plot!(p1, sol.t, sol[G6P], label="G6P", linewidth=2)
    plot!(p1, sol.t, sol[F6P], label="F6P", linewidth=2)
    plot!(p1, sol.t, sol[F16BP], label="F16BP", linewidth=2)
    plot!(p1, sol.t, sol[GAP], label="GAP", linewidth=2)
    plot!(p1, sol.t, sol[Pyruvate], label="Pyruvate", linewidth=2, linestyle=:dash)
    plot!(p1, xlabel="Time (s)", ylabel="Concentration (mM)", title="Glycolysis Metabolites")
    
    # ATP/ADP浓度变化
    p2 = plot(sol.t, sol[ATP], label="ATP", linewidth=2, color=:red)
    plot!(p2, sol.t, sol[ADP], label="ADP", linewidth=2, color=:blue)
    plot!(p2, xlabel="Time (s)", ylabel="Concentration (mM)", title="ATP/ADP Dynamics")
    
    # NAD/NADH浓度变化
    p3 = plot(sol.t, sol[NAD], label="NAD+", linewidth=2, color=:orange)
    plot!(p3, sol.t, sol[NADH], label="NADH", linewidth=2, color=:green)
    plot!(p3, xlabel="Time (s)", ylabel="Concentration (mM)", title="NAD+/NADH Dynamics")
    
    # 热力学通量图 - 简化版本
    p4 = plot()
    colors = [:red, :blue, :green, :orange, :purple, :brown, :pink, :gray, :olive, :cyan]
    for (i, (step, metrics)) in enumerate(step_results)
        if haskey(metrics, "ΔG") && isa(metrics["ΔG"], Vector) && length(metrics["ΔG"]) > 0
            plot!(p4, sol.t, metrics["ΔG"], label=step, linewidth=2, color=colors[i])
        end
    end
    plot!(p4, xlabel="Time (s)", ylabel="ΔG (kJ/mol)", title="Thermodynamic Fluxes")
    hline!(p4, [0], linestyle=:dash, color=:black, alpha=0.5, label="Equilibrium")
    
    combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))
    
    return combined_plot
end

"""
    生成糖酵解反应网络图

使用Graphviz创建反应网络的可视化
"""
function plot_glycolysis_network()
    try
        # 固定坐标布局（简单清晰，避免额外依赖）
        coords = Dict(
            :Glucose => (0.0, 0.0),
            :G6P => (1.5, 0.0),
            :F6P => (3.0, 0.0),
            :F16BP => (4.5, 0.0),
            :DHAP => (6.0, 0.8),
            :GAP => (6.0, -0.8),
            :BPG13 => (7.5, -0.8),
            :PG3 => (9.0, -0.8),
            :PG2 => (10.5, -0.8),
            :PEP => (12.0, -0.8),
            :Pyruvate => (13.5, -0.8),
            :ATP => (1.5, 1.4),
            :ADP => (3.0, 1.4),
            :NAD => (6.0, -2.2),
            :NADH => (7.5, -2.2),
            :Pi => (6.0, -3.2),
            :H2O => (10.5, -2.0),
            :H => (7.5, -3.0)
        )

        # 反应边（仅主代谢物主路径 + 必要支路与辅因子）
        edges = [
            (:Glucose, :G6P),    # 1
            (:G6P, :F6P),        # 2
            (:F6P, :F16BP),      # 3
            (:F16BP, :DHAP),     # 4 (裂解成两条)
            (:F16BP, :GAP),      # 4
            (:DHAP, :GAP),       # 5
            (:GAP, :BPG13),      # 6
            (:BPG13, :PG3),      # 7
            (:PG3, :PG2),        # 8
            (:PG2, :PEP),        # 9
            (:PEP, :Pyruvate)    # 10
        ]

        # 创建画布
        p = Plots.plot(xlim=(-0.5, 14.0), ylim=(-3.8, 2.0),
                       legend=false, framestyle=:none, size=(1200, 500))

        # 绘制节点
        for (name, (x, y)) in coords
            Plots.scatter!(p, [x], [y], ms=6, c=:black)
            Plots.annotate!(p, x, y + 0.18, text(string(name), 9, :black))
        end

        # 绘制有向边（箭头）
        for (src, dst) in edges
            (x1, y1) = coords[src]
            (x2, y2) = coords[dst]
            Plots.plot!(p, [x1, x2], [y1, y2], lw=2, c=:steelblue, arrow=:arrow)
        end

        # 辅因子与能量耦合可视化（细线表示）
        # 1: ATP -> ADP at step3 (F6P -> F16BP)
        Plots.plot!(p, [coords[:ATP][1], coords[:F6P][1]], [coords[:ATP][2], coords[:F6P][2]],
                    lw=1, lc=:red, ls=:dash, arrow=:arrow)
        Plots.plot!(p, [coords[:F16BP][1], coords[:ADP][1]], [coords[:F16BP][2], coords[:ADP][2]],
                    lw=1, lc=:red, ls=:dash, arrow=:arrow)

        # 2: NAD + Pi -> NADH + H at step6 (GAP -> BPG13)
        Plots.plot!(p, [coords[:NAD][1], coords[:GAP][1]], [coords[:NAD][2], coords[:GAP][2]],
                    lw=1, lc=:green, ls=:dot, arrow=:arrow)
        Plots.plot!(p, [coords[:Pi][1], coords[:GAP][1]], [coords[:Pi][2], coords[:GAP][2]],
                    lw=1, lc=:green, ls=:dot, arrow=:arrow)
        Plots.plot!(p, [coords[:BPG13][1], coords[:NADH][1]], [coords[:BPG13][2], coords[:NADH][2]],
                    lw=1, lc=:green, ls=:dot, arrow=:arrow)
        Plots.plot!(p, [coords[:BPG13][1], coords[:H][1]], [coords[:BPG13][2], coords[:H][2]],
                    lw=1, lc=:green, ls=:dot, arrow=:arrow)

        # 3: H2O 在 step9 (PG2 -> PEP) 作为副产物显示
        Plots.plot!(p, [coords[:PG2][1], coords[:H2O][1]], [coords[:PG2][2], coords[:H2O][2]],
                    lw=1, lc=:gray, ls=:dashdot, arrow=:arrow)

        Plots.savefig(p, "glycolysis_network.png")
        println("反应网络图已保存为 glycolysis_network.png")
        return p
    catch e
        println("⚠️ 反应网络绘制失败，改为文本输出: $e")
        println("糖酵解反应网络结构:")
        println("葡萄糖 → G6P → F6P → F16BP → DHAP + GAP → BPG13 → PG3 → PG2 → PEP → 丙酮酸")
        println("包含10个反应步骤，18个代谢物")
    end
end


function write_glycolysis_results_to_toml(results)
    toml_results = Dict{String, Any}()
    for (step, result) in results
        toml_results[string(step)] = result
    end
    output_path = joinpath(@__DIR__, "glycolysis_results.toml")
    open(output_path, "w") do io
        TOML.print(io, toml_results)
    end
    println("结果已写入: $(output_path)")
    return toml_results
end

println("开始构建糖酵解反应网络...")

# 创建反应网络
glycolysis_rn = create_glycolysis_network()
println("糖酵解反应网络已创建，包含 $(length(species(glycolysis_rn))) 个物种和 $(length(reactions(glycolysis_rn))) 个反应")

# 加载配置
println("加载TOML配置文件...")
config = load_glycolysis_config()
deltaG_data = load_thermodynamic_data(config)
println("配置和热力学数据已加载")

# 运行模拟
println("开始模拟糖酵解反应...")




sol = simulate_glycolysis()
println("模拟完成！")

# 计算热力学通量
println("计算热力学通量...")
fluxes = calculate_thermodynamic_fluxes(sol, deltaG_data)

# 验证热力学通量
println("验证热力学通量...")
validation_results = validate_thermodynamic_fluxes(fluxes)


# 显示结果摘要
println("\n=== 糖酵解模拟结果摘要 ===")
println("模拟时间范围: $(sol.t[1]) - $(sol.t[end]) 秒")
println("最终丙酮酸浓度: $(round(sol[Pyruvate][end], digits=3)) mM")
println("最终ATP浓度: $(round(sol[ATP][end], digits=3)) mM")
println("最终NADH浓度: $(round(sol[NADH][end], digits=3)) mM")

# 显示热力学验证结果
println("\n=== 热力学通量验证结果 ===")
for step in sort(collect(keys(validation_results)))
    println("$(step): $(validation_results[step])")
end

# 创建可视化
println("\n生成可视化图表...")
plot_result = visualize_glycolysis_results(sol, fluxes)
savefig(plot_result, "glycolysis_simulation_results.png")
println("图表已保存为 glycolysis_simulation_results.png")
println("将结果写入 toml 文件...")
# 显示网络结构
plot_glycolysis_network()
write_glycolysis_results_to_toml(validation_results)

    

