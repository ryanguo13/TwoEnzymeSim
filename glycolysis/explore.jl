## 识别热力学瓶颈和限速步骤（探索脚本）
# 说明：
# - 复用 main.jl 中已实现的网络/仿真/热力学计算。
# - 定义“v_thermo”为 −ΔG（放能越大，v_thermo 越高，越远离平衡、越可能是控制点）。
# - 对比“近似有氧”(较高 NADH 清除) 与 “近似无氧”(NADH 不清除) 两种条件。
# - 做一次葡萄糖输入扰动（仅改变初始 [Glucose]）并观察 v_thermo 的变化。

using TOML
using Plots
using DifferentialEquations

# 警告：include main.jl 会执行其中的演示脚本（会跑一遍默认仿真与作图）。
# 这里为了重用函数，容忍该副作用。如果不想要，可把 main.jl 的底部执行段落改成模块或加条件守卫。
include("main.jl")


"""
创建一个基于现有 TOML 的场景配置副本，并按需修改。
返回：临时 TOML 文件路径（写在当前目录下）。
"""
function create_scenario_config(; base_path::String = joinpath(@__DIR__, "deltaG.toml"),
                                  nadh_out::Union{Nothing,Float64}=nothing,
                                  nad_in::Union{Nothing,Float64}=nothing,
                                  glucose0::Union{Nothing,Float64}=nothing,
                                  tag::String="scenario")
    cfg = TOML.parsefile(base_path)

    # 修改 supply（近似有氧/无氧通过调节 NADH 清除/ NAD+ 供给）
    if !haskey(cfg, "supply")
        cfg["supply"] = Dict{String,Any}()
    end
    if nadh_out !== nothing
        cfg["supply"]["nadh_out"] = nadh_out
    end
    if nad_in !== nothing
        cfg["supply"]["nad_in"] = nad_in
    end

    # 扰动：调整初始葡萄糖浓度
    if glucose0 !== nothing
        cfg["initial_conditions"]["Glucose"] = glucose0
    end

    out_path = joinpath(@__DIR__, "$(tag).toml")
    open(out_path, "w") do io
        TOML.print(io, cfg)
    end
    return out_path
end


"""
运行一个场景：
 - 使用给定的临时 TOML 配置路径
 - 运行仿真，计算热力学数据，派生 v_thermo = -ΔG
返回：(sol, step_results, vthermo_last, vthermo_mean)
"""
function run_scenario(config_path::String; tspan=(0.0, 100.0), saveat=0.2)
    sol = simulate_glycolysis(; tspan=tspan, saveat=saveat, config_path=config_path)
    cfg = load_glycolysis_config(config_path)
    dG = load_thermodynamic_data(cfg)
    step_results = calculate_thermodynamic_fluxes(sol, dG)

    # v_thermo = -ΔG（向正值表示放能强、远离平衡）
    vthermo_series = Dict{String,Vector{Float64}}()
    vthermo_last = Dict{String,Float64}()
    vthermo_mean = Dict{String,Float64}()
    for (step, metrics) in step_results
        if haskey(metrics, "ΔG")
            dG_series = Vector{Float64}(metrics["ΔG"])
            v = -dG_series
            vthermo_series[step] = v
            vthermo_last[step] = v[end]
            vthermo_mean[step] = sum(v) / length(v)
        end
    end

    return sol, step_results, vthermo_last, vthermo_mean
end


"""
打印场景对比摘要，并绘制柱状图（最终 v_thermo 与均值）。
"""
function summarize_and_plot!(title_suffix::String,
                             v_last::Dict{String,Float64},
                             v_mean::Dict{String,Float64};
                             saveprefix::String)
    steps = sort(collect(keys(v_last)))
    last_vals = [v_last[s] for s in steps]
    mean_vals = [v_mean[s] for s in steps]

    println("\n=== v_thermo 摘要 ($(title_suffix)) ===")
    # 找潜在控制点（远离平衡：v_thermo 高）与近平衡步骤（v_thermo 低）
    sorted_idx = sortperm(last_vals; rev=true)
    top3 = steps[sorted_idx[1:min(3, length(steps))]]
    low3 = steps[reverse!(sortperm(last_vals))[1:min(3, length(steps))]]
    println("高 v_thermo（远离平衡，可能控制点）: ", top3)
    println("低 v_thermo（近平衡）: ", low3)

    p1 = bar(steps, last_vals, legend=false, ylabel="v_thermo (final)",
             title="Final v_thermo $(title_suffix)", xticks=(1:length(steps), steps), rotation=45)
    p2 = bar(steps, mean_vals, legend=false, ylabel="v_thermo (mean)",
             title="Mean v_thermo $(title_suffix)", xticks=(1:length(steps), steps), rotation=45)
    plt = plot(p1, p2, layout=(2,1), size=(1000, 700))
    savefig(plt, "$(saveprefix)_vthermo.png")
    println("图已保存: $(saveprefix)_vthermo.png")
end


"""
主流程：
 - 近似有氧：较大的 nadh_out 和适度的 nad_in（使用默认值或更高）
 - 近似无氧：nadh_out = 0.0（不清除 NADH）
 - 扰动：把初始 [Glucose] 从 5 mM 提高到 10 mM（以近似有氧为基线）
"""
function main()
    base = joinpath(@__DIR__, "deltaG.toml")

    # 有氧（默认文件已包含一定 nadh_out，这里确保不小于 0.15）
    aerobic_path = create_scenario_config(; base_path=base, nadh_out=0.15, nad_in=0.10, tag="aerobic")
    sol_a, res_a, vlast_a, vmean_a = run_scenario(aerobic_path)
    summarize_and_plot!("(Aerobic)", vlast_a, vmean_a; saveprefix="aerobic")

    # 无氧（不清除 NADH）
    anaerobic_path = create_scenario_config(; base_path=base, nadh_out=0.0, nad_in=0.0, tag="anaerobic")
    sol_an, res_an, vlast_an, vmean_an = run_scenario(anaerobic_path)
    summarize_and_plot!("(Anaerobic)", vlast_an, vmean_an; saveprefix="anaerobic")

    # 扰动：增强葡萄糖输入（通过初始条件近似）
    perturb_path = create_scenario_config(; base_path=aerobic_path, glucose0=10.0, tag="aerobic_glc10mM")
    sol_p, res_p, vlast_p, vmean_p = run_scenario(perturb_path)
    summarize_and_plot!("(Aerobic + Glc↑)", vlast_p, vmean_p; saveprefix="aerobic_glc10mM")

    # 简要比较：以最终 v_thermo 为基准
    steps = sort(collect(keys(vlast_a)))
    println("\n=== 条件对比（final v_thermo）===")
    println("step\tAerobic\tAnaerobic\tAerobic+Glc↑")
    for s in steps
        va = round(vlast_a[s], digits=3)
        vx = round(get(vlast_an, s, NaN), digits=3)
        vp = round(get(vlast_p, s, NaN), digits=3)
        println("$(s)\t$(va)\t$(vx)\t$(vp)")
    end

    println("\n提示：高 v_thermo = 远离平衡（高耗散，潜在控制点）；低 v_thermo = 近平衡。")
end

main()
