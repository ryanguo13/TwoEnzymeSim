"""
Plotting utilities for model and Bayesian optimization results.

Outputs directory: /home/ryankwok/Documents/TwoEnzymeSim/ML/result
"""

using JLD2
using Statistics
using Printf
using Plots

include("surrogate_model.jl")
if isfile("bayesian_optimization.jl")
    include("bayesian_optimization.jl")
end

"""
    compute_thermo_steady_for_X(X, ps; tail_frac=0.2)

Return (v1s, v2s) steady means for each row of X using simulate_system and calculate_thermo_fluxes.
"""
function compute_thermo_steady_for_X(X::AbstractMatrix, ps; tail_frac::Float64=0.2)
    n = size(X, 1)
    v1s = fill(NaN, n); v2s = fill(NaN, n)
    for i in 1:n
        x = X[i, :]
        k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = x[1:8]
        A0, B0, C0, E1_0, E2_0 = x[9:13]
        rate_params = Dict(:k1f=>k1f, :k1r=>k1r, :k2f=>k2f, :k2r=>k2r,
                           :k3f=>k3f, :k3r=>k3r, :k4f=>k4f, :k4r=>k4r)
        initial_conditions = [A=>A0, B=>B0, C=>C0, E1=>E1_0, E2=>E2_0, AE1=>0.0, BE2=>0.0]
        try
            sol = simulate_system(rate_params, initial_conditions, ps.tspan, saveat=0.05)
            th = calculate_thermo_fluxes(sol, rate_params)
            v1 = th["v1_thermo"]; v2 = th["v2_thermo"]
            m = length(v1); w = max(1, Int(clamp(round(m * tail_frac), 1, m)))
            v1s[i] = mean(view(v1, m-w+1:m))
            v2s[i] = mean(view(v2, m-w+1:m))
        catch
            v1s[i] = NaN; v2s[i] = NaN
        end
    end
    return v1s, v2s
end

"""
    plot_thermo_dashboard_overview(bo_data; tail_frac=0.2)

Create a compact dashboard with:
 - v1/v2 global correlation bars (side-by-side)
 - v1/v2 k1f×k1r steady heatmaps
Saves: thermo_dashboard_overview.png
"""
function plot_thermo_dashboard_overview(bo_data; tail_frac::Float64=0.2)
    if !haskey(bo_data, "X_evaluated") || !haskey(bo_data, "param_space")
        return
    end
    if !@isdefined(simulate_system) || !@isdefined(calculate_thermo_fluxes)
        return
    end

    X = bo_data["X_evaluated"]
    ps = bo_data["param_space"]

    # Compute steady means and correlations
    v1s, v2s = compute_thermo_steady_for_X(X, ps; tail_frac=tail_frac)
    valid = findall(i -> isfinite(v1s[i]) && isfinite(v2s[i]) && all(isfinite.(X[i, :])), 1:size(X,1))
    Xv = X[valid, :]; v1v = v1s[valid]; v2v = v2s[valid]
    names = ["k1f","k1r","k2f","k2r","k3f","k3r","k4f","k4r","A","B","C","E1","E2"]
    cor_v1 = [var(Xv[:, j])>1e-10 ? cor(Xv[:, j], v1v) : 0.0 for j in 1:size(X,2)]
    cor_v2 = [var(Xv[:, j])>1e-10 ? cor(Xv[:, j], v2v) : 0.0 for j in 1:size(X,2)]

    p_cor_v1 = bar(names, cor_v1, xlabel="Parameter", ylabel="Corr v1", title="Global Corr v1",
                   xrotation=45, color=:blues)
    p_cor_v2 = bar(names, cor_v2, xlabel="Parameter", ylabel="Corr v2", title="Global Corr v2",
                   xrotation=45, color=:greens)

    # Heatmaps for k1f×k1r
    steps = 40
    mid = r -> 0.5 * (minimum(r) + maximum(r))
    defaults = Dict(:k2f=>mid(ps.k2f_range), :k2r=>mid(ps.k2r_range),
                    :k3f=>mid(ps.k3f_range), :k3r=>mid(ps.k3r_range),
                    :k4f=>mid(ps.k4f_range), :k4r=>mid(ps.k4r_range),
                    :A=>mid(ps.A_range), :B=>mid(ps.B_range), :C=>mid(ps.C_range),
                    :E1=>mid(ps.E1_range), :E2=>mid(ps.E2_range))
    k1f_vals = range(minimum(ps.k1f_range), maximum(ps.k1f_range), length=steps)
    k1r_vals = range(minimum(ps.k1r_range), maximum(ps.k1r_range), length=steps)
    Z1 = fill(NaN, steps, steps); Z2 = fill(NaN, steps, steps)
    for (ix, k1f_v) in enumerate(k1f_vals), (iy, k1r_v) in enumerate(k1r_vals)
        rate_params = Dict(:k1f=>k1f_v, :k1r=>k1r_v, :k2f=>defaults[:k2f], :k2r=>defaults[:k2r],
                           :k3f=>defaults[:k3f], :k3r=>defaults[:k3r], :k4f=>defaults[:k4f], :k4r=>defaults[:k4r])
        initial_conditions = [A=>defaults[:A], B=>defaults[:B], C=>defaults[:C], E1=>defaults[:E1], E2=>defaults[:E2], AE1=>0.0, BE2=>0.0]
        try
            sol = simulate_system(rate_params, initial_conditions, ps.tspan, saveat=0.05)
            th = calculate_thermo_fluxes(sol, rate_params)
            v1 = th["v1_thermo"]; v2 = th["v2_thermo"]
            m = length(v1); w = max(1, Int(clamp(round(m * tail_frac), 1, m)))
            Z1[iy, ix] = mean(view(v1, m-w+1:m))
            Z2[iy, ix] = mean(view(v2, m-w+1:m))
        catch
            Z1[iy, ix] = NaN; Z2[iy, ix] = NaN
        end
    end
    p_h1 = heatmap(k1f_vals, k1r_vals, Z1, xlabel="k1f", ylabel="k1r", title="v1 steady heatmap")
    p_h2 = heatmap(k1f_vals, k1r_vals, Z2, xlabel="k1f", ylabel="k1r", title="v2 steady heatmap")

    l = @layout [a b; c d]
    p = plot(p_cor_v1, p_cor_v2, p_h1, p_h2, layout=l, size=(1600, 1200))
    savefig_safely(p, "thermo_dashboard_overview")
end

"""
    plot_thermo_dashboard_scatter(bo_data; tail_frac=0.2, topk=6)

Create a dashboard with top-K parameters by |corr| for v1 and v2 and their scatters.
Saves: thermo_dashboard_scatter.png
"""
function plot_thermo_dashboard_scatter(bo_data; tail_frac::Float64=0.2, topk::Int=6)
    if !haskey(bo_data, "X_evaluated") || !haskey(bo_data, "param_space")
        return
    end
    if !@isdefined(simulate_system) || !@isdefined(calculate_thermo_fluxes)
        return
    end
    X = bo_data["X_evaluated"]; ps = bo_data["param_space"]
    names = ["k1f","k1r","k2f","k2r","k3f","k3r","k4f","k4r","A","B","C","E1","E2"]
    v1s, v2s = compute_thermo_steady_for_X(X, ps; tail_frac=tail_frac)
    valid = findall(i -> isfinite(v1s[i]) && isfinite(v2s[i]) && all(isfinite.(X[i, :])), 1:size(X,1))
    Xv = X[valid, :]; v1v = v1s[valid]; v2v = v2s[valid]
    cor_v1 = [var(Xv[:, j])>1e-10 ? cor(Xv[:, j], v1v) : 0.0 for j in 1:size(X,2)]
    cor_v2 = [var(Xv[:, j])>1e-10 ? cor(Xv[:, j], v2v) : 0.0 for j in 1:size(X,2)]
    idx_v1 = sortperm(abs.(cor_v1), rev=true)[1:min(topk, length(cor_v1))]
    idx_v2 = sortperm(abs.(cor_v2), rev=true)[1:min(topk, length(cor_v2))]

    plots_v1 = [scatter(Xv[:, j], v1v, xlabel=names[j], ylabel="v1 steady", title=names[j], ms=3, alpha=0.8) for j in idx_v1]
    plots_v2 = [scatter(Xv[:, j], v2v, xlabel=names[j], ylabel="v2 steady", title=names[j], ms=3, alpha=0.8) for j in idx_v2]
    grid_v1 = plot(plots_v1..., layout=(ceil(Int, length(plots_v1)/3), 3), size=(1600, 800), title="Top-$(length(plots_v1)) params vs v1")
    grid_v2 = plot(plots_v2..., layout=(ceil(Int, length(plots_v2)/3), 3), size=(1600, 800), title="Top-$(length(plots_v2)) params vs v2")
    l = @layout [a; b]
    p = plot(grid_v1, grid_v2, layout=l, size=(1600, 1600))
    savefig_safely(p, "thermo_dashboard_scatter")
end

"""
    plot_thermo_dashboard_heatmaps(bo_data; steps=40, tail_frac=0.2)

Generate steady thermo heatmaps for k2 and k3 and k4 pairs, and a combined E heatmap (E1×E2).
Saves: thermo_dashboard_heatmaps.png
"""
function plot_thermo_dashboard_heatmaps(bo_data; steps::Int=40, tail_frac::Float64=0.2)
    if !haskey(bo_data, "param_space")
        return
    end
    if !@isdefined(simulate_system) || !@isdefined(calculate_thermo_fluxes)
        return
    end
    ps = bo_data["param_space"]
    mid = r -> 0.5 * (minimum(r) + maximum(r))
    defaults = Dict(
        :k1f=>mid(ps.k1f_range), :k1r=>mid(ps.k1r_range),
        :k2f=>mid(ps.k2f_range), :k2r=>mid(ps.k2r_range),
        :k3f=>mid(ps.k3f_range), :k3r=>mid(ps.k3r_range),
        :k4f=>mid(ps.k4f_range), :k4r=>mid(ps.k4r_range),
        :A=>mid(ps.A_range), :B=>mid(ps.B_range), :C=>mid(ps.C_range),
        :E1=>mid(ps.E1_range), :E2=>mid(ps.E2_range)
    )

    function heat2(namex::Symbol, namey::Symbol)
        xvals = range(minimum(getfield(ps, Symbol(namex, :_range))), maximum(getfield(ps, Symbol(namex, :_range))), length=steps)
        yvals = range(minimum(getfield(ps, Symbol(namey, :_range))), maximum(getfield(ps, Symbol(namey, :_range))), length=steps)
        Z1 = fill(NaN, steps, steps); Z2 = fill(NaN, steps, steps)
        for (ix, xv) in enumerate(xvals), (iy, yv) in enumerate(yvals)
            rate_params = Dict(
                :k1f=>defaults[:k1f], :k1r=>defaults[:k1r],
                :k2f=>defaults[:k2f], :k2r=>defaults[:k2r],
                :k3f=>defaults[:k3f], :k3r=>defaults[:k3r],
                :k4f=>defaults[:k4f], :k4r=>defaults[:k4r]
            )
            rate_params[namex] = xv
            rate_params[namey] = yv
            initial_conditions = [A=>defaults[:A], B=>defaults[:B], C=>defaults[:C],
                                  E1=>defaults[:E1], E2=>defaults[:E2], AE1=>0.0, BE2=>0.0]
            try
                sol = simulate_system(rate_params, initial_conditions, ps.tspan, saveat=0.05)
                th = calculate_thermo_fluxes(sol, rate_params)
                v1 = th["v1_thermo"]; v2 = th["v2_thermo"]
                m = length(v1); w = max(1, Int(clamp(round(m * tail_frac), 1, m)))
                Z1[iy, ix] = mean(view(v1, m-w+1:m))
                Z2[iy, ix] = mean(view(v2, m-w+1:m))
            catch
                Z1[iy, ix] = NaN; Z2[iy, ix] = NaN
            end
        end
        p1 = heatmap(xvals, yvals, Z1, xlabel=String(namex), ylabel=String(namey), title="v1 steady $(namex)×$(namey)")
        p2 = heatmap(xvals, yvals, Z2, xlabel=String(namex), ylabel=String(namey), title="v2 steady $(namex)×$(namey)")
        return p1, p2
    end

    # k2 pair, k3 pair, k4 pair, and E1×E2
    p2_v1, p2_v2 = heat2(:k2f, :k2r)
    p3_v1, p3_v2 = heat2(:k3f, :k3r)
    p4_v1, p4_v2 = heat2(:k4f, :k4r)
    pE_v1, pE_v2 = heat2(:E1, :E2)

    l = @layout [a b; c d; e f; g h]
    p = plot(p2_v1, p2_v2, p3_v1, p3_v2, p4_v1, p4_v2, pE_v1, pE_v2, layout=l, size=(1600, 2000))
    savefig_safely(p, "thermo_dashboard_heatmaps")
end

"""
    plot_bo_thermo_global_influence(bo_data; tail_frac=0.2)

Compute steady thermo means per evaluated point, then compute correlation of each of the 13 params with v1/v2.
Plot bar charts and highlight top-4 parameters with scatter plots.
"""
function plot_bo_thermo_global_influence(bo_data; tail_frac::Float64=0.2)
    if !haskey(bo_data, "X_evaluated") || !haskey(bo_data, "param_space")
        println("⚠️  Missing X_evaluated/param_space; skip global influence")
        return
    end
    if !@isdefined(simulate_system) || !@isdefined(calculate_thermo_fluxes)
        println("⚠️  simulate_system/calculate_thermo_fluxes not available; skip global influence")
        return
    end

    X = bo_data["X_evaluated"]
    ps = bo_data["param_space"]
    n = size(X, 1); d = size(X, 2)
    v1s = fill(NaN, n); v2s = fill(NaN, n)

    for i in 1:n
        x = X[i, :]
        k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = x[1:8]
        A0, B0, C0, E1_0, E2_0 = x[9:13]
        rate_params = Dict(:k1f=>k1f, :k1r=>k1r, :k2f=>k2f, :k2r=>k2r,
                           :k3f=>k3f, :k3r=>k3r, :k4f=>k4f, :k4r=>k4r)
        initial_conditions = [A=>A0, B=>B0, C=>C0, E1=>E1_0, E2=>E2_0, AE1=>0.0, BE2=>0.0]
        try
            sol = simulate_system(rate_params, initial_conditions, ps.tspan, saveat=0.05)
            th = calculate_thermo_fluxes(sol, rate_params)
            v1 = th["v1_thermo"]; v2 = th["v2_thermo"]
            m = length(v1); w = max(1, Int(clamp(round(m * tail_frac), 1, m)))
            v1s[i] = mean(view(v1, m-w+1:m))
            v2s[i] = mean(view(v2, m-w+1:m))
        catch
            v1s[i] = NaN; v2s[i] = NaN
        end
    end

    # 过滤有效样本
    valid = findall(i -> isfinite(v1s[i]) && isfinite(v2s[i]) && all(isfinite.(X[i, :])), 1:n)
    if isempty(valid)
        println("⚠️  No valid samples for global influence")
        return
    end
    Xv = X[valid, :]; v1v = v1s[valid]; v2v = v2s[valid]

    param_names = ["k1f","k1r","k2f","k2r","k3f","k3r","k4f","k4r","A","B","C","E1","E2"]
    cor_v1 = zeros(d); cor_v2 = zeros(d)
    for j in 1:d
        if var(Xv[:, j]) > 1e-10
            cor_v1[j] = cor(Xv[:, j], v1v)
            cor_v2[j] = cor(Xv[:, j], v2v)
        else
            cor_v1[j] = 0.0; cor_v2[j] = 0.0
        end
    end

    p1 = bar(param_names, cor_v1, xlabel="Parameter", ylabel="Corr with v1_thermo (steady)",
             title="Thermo v1 global influence (correlation)", xrotation=45, color=:blues)
    savefig_safely(p1, "thermo_v1_global_correlation")

    p2 = bar(param_names, cor_v2, xlabel="Parameter", ylabel="Corr with v2_thermo (steady)",
             title="Thermo v2 global influence (correlation)", xrotation=45, color=:greens)
    savefig_safely(p2, "thermo_v2_global_correlation")

    # 选取绝对相关性前4，画散点
    top_idx_v1 = sortperm(abs.(cor_v1), rev=true)[1:min(4, d)]
    top_idx_v2 = sortperm(abs.(cor_v2), rev=true)[1:min(4, d)]

    for j in top_idx_v1
        p = scatter(Xv[:, j], v1v, xlabel=param_names[j], ylabel="v1_thermo_mean (steady)",
                    title="v1_thermo vs $(param_names[j])", ms=4, alpha=0.8)
        savefig_safely(p, "thermo_v1_vs_$(param_names[j])_global")
    end
    for j in top_idx_v2
        p = scatter(Xv[:, j], v2v, xlabel=param_names[j], ylabel="v2_thermo_mean (steady)",
                    title="v2_thermo vs $(param_names[j])", ms=4, alpha=0.8)
        savefig_safely(p, "thermo_v2_vs_$(param_names[j])_global")
    end
end

try
    include(joinpath(@__DIR__, "..", "src", "simulation.jl"))
catch e
end

# --- New: Thermo flux visualizations ---

"""
    plot_bo_thermo_flux_history(bo_data)

Plot v1_thermo_mean and v2_thermo_mean versus evaluation index.
"""
function plot_bo_thermo_flux_history(bo_data)
    if !haskey(bo_data, "thermo_v1_mean_history") || !haskey(bo_data, "thermo_v2_mean_history")
        println("⚠️  No thermo flux history in BO results; skip")
        return
    end

    v1 = bo_data["thermo_v1_mean_history"]
    v2 = bo_data["thermo_v2_mean_history"]

    n = max(length(v1), length(v2))
    x = 1:n

    p = plot(x, v1, lw=2, label="v1_thermo_mean", color=:blue,
             xlabel="Evaluation", ylabel="Thermo flux mean",
             title="Thermo Flux Means over Evaluations")
    plot!(p, x, v2, lw=2, label="v2_thermo_mean", color=:red)

    savefig_safely(p, "thermo_flux_history")
end

"""
    plot_bo_thermo_vs_params(bo_data)

Scatter thermo v1/v2 means against k1f and k1r to reveal relationships.
"""
function plot_bo_thermo_vs_params(bo_data)
    if !haskey(bo_data, "X_evaluated") || !haskey(bo_data, "thermo_v1_mean_history")
        println("⚠️  Missing BO parameters or thermo flux history; skip")
        return
    end

    X = bo_data["X_evaluated"]
    v1 = bo_data["thermo_v1_mean_history"]
    v2 = bo_data["thermo_v2_mean_history"]

    if size(X, 2) < 2
        println("⚠️  Less than 2 parameters; skip thermo vs params plots")
        return
    end

    k1f = X[:, 1]
    k1r = X[:, 2]

    # 过滤非有限值
    finite_idx = findall(i -> isfinite(v1[i]) && isfinite(v2[i]) && isfinite(k1f[i]) && isfinite(k1r[i]), eachindex(v1))
    k1f = k1f[finite_idx]; k1r = k1r[finite_idx]; v1 = v1[finite_idx]; v2 = v2[finite_idx]

    p1 = scatter(k1f, v1, xlabel="k1f", ylabel="v1_thermo_mean",
                 title="v1_thermo_mean vs k1f", ms=4, alpha=0.8)
    savefig_safely(p1, "thermo_v1_vs_k1f")

    p2 = scatter(k1r, v1, xlabel="k1r", ylabel="v1_thermo_mean",
                 title="v1_thermo_mean vs k1r", ms=4, alpha=0.8)
    savefig_safely(p2, "thermo_v1_vs_k1r")

    p3 = scatter(k1f, v2, xlabel="k1f", ylabel="v2_thermo_mean",
                 title="v2_thermo_mean vs k1f", ms=4, alpha=0.8)
    savefig_safely(p3, "thermo_v2_vs_k1f")

    p4 = scatter(k1r, v2, xlabel="k1r", ylabel="v2_thermo_mean",
                 title="v2_thermo_mean vs k1r", ms=4, alpha=0.8)
    savefig_safely(p4, "thermo_v2_vs_k1r")
end

"""
    plot_bo_thermo_vs_params_recomputed(bo_data; tail_frac=0.2)

Recompute thermo fluxes per evaluated point via simulate_system and use steady tail mean.
Generates *_recomputed_steady figures.
"""
function plot_bo_thermo_vs_params_recomputed(bo_data; tail_frac::Float64=0.2)
    if !haskey(bo_data, "X_evaluated") || !haskey(bo_data, "param_space")
        println("⚠️  Missing X_evaluated or param_space; skip recomputed thermo plots")
        return
    end

    if !@isdefined(simulate_system) || !@isdefined(calculate_thermo_fluxes)
        println("⚠️  simulate_system/calculate_thermo_fluxes not available; skip recomputed thermo plots")
        return
    end

    X = bo_data["X_evaluated"]
    ps = bo_data["param_space"]
    n = size(X, 1)
    v1s = fill(NaN, n)
    v2s = fill(NaN, n)

    for i in 1:n
        x = X[i, :]
        # 参数向量顺序需与优化一致
        k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = x[1:8]
        A0, B0, C0, E1_0, E2_0 = x[9:13]

        rate_params = Dict(
            :k1f=>k1f, :k1r=>k1r, :k2f=>k2f, :k2r=>k2r,
            :k3f=>k3f, :k3r=>k3r, :k4f=>k4f, :k4r=>k4r
        )
        initial_conditions = [
            A=>A0, B=>B0, C=>C0, E1=>E1_0, E2=>E2_0, AE1=>0.0, BE2=>0.0
        ]

        try
            sol = simulate_system(rate_params, initial_conditions, ps.tspan, saveat=0.05)
            thermo = calculate_thermo_fluxes(sol, rate_params)
            v1 = thermo["v1_thermo"]; v2 = thermo["v2_thermo"]
            m = length(v1)
            w = max(1, Int(clamp(round(m * tail_frac), 1, m)))
            v1s[i] = mean(view(v1, m-w+1:m))
            v2s[i] = mean(view(v2, m-w+1:m))
        catch
            v1s[i] = NaN
            v2s[i] = NaN
        end
    end

    k1f = X[:, 1]; k1r = X[:, 2]
    finite_idx = findall(i -> isfinite(v1s[i]) && isfinite(v2s[i]) && isfinite(k1f[i]) && isfinite(k1r[i]), 1:length(v1s))
    k1f = k1f[finite_idx]; k1r = k1r[finite_idx]; v1s = v1s[finite_idx]; v2s = v2s[finite_idx]

    p1 = scatter(k1f, v1s, xlabel="k1f", ylabel="v1_thermo_mean (steady)",
                 title="v1_thermo_mean vs k1f (recomputed steady)", ms=4, alpha=0.8)
    savefig_safely(p1, "thermo_v1_vs_k1f_recomputed_steady")

    p2 = scatter(k1r, v1s, xlabel="k1r", ylabel="v1_thermo_mean (steady)",
                 title="v1_thermo_mean vs k1r (recomputed steady)", ms=4, alpha=0.8)
    savefig_safely(p2, "thermo_v1_vs_k1r_recomputed_steady")

    p3 = scatter(k1f, v2s, xlabel="k1f", ylabel="v2_thermo_mean (steady)",
                 title="v2_thermo_mean vs k1f (recomputed steady)", ms=4, alpha=0.8)
    savefig_safely(p3, "thermo_v2_vs_k1f_recomputed_steady")

    p4 = scatter(k1r, v2s, xlabel="k1r", ylabel="v2_thermo_mean (steady)",
                 title="v2_thermo_mean vs k1r (recomputed steady)", ms=4, alpha=0.8)
    savefig_safely(p4, "thermo_v2_vs_k1r_recomputed_steady")
end

"""
    plot_bo_thermo_heatmap_k1f_k1r_recomputed(bo_data; steps=40, tail_frac=0.2)

Compute thermo flux steady means on a k1f×k1r grid with other params fixed to mid-range.
"""
function plot_bo_thermo_heatmap_k1f_k1r_recomputed(bo_data; steps::Int=40, tail_frac::Float64=0.2)
    if !haskey(bo_data, "param_space")
        println("⚠️  No param_space in BO results; skip heatmap")
        return
    end
    if !@isdefined(simulate_system) || !@isdefined(calculate_thermo_fluxes)
        println("⚠️  simulate_system/calculate_thermo_fluxes not available; skip heatmap")
        return
    end
    ps = bo_data["param_space"]
    # 取其它参数为区间中点
    mid = r -> 0.5 * (minimum(r) + maximum(r))
    defaults = Dict(
        :k2f=>mid(ps.k2f_range), :k2r=>mid(ps.k2r_range),
        :k3f=>mid(ps.k3f_range), :k3r=>mid(ps.k3r_range),
        :k4f=>mid(ps.k4f_range), :k4r=>mid(ps.k4r_range),
        :A=>mid(ps.A_range), :B=>mid(ps.B_range), :C=>mid(ps.C_range),
        :E1=>mid(ps.E1_range), :E2=>mid(ps.E2_range)
    )

    k1f_vals = range(minimum(ps.k1f_range), maximum(ps.k1f_range), length=steps)
    k1r_vals = range(minimum(ps.k1r_range), maximum(ps.k1r_range), length=steps)
    Z1 = fill(NaN, steps, steps)
    Z2 = fill(NaN, steps, steps)

    for (ix, k1f_v) in enumerate(k1f_vals)
        for (iy, k1r_v) in enumerate(k1r_vals)
            rate_params = Dict(
                :k1f=>k1f_v, :k1r=>k1r_v,
                :k2f=>defaults[:k2f], :k2r=>defaults[:k2r],
                :k3f=>defaults[:k3f], :k3r=>defaults[:k3r],
                :k4f=>defaults[:k4f], :k4r=>defaults[:k4r]
            )
            initial_conditions = [
                A=>defaults[:A], B=>defaults[:B], C=>defaults[:C],
                E1=>defaults[:E1], E2=>defaults[:E2], AE1=>0.0, BE2=>0.0
            ]
            try
                sol = simulate_system(rate_params, initial_conditions, ps.tspan, saveat=0.05)
                th = calculate_thermo_fluxes(sol, rate_params)
                v1 = th["v1_thermo"]; v2 = th["v2_thermo"]
                m = length(v1)
                w = max(1, Int(clamp(round(m * tail_frac), 1, m)))
                Z1[iy, ix] = mean(view(v1, m-w+1:m))
                Z2[iy, ix] = mean(view(v2, m-w+1:m))
            catch
                Z1[iy, ix] = NaN; Z2[iy, ix] = NaN
            end
        end
    end

    p1 = heatmap(k1f_vals, k1r_vals, Z1, xlabel="k1f", ylabel="k1r",
                 title="v1_thermo_mean (steady) heatmap", color=:viridis)
    savefig_safely(p1, "thermo_v1_heatmap_k1f_k1r_recomputed_steady")

    p2 = heatmap(k1f_vals, k1r_vals, Z2, xlabel="k1f", ylabel="k1r",
                 title="v2_thermo_mean (steady) heatmap", color=:viridis)
    savefig_safely(p2, "thermo_v2_heatmap_k1f_k1r_recomputed_steady")
end


const RESULT_DIR = joinpath(@__DIR__, "result")
const MODEL_DIR  = joinpath(@__DIR__, "model")

function ensure_dir(path::String)
    if !isdir(path)
        mkpath(path)
    end
end

function savefig_safely(p::Plots.Plot, filename::String)
    ensure_dir(RESULT_DIR)
    filepath = joinpath(RESULT_DIR, filename)
    png(p, filepath)
    println("📁 Saved figure: $(filepath).png")
end

"""
    load_artifacts(; model_path=..., results_path=...)

Load trained model and scan results.
Defaults: ML/model/cuda_integrated_surrogate.jld2 and ML/model/large_scale_scan_results.jld2
"""
function load_artifacts(; model_path::String=joinpath(MODEL_DIR, "cuda_integrated_surrogate.jld2"),
                           results_path::String=joinpath(MODEL_DIR, "large_scale_scan_results.jld2"))
    surrogate = load_surrogate_model(model_path)

    scan_results = nothing
    scan_config = nothing
    comparison_results = nothing
    if isfile(results_path)
        data = JLD2.load(results_path)
        scan_results = get(data, "scan_results", nothing)
        scan_config = get(data, "scan_config", nothing)
        comparison_results = get(data, "comparison_results", nothing)
    end

    return surrogate, scan_results, scan_config, comparison_results
end

"""
    plot_training_history(surrogate)
"""
function plot_training_history(surrogate)
    h = surrogate.training_history
    if isempty(h)
        println("⚠️  No training history; skip")
        return
    end
    p = plot(h, xlabel="Epoch", ylabel="MSE Loss", title="Training Loss", lw=2, label="train")
    try
        if size(surrogate.X_val, 1) > 0
            y_pred, _ = predict_with_uncertainty(surrogate, surrogate.X_val, n_samples=20)
            y_val_true = surrogate.y_val .* surrogate.output_scaler.std .+ surrogate.output_scaler.mean
            val_mse = mean((y_pred - y_val_true).^2)
            scatter!([length(h)], [val_mse], label="val (current)", ms=6)
        end
    catch
    end
    savefig_safely(p, "training_loss")
end

"""
    plot_prediction_vs_truth(surrogate; n=200)
"""
function plot_prediction_vs_truth(surrogate; n::Int=200)
    ps = surrogate.param_space
    X = generate_lhs_samples(ps, n)
    y_true = simulate_parameter_batch(X, ps.tspan, surrogate.config.target_variables)
    y_pred, _ = predict_with_uncertainty(surrogate, X, n_samples=50)

    for (j, outvar) in enumerate(surrogate.config.target_variables)
        tvals = y_true[:, j]
        pvals = y_pred[:, j]

        p = scatter(tvals, pvals, xlabel="True $(outvar)", ylabel="Pred $(outvar)", title="Prediction vs Truth ($(outvar))",
                    legend=:topright, ms=3, alpha=0.6, label="samples")
        lo = min(minimum(tvals), minimum(pvals))
        hi = max(maximum(tvals), maximum(pvals))
        plot!(p, [lo, hi], [lo, hi], lw=2, lc=:gray, ls=:dash, label="y = x")
        var_t = var(tvals)
        if isfinite(var_t) && var_t > 0
            β1 = cov(tvals, pvals) / var_t
            β0 = mean(pvals) - β1 * mean(tvals)
            plot!(p, [lo, hi], β0 .+ β1 .* [lo, hi], lw=2, lc=:red, label=@sprintf("fit: y=%.3f+%.3fx", β0, β1))
        end

        savefig_safely(p, @sprintf("pred_vs_true_%s", string(outvar)))
    end
end

"""
    plot_uncertainty_histograms(surrogate; n=1000)
"""
function plot_uncertainty_histograms(surrogate; n::Int=1000)
    ps = surrogate.param_space
    X = generate_lhs_samples(ps, n)
    _, y_std = predict_with_uncertainty(surrogate, X, n_samples=50)
    for (j, var) in enumerate(surrogate.config.target_variables)
        p = histogram(y_std[:, j], bins=40, xlabel="Std of $(var)", ylabel="Count",
                      title="Uncertainty Histogram ($(var))")
        savefig_safely(p, @sprintf("uncertainty_hist_%s", string(var)))
    end
end

"""
    plot_scan_distributions(scan_results, target_vars)
"""
function plot_scan_distributions(scan_results, target_vars)
    if scan_results === nothing || length(scan_results) == 0
        println("⚠️  No scan results; skip")
        return
    end
    for var in target_vars
        vals = [r.predictions[var] for r in scan_results]
        p = histogram(vals, bins=60, xlabel=string(var), ylabel="Count", title="Distribution of $(var)")
        savefig_safely(p, @sprintf("scan_dist_%s", string(var)))
    end
end

"""
    plot_pairwise_heatmaps(surrogate; fixed_defaults=true)
"""
function plot_pairwise_heatmaps(surrogate; fixed_defaults::Bool=true)
    if :C_final ∉ surrogate.config.target_variables
        println("⚠️  C_final not configured; skip heatmaps")
        return
    end
    c_idx = findfirst(==( :C_final ), surrogate.config.target_variables)

    ps = surrogate.param_space
    ranges = (
        A=ps.A_range, B=ps.B_range, E1=ps.E1_range,
        k1f=ps.k1f_range, k1r=ps.k1r_range
    )

    function grid_plot(p1::Symbol, p2::Symbol, filename::String; steps::Int=50)
        defaults = [mean(r) for r in (
            ps.k1f_range, ps.k1r_range, ps.k2f_range, ps.k2r_range,
            ps.k3f_range, ps.k3r_range, ps.k4f_range, ps.k4r_range,
            ps.A_range, ps.B_range, ps.C_range, ps.E1_range, ps.E2_range
        )]

        param_index = Dict(
            :k1f=>1, :k1r=>2, :k2f=>3, :k2r=>4, :k3f=>5, :k3r=>6, :k4f=>7, :k4r=>8,
            :A=>9, :B=>10, :C=>11, :E1=>12, :E2=>13
        )

        xvals = range(minimum(ranges[p1]), maximum(ranges[p1]), length=steps)
        yvals = range(minimum(ranges[p2]), maximum(ranges[p2]), length=steps)
        Z = Array{Float64}(undef, steps, steps)

        for (ix, xv) in enumerate(xvals)
            for (iy, yv) in enumerate(yvals)
                v = copy(defaults)
                v[param_index[p1]] = xv
                v[param_index[p2]] = yv
                y_pred, _ = predict_with_uncertainty(surrogate, reshape(v, 1, :), n_samples=20)
                Z[iy, ix] = y_pred[1, c_idx]
            end
        end

        p = heatmap(xvals, yvals, Z, xlabel=string(p1), ylabel=string(p2),
                    title="C_final heatmap: $(p1) vs $(p2)", color=:viridis)
        savefig_safely(p, filename)
    end

    grid_plot(:A, :B, "heatmap_C_A_vs_B")
    grid_plot(:A, :E1, "heatmap_C_A_vs_E1")
    grid_plot(:k1f, :k1r, "heatmap_C_k1f_vs_k1r")
end

# ===== Bayesian Optimization Plots =====

"""
    load_bayesian_optimization_results()
"""
function load_bayesian_optimization_results()
    bo_results_path = joinpath(MODEL_DIR, "bayesian_optimization_results.jld2")
    chosen_path = nothing

    if isfile(bo_results_path)
        chosen_path = bo_results_path
    else
        # Fallback: try result directory latest bayesian_opt_iter_*.jld2
        try
            result_files = readdir(RESULT_DIR; join=true)
            cand = filter(f -> endswith(f, ".jld2") && occursin(r"bayesian_opt|bo_results", basename(f)), result_files)
            if !isempty(cand)
                # pick most recent by mtime
                times = stat.(cand)
                idx = argmax(map(x -> x.mtime, times))
                chosen_path = cand[idx]
            end
        catch
            # ignore directory read errors
        end
    end

    if chosen_path === nothing
        println("⚠️  BO results file not found in model or result directories")
        return nothing
    end

    try
        data = JLD2.load(chosen_path)
        println("📂 Loaded BO results from: $(chosen_path)")
        return data
    catch e
        println("❌ Failed to load BO results from $(chosen_path): $e")
        return nothing
    end
end

"""
    plot_bayesian_optimization_results()
"""
function plot_bayesian_optimization_results()
    println("📈 Generating Bayesian optimization visualizations...")
    
    bo_data = load_bayesian_optimization_results()
    if bo_data === nothing
        return
    end
    
    plot_bo_convergence(bo_data)
    
    plot_bo_exploration_path(bo_data)
    
    plot_bo_acquisition_evolution(bo_data)
    
    plot_bo_parameter_importance(bo_data)

    plot_bo_thermo_flux_history(bo_data)
    plot_bo_thermo_vs_params(bo_data)
    plot_bo_thermo_vs_params_recomputed(bo_data)
    plot_bo_thermo_heatmap_k1f_k1r_recomputed(bo_data)

    plot_bo_thermo_global_influence(bo_data)
    # Dashboards
    try
        plot_thermo_dashboard_overview(bo_data)
        plot_thermo_dashboard_scatter(bo_data)
        plot_thermo_dashboard_heatmaps(bo_data)
    catch e
        println("⚠️  Thermo dashboards failed: $e")
    end
    
    println("✅ Bayesian optimization visualizations completed")
end

"""
    plot_bo_convergence(bo_data)
"""
function plot_bo_convergence(bo_data)
    y_values = bo_data["y_evaluated"]
    config = bo_data["config"]
    n_points = length(y_values)
    
    cumulative_best = zeros(n_points)
    cumulative_best[1] = y_values[1]
    
    for i in 2:n_points
        cumulative_best[i] = max(cumulative_best[i-1], y_values[i])
    end
    
    p = plot(1:n_points, cumulative_best, 
             xlabel="Evaluations", ylabel="Best objective", 
             title="Bayesian Optimization Convergence", 
             lw=3, label="Cumulative best", color=:blue)
    
    if hasproperty(config, :n_initial_points)
        vline!([config.n_initial_points], 
               label="Initial exploration end", color=:red, ls=:dash, lw=2)
    end
    
    scatter!(1:n_points, y_values, 
             label="Evaluations", alpha=0.4, ms=2, color=:green)
    
    savefig_safely(p, "bayesian_convergence_detailed")
end

"""
    plot_bo_exploration_path(bo_data)
"""
function plot_bo_exploration_path(bo_data)
    X_evaluated = bo_data["X_evaluated"]
    y_evaluated = bo_data["y_evaluated"]
    
    if size(X_evaluated, 2) < 2
        println("⚠️  Not enough parameter dimensions; skip exploration path plot")
        return
    end
    
    x1 = X_evaluated[:, 1]
    x2 = X_evaluated[:, 2]
    
    n_points = length(y_evaluated)
    colors = 1:n_points
    
    p = scatter(x1, x2, 
                xlabel="k1f", ylabel="k1r",
                title="BO Parameter Exploration Path",
                zcolor=colors, 
                colorbar_title="Evaluation order",
                ms=4, alpha=0.8)
    
    plot!(p, x1, x2, 
          color=:gray, alpha=0.3, lw=1, label="Path")
    
    scatter!(p, [x1[1]], [x2[1]], 
             ms=8, color=:green, shape=:star, label="Start")
    scatter!(p, [x1[end]], [x2[end]], 
             ms=8, color=:red, shape=:diamond, label="End")
    
    savefig_safely(p, "bayesian_exploration_path")
end

"""
    plot_bo_acquisition_evolution(bo_data)
"""
function plot_bo_acquisition_evolution(bo_data)
    if !haskey(bo_data, "acquisition_history")
        println("⚠️  No acquisition history")
        return
    end
    
    acq_history = bo_data["acquisition_history"]
    if isempty(acq_history)
        println("⚠️  Acquisition history is empty")
        return
    end
    
    n_acq = length(acq_history)
    
    p = plot(1:n_acq, acq_history,
             xlabel="Optimization iteration", ylabel="Acquisition value",
             title="Acquisition Function Evolution",
             lw=2, label="Acquisition", color=:purple)
    
    if n_acq > 5
        window_size = min(5, n_acq ÷ 3)
        smoothed = [mean(acq_history[max(1, i-window_size+1):i]) for i in window_size:n_acq]
        plot!(p, window_size:n_acq, smoothed,
              lw=2, color=:orange, alpha=0.7, label="Trend")
    end
    
    savefig_safely(p, "bayesian_acquisition_evolution_detailed")
end

"""
    plot_bo_parameter_importance(bo_data)
"""
function plot_bo_parameter_importance(bo_data)
    X_evaluated = bo_data["X_evaluated"]
    y_evaluated = bo_data["y_evaluated"]
    
    n_params = size(X_evaluated, 2)
    param_names = ["k1f", "k1r", "k2f", "k2r", "k3f", "k3r", "k4f", "k4r",
                   "A", "B", "C", "E1", "E2"]
    
    correlations = zeros(n_params)
    
    for i in 1:n_params
        if var(X_evaluated[:, i]) > 1e-8
            correlations[i] = abs(cor(X_evaluated[:, i], y_evaluated))
        end
    end
    
    sorted_indices = sortperm(correlations, rev=true)
    sorted_correlations = correlations[sorted_indices]
    sorted_names = param_names[sorted_indices]
    
    p = bar(sorted_names, sorted_correlations,
            xlabel="Parameter", ylabel="|Correlation|",
            title="Parameter Importance",
            xrotation=45, color=:viridis)
    
    savefig_safely(p, "bayesian_parameter_importance")
end

"""
    generate_all_plots_with_bayesian()
"""
function generate_all_plots_with_bayesian(;
                                          model_path::String=joinpath(MODEL_DIR, "cuda_integrated_surrogate.jld2"),
                                          results_path::String=joinpath(MODEL_DIR, "large_scale_scan_results.jld2"))
    
    println("🎨 Generating full visualization suite...")

    println("\n1️⃣ Surrogate model visualizations...")
    try
        surrogate, scan_results, _, _ = load_artifacts(model_path=model_path, results_path=results_path)
        if surrogate.config.model_type == :neural_network
            plot_training_history(surrogate)
        elseif surrogate.config.model_type == :gaussian_process
            try
                if size(surrogate.X_val, 1) > 0
                    if @isdefined(predict_gaussian_process)
                        y_pred = predict_gaussian_process(surrogate, surrogate.X_val)
                        y_true = surrogate.y_val .* surrogate.output_scaler.std .+ surrogate.output_scaler.mean
                        mses = [mean((y_pred[:, j] - y_true[:, j]).^2) for j in 1:size(y_pred, 2)]
                        labels = string.(surrogate.config.target_variables)
                        p_mse = bar(labels, mses, xlabel="Output", ylabel="MSE", title="GP Validation MSE")
                        savefig_safely(p_mse, "gp_validation_mse")
                    else
                        println("⚠️  predict_gaussian_process not defined; skip GP MSE plot")
                    end
                else
                    println("⚠️  No validation set; skip GP MSE plot")
                end
            catch e
                println("⚠️  GP MSE plotting failed: $e")
            end
        end
        
        plot_prediction_vs_truth(surrogate)
        plot_uncertainty_histograms(surrogate)
        plot_scan_distributions(scan_results, surrogate.config.target_variables)
        plot_pairwise_heatmaps(surrogate)
        
        println("✅ Surrogate visualizations completed")
    catch e
        println("❌ Surrogate visualizations failed: $e")
    end
    
    println("\n2️⃣ Bayesian optimization visualizations...")
    try
        plot_bayesian_optimization_results()
    catch e
        println("❌ Bayesian optimization visualizations failed: $e")
    end
    
    println("\n🎉 Full visualization suite complete!")
    println("📊 Includes:")
    println("  - Surrogate performance analysis")
    println("  - Parameter-space heatmaps")
    println("  - Uncertainty analysis")
    println("  - BO convergence analysis")
    println("  - Parameter exploration path")
    println("  - Acquisition evolution")
    println("  - Parameter importance")
end

"""
    generate_all_plots(; model_path=..., results_path=...)
"""
function generate_all_plots(; model_path::String=joinpath(MODEL_DIR, "cuda_integrated_surrogate.jld2"),
                               results_path::String=joinpath(MODEL_DIR, "large_scale_scan_results.jld2"))
    surrogate, scan_results, _, _ = load_artifacts(model_path=model_path, results_path=results_path)

    if surrogate.config.model_type == :neural_network
        plot_training_history(surrogate)
    elseif surrogate.config.model_type == :gaussian_process
        try
            if size(surrogate.X_val, 1) > 0
                y_pred = predict_gaussian_process(surrogate, surrogate.X_val)
                y_true = surrogate.y_val .* surrogate.output_scaler.std .+ surrogate.output_scaler.mean
                mses = [mean((y_pred[:, j] - y_true[:, j]).^2) for j in eachindex(y_pred, y_true)]
                labels = string.(surrogate.config.target_variables)
                p_mse = bar(labels, mses, xlabel="Output", ylabel="MSE", title="GP Validation MSE")
                savefig_safely(p_mse, "gp_validation_mse")
            else
                println("⚠️  No validation set; skip GP MSE plot")
            end
        catch e
            println("⚠️  GP validation MSE plotting failed: $(e)")
        end
    end

    plot_prediction_vs_truth(surrogate)
    plot_uncertainty_histograms(surrogate)

    plot_scan_distributions(scan_results, surrogate.config.target_variables)
    plot_pairwise_heatmaps(surrogate)

    println("✅ All figures saved to: $(RESULT_DIR)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Run full suite including Bayesian optimization visuals when invoked directly
    generate_all_plots_with_bayesian()
end


