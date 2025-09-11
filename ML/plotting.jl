"""
绘图工具 plotting.jl

根据 cuda_integrated_example.jl 的工作流，生成一系列可视化，结果保存在 result/。

输出目录：/home/ryankwok/Documents/TwoEnzymeSim/result
"""

using JLD2
using Statistics
using Printf
using Plots

include("surrogate_model.jl")

const RESULT_DIR = "ML/result"
const MODEL_DIR  = "ML/model"

function ensure_dir(path::String)
    if !isdir(path)
        mkpath(path)
    end
end

function savefig_safely(p::Plots.Plot, filename::String)
    ensure_dir(RESULT_DIR)
    filepath = joinpath(RESULT_DIR, filename)
    png(p, filepath)
    println("📁 已保存图像: $(filepath).png")
end

"""
    load_artifacts(; model_path=..., results_path=...)

加载已训练模型与扫描结果。
默认从 ML/model/cuda_integrated_surrogate.jld2 与 ML/model/large_scale_scan_results.jld2 读取。
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
        println("⚠️  无训练历史，跳过训练曲线绘制")
        return
    end
    p = plot(h, xlabel="Epoch", ylabel="MSE Loss", title="Training Loss", lw=2, legend=false)
    savefig_safely(p, "training_loss")
end

"""
    plot_prediction_vs_truth(surrogate; n=200)

随机采样小批量，用CPU仿真得到真值，和代理预测对比散点。
"""
function plot_prediction_vs_truth(surrogate; n::Int=200)
    ps = surrogate.param_space
    X = generate_lhs_samples(ps, n)
    y_true = simulate_parameter_batch(X, ps.tspan, surrogate.config.target_variables)
    y_pred, _ = predict_with_uncertainty(surrogate, X, n_samples=50)

    for (j, var) in enumerate(surrogate.config.target_variables)
        tvals = y_true[:, j]
        pvals = y_pred[:, j]
        p = scatter(tvals, pvals, xlabel="True $(var)", ylabel="Pred $(var)", title="Prediction vs Truth ($(var))",
                    legend=false, ms=3, alpha=0.6)
        plot!(minimum(tvals):maximum(tvals), minimum(tvals):maximum(tvals), lw=2, lc=:red)
        savefig_safely(p, @sprintf("pred_vs_true_%s", string(var)))
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
        println("⚠️  无扫描结果，跳过扫描分布绘制")
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

对常见成对参数 (A,B), (A,E1), (k1f,k1r) 生成 2D 网格热力图，显示 C_final。
"""
function plot_pairwise_heatmaps(surrogate; fixed_defaults::Bool=true)
    if :C_final ∉ surrogate.config.target_variables
        println("⚠️  未配置 C_final，跳过热力图")
        return
    end
    c_idx = findfirst(==( :C_final ), surrogate.config.target_variables)

    ps = surrogate.param_space
    ranges = (
        A=ps.A_range, B=ps.B_range, E1=ps.E1_range,
        k1f=ps.k1f_range, k1r=ps.k1r_range
    )

    function grid_plot(p1::Symbol, p2::Symbol, filename::String; steps::Int=50)
        # 构造默认中心
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

"""
    generate_all_plots(; model_path=..., results_path=...)
"""
function generate_all_plots(; model_path::String=joinpath(MODEL_DIR, "cuda_integrated_surrogate.jld2"),
                               results_path::String=joinpath(MODEL_DIR, "large_scale_scan_results.jld2"))
    surrogate, scan_results, _, _ = load_artifacts(model_path=model_path, results_path=results_path)

    plot_training_history(surrogate)
    plot_prediction_vs_truth(surrogate)
    plot_uncertainty_histograms(surrogate)

    plot_scan_distributions(scan_results, surrogate.config.target_variables)
    plot_pairwise_heatmaps(surrogate)

    println("✅ 所有图像已生成到: $(RESULT_DIR)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    generate_all_plots()
end


