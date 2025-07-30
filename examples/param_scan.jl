using Distributed
using ProgressMeter
using Plots
using IterTools
using Statistics 


# include("../src/parameters.jl")
include("../src/simulation.jl")
include("../src/visualization.jl")

# Define the parameters of the reaction rate iteration range
# Use fewer points to cover the full range
k1f_range = 0.1:0.5:10.0  # 20 points
k1r_range = 0.1:0.5:10.0  # 20 points
k2f_range = 0.1:0.5:10.0  # 20 points
k2r_range = 0.1:0.5:10.0  # 20 points
k3f_range = 0.1:0.5:10.0  # 20 points
k3r_range = 0.1:0.5:10.0  # 20 points
k4f_range = 0.1:0.5:10.0  # 20 points
k4r_range = 0.1:0.5:10.0  # 20 points

A_range = 5.0:0.5:20.0
B_range = 0.0:0.5:5.0
C_range = 0.0:0.5:5.0
E1_range = 5.0:0.5:20.0
E2_range = 5.0:0.5:20.0

# Create a grid of parameters (reaction rate constants + initial concentrations)
param_grid = Iterators.product(
    k1f_range, k1r_range, k2f_range, k2r_range, k3f_range, k3r_range,
    k4f_range, k4r_range, A_range, B_range, C_range, E1_range, E2_range
)

# Calculate total combinations for progress reporting
 total_combinations = length(k1f_range) * length(k1r_range) * length(k2f_range) * length(k2r_range) * 
                    length(k3f_range) * length(k3r_range) * length(k4f_range) * length(k4r_range) *
                    length(A_range) * length(B_range) * length(C_range) * length(E1_range) * length(E2_range)
println("Total parameter combinations: $total_combinations")

# Preprocess the solution to get the final concentrations of A, B, C, E1, E2
function preprocess_solution(sol)
    try
        vals = [sol[Symbol("A")][end], sol[Symbol("B")][end], sol[Symbol("C")][end], sol[Symbol("E1")][end], sol[Symbol("E2")][end]]
        if any(x -> isnan(x) || isinf(x) || abs(x) > 1e6, vals)
            return nothing
        end
        return vals
    catch
        return nothing
    end
end

function simulate_reaction(params, tspan)
    # Unpack the parameters (reaction rate constants + initial concentrations)
    k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, A0, B0, C0, E1_0, E2_0 = params
    if any(x -> isnan(x) || isinf(x) || abs(x) > 1e6, params)
        return nothing
    end
    rn = @reaction_network begin
        k1f,  A + E1 --> AE1
        k1r,  AE1 --> A + E1
        k2f,  AE1 --> B + E1
        k2r,  B + E1 --> AE1
        k3f,  B + E2 --> BE2
        k3r,  BE2 --> B + E2
        k4f,  BE2 --> C + E2
        k4r,  C + E2 --> BE2
    end
    p = Dict(:k1f=>k1f, :k1r=>k1r, :k2f=>k2f, :k2r=>k2r, :k3f=>k3f, :k3r=>k3r, :k4f=>k4f, :k4r=>k4r)
    initial_conditions = Dict(:A=>A0, :B=>B0, :C=>C0, :E1=>E1_0, :E2=>E2_0, :AE1=>0.0, :BE2=>0.0)
    try
        sol = solve(ODEProblem(rn, initial_conditions, tspan, p), Tsit5(), saveat=0.1)
        return preprocess_solution(sol)
    catch
        return nothing
    end
end

results = []
global progress = 0
max_progress = min(1000000, total_combinations)  # Limit to 10000 simulations
println("Running $max_progress simulations...")
for (i, params) in enumerate(Iterators.take(param_grid, max_progress))
    global progress += 1
    if progress % 1000 == 0
        println("Progress: $progress/$max_progress")
    end
    res = simulate_reaction(params, (0.0, 5.0))
    if res !== nothing
        push!(results, (params, res))
    end
end

println("\n")
println("Number of results: $(length(results))")

# Plotting and statistics (synchronize with CUDA version)
if length(results) > 0
    # 选择用于可视化的参数索引（如k1f、k1r、A0等）
    x_idx = 1  # k1f
    y_idx = 2  # k1r
    z_idx = 9  # A0（初始A浓度），如需可视化初始浓度对结果的影响可切换

    x_vals = [params[x_idx] for (params, res) in results]
    y_vals = [params[y_idx] for (params, res) in results]
    z_vals = [res[1] for (params, res) in results]

    println("k1f range in results: $(minimum(x_vals)) to $(maximum(x_vals))")
    println("k1r range in results: $(minimum(y_vals)) to $(maximum(y_vals))")
    println("A0 (initial) range in results: $(minimum([params[9] for (params,_) in results])) to $(maximum([params[9] for (params,_) in results]))")

    # 1. 多子热力图
    p1 = plot_multi_species_heatmap(results)
    if p1 !== nothing
        savefig(p1, "multi_species_heatmap.png")
        println("Multi-species heatmap saved as multi_species_heatmap.png")
    else
        println("Warning: Could not create multi-species heatmap - no valid data")
    end

    # 2. 参数敏感性分析
    p2 = plot_parameter_sensitivity_analysis(results)
    if p2 !== nothing
        savefig(p2, "parameter_sensitivity.png")
        println("Parameter sensitivity analysis saved as parameter_sensitivity.png")
    else
        println("Warning: Could not create parameter sensitivity analysis - no valid data")
    end

    # 3. 浓度分布直方图
    p3 = plot_concentration_distributions(results)
    if p3 !== nothing
        savefig(p3, "concentration_distributions.png")
        println("Concentration distributions saved as concentration_distributions.png")
    else
        println("Warning: Could not create concentration distributions - no valid data")
    end

    # 4. 3D参数空间图
    # A浓度
    p4_A = plot_3d_parameter_space(results, x_idx, y_idx, z_idx)
    if p4_A !== nothing
        savefig(p4_A, "3d_parameter_space_A.png")
        println("3D parameter space plot (A) saved as 3d_parameter_space_A.png")
    else
        println("Warning: Could not create 3D parameter space plot for A - no valid data")
    end
    # B浓度
    function plot_3d_parameter_space_B(results, param1_idx, param2_idx, param3_idx)
        if isempty(results)
            println("No results to plot")
            return nothing
        end
        valid_results = []
        for (i, (params, res)) in enumerate(results)
            if res !== nothing && length(res) >= 2
                push!(valid_results, (params, res))
            end
        end
        if isempty(valid_results)
            println("No valid results with sufficient data for B")
            return nothing
        end
        param_names = ["k1f", "k1r", "k2f", "k2r", "k3f", "k3r", "k4f", "k4r", "A", "B", "C", "E1", "E2"]
        x_vals = [params[param1_idx] for (params, res) in valid_results]
        y_vals = [params[param2_idx] for (params, res) in valid_results]
        z_vals = [res[2] for (params, res) in valid_results]  # B浓度
        p = scatter3d(x_vals, y_vals, z_vals,
                      xlabel=param_names[param1_idx],
                      ylabel=param_names[param2_idx],
                      zlabel="[B] final",
                      title="3D Parameter Space: $(param_names[param1_idx]) vs $(param_names[param2_idx]) vs [B]",
                      markersize=2, color=:plasma)
        return p
    end
    p4_B = plot_3d_parameter_space_B(results, x_idx, y_idx, z_idx)
    if p4_B !== nothing
        savefig(p4_B, "3d_parameter_space_B.png")
        println("3D parameter space plot (B) saved as 3d_parameter_space_B.png")
    else
        println("Warning: Could not create 3D parameter space plot for B - no valid data")
    end
    # C浓度
    function plot_3d_parameter_space_C(results, param1_idx, param2_idx, param3_idx)
        if isempty(results)
            println("No results to plot")
            return nothing
        end
        valid_results = []
        for (i, (params, res)) in enumerate(results)
            if res !== nothing && length(res) >= 3
                push!(valid_results, (params, res))
            end
        end
        if isempty(valid_results)
            println("No valid results with sufficient data for C")
            return nothing
        end
        param_names = ["k1f", "k1r", "k2f", "k2r", "k3f", "k3r", "k4f", "k4r", "A", "B", "C", "E1", "E2"]
        x_vals = [params[param1_idx] for (params, res) in valid_results]
        y_vals = [params[param2_idx] for (params, res) in valid_results]
        z_vals = [res[3] for (params, res) in valid_results]  # C浓度
        p = scatter3d(x_vals, y_vals, z_vals,
                      xlabel=param_names[param1_idx],
                      ylabel=param_names[param2_idx],
                      zlabel="[C] final",
                      title="3D Parameter Space: $(param_names[param1_idx]) vs $(param_names[param2_idx]) vs [C]",
                      markersize=2, color=:cividis)
        return p
    end
    p4_C = plot_3d_parameter_space_C(results, x_idx, y_idx, z_idx)
    if p4_C !== nothing
        savefig(p4_C, "3d_parameter_space_C.png")
        println("3D parameter space plot (C) saved as 3d_parameter_space_C.png")
    else
        println("Warning: Could not create 3D parameter space plot for C - no valid data")
    end

    # 5. 规则网格contour图
    k1f_unique = sort(unique(x_vals))
    k1r_unique = sort(unique(y_vals))
    z_grid = zeros(length(k1f_unique), length(k1r_unique))
    a_concentrations = [res[1] for (params, res) in results]
    for i in eachindex(k1f_unique)
        for j in eachindex(k1r_unique)
            distances = [(x_vals[k] - k1f_unique[i])^2 + (y_vals[k] - k1r_unique[j])^2 for k in eachindex(x_vals)]
            closest_idx = argmin(distances)
            z_grid[i, j] = a_concentrations[closest_idx]
        end
    end
    p5 = contour(k1f_unique, k1r_unique, z_grid', 
                xlabel="k1f", ylabel="k1r", 
                title="Parameter Scan Results (A concentration)",
                colorbar_title="[A] final")
    savefig(p5, "param_scan.png")
    println("Contour plot saved as param_scan.png")

    # 6. 统计摘要
    println("\n=== Statistical Summary ===")
    a_vals = [res[1] for (params, res) in results]
    b_vals = [res[2] for (params, res) in results] 
    c_vals = [res[3] for (params, res) in results]
    product_ratio = [(b + c)/max(a, 1e-6) for (a,b,c) in zip(a_vals, b_vals, c_vals)]
    println("A concentration: mean=$(round(mean(a_vals), digits=3)), std=$(round(std(a_vals), digits=3))")
    println("B concentration: mean=$(round(mean(b_vals), digits=3)), std=$(round(std(b_vals), digits=3))")
    println("C concentration: mean=$(round(mean(c_vals), digits=3)), std=$(round(std(c_vals), digits=3))")
    println("Product ratio: mean=$(round(mean(product_ratio), digits=3)), std=$(round(std(product_ratio), digits=3))")

    # 7. 打印前几组参数，明确参数含义
    println("\nFirst 5 results:")
    for (i, (params, res)) in enumerate(results[1:min(5, length(results))])
        println("Result $i:")
        println("  Parameters (k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, A, B, C, E1, E2): $(join(collect(params), ", "))")
        println("  Final concentrations [A, B, C, E1, E2]: $res")
        println()
    end

    # 8. 打印后几组参数，明确参数含义
    println("\nLast 5 results:")
    for (i, (params, res)) in enumerate(results[end-4:end])
        println("Result $i:")
        println("  Parameters (k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r, A, B, C, E1, E2): $(join(collect(params), ", "))")
        println("  Final concentrations [A, B, C, E1, E2]: $res")
        println()
    end
else
    println("No valid results to plot")
end





