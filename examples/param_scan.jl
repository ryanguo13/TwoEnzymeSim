using Distributed
using ProgressMeter
using Plots
using IterTools
using Statistics 


include("../src/parameters.jl")
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

# Create a grid of parameters (only reaction rate constants)
param_grid = Iterators.product(
    k1f_range, k1r_range, k2f_range, k2r_range, k3f_range, k3r_range,
    k4f_range, k4r_range)

# Calculate total combinations for progress reporting
total_combinations = length(k1f_range) * length(k1r_range) * length(k2f_range) * length(k2r_range) * 
                    length(k3f_range) * length(k3r_range) * length(k4f_range) * length(k4r_range)
println("Total parameter combinations: $total_combinations")

# Fixed initial conditions for all simulations
fixed_initial_conditions = Dict(
    Symbol("A") => 5.0,
    Symbol("B") => 0.0,
    Symbol("C") => 0.0,
    Symbol("E1") => 20.0,
    Symbol("E2") => 15.0,
    Symbol("AE1") => 0.0,
    Symbol("BE2") => 0.0
)

# Preprocess the solution to get the final concentrations of A, B, C, E1, E2
function preprocess_solution(sol)
    # Check for overflow or NaN in the solution
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
    # Unpack the parameters (only reaction rate constants)
    k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r = params
    # Check for overflow/invalid parameter values
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
    try
        sol = solve(ODEProblem(rn, fixed_initial_conditions, tspan, p), Tsit5(), saveat=0.1)
        return preprocess_solution(sol)
    catch
        return nothing
    end
end

# Example usage: scan a subset for demonstration
results = []
global progress = 0
max_progress = min(10000, total_combinations)  # Limit to 10000 simulations

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

# Show a report of the results
println("\n")
println("Number of results: $(length(results))")

# Plot the results - create multiple visualization types
if length(results) > 0
    # Extract all parameters and results
    x_vals = [params[1] for (params, res) in results]  # k1f values
    y_vals = [params[2] for (params, res) in results]  # k1r values
    z_vals = [res[1] for (params, res) in results]     # A concentration values
    
    # Print the actual ranges found in the data
    println("k1f range in results: $(minimum(x_vals)) to $(maximum(x_vals))")
    println("k1r range in results: $(minimum(y_vals)) to $(maximum(y_vals))")
    
    # 1. 多子热力图 - 显示所有物种的浓度
    p1 = plot_multi_species_heatmap(results)
    savefig(p1, "multi_species_heatmap.png")
    println("Multi-species heatmap saved as multi_species_heatmap.png")
    
    # 2. 参数敏感性分析
    p2 = plot_parameter_sensitivity_analysis(results)
    savefig(p2, "parameter_sensitivity.png")
    println("Parameter sensitivity analysis saved as parameter_sensitivity.png")
    
    # 3. 浓度分布直方图
    p3 = plot_concentration_distributions(results)
    savefig(p3, "concentration_distributions.png")
    println("Concentration distributions saved as concentration_distributions.png")
    
    # 4. 3D参数空间图 (k1f vs k1r vs k2f)
    p4 = plot_3d_parameter_space(results, 1, 2, 3)
    savefig(p4, "3d_parameter_space.png")
    println("3D parameter space plot saved as 3d_parameter_space.png")
    
    # 5. 创建规则网格的contour图
    # 首先创建规则网格
    k1f_unique = sort(unique(x_vals))
    k1r_unique = sort(unique(y_vals))
    
    # 创建网格
    k1f_grid = repeat(k1f_unique, outer=length(k1r_unique))
    k1r_grid = repeat(k1r_unique, inner=length(k1f_unique))
    
    # 对每个网格点插值或找到最近的值
    z_grid = zeros(length(k1f_unique), length(k1r_unique))
    
    for i in eachindex(k1f_unique) # using eachaxes method
        for j in eachindex(k1r_unique)
            # 找到最接近的k1f和k1r组合
            distances = [(x_vals[k] - k1f_unique[i])^2 + (y_vals[k] - k1r_unique[j])^2 for k in eachindex(x_vals)] # using eachindex method
            closest_idx = argmin(distances)
            z_grid[i, j] = z_vals[closest_idx]
        end
    end
    
    # 创建contour图
    p5 = contour(k1f_unique, k1r_unique, z_grid, 
                xlabel="k1f", ylabel="k1r", 
                title="Parameter Scan Results (A concentration)",
                colorbar_title="[A] final")
    savefig(p5, "param_scan.png")
    println("Original contour plot saved as param_scan.png")
    
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
    
else
    println("No valid results to plot")
end

# Inspect the first few results 
println("\nFirst 5 results:")
for (i, (params, res)) in enumerate(results[1:min(5, length(results))])
    println("Result $i:")
    println("  Parameters (k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r): $params")
    println("  Final concentrations [A, B, C, E1, E2]: $res")
    println()
end





