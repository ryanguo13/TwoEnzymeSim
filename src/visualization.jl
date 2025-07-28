using Plots
using Statistics  # 添加这行来支持cor函数


"""
    plot_parameter_sensitivity(param_range, steady_vals, param_name)

Plot parameter sensitivity analysis results.
"""
function plot_parameter_sensitivity(param_range, steady_vals, param_name)
    p = plot(param_range, steady_vals[:A], label="[A] Steady State", linewidth=2,
             xlabel=string(param_name), ylabel="Steady State Concentration (M)",
             title="Steady State Sensitivity to $param_name")
    plot!(p, param_range, steady_vals[:B], label="[B] Steady State", linewidth=2)
    plot!(p, param_range, steady_vals[:C], label="[C] Steady State", linewidth=2)
    
    return p
end

"""
    plot_equilibrium_constants(ΔG0_range, equilibrium_constants)

Plot equilibrium constants as a function of ΔG0.
"""
function plot_equilibrium_constants(ΔG0_range, equilibrium_constants)
    p = plot(ΔG0_range ./ 1000, equilibrium_constants, linewidth=2,
             xlabel="ΔG⁰ (kJ/mol)", ylabel="Equilibrium Constant K_eq",
             title="Equilibrium Constants vs. Standard Gibbs Free Energy",
             )
    # plot!([K_eq1], [K_eq2], label="K_eq1 and K_eq2", markersize=8, color=:red)
    return p
end



# Function to plot concentration profiles
function plot_concentrations(sol)
    p = plot(sol.t, [sol[A] sol[B] sol[C]],
        xlabel="Time", 
        ylabel="Concentration",
        title="Concentration Profiles",
        label=["A" "B" "C"],
        linewidth=2
    )
    return p
end

# Function to plot reaction fluxes (now only v1, v2)
function plot_fluxes(v1, v2, sol)
    p = plot(sol.t, [v1 v2],
        xlabel="Time",
        ylabel="Flux",
        title="Reaction Fluxes",
        label=["v₁ (A→B)" "v₂ (B→C)"],
        linewidth=2
    )
    return p
end


function plot_thermo_fluxes(sol, params)
    R = 8.314
    T = 298.15
    # using the reuslt of the calculate_thermo_fluxes
    thermo = calculate_thermo_fluxes(sol, params)
    p = plot(sol.t, [thermo["v1_thermo"] thermo["v2_thermo"]],
        xlabel="Time",
        ylabel="Flux",
        title="Thermodynamic Fluxes/Dissipation Energy",
        label=["v₁ (A→B)" "v₂ (B→C)"],
        linewidth=2
    )
    return p
end

function steady_state_thermo_fluxes(A, B, C, E_tot, k1f, k1r, k2f, k2r, ΔG1_std)
    R = 8.314
    T = 298.15
    expG = exp(ΔG1_std / (R * T))
    num = k1f * k2f * A * E_tot * (1 - expG * B / A)
    denom = (k1r + k2f + k1f * A) * (1 + (k2f / k1r) * expG * B / A)
    return num / denom
end 

# Plot all species concentrations (A, B, C, E1, AE1, E2, BE2)
function plot_all_concentrations(sol)
    p = plot(sol.t, [sol[A] sol[B] sol[C] sol[E1] sol[AE1] sol[E2] sol[BE2]],
        xlabel="Time", ylabel="Concentration",
        title="All Species Concentrations",
        label=["A" "B" "C" "E1" "AE1" "E2" "BE2"],
        linewidth=2)
    return p
end

# Plot fluxes V1, V2 over time
default(; legend=:topright)
function plot_fluxes_time(v1, v2, t)
    p = plot(t, [v1 v2], xlabel="Time", ylabel="Flux",
        title="Fluxes V₁, V₂", label=["V₁" "V₂"], linewidth=2)
    return p
end

# Plot ΔG1, ΔG2 over time
function plot_dG_time(dG1, dG2, t)
    p = plot(t, [dG1 dG2], xlabel="Time", ylabel="ΔG (J/mol)",
        title="Reaction Free Energy Changes", label=["ΔG₁" "ΔG₂"], linewidth=2)
    return p
end

# Plot R1, R2 (forward/reverse flux ratio) over time
function plot_R_time(R1, R2, t)  
    R1_plot = max.(R1, 1e-12)
    R2_plot = max.(R2, 1e-12)
    p = plot(t, [R1_plot R2_plot], xlabel="Time", ylabel="J⁺/J⁻",
        title="Forward/Reverse Flux Ratio", label=["R₁" "R₂"], linewidth=2, yscale=:log10)
    return p
end

# Plot the derivative of the thermodynamic fluxes over time
function plot_derivative_thermo_fluxes(sol, params)
    thermo = calculate_thermo_fluxes(sol, params)
    # calculate the derivative of the thermodynamic fluxes
    dG_diss_1 = thermo["ΔG1"] 
    dG_diss_2 = thermo["ΔG2"]
    dG_diss_dt_1 = diff(dG_diss_1) ./ diff(sol.t)
    dG_diss_dt_2 = diff(dG_diss_2) ./ diff(sol.t)
    p = plot(sol.t[2:end], dG_diss_dt_1, xlabel="Time", ylabel="dG_diss/dt",
        title="Derivative of Thermodynamic Fluxes (dG_diss)/dt", label=["ΔG_diss₁"], linewidth=2)
    plot!(p, sol.t[2:end], dG_diss_dt_2, label=["ΔG_diss₂"], linewidth=2)
    
    # Find zero-crossing points (where derivative changes sign)
    # For ΔG_diss₁
    zero_crossings_1 = findall(diff(sign.(dG_diss_dt_1)) .!= 0)
    # For ΔG_diss₂  
    zero_crossings_2 = findall(diff(sign.(dG_diss_dt_2)) .!= 0)
    
    # Plot zero-crossing points
    if !isempty(zero_crossings_1)
        scatter!(p, sol.t[zero_crossings_1 .+ 1], zeros(length(zero_crossings_1)), 
                label="Zero Crossing ΔG₁", markersize=6, color=:red, marker=:circle)
    end
    if !isempty(zero_crossings_2)
        scatter!(p, sol.t[zero_crossings_2 .+ 1], zeros(length(zero_crossings_2)), 
                label="Zero Crossing ΔG₂", markersize=6, color=:blue, marker=:diamond)
    end
    
    # Also find points where both derivatives are close to zero (steady state)
    tolerance = 1e-6 * maximum(abs.(dG_diss_dt_1))
    steady_state_points = findall(abs.(dG_diss_dt_1) .< tolerance .&& abs.(dG_diss_dt_2) .< tolerance)
    
    # if !isempty(steady_state_points)
    #     scatter!(p, sol.t[steady_state_points .+ 1], zeros(length(steady_state_points)), 
    #             label="Steady State", markersize=8, color=:green, marker=:star)
    # end
    # # Also show the value of the dG_diss_1 and dG_diss_2 at the zero-crossing points using arrow with text using eachindex 
    # # keep some distance between the text and the point
    # # using the distance 100
    # for i in eachindex(zero_crossings_1)
    #     annotate!(p, sol.t[zero_crossings_1[i] .+ 1], dG_diss_1[zero_crossings_1[i] .+ 1], 
    #             text("ΔG₁ = $(dG_diss_1[zero_crossings_1[i] .+ 1])", 10, :red)) # using the font size 10 and the color red
    #     # using quiver! to show the arrow
    #     quiver!(p, sol.t[zero_crossings_1[i] .+ 1], dG_diss_1[zero_crossings_1[i] .+ 1], 
    #             quiver=([0, 100], [0, 0]), color=:red)
    # end
    # for i in eachindex(zero_crossings_2)
    #     annotate!(p, sol.t[zero_crossings_2[i] .+ 1], dG_diss_2[zero_crossings_2[i] .+ 1], 
    #             text("ΔG₂ = $(dG_diss_2[zero_crossings_2[i] .+ 1])", 10, :blue))
    #     quiver!(p, sol.t[zero_crossings_2[i] .+ 1], dG_diss_2[zero_crossings_2[i] .+ 1], 
    #             quiver=([0, 100], [0, 0]), color=:blue)
    # end
    
    return p
end


# Plot the contour graph of the parameter scan of the final concentrations of A, B, C
function plot_param_scan(param_grid, final_concentrations)
    p = contour(param_grid, final_concentrations,
        xlabel="Parameter", ylabel="Time",
        title="Parameter Scan",
        label=["A" "B" "C"],
        linewidth=2
    )
    return p
end

# Plot the contour graph of the E1 and E2 concentrations
function plot_E1_E2_contour(param_grid, sol)
    p = contour(param_grid, sol.t, [sol[E1] sol[E2]],
        xlabel="Parameter", ylabel="Time",
        title="E1 and E2 Concentrations",
        label=["E1" "E2"],
        linewidth=2
    )
    return p
end

# Plot the contour graph of the

"""
    plot_multi_species_heatmap(results)

Create a multi-species heatmap showing the final concentrations of all species.
"""
function plot_multi_species_heatmap(results)
    if isempty(results)
        println("No results to plot")
        return nothing
    end
    
    # Validate data structure and handle different array lengths
    valid_results = []
    for (i, (params, res)) in enumerate(results)
        if res !== nothing && length(res) >= 2  # At least need A and B
            push!(valid_results, (params, res))
        else
            if res === nothing
                println("Warning: Result $i has no data")
            else
                println("Warning: Result $i has insufficient data (length: $(length(res)), expected: 2+)")
            end
        end
    end
    
    if isempty(valid_results)
        println("No valid results with sufficient data")
        return nothing
    end
    
    # Extract data with safe indexing
    x_vals = [params[1] for (params, res) in valid_results]  # k1f values
    y_vals = [params[2] for (params, res) in valid_results]  # k1r values
    a_vals = [res[1] isa Vector ? res[1][end] : (res[1] isa Tuple ? res[1][end] : res[1]) for (params, res) in valid_results]     # A concentration values
    b_vals = [length(res) >= 2 ? (res[2] isa Vector ? res[2][end] : (res[2] isa Tuple ? res[2][end] : res[2])) : 0.0 for (params, res) in valid_results]     # B concentration values
    c_vals = [length(res) >= 3 ? (res[3] isa Vector ? res[3][end] : (res[3] isa Tuple ? res[3][end] : res[3])) : 0.0 for (params, res) in valid_results]     # C concentration values
    e1_vals = [length(res) >= 4 ? (res[4] isa Vector ? res[4][end] : (res[4] isa Tuple ? res[4][end] : res[4])) : 0.0 for (params, res) in valid_results]    # E1 concentration values
    e2_vals = [length(res) >= 5 ? (res[5] isa Vector ? res[5][end] : (res[5] isa Tuple ? res[5][end] : res[5])) : 0.0 for (params, res) in valid_results]    # E2 concentration values
    
    # Calculate product ratio (B+C)/A
    product_ratio = [(b + c)/max(a, 1e-6) for (a,b,c) in zip(a_vals, b_vals, c_vals)]
    
    # Create 2x3 subplot layout
    p = plot(layout=(2,3), size=(1200,800), 
             plot_title="Multi-Species Parameter Sensitivity Analysis")
    
    # A concentration heatmap
    scatter!(p[1], x_vals, y_vals, zcolor=a_vals, 
             xlabel="k1f", ylabel="k1r", 
             title="[A] Final Concentration",
             colorbar_title="[A] final",
             markersize=4, color=:viridis)
    
    # B concentration heatmap
    scatter!(p[2], x_vals, y_vals, zcolor=b_vals, 
             xlabel="k1f", ylabel="k1r", 
             title="[B] Final Concentration",
             colorbar_title="[B] final",
             markersize=4, color=:plasma)
    
    # C concentration heatmap
    scatter!(p[3], x_vals, y_vals, zcolor=c_vals, 
             xlabel="k1f", ylabel="k1r", 
             title="[C] Final Concentration",
             colorbar_title="[C] final",
             markersize=4, color=:inferno)
    
    # E1 concentration heatmap
    scatter!(p[4], x_vals, y_vals, zcolor=e1_vals, 
             xlabel="k1f", ylabel="k1r", 
             title="[E1] Final Concentration",
             colorbar_title="[E1] final",
             markersize=4, color=:magma)
    
    # E2 concentration heatmap
    scatter!(p[5], x_vals, y_vals, zcolor=e2_vals, 
             xlabel="k1f", ylabel="k1r", 
             title="[E2] Final Concentration",
             colorbar_title="[E2] final",
             markersize=4, color=:cividis)
    
    # Product ratio heatmap
    scatter!(p[6], x_vals, y_vals, zcolor=product_ratio, 
             xlabel="k1f", ylabel="k1r", 
             title="Product Ratio (B+C)/A",
             colorbar_title="Ratio",
             markersize=4, color=:turbo)
    
    return p
end

"""
    plot_parameter_sensitivity_analysis(results)

Create a bar plot showing the sensitivity of each parameter to the final A concentration.
"""
function plot_parameter_sensitivity_analysis(results)
    if isempty(results)
        println("No results to analyze")
        return nothing
    end
    
    # Validate data structure and handle different array lengths
    valid_results = []
    for (i, (params, res)) in enumerate(results)
        if res !== nothing && length(res) >= 1
            push!(valid_results, (params, res))
        else
            if res === nothing
                println("Warning: Result $i has no data")
            else
                println("Warning: Result $i has insufficient data (length: $(length(res)), expected: 1+)")
            end
        end
    end
    
    if isempty(valid_results)
        println("No valid results with sufficient data")
        return nothing
    end
    
    param_names = ["k1f", "k1r", "k2f", "k2r", "k3f", "k3r", "k4f", "k4r"]
    sensitivities = []
    
    for param_idx in 1:8
        param_values = [params[param_idx] for (params, res) in valid_results]
        a_concentrations = [res[1] isa Vector ? res[1][end] : (res[1] isa Tuple ? res[1][end] : res[1]) for (params, res) in valid_results]
        
        # Calculate correlation as a simple sensitivity measure
        correlation = cor(param_values, a_concentrations)
        push!(sensitivities, abs(correlation))
    end
    
    p = bar(param_names, sensitivities, 
             xlabel="Parameters", ylabel="Sensitivity (|Correlation|)",
             title="Parameter Sensitivity Analysis",
             legend=false, color=:steelblue)
    
    return p
end

"""
    plot_concentration_distributions(results)

Create histograms showing the distribution of final concentrations for all species.
"""
function plot_concentration_distributions(results)
    if isempty(results)
        println("No results to analyze")
        return nothing
    end
    
    # Validate data structure and handle different array lengths
    valid_results = []
    for (i, (params, res)) in enumerate(results)
        if res !== nothing && length(res) >= 2  # At least need A and B
            push!(valid_results, (params, res))
        else
            if res === nothing
                println("Warning: Result $i has no data")
            else
                println("Warning: Result $i has insufficient data (length: $(length(res)), expected: 2+)")
            end
        end
    end
    
    if isempty(valid_results)
        println("No valid results with sufficient data")
        return nothing
    end
    
    # Extract concentration data with safe indexing
    a_vals = [res[1] isa Vector ? res[1][end] : (res[1] isa Tuple ? res[1][end] : res[1]) for (params, res) in valid_results]
    b_vals = [length(res) >= 2 ? (res[2] isa Vector ? res[2][end] : (res[2] isa Tuple ? res[2][end] : res[2])) : 0.0 for (params, res) in valid_results]
    c_vals = [length(res) >= 3 ? (res[3] isa Vector ? res[3][end] : (res[3] isa Tuple ? res[3][end] : res[3])) : 0.0 for (params, res) in valid_results]
    e1_vals = [length(res) >= 4 ? (res[4] isa Vector ? res[4][end] : (res[4] isa Tuple ? res[4][end] : res[4])) : 0.0 for (params, res) in valid_results]
    e2_vals = [length(res) >= 5 ? (res[5] isa Vector ? res[5][end] : (res[5] isa Tuple ? res[5][end] : res[5])) : 0.0 for (params, res) in valid_results]
    
    # Calculate product ratio
    product_ratio = [(b + c)/max(a, 1e-6) for (a,b,c) in zip(a_vals, b_vals, c_vals)]
    
    # Create 2x3 subplot layout
    p = plot(layout=(2,3), size=(1200,800),
             plot_title="Concentration Distributions")
    
    histogram!(p[1], a_vals, xlabel="[A] final", ylabel="Frequency", 
               title="Distribution of [A] Final Concentration", color=:steelblue)
    histogram!(p[2], b_vals, xlabel="[B] final", ylabel="Frequency", 
               title="Distribution of [B] Final Concentration", color=:orange)
    histogram!(p[3], c_vals, xlabel="[C] final", ylabel="Frequency", 
               title="Distribution of [C] Final Concentration", color=:green)
    histogram!(p[4], e1_vals, xlabel="[E1] final", ylabel="Frequency", 
               title="Distribution of [E1] Final Concentration", color=:red)
    histogram!(p[5], e2_vals, xlabel="[E2] final", ylabel="Frequency", 
               title="Distribution of [E2] Final Concentration", color=:purple)
    histogram!(p[6], product_ratio, xlabel="(B+C)/A ratio", ylabel="Frequency", 
               title="Distribution of Product Ratio", color=:brown)
    
    return p
end

"""
    plot_3d_parameter_space(results, param1_idx=1, param2_idx=2, param3_idx=3)

Create a 3D scatter plot showing the relationship between three parameters and A concentration.
"""
function plot_3d_parameter_space(results, param1_idx=1, param2_idx=2, param3_idx=3)
    if isempty(results)
        println("No results to plot")
        return nothing
    end
    
    # Validate data structure and handle different array lengths
    valid_results = []
    for (i, (params, res)) in enumerate(results)
        if res !== nothing && length(res) >= 1
            push!(valid_results, (params, res))
        else
            if res === nothing
                println("Warning: Result $i has no data")
            else
                println("Warning: Result $i has insufficient data (length: $(length(res)), expected: 1+)")
            end
        end
    end
    
    if isempty(valid_results)
        println("No valid results with sufficient data")
        return nothing
    end
    
    param_names = ["k1f", "k1r", "k2f", "k2r", "k3f", "k3r", "k4f", "k4r"]
    
    x_vals = [params[param1_idx] for (params, res) in valid_results]
    y_vals = [params[param2_idx] for (params, res) in valid_results]
    z_vals = [res[1] isa Vector ? res[1][end] : (res[1] isa Tuple ? res[1][end] : res[1]) for (params, res) in valid_results]  # A concentration
    
    p = scatter3d(x_vals, y_vals, z_vals, 
                  xlabel=param_names[param1_idx], 
                  ylabel=param_names[param2_idx], 
                  zlabel="[A] final",
                  title="3D Parameter Space: $(param_names[param1_idx]) vs $(param_names[param2_idx]) vs [A]",
                  markersize=2, color=:viridis)
    
    return p
end