
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

using Plots

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
    
    if !isempty(steady_state_points)
        scatter!(p, sol.t[steady_state_points .+ 1], zeros(length(steady_state_points)), 
                label="Steady State", markersize=8, color=:green, marker=:star)
    end
    # Also show the value of the dG_diss_1 and dG_diss_2 at the zero-crossing points using arrow with text using eachindex 
    # keep some distance between the text and the point
    # using the distance 100
    for i in eachindex(zero_crossings_1)
        annotate!(p, sol.t[zero_crossings_1[i] .+ 1], dG_diss_1[zero_crossings_1[i] .+ 1], 
                text("ΔG₁ = $(dG_diss_1[zero_crossings_1[i] .+ 1])", 10, :red)) # using the font size 10 and the color red
        # using quiver! to show the arrow
        quiver!(p, sol.t[zero_crossings_1[i] .+ 1], dG_diss_1[zero_crossings_1[i] .+ 1], 
                quiver=([0, 100], [0, 0]), color=:red)
    end
    for i in eachindex(zero_crossings_2)
        annotate!(p, sol.t[zero_crossings_2[i] .+ 1], dG_diss_2[zero_crossings_2[i] .+ 1], 
                text("ΔG₂ = $(dG_diss_2[zero_crossings_2[i] .+ 1])", 10, :blue))
        quiver!(p, sol.t[zero_crossings_2[i] .+ 1], dG_diss_2[zero_crossings_2[i] .+ 1], 
                quiver=([0, 100], [0, 0]), color=:blue)
    end
    
    return p
end
