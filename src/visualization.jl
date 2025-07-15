"""
    plot_results(sol, v1_time, v2_time, steady_sol)

Plot the simulation results including concentrations, fluxes, and phase space.
"""
function plot_results(sol, v1_time, v2_time, steady_sol)
    # Plot concentrations over time
    p1 = plot(sol.t, sol[A], label="[A]", linewidth=2, 
              xlabel="Time (s)", ylabel="Concentration (M)", 
              title="Chemical Species Concentration")
    plot!(p1, sol.t, sol[B], label="[B]", linewidth=2)
    plot!(p1, sol.t, sol[C], label="[C]", linewidth=2)

    # Plot reaction fluxes over time
    p2 = plot(sol.t, v1_time, label="v₁ (A→B)", linewidth=2,
              xlabel="Time (s)", ylabel="Reaction Flux (M/s)",
              title="Reaction Fluxes")
    plot!(p2, sol.t, v2_time, label="v₂ (B→C)", linewidth=2)

    # Plot phase space
    p3 = plot(sol[A], sol[B], label="A-B Phase Space", linewidth=2,
              xlabel="[A] (M)", ylabel="[B] (M)",
              title="System Phase Space")
    plot!(p3, sol[B], sol[C], label="B-C Phase Space", linewidth=2)
    
    # Mark steady state point
    scatter!(p3, [steady_sol[A]], [steady_sol[B]], 
             markersize=8, color=:red, label="Steady State")

    return p1, p2, p3
end

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
             yscale=:log10)
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

# Function to plot phase portrait
# function plot_phase_portrait(sol)
#     plotly()
#     p = plot(sol[A], sol[B], sol[C],
#         xlabel="[A]",
#         ylabel="[B]",
#         zlabel="[C]",
#         title="Phase Portrait",
#         label="Trajectory",
#         linewidth=2,
#         camera=(30, 30),
#         legend=:topright
#     )
#     scatter!([sol[A][1]], [sol[B][1]], [sol[C][1]], 
#              label="Start", markersize=4, color=:green)
#     scatter!([sol[A][end]], [sol[B][end]], [sol[C][end]], 
#              label="End", markersize=4, color=:red)
#     return p
# end 

function steady_state_thermo_fluxes(A, B, C, E_tot, k1f, k1r, k2f, k2r, ΔG1_std, R, T)
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
    # 过滤掉非正数，避免 log10 报错
    R1_plot = max.(R1, 1e-12)
    R2_plot = max.(R2, 1e-12)
    p = plot(t, [R1_plot R2_plot], xlabel="Time", ylabel="J⁺/J⁻",
        title="Forward/Reverse Flux Ratio", label=["R₁" "R₂"], linewidth=2, yscale=:log10)
    return p
end 