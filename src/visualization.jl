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
    p = plot(ΔG0_range, equilibrium_constants, linewidth=2,
             xlabel="ΔG⁰ (J/mol)", ylabel="Equilibrium Constant K_eq",
             title="Equilibrium Constants vs. Standard Gibbs Free Energy")
    
    return p
end

using Plots

# Function to plot concentration profiles
function plot_concentrations(sol)
    p = plot(sol, 
        xlabel="Time", 
        ylabel="Concentration",
        title="Concentration Profiles",
        label=["A" "B" "C"],
        linewidth=2
    )
    return p
end

# Function to plot reaction fluxes
function plot_fluxes(fluxes, sol)
    p = plot(sol.t, 
        [fluxes["A→B"] fluxes["B→A"] fluxes["B→C"] fluxes["C→B"] fluxes["A→C"] fluxes["C→A"]],
        xlabel="Time",
        ylabel="Flux",
        title="Reaction Fluxes",
        label=["A→B" "B→A" "B→C" "C→B" "A→C" "C→A"],
        linewidth=2
    )
    return p
end

# Function to plot phase portrait
function plot_phase_portrait(sol)
    # Set plotly as the backend for interactive 3D plots
    plotly()
    
    p = plot(sol[A], sol[B], sol[C],
        xlabel="[A]",
        ylabel="[B]",
        zlabel="[C]",
        title="Phase Portrait",
        label="Trajectory",
        linewidth=2,
        camera=(30, 30),  # Set initial camera angle
        legend=:topright
    )
    
    # Add markers at start and end points
    scatter!([sol[A][1]], [sol[B][1]], [sol[C][1]], 
             label="Start", markersize=8, color=:green)
    scatter!([sol[A][end]], [sol[B][end]], [sol[C][end]], 
             label="End", markersize=8, color=:red)
    
    return p
end 