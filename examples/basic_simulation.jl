using Plots
include("../src/simulation.jl")
include("../src/parameters.jl")
include("../src/analysis.jl")
include("../src/visualization.jl")

# Run simulation
sol = simulate_system(params, initial_conditions, tspan)

# Calculate fluxes
fluxes = calculate_fluxes(sol, params)

# Create 2D plots with GR backend
gr()
p1 = plot_concentrations(sol)
p2 = plot_fluxes(fluxes, sol)

# Create 3D plot with Plotly backend
plotly()
p3 = plot_phase_portrait(sol)

# Display and save 2D plots
plot(p1, p2, layout=(2,1), size=(800,800))
savefig("concentration_and_fluxes.png")

# Save 3D plot separately
savefig(p3, "phase_portrait.html")

# Create equilibrium constants plot
ΔG0_range = -40000:10:40000  # -40 kJ/mol 到 +40 kJ/mol
_, equilibrium_constants = analyze_equilibrium_constants(ΔG0_range)
plot_equilibrium_constants(ΔG0_range, equilibrium_constants)
savefig("equilibrium_constants.png")

# plot_parameter_sensitivity(params, initial_conditions, tspan)
# savefig("parameter_sensitivity.png")



println("Simulation complete")
