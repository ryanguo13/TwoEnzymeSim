using Plots
include("../src/simulation.jl")
include("../src/parameters.jl")
include("../src/analysis.jl")
include("../src/visualization.jl")

# Run simulation
sol = simulate_system(params, initial_conditions, tspan)

# Calculate thermodynamic fluxes and related quantities
thermo = calculate_thermo_fluxes(sol, params)

# Create 2D plots with GR backend
gr()
p1 = plot_all_concentrations(sol)
p2 = plot_fluxes_time(thermo["v1_thermo"], thermo["v2_thermo"], sol.t)
p3 = plot_dG_time(thermo["ΔG1"], thermo["ΔG2"], sol.t)
p4 = plot_R_time(thermo["R1"], thermo["R2"], sol.t)

# Save all 2D plots
savefig(p1, "all_concentrations.png")
savefig(p2, "fluxes_time.png")
savefig(p3, "dG_time.png")
savefig(p4, "R_time.png")

# Create 3D plot with Plotly backend
# plotly()
# p5 = plot_phase_portrait(sol)
# savefig(p5, "phase_portrait.html")

# Create equilibrium constants plot
ΔG0_range = -40000:10:40000  # -40 kJ/mol 到 +40 kJ/mol
_, equilibrium_constants = analyze_equilibrium_constants(ΔG0_range)
p6 = plot_equilibrium_constants(ΔG0_range, equilibrium_constants)
savefig(p6, "equilibrium_constants.png")

println("Simulation complete")
