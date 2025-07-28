using Plots
using ProgressMeter

include("../src/simulation.jl")
include("../src/parameters.jl")
include("../src/analysis.jl")
include("../src/visualization.jl")

# Set the progress meter
@showprogress for i in 1:100
    sleep(0.01)
end


# Run simulation
sol = simulate_system(params, initial_conditions, tspan)

# Calculate thermodynamic fluxes and related quantities
thermo = calculate_thermo_fluxes(sol, params)

# Show all the parameters loaded and the solution of the simulation
println("--------------------------------")
println("All parameters loaded:")
for (k, v) in params
    println("$k: $v")
end
println("--------------------------------")
println("Solution of the simulation:")
println("A: $(sol[A][end])")
println("B: $(sol[B][end])")
println("C: $(sol[C][end])")
println("E1: $(sol[E1][end])")
println("AE1: $(sol[AE1][end])")
println("E2: $(sol[E2][end])")
println("BE2: $(sol[BE2][end])")

# Show the result of the calculate_thermo_fluxes function
println("--------------------------------")
println("Result of the calculate_thermo_fluxes function:")
println("v1_thermo: $(thermo["v1_thermo"][end])")
println("v2_thermo: $(thermo["v2_thermo"][end])")
println("ΔG1: $(thermo["ΔG1"][end])")
println("ΔG2: $(thermo["ΔG2"][end])")
println("R1: $(thermo["R1"][end])")
println("R2: $(thermo["R2"][end])")
println("--------------------------------")
# Create 2D plots with GR backend
gr()
p1 = plot_all_concentrations(sol)
p2 = plot_fluxes_time(thermo["v1_thermo"], thermo["v2_thermo"], sol.t)
p3 = plot_dG_time(thermo["ΔG1"], thermo["ΔG2"], sol.t)
p4 = plot_R_time(thermo["R1"], thermo["R2"], sol.t)
# p5 = steady_state_thermo_fluxes(sol[A], sol[B], sol[C], sol[E1], params[:k1f], params[:k1r], params[:k2f], params[:k2r], params[:ΔG1], params[:R], params[:T])
# p7 = plot_thermo_fluxes(sol, params)
p8 = plot_derivative_thermo_fluxes(sol, params)
# Save all 2D plots
savefig(p1, "all_concentrations.png")
savefig(p2, "fluxes_time.png")
savefig(p3, "dG_time.png")
savefig(p4, "R_time.png")
# savefig(p5, "steady_state_thermo_fluxes.png")
# savefig(p7, "thermo_fluxes.png")
savefig(p8, "derivative_thermo_fluxes.png")

_, equilibrium_constants = analyze_equilibrium_constants(ΔG0_range)
p6 = plot_equilibrium_constants(ΔG0_range, equilibrium_constants)
savefig(p6, "equilibrium_constants.png")



# Also save all the plots to results_svg folder
mkpath("results_svg")
savefig(p1, "results_svg/all_concentrations.svg")
savefig(p2, "results_svg/fluxes_time.svg")
savefig(p3, "results_svg/dG_time.svg")
savefig(p4, "results_svg/R_time.svg")
savefig(p7, "results_svg/thermo_fluxes.svg")
savefig(p8, "results_svg/derivative_thermo_fluxes.svg")
savefig(p6, "results_svg/equilibrium_constants.svg")

println("\n\n")
println("Simulation complete")
