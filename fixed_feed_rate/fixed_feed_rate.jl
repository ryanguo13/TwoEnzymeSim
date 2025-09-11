using Plots
using ProgressMeter

include("../src/simulation.jl")
include("../src/parameters.jl")
include("../src/analysis.jl")
include("../src/visualization.jl")

# Fixed feed rate for A (units consistent with our system, e.g., μM/s)
const FEED_RATE_A = 5.0
# Choose some different feed rates for A, but keep the rest of the parameters the same
FEED_RATE_A_range = 0.1:0.5:10.0


# Define a reaction network that adds a constant inflow of A: ∅ → A
# Copy the base network and add the inflow term
rn_feed = @reaction_network begin
    k1f,  A + E1 --> AE1
    k1r,  AE1 --> A + E1
    k2f,  AE1 --> B + E1
    k2r,  B + E1 --> AE1

    k3f,  B + E2 --> BE2
    k3r,  BE2 --> B + E2
    k4f,  BE2 --> C + E2q
    k4r,  C + E2 --> BE2

    k_in, ∅ --> A
end

# Merge base parameters with the feed-rate parameter
params_feed = Dict(params...)
params_feed[:k_in] = FEED_RATE_A

# Quick diagnostic: compare feed vs. initial consumption scale
A0   = first(filter(x -> x.first === A, initial_conditions)).second
E1_0 = first(filter(x -> x.first === E1, initial_conditions)).second
AE1_0 = first(filter(x -> x.first === AE1, initial_conditions)).second
init_consume_est = params_feed[:k1f] * A0 * E1_0 - params_feed[:k1r] * AE1_0
println("[diagnostic] k_in = ", params_feed[:k_in], ", initial net consumption estimate ≈ ", init_consume_est,
        " (positive means net drain). If k_in << consumption, A will drop fast.")


# Run the simulation with the feed network
ode_prob = ODEProblem(rn_feed, initial_conditions, tspan, params_feed)
sol = solve(ode_prob, Tsit5(), saveat=0.1)

@showprogress for i in 1:length(FEED_RATE_A_range)
    # Iterate over the feed rate range
        for FEED_RATE_A in FEED_RATE_A_range
            # Run the simulation
            sol = simulate_system(params, fixed_initial_conditions, tspan, saveat=0.1)
            # Plot the results
            plot(sol)
    
        end
end

# Compute thermodynamic metrics for plotting
thermo = calculate_thermo_fluxes(sol, params_feed)

# Print a brief summary
println("--------------------------------")
println("Fixed-feed simulation summary (A is supplied at a constant rate):")
println("feed rate k_in = $(params_feed[:k_in])")
println("Final concentrations:")
println("A: $(sol[A][end])")
println("B: $(sol[B][end])")
println("C: $(sol[C][end])")
println("E1: $(sol[E1][end])")
println("AE1: $(sol[AE1][end])")
println("E2: $(sol[E2][end])")
println("BE2: $(sol[BE2][end])")

# Plots
p1 = plot_all_concentrations(sol)
p2 = plot_fluxes_time(thermo["v1_thermo"], thermo["v2_thermo"], sol.t)
p3 = plot_dG_time(thermo["ΔG1"], thermo["ΔG2"], sol.t)
p4 = plot_R_time(thermo["R1"], thermo["R2"], sol.t)

# Save figures
savefig(p1, "all_concentrations_fixed_feed_rate.png")
savefig(p2, "fluxes_time_fixed_feed_rate.png")
savefig(p3, "dG_time_fixed_feed_rate.png")
savefig(p4, "R_time_fixed_feed_rate.png")
