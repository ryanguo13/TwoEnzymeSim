using Catalyst
using DifferentialEquations

# Define parameters and variables
@parameters k1f k1r k3 k2f k2r k4 E1_tot E2_tot
@variables t A(t) B(t) C(t) AE1(t) BE2(t)

# Define the reaction network with enzyme conservation
rn = @reaction_network begin
    k1f*(E1_tot - AE1), A --> AE1
    k1r, AE1 --> A
    k3, AE1 --> B

    k2f*(E2_tot - BE2), B --> BE2
    k2r, BE2 --> B
    k4, BE2 --> C
end

"""
    simulate_system(params, initial_conditions, tspan; saveat=0.1)

Simulate the enzyme system and return the solution.

# Arguments
- `params`: Dictionary of parameter values
- `initial_conditions`: Dictionary of initial conditions (for A, B, C, AE1, BE2)
- `tspan`: Time span tuple (t_start, t_end)
- `saveat`: Time points to save the solution at (default: 0.1)
"""
function simulate_system(params, initial_conditions, tspan; saveat=0.1)
    # Create ODE problem
    ode_prob = ODEProblem(rn, initial_conditions, tspan, params)
    
    # Solve ODE
    sol = solve(ode_prob, Tsit5(), saveat=saveat)
    
    return sol
end

"""
    calculate_fluxes(sol, params)

Calculate reaction fluxes for the given solution and parameters.
"""
function calculate_fluxes(sol, params)
    # Convert params vector to dictionary for easier access
    p_dict = Dict(first(p) => last(p) for p in params)
    
    # Calculate fluxes at each time point
    fluxes = Dict(
        "A+E1→AE1" => p_dict[k1f] .* (p_dict[E1_tot] .- sol[AE1]) .* sol[A],
        "AE1→A" => p_dict[k1r] .* sol[AE1],
        "AE1→B" => p_dict[k3] .* sol[AE1],
        "B+E2→BE2" => p_dict[k2f] .* (p_dict[E2_tot] .- sol[BE2]) .* sol[B],
        "BE2→B" => p_dict[k2r] .* sol[BE2],
        "BE2→C" => p_dict[k4] .* sol[BE2]
    )
    
    return fluxes
end 