using Catalyst
using DifferentialEquations


# Define parameters and variables
@parameters k_f k_yr k_ry k_r k_y k_rf E_tot ΔG0 R T
@variables t A(t) B(t) C(t)

# Define the reaction network
rn = @reaction_network begin
    # Forward reaction
    k_f, A --> B
    # Reverse reaction
    k_yr, B --> A
    # Additional reactions
    k_ry, B --> C
    k_r, C --> B
    k_y, A --> C
    k_rf, C --> A
end

"""
    simulate_system(params, initial_conditions, tspan; saveat=0.1)

Simulate the enzyme system and return the solution.

# Arguments
- `params`: Dictionary of parameter values
- `initial_conditions`: Dictionary of initial conditions
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
        "A→B" => p_dict[k_f] .* sol[A],
        "B→A" => p_dict[k_yr] .* sol[B],
        "B→C" => p_dict[k_ry] .* sol[B],
        "C→B" => p_dict[k_r] .* sol[C],
        "A→C" => p_dict[k_y] .* sol[A],
        "C→A" => p_dict[k_rf] .* sol[C]
    )
    
    return fluxes
end 