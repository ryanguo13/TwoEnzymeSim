"""
    analyze_steady_state()

Analyze the steady state of the system and return the steady state solution.
"""
function analyze_steady_state()
    # Create ODE problem
    ode_prob = ODEProblem(reaction_network, initial_conditions, tspan, params)
    
    # Convert to steady state problem
    steady_prob = SteadyStateProblem(ode_prob)
    
    # Solve steady state
    steady_sol = solve(steady_prob, DynamicSS(Tsit5()))
    
    return steady_sol
end

"""
    parameter_sensitivity(param_name, range)

Perform parameter sensitivity analysis for the given parameter.
"""
function parameter_sensitivity(param_name, range)
    steady_vals = Dict{Symbol, Vector{Float64}}(
        :A => Float64[],
        :B => Float64[],
        :C => Float64[]
    )
    
    for val in range
        params_temp = [p for p in params if p.first != param_name]
        push!(params_temp, param_name => val)
        
        prob_temp = SteadyStateProblem(reaction_network, initial_conditions, params_temp)
        sol_temp = solve(prob_temp, DynamicSS(Tsit5()))
        
        push!(steady_vals[:A], sol_temp[A])
        push!(steady_vals[:B], sol_temp[B])
        push!(steady_vals[:C], sol_temp[C])
    end
    
    return range, steady_vals
end

"""
    analyze_equilibrium_constants(ΔG0_range)

Analyze equilibrium constants for different ΔG0 values.
"""
function analyze_equilibrium_constants(ΔG0_range)
    equilibrium_constants = Float64[]
    
    for ΔG0_val in ΔG0_range
        R_val = 8.314
        T_val = 298.0
        K_eq = exp(-ΔG0_val/(R_val * T_val))
        push!(equilibrium_constants, K_eq)
    end
    
    return ΔG0_range, equilibrium_constants
end 