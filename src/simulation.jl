using Catalyst
using DifferentialEquations


# Define parameters and variables (using k1f, k1r, ... naming)
@parameters k1f k1r k2f k2r k3f k3r k4f k4r
@variables t A(t) B(t) C(t) E1(t) E2(t) AE1(t) BE2(t)

# Define the reaction network with all reversible steps and explicit enzyme species
rn = @reaction_network begin
    k1f,  A + E1 --> AE1
    k1r,  AE1 --> A + E1
    k2f,  AE1 --> B + E1
    k2r,  B + E1 --> AE1

    k3f,  B + E2 --> BE2
    k3r,  BE2 --> B + E2
    k4f,  BE2 --> C + E2
    k4r,  C + E2 --> BE2
end

"""
    simulate_system(params, initial_conditions, tspan; saveat=0.1)

Simulate the enzyme system and return the solution.

# Arguments
- `params`: Dictionary or NamedTuple of parameter values (k1f, k1r, k2f, k2r, k3f, k3r, k4f, k4r)
- `initial_conditions`: Dictionary or NamedTuple of initial conditions (A, B, C, E1, E2, AE1, BE2)
- `tspan`: Time span tuple (t_start, t_end)
- `saveat`: Time points to save the solution at (default: 0.1)
"""
function simulate_system(params, initial_conditions, tspan; saveat=0.1)
    ode_prob = ODEProblem(rn, initial_conditions, tspan, params)
    sol = solve(ode_prob, Tsit5(), saveat=saveat)
    return sol
end

"""
    calculate_kinetic_fluxes(sol, params)

Calculate the kinetic fluxes v1 and v2 at each time point, strictly following the LaTeX equations.

Returns a Dict with keys "v1" and "v2" (arrays over time).

Note: Not the case like the LaTeX equuations since it calculates the transisant state.
"""
function calculate_kinetic_fluxes(sol, params)
    p = params isa Dict ? params : Dict(Symbol(k)=>v for (k,v) in pairs(params))
    A = sol[Symbol("A")]
    B = sol[Symbol("B")]
    C = sol[Symbol("C")]
    E1 = sol[Symbol("E1")]
    E2 = sol[Symbol("E2")]
    AE1 = sol[Symbol("AE1")]
    BE2 = sol[Symbol("BE2")]

    # v1 = k2f*[AE1] - k2r*[B]*[E1]
    v1 = p[:k2f] .* AE1 .- p[:k2r] .* B .* E1
    # v2 = k4f*[BE2] - k4r*[C]*[E2]
    v2 = p[:k4f] .* BE2 .- p[:k4r] .* C .* E2

    return Dict("v1" => v1, "v2" => v2)
end

function calculate_thermo_fluxes(sol, params; T=298.15)
    R = 8.314
    p = params isa Dict ? params : Dict(Symbol(k)=>v for (k,v) in pairs(params))
    k1f, k1r = p[:k1f], p[:k1r]
    k2f, k2r = p[:k2f], p[:k2r]
    k3f, k3r = p[:k3f], p[:k3r]
    k4f, k4r = p[:k4f], p[:k4r]

    A = sol[Symbol("A")]
    B = sol[Symbol("B")]
    C = sol[Symbol("C")]
    E1 = sol[Symbol("E1")]
    E2 = sol[Symbol("E2")]
    AE1 = sol[Symbol("AE1")]
    BE2 = sol[Symbol("BE2")]

    # A + E1 <=> AE1 <=> B + E1
    K_eq1 = (k1f * k2f) / (k1r * k2r)
    ΔG1_std = -R * T * log(K_eq1)
    Q1 = (B .* E1) ./ (A .* E1 .+ 1e-12)
    ΔG1 = ΔG1_std .+ R * T * log.(Q1)

    # B + E2 <=> BE2 <=> C + E2
    K_eq2 = (k3f * k4f) / (k3r * k4r)
    ΔG2_std = -R * T * log(K_eq2)
    Q2 = (C .* E2) ./ (B .* E2 .+ 1e-12)
    ΔG2 = ΔG2_std .+ R * T * log.(Q2)

    v1_thermo = k2f * AE1 .- k2r * B .* E1
    v2_thermo = k4f * BE2 .- k4r * C .* E2

    # Forward/reverse flux ratios
    J1_plus = k2f * AE1
    J1_minus = k2r * B .* E1
    R1 = J1_plus ./ (J1_minus .+ 1e-12)
    J2_plus = k4f * BE2
    J2_minus = k4r * C .* E2
    R2 = J2_plus ./ (J2_minus .+ 1e-12)

    return Dict(
        "K_eq1" => K_eq1,
        "ΔG1_std" => ΔG1_std,
        "Q1" => Q1,
        "ΔG1" => ΔG1,
        "v1_thermo" => v1_thermo,
        "K_eq2" => K_eq2,
        "ΔG2_std" => ΔG2_std,
        "Q2" => Q2,
        "ΔG2" => ΔG2,
        "v2_thermo" => v2_thermo,
        "R1" => R1,
        "R2" => R2
    )
end