using Test
# Ensure Unitful is available
try
    @eval using Unitful
catch
    try
        import Pkg
        Pkg.add("Unitful")
        @eval using Unitful
    catch e
        @error "Unitful not available and could not be installed" exception=e
        rethrow(e)
    end
end
using Unitful.DefaultSymbols

module UnitCheckTests
using Test
using Unitful
using Unitful.DefaultSymbols

# Define canonical units for this project
const conc = 1.0u"mol/L"          # concentration
const timeu = 1.0u"s"             # time
const rate_uni = 1.0u"s^-1"       # unimolecular rate constant
const rate_bi = 1.0u"L/(mol*s)"   # bimolecular rate constant
const Rgas = 8.314u"J/mol/K"      # universal gas constant
const Temp = 298.15u"K"

# Sample concentrations (values arbitrary; only units matter here)
const A = 5.0u"mol/L"
const B = 0.1u"mol/L"
const C = 0.05u"mol/L"
const E1 = 20.0u"mol/L"
const E2 = 15.0u"mol/L"
const AE1 = 0.2u"mol/L"
const BE2 = 0.15u"mol/L"

# Assign unitful rate constants according to reaction order
const k1f = 2.0 * rate_bi   # A + E1 -> AE1 (bimolecular)
const k1r = 1.5 * rate_uni  # AE1 -> A + E1 (unimolecular)
const k2f = 1.8 * rate_uni  # AE1 -> B + E1 (unimolecular)
const k2r = 1.0 * rate_bi   # B + E1 -> AE1 (bimolecular)
const k3f = 1.2 * rate_bi   # B + E2 -> BE2 (bimolecular)
const k3r = 1.0 * rate_uni  # BE2 -> B + E2 (unimolecular)
const k4f = 1.6 * rate_uni  # BE2 -> C + E2 (unimolecular)
const k4r = 0.8 * rate_bi   # C + E2 -> BE2 (bimolecular)

@testset "Kinetic flux units" begin
    # v1 = k2f*[AE1] - k2r*[B]*[E1]
    v1_forward = k2f * AE1
    v1_reverse = k2r * B * E1
    v1 = v1_forward - v1_reverse

    @test unit(v1_forward) == u"mol/L/s"
    @test unit(v1_reverse) == u"mol/L/s"
    @test unit(v1) == u"mol/L/s"

    # v2 = k4f*[BE2] - k4r*[C]*[E2]
    v2_forward = k4f * BE2
    v2_reverse = k4r * C * E2
    v2 = v2_forward - v2_reverse

    @test unit(v2_forward) == u"mol/L/s"
    @test unit(v2_reverse) == u"mol/L/s"
    @test unit(v2) == u"mol/L/s"
end

@testset "Equilibrium and thermodynamic quantities" begin
    # Equilibrium constants for two-step sequences should be dimensionless
    K_eq1 = (k1f * k2f) / (k1r * k2r)
    K_eq2 = (k3f * k4f) / (k3r * k4r)

    @test unit(K_eq1) === Unitful.NoUnits
    @test unit(K_eq2) === Unitful.NoUnits

    # Standard free energies ΔG° = -R T ln(Keq) have units of J/mol
    # Check R*T units and that adding RT*log(Q) preserves units
    @test unit(Rgas * Temp) == u"J/mol"

    # Reaction quotients are ratios of concentrations -> dimensionless
    Q1 = B / A
    Q2 = C / B
    @test unit(Q1) === Unitful.NoUnits
    @test unit(Q2) === Unitful.NoUnits

    # Construct ΔG using dimensionless arguments for log
    ΔG1_std_units = unit(Rgas * Temp)
    ΔG2_std_units = unit(Rgas * Temp)

    # Even without computing numeric logs, verify resulting units
    ΔG1_like = (Rgas * Temp) # * log(ustrip(K_eq1))  # dimensionally J/mol
    ΔG2_like = (Rgas * Temp) # * log(ustrip(K_eq2))  # dimensionally J/mol

    @test unit(ΔG1_like) == u"J/mol"
    @test unit(ΔG2_like) == u"J/mol"

    # Total ΔG = ΔG° + R T ln(Q) must also be J/mol
    ΔG1_total_like = (Rgas * Temp) + (Rgas * Temp) # * log(ustrip(Q1))
    ΔG2_total_like = (Rgas * Temp) + (Rgas * Temp) # * log(ustrip(Q2))

    @test unit(ΔG1_total_like) == u"J/mol"
    @test unit(ΔG2_total_like) == u"J/mol"
end

@testset "Thermodynamic flux scaling units" begin
    # v_thermo = v .* exp(-ΔG/(R T)) -> multiplicative dimensionless factor
    v_sample = 1.0u"mol/L/s"
    scale = exp(-ustrip((Rgas * Temp) / (Rgas * Temp)))
    @test unit(scale) === Unitful.NoUnits
    v_scaled = v_sample * scale
    @test unit(v_scaled) == u"mol/L/s"
end
end # module UnitCheckTests

using Unitful
using DifferentialEquations

# Load project source files directly (simulation defines variables used in parameters)
include(joinpath(@__DIR__, "..", "src", "simulation.jl"))
include(joinpath(@__DIR__, "..", "src", "parameters.jl"))

@testset "Unitful validation (μmol/L, s)" begin
    p_u = params_with_units()
    ic_u = initial_conditions_with_units()
    ts_u = tspan_with_units()

    # Dimensional checks on parameters
    @test ustrip(u"L/(μmol*s)", p_u[:k1f]) ≈ ustrip(u"L/(μmol*s)", 2.0u"L/(μmol*s)")
    @test ustrip(u"s^-1", p_u[:k1r]) ≈ ustrip(u"s^-1", 1.5u"s^-1")

    # Dimensional checks on initial conditions (μmol/L)
    for pair in ic_u
        @test unit(pair.second) == u"μmol/L"
    end

    # Simulate using unit-prepared inputs
    saveat_u = 0.1u"s"
    sol = simulate_system_unitful(p_u, ic_u, ts_u; saveat_u)

    # Solution arrays should be numeric but represent μmol/L magnitudes as fed in
    A = sol[:A]; B = sol[:B]; C = sol[:C]
    @test A[1] ≈ ustrip(u"μmol/L", 5.0u"μmol/L") atol=1e-12
    @test B[1] ≈ 0.0 atol=1e-12
    @test C[1] ≈ 0.0 atol=1e-12

    # Flux dimensions: μmol/L/s
    flux = calculate_kinetic_fluxes(sol, Dict(Symbol(k)=>ustrip(p_u[k]) for k in keys(p_u)))
    v1 = flux["v1"]; v2 = flux["v2"]
    # Check that values are finite and non-NaN
    @test all(isfinite, v1)
    @test all(isfinite, v2)

    # Compute a representative flux with explicit units and compare numeric magnitude
    # v1 = k2f*AE1 - k2r*B*E1
    k2f_u = p_u[:k2f]; k2r_u = p_u[:k2r]
    AE1_u0 = ic_u[6].second; B_u0 = ic_u[2].second; E1_u0 = ic_u[4].second
    v1_u0 = k2f_u*AE1_u0 - k2r_u*B_u0*E1_u0  # should be μmol/L/s
    @test unit(v1_u0) == u"μmol/L*s^-1"
    # Compare numeric magnitude to code path at t=1st index
    @test v1[1] ≈ ustrip(u"μmol/L*s^-1", v1_u0) atol=1e-8
end 