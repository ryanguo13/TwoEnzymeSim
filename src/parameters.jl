using Catalyst
using Unitful

# System parameters (using k1f, k1r, ... naming)
params = Dict(
    :k1f    => 2.0,      # A + E1 -> AE1
    :k1r    => 1.5,      # AE1 -> A + E1
    :k2f    => 1.8,      # AE1 -> B + E1
    :k2r    => 1.0,      # B + E1 -> AE1
    :k3f    => 1.2,      # B + E2 -> BE2
    :k3r    => 1.0,      # BE2 -> B + E2
    :k4f    => 1.6,      # BE2 -> C + E2
    :k4r    => 0.8,      # C + E2 -> BE2
)

# Initial conditions
initial_conditions = [
    A    => 5.0,
    B    => 0.0,
    C    => 0.0,
    E1   => 20.0,   # All enzyme free at t=0
    E2   => 15.0,
    AE1  => 0.0,
    BE2  => 0.0
]


ΔG0_range = -80000:10:40000  # -80 kJ/mol 到 +40 kJ/mol


# Time span
tspan = (0.0, 5.0) 

# === Unitful helpers for validation/tests (concentrations in μmol/L, time in s) ===
# These do not change the default numeric values above; they provide unitful variants
# for tests and users who want unit-checked simulations.

function params_with_units()
    # Bimolecular rate constants have units L/(μmol*s); unimolecular have 1/s
    return Dict(
        :k1f => params[:k1f] * u"L/(μmol*s)",
        :k1r => params[:k1r] * u"s^-1",
        :k2f => params[:k2f] * u"s^-1",
        :k2r => params[:k2r] * u"L/(μmol*s)",
        :k3f => params[:k3f] * u"L/(μmol*s)",
        :k3r => params[:k3r] * u"s^-1",
        :k4f => params[:k4f] * u"s^-1",
        :k4r => params[:k4r] * u"L/(μmol*s)"
    )
end

function initial_conditions_with_units()
    return [
        A    => initial_conditions[1].second * u"μmol/L",
        B    => initial_conditions[2].second * u"μmol/L",
        C    => initial_conditions[3].second * u"μmol/L",
        E1   => initial_conditions[4].second * u"μmol/L",
        E2   => initial_conditions[5].second * u"μmol/L",
        AE1  => initial_conditions[6].second * u"μmol/L",
        BE2  => initial_conditions[7].second * u"μmol/L"
    ]
end

function tspan_with_units()
    return (tspan[1] * u"s", tspan[2] * u"s")
end 