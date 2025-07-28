using Catalyst

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