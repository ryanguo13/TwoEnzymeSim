using Catalyst

# System parameters
params = [
    k1f => 2.0,      # A + E1 -> AE1
    k1r => 1.5,      # AE1 -> A
    k3  => 1.8,      # AE1 -> B
    k2f => 1.2,      # B + E2 -> BE2
    k2r => 1.0,      # BE2 -> B
    k4  => 1.6,      # BE2 -> C
    E1_tot => 20.0,  # Total E1
    E2_tot => 15.0   # Total E2
]

# Initial conditions
initial_conditions = [
    A => 5.0,
    B => 2.0,
    C => 1.0,
    AE1 => 0.0,
    BE2 => 0.0
]

# Time span
tspan = (0.0, 5.0) 