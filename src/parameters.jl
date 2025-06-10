using Catalyst

# System parameters
params = [
    k_f => 2.0,      # Forward rate constant (positive)
    k_yr => 1.5,     # Reverse rate constant (positive)  
    k_ry => 1.8,     # B→C rate constant
    k_r => 0.5,      # Enzyme release rate constant
    k_y => 0.3,      # Enzyme binding rate constant
    k_rf => 0.4,     # Other rate constant
    E_tot => 10.0,   # Total enzyme concentration
    ΔG0 => -2500.0,  # Standard Gibbs free energy change (negative)
    R => 8.314,      # Gas constant
    T => 298.0       # Temperature (K)
]

# Initial conditions
initial_conditions = [
    A => 5.0,
    B => 2.0, 
    C => 1.0
]

# Time span
tspan = (0.0, 50.0) 