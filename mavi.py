import numpy as np
import matplotlib.pyplot as plt

# Constants
k0 = 8.9875517873681764e9  # Coulomb's constant
e = 1.602176634e-19  # Electron charge
m = 9.1093837015e-31  # Electron mass
c = 299792458  # Speed of light
N_A = 6.02214076e23  # Avogadro's number

# Functions
def mean_excitation_energy(Z):
    if Z == 1:
        return 19.0
    elif 2 <= Z <= 13:
        return 11.2 + 11.7 * Z
    else:
        return 52.8 + 8.71 * Z

def stopping_power(Z, E, n):
    beta = np.sqrt(1 - (m * c**2 / E)**2)
    I = mean_excitation_energy(Z)
    return (4 * np.pi * k0**2 * Z**2 * e**4 * n) / (m * c**2 * beta**2) * (np.log(2 * m * c**2 * beta**2 / I * (1 - beta**2)) - beta**2)

def range_calculation(Z, E0, n, delta_x):
    E = E0
    x = 0
    while E > 0:
        dE = stopping_power(Z, E, n) * delta_x
        E -= dE
        x += delta_x
    return x

# Particle parameters
alpha_mass = 6.64465723e-27  # Alpha particle mass
proton_mass = 1.67262192e-27  # Proton mass

# Medium parameters
water_density = 1000  # kg/m^3
hydrogen_density = 0.0899  # kg/m^3
water_N = water_density * N_A / 18.01528  # Number of water molecules per m^3
hydrogen_N = hydrogen_density * N_A / 2.01588  # Number of hydrogen atoms per m^3

# Energy ranges
alpha_energies = np.logspace(-1, 3, 1000)  # 0.1 MeV to 1000 MeV
proton_energies = np.logspace(-2, 3, 1000)  # 0.01 MeV to 1000 MeV

# Stopping power calculations
alpha_water_stopping_power = stopping_power(2, alpha_energies, water_N)
alpha_hydrogen_stopping_power = stopping_power(1, alpha_energies, hydrogen_N)
proton_water_stopping_power = stopping_power(1, proton_energies, water_N)
proton_hydrogen_stopping_power = stopping_power(1, proton_energies, hydrogen_N)

# Range calculations
alpha_water_ranges = [range_calculation(2, E, water_N, 1e-6) for E in [0.5, 5, 50]]
alpha_hydrogen_ranges = [range_calculation(2, E, hydrogen_N, 1e-6) for E in [0.5, 5, 50]]
proton_water_ranges = [range_calculation(1, E, water_N, 1e-6) for E in [0.5, 5, 50]]
proton_hydrogen_ranges = [range_calculation(1, E, hydrogen_N, 1e-6) for E in [0.5, 5, 50]]

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.loglog(alpha_energies, alpha_water_stopping_power, label="Alpha in water")
plt.loglog(alpha_energies, alpha_hydrogen_stopping_power, label="Alpha in hydrogen")
plt.xlabel("Energy (MeV)")
plt.ylabel("Stopping power (MeV/cm)")
plt.legend()
plt.title("Stopping power of alpha particles")

plt.subplot(1, 2, 2)
plt.loglog(proton_energies, proton_water_stopping_power, label="Proton in water")
plt.loglog(proton_energies, proton_hydrogen_stopping_power, label="Proton in hydrogen")
plt.xlabel("Energy (MeV)")
plt.ylabel("Stopping power (MeV/cm)")
plt.legend()
plt.title("Stopping power of protons")

plt.tight_layout()
plt.show()

print("Alpha particle ranges in water:", alpha_water_ranges)
print("Alpha particle ranges in hydrogen:", alpha_hydrogen_ranges)
print("Proton ranges in water:", proton_water_ranges)
print("Proton ranges in hydrogen:", proton_hydrogen_ranges)