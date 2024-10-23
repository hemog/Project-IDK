import numpy as np
import matplotlib.pyplot as plt

# Constants
e_charge = 1.602e-19  # Coulombs
m_e = 9.109e-31  # kg (electron mass)
c = 2.98e8  # speed of light in m/s
m_proton = 1.673e-27  # kg (proton mass)
m_alpha = 4 * m_proton  # kg (alpha particle mass)
n_water = 3.34e28  # 1/m^3 (number density of water molecules)

# Energy ranges for alpha particles and protons (in MeV)
E_alpha = np.logspace(np.log10(0.2), np.log10(400), 100)  # 100 energy points from 0.2 to 400 MeV
E_proton = np.logspace(np.log10(0.01), np.log10(1000), 100)  # 100 energy points from 0.01 to 1000 MeV

# Mean excitation energy function
def mean_excitation_energy(Z):
    if Z == 1:
        I = 19.0 * e_charge  # For hydrogen (Z=1)
    elif 2 <= Z <= 13:
        I = (11.2 + 11.7 * Z) * e_charge  # For elements with Z from 2 to 13
    else:
        I = (52.8 + 8.71 * Z) * e_charge  # For heavier elements
    return I

# Stopping Power function (MeV/cm)
def stopping_power(Z, n, beta, I):
    return (5.08e-31 * Z**2 * n / beta**2) * (np.log(2 * m_e * c**2 * beta**2 / I) - beta**2)


stopping_water_alpha = np.full(len(E_alpha), np.nan)
stopping_water_proton = np.full(len(E_proton), np.nan)


for i in range(4, len(E_alpha)):
    energy = E_alpha[i]
    gamma = 1 + (energy * 1e6 * e_charge / (m_alpha * c**2))  # Relativistic gamma
    beta = np.sqrt(1 - (1 / gamma**2))
    stopping_power_value = stopping_power(2, n_water, beta, mean_excitation_energy(8))  # Z=2 (alpha), Z=8 (water)
    
    if stopping_power_value > 0:
        stopping_water_alpha[i] = stopping_power_value

# Stopping Power for Protons
for i in range(len(E_proton)):
    energy = E_proton[i]
    gamma = 1 + (energy * 1e6 * e_charge / (m_proton * c**2))  # Relativistic gamma
    beta = np.sqrt(1 - (1 / gamma**2))  # Relativistic beta
    stopping_power_value = stopping_power(1, n_water, beta, mean_excitation_energy(1))  # Z=1 (proton), Z=1 (water)
    
    if stopping_power_value > 0:
        stopping_water_proton[i] = stopping_power_value

# Plotting
plt.figure()
plt.loglog(E_alpha, stopping_water_alpha, 'b', label='Alpha Particles (Water)')
plt.loglog(E_proton, stopping_water_proton, 'r', label='Protons (Water)')
plt.xlabel('Energy (MeV)')
plt.ylabel('Stopping Power (MeV/cm)')
plt.title('Stopping Power of Alpha Particles and Protons (Water)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
