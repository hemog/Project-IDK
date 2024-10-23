import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

k0 = 8.9875517873681764e9
e = 1.602176634e-19 
m_e = 9.10938356e-31 
c = 299792458 

m_alpha = 6.644657230e-27
z_alpha = 2


def mean_excitation_energy_element(Z):
    if Z == 1:
        return 19.0
    elif 2 <= Z <= 13:
        return 11.2 + 11.7 * Z
    else:
        return 52.8 + 8.71 * Z


def mean_excitation_energy_mixture(elements, fractions):
    nlnI_sum = 0
    total_NZ = 0
    for i in range(len(elements)):
        Z_i = elements[i]
        N_i = fractions[i]
        I_i = mean_excitation_energy_element(Z_i)
        nlnI_sum += N_i * Z_i * np.log(I_i)
    
    total_NZ = sum([N_i * Z_i for N_i, Z_i in zip(fractions, elements)])
    I_compound = np.exp(nlnI_sum / total_NZ)
    return I_compound


def get_beta(E_value, mass):
    gamma = 1 + (E_value * 10**6 * e) / (mass * c**2)
    beta = np.sqrt(1 - (1/gamma**2))
    return beta

def stopping_power(z, n, beta, I_ev):
    coefficient = 5.08e-31
    beta_square = beta**2
    F_beta = np.log((1.02e6 * beta_square) / (1 - beta_square)) - beta_square
    dE_dx = (coefficient * z**2 * n / beta_square) * (F_beta - np.log(I_ev))
    return dE_dx

I_water = mean_excitation_energy_mixture([1, 8], [2, 1])
I_hydrogen = mean_excitation_energy_element(1)
n_water = 3.34e28
n_hydrogen = 5.40596e28

E_alpha_values = np.logspace(np.log10(0.2), np.log10(400), 100)


stopping_powers_water_alpha = [stopping_power(z_alpha, n_water, get_beta(E, m_alpha), I_water) for E in E_alpha_values]
stopping_powers_hydrogen_alpha = [stopping_power(z_alpha, n_hydrogen, get_beta(E, m_alpha), I_hydrogen) for E in E_alpha_values]


plt.figure(figsize=(10, 6))
plt.loglog(E_alpha_values, stopping_powers_water_alpha, label='Alpha in Water', color='blue', linewidth=2)
plt.loglog(E_alpha_values, stopping_powers_hydrogen_alpha, label='Alpha in Hydrogen', color='red', linewidth=2)


plt.xlabel('Energy (MeV)', fontsize=14)
plt.ylabel('Stopping Power (MeV cm$^{-1}$)', fontsize=14)
plt.title('Stopping Power of Alpha Particles in Water and Hydrogen', fontsize=16)
plt.legend()


plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

m_proton = 1.67262192369e-27
z_proton = 1


E_proton_values = np.logspace(np.log10(0.01), np.log10(1000), 100)


stopping_powers_water_proton = [stopping_power(z_proton, n_water, get_beta(E, m_proton), I_water) for E in E_proton_values]
stopping_powers_hydrogen_proton = [stopping_power(z_proton, n_hydrogen, get_beta(E, m_proton), I_hydrogen) for E in E_proton_values]


plt.figure(figsize=(10, 6))
plt.loglog(E_proton_values, stopping_powers_water_proton, label='Proton in Water', color='green', linewidth=2)
plt.loglog(E_proton_values, stopping_powers_hydrogen_proton, label='Proton in Hydrogen', color='orange', linewidth=2)

plt.xlabel('Energy (MeV)', fontsize=14)
plt.ylabel('Stopping Power (MeV cm$^{-1}$)', fontsize=14)
plt.title('Stopping Power of Protons in Water and Hydrogen', fontsize=16)
plt.legend()

plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
