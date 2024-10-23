import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Constants
k0 = 8.9875517873681764e9
e = 1.602176634e-19
m_e = 9.10938356e-31
c = 299792458
m_a = 3727.379

def get_beta(E_value):

    df = pd.DataFrame({
        'E': [0.05, 0.08, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00, 2.00, 4.00, 6.00, 8.00, 10.00, 20.00, 40.00, 60.00, 80.00, 100.00],
        'beta': [127e-6, 171e-6, 213e-6, 426e-6, 852e-6, 1278e-6, 1703e-6, 2129e-6, 4252e-6, 8476e-6, 0.01267, 0.01685, 0.02099, 0.04133, 0.08014, 0.1166, 0.1510, 0.1834]
    })

    beta_value = df[df['E'] == E_value]['beta'].values[0]
    
    return beta_value

"""def get_beta(E_value):
    beta = np.sqrt(2*E_value*m_a**(-1))*c**(-1)
    return beta"""


# Mean excitation energy for an element
def mean_excitation_energy_element(Z):

    if Z == 1:
        return 19.0
    elif 2 <= Z <= 13:
        return 11.2 + 11.7 * Z
    else:
        return 52.8 + 8.71 * Z

# Mean excitation energy for a compound or mixture
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

# Bethe formula for stopping power
"""def stopping_power(z, n, beta, I):
    
    term1 = (4 * np.pi * k0**2 * z**2 * e**4 * n) / (m_e * c**2 * beta**2)
    term2 = np.log((2 * m_e * c**2 * beta**2) / (I * (1 - beta**2))) - beta**2
    stopping_power_in_mev = (term1 * term2)
    return stopping_power_in_mev/1.602176634e-19"""

def stopping_power(z, n, beta, I_ev):
    # Constants
    coefficient = 5.08e-31  # constant factor in MeV cm^-1
    beta_square = beta**2

    # F(beta) calculation
    F_beta = np.log((1.02e6 * beta_square) / (1 - beta_square)) - beta_square

    # Energy loss per unit distance formula (in MeV cm^-1)
    dE_dx = (coefficient * z**2 * n / beta_square) * (F_beta - np.log(I_ev))
    
    return dE_dx

I_hydrogen = mean_excitation_energy_element(1)
I_water = mean_excitation_energy_mixture([1,8],[2,1])
n_water = 3.34e28
n_hydrogen = 5.40596*10**22 #m^(-3)

"""stopping_power_water = stopping_power(2, n_water, get_beta(10), I_water)
stopping_power_hydrogen = stopping_power(2, n_hydrogen, get_beta(0.80), I_hydrogen)"""

E_values = np.array([0.05, 0.08, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00, 2.00, 4.00, 6.00, 8.00, 10.00, 20.00, 40.00, 60.00, 80.00, 100.00])

# Stopping power for each energy value for water
stopping_power_values_water = [stopping_power(2, n_water, get_beta(E), I_water) for E in E_values]

# Stopping power for each energy value for hydrogen
stopping_power_values_hydrogen = [stopping_power(2, n_hydrogen, get_beta(E), I_hydrogen) for E in E_values]

print(stopping_power_values_water)
