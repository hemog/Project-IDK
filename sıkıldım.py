import numpy as np
import matplotlib.pyplot as plt

# Temel sabitler
e = 1.602176634e-19  # C
c = 299792458  # m/s
m_e = 9.10938356e-31  # kg (elektron kütlesi)

# Parçacık özellikleri
alpha_mass = 6.644657230e-27  # kg
alpha_charge = 2
proton_mass = 1.67262192369e-27  # kg
proton_charge = 1

# Elektron yoğunlukları
n_water = 3.34e28
n_hydrogen = 5.40596e28

# Ortalama uyarılma enerjisi fonksiyonları
def excitation_energy(Z):
    if Z == 1:
        return 19.0
    elif 2 <= Z <= 13:
        return 11.2 + 11.7 * Z
    else:
        return 52.8 + 8.71 * Z

def compound_excitation_energy(elements, fractions):
    log_I = 0
    total_NZ = 0
    for Z, frac in zip(elements, fractions):
        I_element = excitation_energy(Z)
        log_I += frac * Z * np.log(I_element)
        total_NZ += frac * Z
    return np.exp(log_I / total_NZ)

# Hız beta fonksiyonu
def beta_from_energy(E_MeV, mass):
    energy_joules = E_MeV * 1e6 * e
    gamma = 1 + energy_joules / (mass * c**2)
    return np.sqrt(1 - (1 / gamma**2))

# Durdurma gücü hesaplama fonksiyonu
def compute_stopping_power(Z, n, beta, I):
    coeff = 5.08e-31
    beta_square = beta**2
    log_term = np.log((1.02e6 * beta_square) / (1 - beta_square))
    F_beta = log_term - beta_square
    return (coeff * Z**2 * n / beta_square) * (F_beta - np.log(I))

# Enerji değerleri ve durdurma gücü hesaplamaları
def calculate_stopping_power(energies, mass, charge, n, I):
    stopping_powers = []
    for E in energies:
        beta = beta_from_energy(E, mass)
        stopping_power = compute_stopping_power(charge, n, beta, I)
        stopping_powers.append(stopping_power)
    return stopping_powers

# Su ve hidrojen için ortalama uyarılma enerjileri
I_water = compound_excitation_energy([1, 8], [2, 1])
I_hydrogen = excitation_energy(1)

# Enerji aralıkları
E_alpha = np.logspace(np.log10(0.2), np.log10(400), 100)
E_proton = np.logspace(np.log10(0.01), np.log10(1000), 100)

# Durdurma güçlerini hesapla
stopping_alpha_water = calculate_stopping_power(E_alpha, alpha_mass, alpha_charge, n_water, I_water)
stopping_alpha_hydrogen = calculate_stopping_power(E_alpha, alpha_mass, alpha_charge, n_hydrogen, I_hydrogen)

stopping_proton_water = calculate_stopping_power(E_proton, proton_mass, proton_charge, n_water, I_water)
stopping_proton_hydrogen = calculate_stopping_power(E_proton, proton_mass, proton_charge, n_hydrogen, I_hydrogen)

# Grafikleri oluştur
def plot_stopping_powers(E_values, stopping_powers, labels, title):
    plt.figure(figsize=(10, 6))
    for E, stopping, label in zip(E_values, stopping_powers, labels):
        plt.loglog(E, stopping, label=label, linewidth=2)
    
    plt.xlabel('Energy (MeV)', fontsize=14)
    plt.ylabel('Stopping Power (MeV cm$^{-1}$)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

# Alfa parçacıkları için grafik
plot_stopping_powers(
    [E_alpha, E_alpha],
    [stopping_alpha_water, stopping_alpha_hydrogen],
    ['Alpha in Water', 'Alpha in Hydrogen'],
    'Stopping Power of Alpha Particles in Water and Hydrogen'
)

# Protonlar için grafik
plot_stopping_powers(
    [E_proton, E_proton],
    [stopping_proton_water, stopping_proton_hydrogen],
    ['Proton in Water', 'Proton in Hydrogen'],
    'Stopping Power of Protons in Water and Hydrogen'
)
