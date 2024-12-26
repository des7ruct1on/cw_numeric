from ThermodynamicData import thermodynamic_properties_o2, get_thermodynamic_arrays


def calculate_gibbs(T, H, S):
    """
    Вычисление приведённого потенциала Гиббса Φ(T).
    """
    enthalpy = H
    entropy = S
    gibbs = []
    for i in range(len(T)):
        g = enthalpy[i] - entropy[i] * T[i]
        gibbs.append(g)

    return gibbs

def calculate_phi(H, G, T):
    """
    Вычисление приведённого потенциала Гиббса Φ(T) для конкретных значений H, G и T.
    """
    print()
    print(f"H = ", H)
    print(f"G = ", G)
    print(f"T = ", T)
    return (493.566 - H * 100 - G) / T

T_data, Cp_data, Phi_data, S_data, H_data = get_thermodynamic_arrays(thermodynamic_properties_o2)
G_data = calculate_gibbs(T_data, H_data, S_data)
for i in range(5):
    print(f"Phi = ", calculate_phi(H_data[i], G_data[i], T_data[i]))