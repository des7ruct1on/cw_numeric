import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ThermodynamicModel:
    def __init__(self, elem, T_data, Cp_data, H0 = 0, S0 = 0, H_del = 0):
        """
        Класс для работы с термодинамическими свойствами, такими как Cp(T), энтальпия, энтропия и потенциал Гиббса.

        Parameters:
            T_data (array-like): Температурные данные.
            Cp_data (array-like): Значения теплоёмкости при соответствующих температурах.
            H0 (float): Энтальпия при базовой температуре.
            S0 (float): Энтропия при базовой температуре.
        """
        self.T_data = np.array(T_data)
        self.Cp_data = np.array(Cp_data)
        self.H0 = H0
        self.S0 = S0
        self.H_del = H_del
        self.elem = elem
        self.Cp_params = None

    @staticmethod
    def Cp_model(T, a, b, c, d, e):
        """
        Модель теплоёмкости Cp(T) = a + b / T^2 + c * T + d * T^2 + e * T^3.
        """
        return a + b / (T**2) + c * T + d * T**2 + e * T**3

    def fit_Cp_model(self):
        """
        Аппроксимация данных Cp(T) методом наименьших квадратов.
        """
        T = self.T_data.astype(float) 
        Cp = self.Cp_data
        X = np.vstack([
            np.ones(len(T)),       # a
            T**-2,                 # b / T^2
            T,                     # c * T
            T**2,                  # d * T^2
            T**3                   # e * T^3
        ]).T

        # Решение системы линейных уравнений: (XᵀX)a = Xᵀy
        XtX = np.dot(X.T, X)       # XᵀX
        Xty = np.dot(X.T, Cp)      # Xᵀy
        self.Cp_params = np.linalg.solve(XtX, Xty)  # Решение линейной системы
        return self.Cp_params

    @staticmethod
    def trapezoidal_integral(func, x_start, x_end, num_points=1000):
        """
        Численное интегрирование методом трапеций.
        """
        x = np.linspace(x_start, x_end, num_points)
        y = func(x)
        dx = (x_end - x_start) / (num_points - 1)
        return np.sum((y[:-1] + y[1:]) / 2) * dx

    def calculate_enthalpy(self, T):
        """
        Вычисление энтальпии H(T).
        """
        if self.Cp_params is None:
            raise ValueError("Model is not fitted yet. Call `fit_Cp_model` first.")
        Cp_func = lambda t: self.Cp_model(t, *self.Cp_params)
        integral = self.trapezoidal_integral(Cp_func, 100, T)
        return integral + self.H0

    def calculate_entropy(self, T):
        """
        Вычисление энтропии S(T).
        """
        if self.Cp_params is None:
            raise ValueError("Model is not fitted yet. Call `fit_Cp_model` first.")
        entropy_integrand = lambda t: self.Cp_model(t, *self.Cp_params) / t
        integral = self.trapezoidal_integral(entropy_integrand, 100, T)
        return integral + self.S0

    def calculate_gibbs(self, T):
        """
        Вычисление приведённого потенциала Гиббса Φ(T).
        """
        H = self.calculate_enthalpy(T)
        S = self.calculate_entropy(T)
        return self.H_del - H * 1000 - T * S

    def plot_properties(self):
        """
        Визуализация термодинамических свойств.
        """
        Cp_fitted = self.Cp_model(self.T_data, *self.Cp_params)
        enthalpy = [self.calculate_enthalpy(T) / 1000.0 for T in self.T_data]
        entropy = [self.calculate_entropy(T) for T in self.T_data]
        gibbs = [self.calculate_gibbs(T) / 100 for T in self.T_data]

        plt.figure(figsize=(12, 8))

        # Теплоёмкость
        plt.subplot(2, 2, 1)
        plt.scatter(self.T_data, self.Cp_data, label="Experimental Data for " + self.elem, s=10, color="blue")
        plt.plot(self.T_data, Cp_fitted, label="Fitted Model", color="red")
        plt.title("Heat Capacity (Cp)")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Cp")
        plt.legend()

        # Энтальпия
        plt.subplot(2, 2, 2)
        plt.plot(self.T_data, enthalpy, label="Enthalpy (H)", color="blue")
        plt.title("Enthalpy (H)")
        plt.xlabel("Temperature (K)")
        plt.ylabel("H")

        # Энтропия
        plt.subplot(2, 2, 3)
        plt.plot(self.T_data, entropy, label="Entropy (S)", color="green")
        plt.title("Entropy (S)")
        plt.xlabel("Temperature (K)")
        plt.ylabel("S")

        # Приведённый потенциал Гиббса
        plt.subplot(2, 2, 4)
        plt.plot(self.T_data, gibbs, label="Gibbs Potential (Φ)", color="purple")
        plt.title("Gibbs Potential (Φ)")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Φ")

        plt.tight_layout()
        plt.show()

    def get_table(self):
        """
        Вывод таблицы с термодинамическими свойствами.
        """
        if self.Cp_params is None:
            raise ValueError("Model is not fitted yet. Call `fit_Cp_model` first.")

        Cp_fitted = self.Cp_model(self.T_data, *self.Cp_params)
        enthalpy = [self.calculate_enthalpy(T) / 1000.0 for T in self.T_data]
        entropy = [self.calculate_entropy(T) for T in self.T_data]
        gibbs = [self.calculate_gibbs(T) for T in self.T_data]

        # Создание таблицы
        data_table = pd.DataFrame({
            "Temperature (K)": self.T_data,
            "Heat Capacity (Cp)": self.Cp_data,
            "Fitted Cp": Cp_fitted,
            "Enthalpy (H)": enthalpy,
            "Entropy (S)": entropy,
            "Gibbs Potential (Φ)": gibbs
        })

        return data_table

    def get_name_elem(self) -> str:
        return self.elem
