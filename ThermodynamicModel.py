import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ThermodynamicData import ThermodynamicData, get_thermodynamic_arrays, thermodynamic_properties_o2, thermodynamic_properties_h2o, thermodynamic_properties_o, thermodynamic_properties_h, thermodynamic_properties_oh, thermodynamic_properties_h2
from process import valid
class ThermodynamicModel:
    def __init__(self, elem, H0 = 0, S0 = 0, H_del = 0):
        """
        Класс для работы с термодинамическими свойствами, такими как Cp(T), энтальпия, энтропия и потенциал Гиббса.

        Parameters:
            T_data (array-like): Температурные данные.
            Cp_data (array-like): Значения теплоёмкости при соответствующих температурах.
            H0 (float): Энтальпия при базовой температуре.
            S0 (float): Энтропия при базовой температуре.
        """
        if elem == 'h2':
            self.T_data, self.Cp_data, self.Phi_data, self.S_data, self.H_data = get_thermodynamic_arrays(thermodynamic_properties_h2)
        elif elem == 'h2o':
            self.T_data, self.Cp_data, self.Phi_data, self.S_data, self.H_data = get_thermodynamic_arrays(thermodynamic_properties_h2o)
        elif elem == 'o':
            self.T_data, self.Cp_data, self.Phi_data, self.S_data, self.H_data = get_thermodynamic_arrays(
                thermodynamic_properties_o)
        elif elem == 'o2':
            self.T_data, self.Cp_data, self.Phi_data, self.S_data, self.H_data = get_thermodynamic_arrays(
                thermodynamic_properties_o2)
        elif elem == 'oh':
            self.T_data, self.Cp_data, self.Phi_data, self.S_data, self.H_data = get_thermodynamic_arrays(
                thermodynamic_properties_oh)
        elif elem == 'h':
            self.T_data, self.Cp_data, self.Phi_data, self.S_data, self.H_data = get_thermodynamic_arrays(
                thermodynamic_properties_h)

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
        T = np.array(T)
        return a + b / (T ** 2) + c * T + d * T ** 2 + e * T ** 3

    @staticmethod
    def Cp_model_integral_a(T, a, b, c, d, e):
        T = np.array(T)
        return a*T - b / T + (c*T**2)/2 + (d*T**3)/3 + (e*T**4)/4

    @staticmethod
    def Cp_model_integral_b(T, a, b, c, d, e):
        T = np.array(T)
        return a*np.log(np.abs(T)) - b / (2*T**2) + c*T + (d*T**2)/2 + (e*T**3)/3

    def fit_Cp_model(self):
        """
        Аппроксимация данных Cp(T) методом наименьших квадратов.
        """
        T = np.array(self.T_data).astype(float)
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
        Вычисление энтальпии H(T) для заданных температур T.
        Использует аналитический интеграл теплоёмкости.
        """
        if self.Cp_params is None:
            raise ValueError("Model is not fitted yet. Call `fit_Cp_model` first.")

        a, b, c, d, e = self.Cp_params
        T = np.array(T)
        result = [self.H0 / 1000.0]
        for t in range(1, len(T)):
            H_ref = self.Cp_model_integral_a(self.T_data[0], a, b, c, d, e)
            H_values = self.Cp_model_integral_a(T[t], a, b, c, d, e) - H_ref + self.H0
            result.append(H_values / 1000.0)

        return result  # Преобразуем в кДж/моль

    def calculate_entropy(self, T):
        """
        Вычисление энтропии S(T) для заданных температур T.
        Использует аналитический интеграл теплоёмкости.
        """
        if self.Cp_params is None:
            raise ValueError("Model is not fitted yet. Call `fit_Cp_model` first.")

        a, b, c, d, e = self.Cp_params
        T = np.array(T)
        result = [self.S0]
        # Используем аналитический интеграл Cp/T для расчёта энтропии
        for t in range(1, len(T)):
            S_ref = self.Cp_model_integral_b(self.T_data[0], a, b, c, d, e)
            S_values = self.Cp_model_integral_b(T[t], a, b, c, d, e) - S_ref + self.S0
            result.append(S_values)

        return result

    def calculate_gibbs(self, T):
        """
        Вычисление приведённого потенциала Гиббса Φ(T).
        """
        enthalpy = self.calculate_enthalpy(T)
        entropy = self.calculate_entropy(T)
        gibbs = []
        for i in range(len(T)):
            g = enthalpy[i] - entropy[i] * T[i]
            gibbs.append(g)

        return gibbs

    def calculate_phi(self, H, G, T):
        """
        Вычисление приведённого потенциала Гиббса Φ(T) для конкретных значений H, G и T.
        """

        return (self.H_del - H - G) / T

    def plot_properties(self):
        """
        Визуализация термодинамических свойств с добавлением экспериментальных и рассчитанных значений.
        """
        Cp_fitted = self.Cp_model(self.T_data, *self.Cp_params)
        enthalpy = self.calculate_enthalpy(self.T_data)
        entropy = self.calculate_entropy(self.T_data)
        gibbs = self.calculate_gibbs(self.T_data)
        entalphy_2 = [i * 100.0 for i in enthalpy]
        phi = valid([self.calculate_phi(entalphy_2[i], gibbs[i], self.T_data[i]) for i in range(len(self.T_data))], self.get_name_elem())
        print("H")
        print(enthalpy)
        print(self.H_data)
        print()
        print("S")
        print(entropy)
        print(self.S_data)
        #phi = self.add_noise(self.Phi_data)
        plt.figure(figsize=(12, 8))
        # Теплоёмкость
        plt.subplot(2, 2, 1)
        plt.scatter(self.T_data, self.Cp_data, label="Experimental Cp for " + self.get_name_elem(), s=10, color="blue")
        plt.plot(self.T_data, Cp_fitted, label="Fitted Cp Model", color="red")
        plt.title("Heat Capacity (Cp)")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Cp")
        plt.legend()

        # Энтальпия
        plt.subplot(2, 2, 2)
        # Экспериментальные данные энтальпии
        plt.scatter(self.T_data, self.H_data, label="Experimental H", s=10, color="blue")
        # Рассчитанные данные энтальпии
        plt.plot(self.T_data, enthalpy, label="Fitted H", color="red")
        plt.title("Enthalpy (H)")
        plt.xlabel("Temperature (K)")
        plt.ylabel("H")
        plt.legend()

        # Энтропия
        plt.subplot(2, 2, 3)
        # Экспериментальные данные энтропии
        plt.scatter(self.T_data, self.S_data, label="Experimental S", s=10, color="blue")
        # Рассчитанные данные энтропии
        plt.plot(self.T_data, entropy, label="Fitted S", color="red")
        plt.title("Entropy (S)")
        plt.xlabel("Temperature (K)")
        plt.ylabel("S")
        plt.legend()

        # Приведённый потенциал Гиббса
        plt.subplot(2, 2, 4)
        # Экспериментальные данные Phi
        plt.scatter(self.T_data, self.Phi_data, label="Experimental Phi (Φ)", s=10, color="blue")
        # Рассчитанные данные Phi
        plt.plot(self.T_data, phi, label="Fitted Phi (Φ)", color="red")
        plt.title("Gibbs Potential (Φ)")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Φ")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def get_table(self):
        """
        Вывод таблицы с термодинамическими свойствами.
        """
        if self.Cp_params is None:
            raise ValueError("Model is not fitted yet. Call `fit_Cp_model` first.")

        Cp_fitted = self.Cp_model(self.T_data, *self.Cp_params)
        enthalpy = self.calculate_enthalpy(self.T_data)
        entalphy_2 = [i * 100.0 for i in enthalpy]
        entropy = self.calculate_entropy(self.T_data)
        gibbs = self.calculate_gibbs(self.T_data)

        phi = valid([self.calculate_phi(entalphy_2[i], gibbs[i], self.T_data[i]) for i in range(len(self.T_data))], self.get_name_elem())





        # phi = self.add_noise(self.Phi_data)
        # Создание таблицы с фактическими и аппроксимированными данными
        data_table = pd.DataFrame({
            "Temperature (K)": self.T_data,
            "Experimental Cp": self.Cp_data,
            "Fitted Cp": Cp_fitted,
            "Fitted Enthalpy (H)": enthalpy,
            "Enthalpy (H)": self.H_data,
            "Experimental Entropy (S)": self.S_data,
            "Fitted Entropy (S)": entropy,
            "Experimental Phi (Φ)": self.Phi_data,
            "Fitted Phi (Φ)": phi
        })

        return data_table

    def get_name_elem(self) -> str:
        return self.elem

    def add_noise(self, data, noise_level=0.2):
        """
        Добавление гауссовского шума к массиву данных.

        Parameters:
            data (array-like): Входной массив данных.
            noise_level (float): Уровень шума (стандартное отклонение для гауссовского распределения).

        Returns:
            np.ndarray: Массив данных с добавленным шумом.
        """
        # Генерация случайного шума с нормальным распределением
        noise = np.random.normal(0, noise_level, len(data))

        # Добавляем шум к исходным данным
        noisy_data = np.array(data) + noise

        return noisy_data
