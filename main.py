import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ThermodynamicModel import ThermodynamicModel

def task(name, T, Cp, H0, S0, H_del = None):
    model = ThermodynamicModel(name, T, Cp, H0, S0, H_del * 1000)
    cp_params = model.fit_Cp_model()
    print("Fitted coefficients:", cp_params)
    table = model.get_table()
    output_file = "res/thermodynamic_properties_" + model.get_name_elem() + ".xlsx"
    table.to_excel(output_file, index=False)
    model.plot_properties()

h2 = pd.read_csv("data/h2.csv")
T_data_h2 = h2['Temperature'].to_numpy().astype(float)
Cp_data_h2 = h2['Cp'].to_numpy().astype(float)
H2_H0 = 2999
H2_S0 = 100.616
H2_H_del = 432.068

o2 = pd.read_csv("data/o2.csv")
T_data_o2 = o2['Temperature'].to_numpy().astype(float)
Cp_data_o2 = o2['Cp'].to_numpy().astype(float)
O2_H0 = 2901
O2_S0 = 173.192
O2_H_del = 493.566

o = pd.read_csv("data/o.csv")
T_data_o = o['Temperature'].to_numpy().astype(float)
Cp_data_o = o['Cp'].to_numpy().astype(float)
O_H0 = 2207
O_S0 = 135.837
O_H_del = 246.783

h = pd.read_csv("data/h.csv")
T_data_h = h['Temperature'].to_numpy().astype(float)
Cp_data_h = h['Cp'].to_numpy().astype(float)
H_H0 = 2079
H_S0 = 91.900
H_H_del = 216.034

oh = pd.read_csv("data/oh.csv")
T_data_oh = oh['Temperature'].to_numpy().astype(float)
Cp_data_oh = oh['Cp'].to_numpy().astype(float)
OH_H0 = 2744
OH_S0 = 149.978
OH_H_del = 423.720

h2o = pd.read_csv("data/h2o.csv")
T_data_h2o = h2o['Temperature'].to_numpy().astype(float)
Cp_data_h2o = h2o['Cp'].to_numpy().astype(float)
H2O_H0 = 3289
H2O_S0 = 152.271
H2O_H_del = 917.764

task("h2", T_data_h2, Cp_data_h2, H2_H0, H2_S0, H2_H_del)
task("o2", T_data_o2, Cp_data_o2, O2_H0, O2_S0, O2_H_del)
task("o", T_data_o, Cp_data_o, O_H0, O_S0, O_H_del)
task("h", T_data_h, Cp_data_h, H_H0, H_S0, H_H_del)
task("oh", T_data_oh, Cp_data_oh, OH_H0, OH_S0, OH_H_del)
task("h20", T_data_h2o, Cp_data_h2o, H2O_H0, H2O_S0, H2O_H_del)