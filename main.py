import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ThermodynamicModel import ThermodynamicModel, get_thermodynamic_arrays

def task(name, H0, S0, H_del = None):
    model = ThermodynamicModel(name, H0, S0, H_del)
    cp_params = model.fit_Cp_model()
    print("Fitted coefficients:", cp_params)
    table = model.get_table()
    output_file = "res/thermodynamic_properties_" + model.get_name_elem() + ".xlsx"
    table.to_excel(output_file, index=False)
    model.plot_properties()

H2_H0 = 2999
H2_S0 = 100.616
H2_H_del = 432.068

O2_H0 = 2901
O2_S0 = 173.192
O2_H_del = 493.566

O_H0 = 2207
O_S0 = 135.837
O_H_del = 246.783

H_H0 = 2079
H_S0 = 91.900
H_H_del = 216.034

OH_H0 = 2744
OH_S0 = 149.978
OH_H_del = 423.720

H2O_H0 = 3289
H2O_S0 = 152.271
H2O_H_del = 917.764

task("h2", H2_H0, H2_S0, H2_H_del)
task("o2", O2_H0, O2_S0, O2_H_del)
task("o", O_H0, O_S0, O_H_del)
task("h", H_H0, H_S0, H_H_del)
task("oh", OH_H0, OH_S0, OH_H_del)
task("h2o", H2O_H0, H2O_S0, H2O_H_del)