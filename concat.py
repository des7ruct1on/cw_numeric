import pandas as pd

def concat_txt(a, b, output_file):
    temp = pd.read_csv(a, names=["Temperature"])  # Чтение температур
    cp = pd.read_csv(b, names=["Cp"])  # Чтение теплоёмкости

    result = pd.concat([temp, cp], axis=1)

    result.to_csv(output_file, index=False)

    return result

output_file = "data/h2o.csv"
result = concat_txt("a.txt", "b.txt", output_file)
print(result)  # Вывод результата
