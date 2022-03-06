import pandas as pd
import os

flower_dir = r"./flower.csv"

flower = pd.read_csv(flower_dir, header=None)

flower_name = flower.iloc[:, 0]
flower_label = flower.iloc[:, 1]
for i, name in enumerate(flower_name):
    with open(os.path.join(r"D:\deepL\dataset\flower_dataset\test\label", name.split('.')[0] + ".txt"), 'w') as fp:
        fp.write(flower_label[i].split('[')[1].split(']')[0])
