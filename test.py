import pandas as pd
from tqdm import tqdm
import random

temp_list = list()
for i in tqdm(range(10000000)):
    if random.random() > 0.1:
        temp_list.append(("asdfasdfasdfasdfasdf", 0.234235, 234.23423, 54.234, 5))
    else:
        temp_list.append((None, 0.234235, 234.23423, 54.234, 5))


b = pd.DataFrame.from_records(temp_list)
b = b.dropna()
print(b)