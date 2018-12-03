import pandas as pd
from tqdm import tqdm
import random
import re

def extract_num(st):
    result = [int(s) for s in re.findall(r'-?\d+\.?\d*', st)][0]
    return result

print(extract_num('train 00001 .csv'))

a = 5
b = 4

print('this is multi-gpu')