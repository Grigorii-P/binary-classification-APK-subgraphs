from utils import batch, get_data
import numpy as np
import pandas as pd

x,y = get_data()

print(np.amax(y),np.amin(y))
print(len(y[y==1]),len(y[y==0]))

print()