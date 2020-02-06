import numpy as np
import pandas as pd
import scipy.sparse as sparse

from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()
print(lb.fit_transform(['yes', 'no', 'no', 'yes']))
