#!/usr/bin/env python3

# import vaex
import numpy as np
from datetime import datetime
from dateutil.parser import parse
import pandas as pd

# build dataframe
dates = pd.date_range('2023-01-01', '2023-09-01', 100)
imgs = list(np.random.random((len(dates), 100, 100)))
df = pd.DataFrame({'img': imgs}, index=dates)

# get dates
result_dates = df.index.get_indexer(
    pd.date_range('2023-03-01', '2023-05-01', 10),
    method='nearest'
)
result = df.iloc[result_dates]
