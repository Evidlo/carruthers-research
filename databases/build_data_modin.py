#!/usr/bin/env python3

import modin.pandas as pd
import numpy as np

from shutil import rmtree

db = 'test.modin'
rmtree(db, ignore_errors=True)

df = pd.DataFrame(

)
# df = pd.concat([pd.DataFrame(np.random.randint(0, 100, size=(2**20, 2**8))) for _ in range(40)]) # 40x2GB frames -- Working!
# df.info()

# build dataframe
dates = pd.date_range('2025-01-01', '2028-01-01', 1000)

# process dates one chunk at a time
chunk_len = 100
for n in range(0, len(dates), chunk_len):
    print(f'n:{n}')
    dates_chunk = dates[n:n + chunk_len]
    img_a_chunk = list(np.random.random((len(dates_chunk), 1000, 1000)))
    img_b_chunk = list(np.random.random((len(dates_chunk), 1000, 1000)))
    param_chunk = np.random.random(len(dates_chunk))

    df_chunk = pd.DataFrame(
        {
            "img_a": img_a_chunk,
            "img_b": img_b_chunk,
            "param": param_chunk
        },
        index=dates_chunk,
    )

    df = pd.concat([df, df_chunk])

df.to_hdf('modin.hdf', 'hdf')
