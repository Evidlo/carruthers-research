#!/usr/bin/env python3
# Example for running code in parallel with dask
# https://examples.dask.org/applications/embarrassingly-parallel.html

import numpy as np
from dask.distributed import LocalCluster, Client

def rt_code_simple(x):
    return x * 10

def rt_code_xarray(x):
    return x * 10

if __name__ == '__main__':
    # set up a cluster with 8 workers on the local machine
    # this cluster could also consist of multiple machines
    cluster = LocalCluster(n_workers=8, ip='127.0.0.1')
    # connect to the cluster
    client = Client(cluster)

    # run rt_code_simple in parallel with different inputs
    inputs = range(10)
    rt_result = client.map(rt_code_simple, inputs)
    print(client.gather(rt_result))

    # run rt_code_xarray in parallel with more complicated input
    # TODO

    client.close()
    cluster.close()