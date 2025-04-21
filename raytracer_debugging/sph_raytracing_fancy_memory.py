#!/usr/bin/env python3

from contexttimer import Timer
import torch as t
t.cuda.empty_cache()
t.set_num_threads(48)

def memprint():
    print("memory_allocated: %fGB"%(t.cuda.memory_allocated(0)/1024/1024/1024))
    print("memory_reserved: %fGB"%(t.cuda.memory_reserved(0)/1024/1024/1024))
    print("max_memory_reserved: %fGB"%(t.cuda.max_memory_reserved(0)/1024/1024/1024))

device = 'cpu'
# specint = {'dtype':t.int64, 'device':device}
specint = {'dtype':t.int64, 'device':device}
specfloat = {'dtype':t.float16, 'device':device}

vol = (60, 19, 37)
shape=[10, 512, 512, 2*vol[0] + vol[1] + vol[2] // 2]
r=t.randint(0, 60, shape, **specint)
e=t.randint(0, 19, shape, **specint)
a=t.randint(0, 37, shape, **specint)
l = t.rand(shape, **specfloat)

memprint()

with Timer(prefix='line integration'):
    for _ in range(1):
        print(_)
        x = t.rand(vol, **specfloat)
        result = (x[r, e, a] * l).sum(axis=-1)
