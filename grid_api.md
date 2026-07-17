This is an enumeration of SphericalGrid use cases to improve their ergonomics.

Each heading is a usage scenario and each code block is a potential solution for that scenario. Scenarios marked with `[ ]` are considered as having satisfactory solution and `[x]` means they are implemented.

# [ ] Load static/dynamic dataset and trim radial extent to (3, 15) Re

``` python
m = TIMEGCMModel()
m = m.trim(r=(3, 15))
```

- is modifying the grid shape of an instantiated model feasible?
  - can get really nasty for models like SphHarmSplineModel with internally generated stuff that depends on the grid shape (e.g. self.l, self.m, etc.)
  
``` python
m = TIMEGCMModel()
m = TIMEGCMModel(m.grid.trim(r=3, 15))
```

- *TODO*: add trim argument to SphericalGrid

# [ ] Load dynamic dataset and sample 6 hour cadence

``` python
m = TIMEGCMModel(freq='6h')
```

- *TODO*: add freq argument to DataDynamicModel

# [ ] Load dynamic dataset at 6 hour cadence and default spatial grid

``` python
grid = DefaultGrid() # static grid
m = TIMEGCMModel(grid, freq='6h')
```

- this doesn't work
    - is final result static or not?
    - if static, we must load first time index only
  
``` python
m = TIMEGCMModel(freq='6h')
grid = DefaultGrid().makedynamic(m.grid.t)
```

- *TODO*: add makedynamic(...) to SphericalGrid

# [x] Load storm week at 1 hour cadence

``` python
def Pratik25StormModel(*args, offset=6*7, freq='1h', window=14, **kwargs):
    return Pratik25Model(*args, offset=offset, freq=freq, window=window, **kwargs)
    
m = Pratik25StormModel()
```

# [ ] Load dataset at specific times

``` python
grid = DefaultGrid(t=[1, 2, 3, 4, 5])
m = TIMEGCMModel(grid)
```

- *TODO*: change behavior of SphericalGrid to not require all t_b, r, e, a

# [ ] Load dynamic dataset with static grid

```
grid = DefaultGrid()
m = TIMEGCMModel(grid)
```

- takes first time sample

# [ ] Load dynamic dataset with static grid at specific time

```
grid = DefaultGrid()
m = TIMEGCMModel(grid, offset=np.timedelta(1, 'W'))
```

- this may not work in practice
- requires applying `offset` argument first, then selecting first item
  - I don't think Datadynamicmodel works this way
  - Have notes about making this change to handle time first for efficiency
  
- *TODO*: process time args first on DataDynamicModel