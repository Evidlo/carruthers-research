This is an enumeration of SphericalGrid use cases to improve their ergonomics.

Each heading is a usage scenario and each code block is a potential solution for that scenario.  

# [x] Load dataset and trim only radial extent to (3, 15) Re

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

- _add trim argument to SphericalGrid_

# [x] Load dataset at 6 hour cadence

``` python
m = TIMEGCMModel(period=np.timedelta64(6, 'h'))
```

- _add period argument to DataDynamicModel_

# Load dataset at 6 hour cadence and default grid

``` python
grid = DefaultGrid()
m = TIMEGCMModel(grid, period=np.timedelta64(6, 'h'))
```

- is grid static or not?
  - if static, we must load first time index only
  
``` python
m = TIMEGCMModel(period=np.timedelta64(6, 'h'))
grid = DefaultGrid().makedynamic(m.grid.t)
```

- _add makedynamic(...) to SphericalGrid_

# [x] Load dataset at specific times

``` python
grid = DefaultGrid(t=[1, 2, 3, 4, 5])
m = TIMEGCMModel(grid)
```

# [x] Load dynamic dataset with static grid

```
grid = DefaultGrid()
m = TIMEGCMModel(grid)
```

- takes first time sample

# [x] Load dynamic dataset with static grid at specific time

```
grid = DefaultGrid()
m = TIMEGCMModel(grid, offset=np.timedelta(1, 'W'))
```

- this may not work in practice
- requires applying `offset` argument first, then selecting first item
  - I don't think Datadynamicmodel works this way
  - Have notes about making this change to handle time first for efficiency
  
- _process time args first on DataDynamicModel_