# PDEcollector
## Collect PDE and its true solution
&nbsp;&nbsp;
### maybe exsit errors in PDE, please use it carefully !

### PDE

#### - - - PDE_name
``` 
 datafile    # if PDE_analytic_solution = False
 description.pdf
 
 PDE_name.py
    import numpy as np
    
    PDE_datafile = 'datafile' # if PDE_analytic_solution = False
    PDE_dim = 2
    PDE_vars = ['x', 'y']
    PDE_scale = {
        'x': (0, 1),
        'y': (0, 1)
    }
    PDE_analytic_solution = True
    PDE_description = 'Uxx + Uyy = PDE_f1(x, y)'
    PDE_initial_condition = ['PDE_ic1()']
    PDE_boundary_condition = ['PDE_bc1(y, x=0)', 'PDE_bc2(y, x=1)', 'PDE_bc3(x, y=0)', 'PDE_bc4(x, y=1)']

    
    def PDE_f1(x, y):
        return np.exp(-x) * (x - 2 + np.power(y, 3) + 6 * y)
    
    #def PDE_ic1():
    #    pass
        
    def PDE_bc1(y, x=0):
        return np.power(y, 3)

    def PDE_bc2(y, x=1):
        return np.exp(-1) * (1 + np.power(y, 3))

    def PDE_bc3(x, y=0):
        return np.exp(-x) * x

    def PDE_bc4(x, y=1):
        return np.exp(-x) * (x + 1)
    
    # if PDE_analytic_solution = True
    def PDE_u(x, y):
        return np.exp(-x) * (x + np.power(y, 3))
        
    # if PDE_analytic_solution = False
    #def PDE_get_data():

    #    return {
    #        'data': data,
    #        'x': x,
    #        'x_dim': len(x),
    #        'y': y,
    #        'y_dim': len(y),
    #        'solution_type': 'X Y U',
    #        'solution': [X, Y, Exact]
    #    }
    
```

    
             
      
