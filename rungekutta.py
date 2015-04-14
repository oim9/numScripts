# schmithi@student.ethz.ch
# 14.04.2015

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


class RungeKutta():
    """
    Solves Runge-Kutta-Methods.
    
    Use self.integrate()
    """
    
    def __init__(self, butcher_template):
        """
        Build a new Runge-Kutta instance given a Butcher Scheme with s stages

        Arguments:
        butcher_template: tuple(A, b, c) 
            A:  Butcher matrix A of shape (s,s)
            b:  Butcher vector b of shape (s)
            c:  Butcher vector c of shape (s)

            c | A
            __|___
              | b

        """
        A, b, c = butcher_template
        self.A = A.copy()
        self.b = b.copy()
        self.c = c.copy()
        self.stages = self.b.size
    
        
    def integrate(self, f, initial_value, t0, t1, steps):
        """
        integrates f from t0 to t1 with f(t0) = initial_value over (steps) steps
        """
        self.f = f
        if type(initial_value) == int:
            self.dimension = 1
        else:
            self.dimension = initial_value.shape[0]
        
        u = np.zeros((self.dimension+1, steps)) # dimensions + 1 for time
        u[:,0] = np.append(initial_value, t0)
        
        dt = 1.*(t1-t0)/steps
        
        for i in range(steps-1):
            u[:,i+1] = self._step(u[:,i], dt)
            
        return u[-1,:], u[:-1,:] # t, y
        
    def _step(self, u0, dt):
        """
        Makes a single Runge-Kutta step of size dt, starting from current solution u(t).

        Arguments:
        u0:     Current solution u(t) := [y(t), t]
        dt:     Timestep size

        Returns:
        u1:     New solution u(t+dt) := [y(t+dt), t+dt]
        """
        
        y0, t = u0
        
        # System of s equations:
        # x is a s-dimensional vector [k1, .. , ks]
        # vector function:
        #
        # f(y0 + h(A11*k1 + ... + A1s*ks) - k1 = 0
        # f(y0 + h(A21*k1 + ... + A2s*ks) - k2 = 0
        # .
        # ...
        #
        # finds root of helper_function
       
        helper_function = lambda x: self.f(t + dt*self.c ,y0 + dt*np.dot(self.A, x)) - x
        
        rk_coefficients = fsolve(helper_function, np.zeros(stages))
    
        y1 = y0 + dt*np.dot(self.b, rk_coefficients)
    
        return np.append(y1, t+dt)


def main():
    # butcher scheme for simpson rule
    A = np.array([[0.,  0., 0.],
                  [0.5, 0., 0.],
                  [-1., 2., 0.]])
    b = np.array([1/6., 4/6., 1/6.])
    c = np.array([0., 0.5, 1.])
                     
    f = lambda t, y: np.cos(t*y)
    
    rk = RungeKutta((A, b, c))
    t, y = rk.integrate(f, initial_value=3, t0=0, t1=5, steps=1000)
    
    plt.plot(t, y[0]) # because y is 1 dimensional
    plt.show()
    
if __name__ == '__main__':
    # for testing purposes
    main()
