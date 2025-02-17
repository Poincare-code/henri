import numpy as np
import numba

from numba.experimental import jitclass


# Define variable types for numba.jitclass
spec = {
    'polytropic_index': numba.float64,
    'Gamma'           : numba.float64,
    'rho_min'         : numba.float64,
    'rho_max'         : numba.float64,
    'eps_min'         : numba.float64,
    'eps_max'         : numba.float64,
    'v_max'           : numba.float64,
}


# @jitclass(spec) # Does not speed up the code...
class EOS_polytrope:
    """
    Polytropic equation of state.
    """
    def __init__(self, polytropic_index, rho_min=0.0, rho_max=np.inf, eps_min=0.0, eps_max=np.inf, v_max=0.999):
        """
        Initialize polytropic equation of state.
        """
        # Compute polytropic exponent
        self.Gamma = 1.0 + 1.0 / polytropic_index
        # Define limits of eos
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.v_max   = v_max


    def rho(self, p):
        """
        Density as a function of pressure.
        """
        return p**self.Gamma


    def p(self, rho):
        """
        Pressure as a function of density.
        """
        return rho**(1.0 / self.Gamma)