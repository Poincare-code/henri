import numpy          as     np
from   scipy.optimize import root
from   numba          import njit


def cons2prim(D, S_r, tau, eos):
    """
    Conversion of conserved to primitive variables.
    Following Appendix C in Galleazzi et al. (2013).
    """

    # Determine where the density is too low
    low_rho = (D < eos.rho_min)

    # Clip density (this case will be overwritten further down)
    D[low_rho] = eos.rho_min

    # Define helper variables
    R = np.abs(S_r) / D
    Q = tau / D

    # Clip Q to avoid numerical issues
    Q[Q<0.0] = 0.0

    # Define helper variable
    K = R / (1.0 + Q)

    # Clip R and Q to avoid numerical issues
    K_max = 2.0 * eos.v_max / (1.0 + eos.v_max**2)
    R[K>K_max] = K_max * (1.0 + Q[K>K_max])
    K[K>K_max] = K_max
    
    # Define bounds for the root finding
    z_min = 0.5 * K / np.sqrt(1.0 - 0.25*K**2)
    z_max =       K / np.sqrt(1.0 -      K**2)

    # Define helper functions
    def W(z):
        """
        Functional Lorentz factor.
        """
        return np.sqrt(1.0 + z**2)

    def RHO(z):
        """
        Functional density.
        """
        # Compute density
        rho = D / W(z)
        # Clip within physical limits
        rho[rho<eos.rho_min] = eos.rho_min
        rho[rho>eos.rho_max] = eos.rho_max
        # Return result
        return rho
    
    def EPS(z):
        """
        Functional specific internal energy.
        """
        # Compute specific internal energy
        eps = W(z) * Q - z * R + z**2 / (1.0 + W(z))
        # Clip within physical limits
        eps[eps<eos.eps_min] = eos.eps_min
        eps[eps>eos.eps_max] = eos.eps_max
        # Return result
        return eps

    def A(z):
        """
        Functional A.
        """
        # Extract pressure from equation of state
        p = eos.p(RHO(z))
        # Return result
        return p / (RHO(z) * (1.0 + EPS(z)))
    
    def H(z):
        """
        Functional enthalpy.
        """
        return (1.0 + EPS(z)) * (1.0 + A(z))

    def f(z):
        return z - R / H(z)

    # Find root
    z_root = root(fun=f, x0=0.5*(z_min+z_max), method='krylov')

    # Extract primitive variables
    rho = RHO(z_root.x)
    v_r = S_r / (D * W(z_root.x) * H(z_root.x))

    # Limit velocity
    v_r[v_r>eos.v_max] = eos.v_max

    # Exclude too low densities (i.e. basically vacuum)
    rho[low_rho] = eos.rho_min
    v_r[low_rho] = 0.0

    # Derive remaining primitive variables
    p = eos.p(rho)
    w = 1.0 / np.sqrt(1.0 - v_r**2)
    h = 1.0 + p / rho
    
    # Return primitive variables
    return (rho, p, v_r, w, h)