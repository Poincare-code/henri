import numpy as np

from scipy.integrate      import solve_ivp
from henri.grhd.cons2prim import cons2prim


class GRHD:

    def __init__(self, eos):

        self.eos = eos


    def solve_metric(self, tilde_E, tilde_S_r, tilde_S):
        """
        Solve for the metric.
        """
        raise NotImplementedError("solve_metric not implemented.")


    def rhs(self, t, y, r, alpha, beta_r, psi):
        """
        Right-hand side of the conservation equations.
        """

        N = len(y) // 3

        # Unpack conserved variables
        tilde_D   = y[   :  N]
        tilde_S_r = y[  N:2*N]
        tilde_tau = y[2*N:3*N]

        # Define helper variables
        psi4  = psi**4
        psi6  = psi**6
        psi10 = psi**10
        log_r = np.log(r)
        inv_r    = 1.0 / r
        inv_psi6 = 1.0 / psi6

        # Extract sqrt of the (spatial) metric determinant, i.e. sqrt(det(gamma))
        sqrt_gamma = psi6 * r**2

        # Conserved variables
        D   = tilde_D   * inv_psi6
        S_r = tilde_S_r * inv_psi6
        tau = tilde_tau * inv_psi6

        # Convert S_r to S^r
        S_r /= psi4

        # Convert conserved to primitive variables
        (rho, p, v_r, W, h) = cons2prim(D=D, S_r=S_r, tau=tau, eos=self.eos)

        # Variables required by metric solver
        a   = rho * h * W**2
        E   = a - p
        S_r = a * v_r
        S   = S_r * v_r + 3.0 * p

        # Rescaled variables required by metric solver
        tilde_E   = psi6 * E
        tilde_S_r = psi6 * S_r
        tilde_S   = psi6 * S

        # Solve for metric 
        alpha[:], beta_r[:], psi[:], A_rr = self.solve_metric(tilde_E, tilde_S_r, tilde_S)

        # Get metric derivatives
        d_alpha  = np.gradient(alpha,  log_r) * inv_r
        d_beta_r = np.gradient(beta_r, log_r) * inv_r
        d_psi    = np.gradient(psi   , log_r) * inv_r
        
        # Flux variables
        b     = alpha * v_r - beta_r
        F_D   = D          * b
        F_r_S = S_r * psi4 * b + alpha * p
        F_tau = tau        * b + alpha * p * v_r

        # Source variables
        S_r_S = 2 * alpha * (S * d_psi / psi + p * inv_r) + S_r * psi4 * d_beta_r - E * d_alpha
        S_tau =     alpha * rho * h * W**2 * v_r**2 / psi10 * A_rr - S_r * d_alpha

        # Right-hand sides of conservation equations
        dC_D_dt   = -np.gradient(sqrt_gamma * F_D,   log_r) * inv_r**3
        dC_S_r_dt = -np.gradient(sqrt_gamma * F_r_S, log_r) * inv_r**3 + psi6 * S_r_S
        dC_tau_dt = -np.gradient(sqrt_gamma * F_tau, log_r) * inv_r**3 + psi6 * S_tau
        
        # Return the derivatives
        return np.concatenate((dC_D_dt, dC_S_r_dt, dC_tau_dt))
    

    def evolve(self, r, rho_0, v_r_0, psi_0, t_evol):
        """
        Evolve conservations equations forward in time.
        """

        # Helper variables
        p_0 = self.eos.p(rho_0)
        W_0 = 1.0 / np.sqrt(1.0 - v_r_0**2)
        h_0 = 1.0 + p_0 / rho_0

        # Conserved variables
        D_0   = rho_0 * W_0
        S_r_0 = rho_0 * h_0 * W_0**2 * v_r_0
        tau_0 = rho_0 * h_0 * W_0**2 - p_0 - D_0

        # Rescaled conserved variables
        tilde_D_0    = psi_0**6 * D_0
        tilde_S_r_0  = psi_0**6 * S_r_0
        tilde_tau_0  = psi_0**6 * tau_0

        # Initial conditions
        y0 = np.concatenate((tilde_D_0, tilde_S_r_0, tilde_tau_0))

        # Allocate memory for metric variables
        alpha  = np.zeros_like(r)
        beta_r = np.zeros_like(r)
        psi    = psi_0.copy()

        # Solve conservation equations
        self.sol = solve_ivp(
            fun     = self.rhs,
            t_span  = [0.0, t_evol],
            y0      = y0,
            method  = 'RK45',
            rtol    = 1.0e-10,
            atol    = 1.0e-10,
            args    = (r, alpha, beta_r, psi),
        )

        # Extract time
        self.t = self.sol.t

        N = len(self.sol.y) // 3

        # Unpack conserved variables
        self.tilde_D   = self.sol.y[   :  N]
        self.tilde_S_r = self.sol.y[  N:2*N]
        self.tilde_tau = self.sol.y[2*N:3*N]

        print(f'Evolved over t={t_evol}s, in {len(self.t)} steps.')