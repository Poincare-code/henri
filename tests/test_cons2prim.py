import numpy as np

from henri.grhd.cons2prim import cons2prim
from henri.grhd.eos       import EOS_polytrope


def test_cons2prim():

    N = 100

    eos = EOS_polytrope(polytropic_index=1.0)

    rho = np.random.rand(N)
    v_r = np.random.rand(N) * 0.1
    eps = np.random.rand(N)

    W = 1 / np.sqrt(1 - v_r**2)
    p = eos.p(rho)
    h = 1 + eps + p / rho

    # Conserved variables
    D   = rho * W
    S_r = rho * h * W**2 * v_r
    tau = rho * h * W**2 - p - D

    (rho_t, p_t, v_r_t, W_t, h_t) = cons2prim(D=D, S_r=S_r, tau=tau, eos=eos)

    assert np.allclose(rho, rho_t)
    assert np.allclose(p,     p_t)
    assert np.allclose(v_r, v_r_t)
    assert np.allclose(W,     W_t)
    assert np.allclose(h,     h_t)