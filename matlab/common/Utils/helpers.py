import numpy as np
from scipy.stats import beta
from scipy.special import digamma, logsumexp

def convfun(t, x, phi, u, v):
    """
    f = betapdf(t/phi,u(1),v(1)).*betapdf((x-t)/(1-phi),u(2),v(2));
    """
    return beta.pdf(t / phi, u[0], v[0]) * beta.pdf((x - t) / (1 - phi), u[1], v[1])

def fun(t, phi, x, u, v):
    """
    f = t.^(u(1)-1) .* (x-t).^(u(2)-1) .* (phi-t).^(v(1)-1) .* (1-phi-x+t).^(v(2)-1);
    """
    return (t**(u[0] - 1) * (x - t)**(u[1] - 1) * 
            (phi - t)**(v[0] - 1) * (1 - phi - x + t)**(v[1] - 1))

def gfun(t, phi, x, u, v):
    """
    f = t.^(u(1)-1) .* (x-t).^(u(2)-1) .* ( (u(2)-1)*(phi-t).^(v(1)-2) .* (1-phi-x+t).^(v(2)-1) ...
        - (v(2)-1)*(phi-t).^(v(1)-1) .* (1-phi-x+t).^(v(2)-2));
    """
    term1 = (u[1] - 1) * (phi - t)**(v[0] - 2) * (1 - phi - x + t)**(v[1] - 1)
    term2 = (v[1] - 1) * (phi - t)**(v[0] - 1) * (1 - phi - x + t)**(v[1] - 2)
    return t**(u[0] - 1) * (x - t)**(u[1] - 1) * (term1 - term2)

def min_alpha_func(alpha_ip, salpha, xn, zi):
    """
    fx = psi(alpha_ip) - psi(salpha) - sum(zi .* log(xn)) ./ sum(zi);
    In Python, psi is scipy.special.digamma.
    """
    return digamma(alpha_ip) - digamma(salpha) - np.sum(zi * np.log(xn)) / np.sum(zi)

# progressbar.m will be replaced by tqdm in the python codebase.
