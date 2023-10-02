import numpy as np
import scipy as sp
from .lanczos import *

# default number of recurrence coefficients to generate
_default_degree = 1000

# TODO: number of coefficient adaptively? I.e. store iterators fpr coefficients rather than a list?
class reference_density:

    def __init__(self,σ,γδ,mass=1):
        """
        Parameters
        ----------
        σ : function
            unit mass density function
        γδ : tuple of ndarray
            recurrence coeffs
        mass : float
            mass for density function
        """
        γ,δ = γδ
        
        self.s = min(2*len(γ),2*len(δ))
        self.σ = σ
        self.γ = γ[:self.s//2]
        self.δ = δ[:self.s//2]
        self.mass = mass

    def __call__(self,x):
        return self.mass*self.σ(x)

    def __add__(self,σ1):
        return combine_densities([self,σ1])
        
    def __mul__(self,c):
        return reference_density(self.σ,(self.γ,self.δ),c*self.mass)

    def __rmul__(self,c):
        return self*c
        
    def series(self,x,c):
        """
        evalutes orthogonal polynomial series at x
        
        Parameters
        ----------
        x : float or ndarray
            points to evaluate
        c : ndarray
            series coefficients

        Returns
        -------
        ndarray
            orthogonal polynomial series evaluated at x
        
        """
        
        k = len(c)
        assert k<=self.s, f'degree {k-1} of requested series exceeds degree {self.s} of recurrence'

        pn_ = np.zeros_like(x)
        pn = np.ones_like(x)
        y = c[0]*pn
        
        for n in range(k-1):
            pn__ = np.copy(pn)
            pn = (1/self.δ[n])*(x*pn - self.γ[n]*pn - (self.δ[n-1] if n>0 else 0)*pn_)
            pn_ = pn__

            y += c[n+1]*pn

        return y

    def moments(self,A,v,k):
        """
        computes modified moments of spectral density for (A,v) through degree k-1
        
        Parameters
        ----------
        A : (n,n) matrix-like
            matrix
        v : (n,) ndarray
            starting vector
        k : int 
            maximum iteratinos

        Returns
        -------
        ndarray
            modified moments of spectral density wrt. orthogonal polynomials
        
        """
        
        assert k<=self.s//2, f'degree {k-1} of requested moments exceeds degree {self.s//2} of recurrence'

        μ = np.full(k,np.nan)
        
        pn_ = np.zeros_like(v)
        pn = np.copy(v)
        μ[0] = v@pn
        
        for n in range(k-1):
            pn__ = np.copy(pn)
            pn = (1/self.δ[n])*(A@pn - self.γ[n]*pn - (self.δ[n-1] if n>0 else 0)*pn_)
            pn_ = pn__

            μ[n+1] = v@pn

        return μ

def combine_densities(densities):
    """
    Combine densities 
    
    Parameters
    ----------
    densities : list of reference_density
        list of densities to be added

    Returns
    -------
    reference_density
        resulting density
    """

    n_densities = len(densities)

    mass = sum([d.mass for d in densities])
    
    if n_densities == 1:
        return densities[0]
        
    def σ(x):
        y = np.zeros_like(x)
        for d in densities:
            y += d(x)
        return y / mass

    s = min([d.s for d in densities])

    γ_raw = np.zeros((n_densities,s//2))
    δ_raw = np.zeros((n_densities,s//2))
    v = np.zeros((n_densities,s//2))
    for i,d in enumerate(densities):
        γ_raw[i] = d.γ[:s//2]
        δ_raw[i,:-1] = d.δ[:s//2-1]

        v[i,0] = np.sqrt(d.mass) / mass
    
    γ_raw = γ_raw.reshape(-1)
    δ_raw = δ_raw.reshape(-1)[:-1]
    v = v.reshape(-1)

    A = sp.sparse.diags([γ_raw,δ_raw,δ_raw],[0,-1,1])

    γ,δ = lanczos(A,v,s//2,reorth=False)

    return reference_density(σ,(γ,δ),mass)


def get_arcsin_density(a,b,k=_default_degree):
    r"""
    Return reference_density object for arcsin distribution on \([a,b]\):

    \[\sigma(x) = \frac{1}{\pi} \frac{1}{\sqrt{(b-x)(x-a)}}\]

    Note: this is the unit-mass orthogonality measure for the Chebyshev polynomials of the first kind shifted and scaled to \([a,b]\).


    Parameters
    ----------
    a : float
    b : float
    k : int
        maximum number of coefficients

    Returns
    -------
    reference_density
        arcsin density
    """

    assert a<b, 'left endpoint a must be less than right endpoint b'
    
    def σ(x):
        supp = (x>a)*(x<b)
        return (1/np.pi)/(np.sqrt((b-x)*(x-a)*supp)+1*(1-supp))*supp

    γ = (a+b)/2 * np.ones(k)
    δ = (b-a)/2 * np.ones(k)/2
    δ[0] *= np.sqrt(2)

    return reference_density(σ,(γ,δ))

def get_semicircle_density(a,b,k=_default_degree):
    r"""
    return reference_density object for semicircle distribution on \([a,b]\)

    \[\sigma(x) = \frac{8}{\pi(b-a)^2} \sqrt{(b-x)(x-a)}\]

    Note: this is the unit-mass orthogonality measure for the Chebyshev polynomials of the second kind shifted and scaled to \([a,b]\).

    Parameters
    ----------
    a : float
    b : float
    k : int
        maximum number of coefficients

    Returns
    -------
    reference_density
        semicircle density
    """
    def σ(x):
        supp = (x>a)*(x<b)
        return 8/(np.pi*(b-a)**2)*(np.sqrt((b-x)*(x-a)*supp))

    γ = (a+b)/2 * np.ones(k)
    δ = (b-a)/2 * np.ones(k)/2
    
    return reference_density(σ,(γ,δ))

def get_uniform_density(a,b,k=_default_degree):
    r"""
    Return reference_density object for uniform distribution on \([a,b]\):

    \[\sigma(x) = \frac{1}{b-a} \]

    Note: this is the unit-mass orthogonality measure for the Legendre polynomials shifted and scaled to \([a,b]\).


    Parameters
    ----------
    a : float
    b : float
    k : int
        maximum number of coefficients

    Returns
    -------
    reference_density
        arcsin density
    """

    assert a<b, 'left endpoint a must be less than right endpoint b'
    
    def σ(x):
        supp = (x>a)*(x<b)
        return supp/(b-a)

    n = np.arange(k)
    γ = (a+b)/2 * np.ones(k)
    δ = (b-a)/2 * (n+1) / (np.sqrt(2*n+3) * np.sqrt(2*n+1))
    δ[0] = (b-a)/2 * np.sqrt(1/3)

    return reference_density(σ,(γ,δ))
