import numpy as np
import scipy as sp

class KPM:

    def __init__(self,αβ_list,σ):
        r"""        
        Parameters
        ----------
        αβ_list : list of tuple of ndarray
            Lanczos recurrence coeffs from m runs
        σ : reference_density
            reference density for KPM approximation
        """

        self.σ = σ
        
        self.m = len(αβ_list)

        k_list = [min(len(α),len(β)+1) for α,β in αβ_list]
        k = min(k_list)
        self.s = 2*k-1

        self.μ = np.full((self.m,self.s),np.nan)
        for i,(α,β) in enumerate(αβ_list):
            T = sp.sparse.diags([α,β[:k-1],β[:k-1]],[0,1,-1])
            e1 = np.zeros(k)
            e1[0] = 1
            self.μ[i] = σ.moments(T,e1,self.s)


    def _jackson_weights(self,k):
        return (1/(k+1))*((k-np.arange(k)+1)*np.cos(np.pi*np.arange(k)/(k+1))+np.sin(np.pi*np.arange(k)/(k+1))/np.tan(np.pi/(k+1)))

        
    def __call__(self,x,degree=None,damping=''):
        """
        evaluate KPM approximation of specified degree

        Parameters
        -----
        x : float or ndarray
            points at which to evaluate KPM approximation
        degree : int
            maximum degree of approximation, defaults to maximum possible degree
        damping : {'','Jackson'}, optional
            damping type to use. Jackson's damping only gurantees positivity for arcsin densities

        Returns
        -------
        float or ndarray
            KPM approximation at x
        
        """
        
        if degree is None:
            degree = self.s
        assert degree <= self.s, 'approximation degree exceeds maximum possible degree'

        damp_coeff = np.ones(degree)
        if damping == 'Jackson':
            damp_coeff = self._jackson_weights(degree)
            
        μ_ave = damp_coeff*np.mean(self.μ[:,:degree],axis=0)
        
        return self.σ(x)*self.σ.series(x,μ_ave)


    def var(self,x,degree=None,damping=None):
        """
        evaluate jackknife variance of KPM approximation of specified degree

        Warning: the jackknife error estimate is an estimate of the error due to the statistical noise in in the Lanczos recurrence coefficients. 
        This is NOT AN ESTIMATE of the overall error in the algorithm.
        
        Parameters
        -----
        x : float or ndarray
            points at which to evaluate KPM approximation
        degree : int
            maximum degree of approximation, defaults to maximum possible degree
        damping : TODO

        Returns
        -------
        float or ndarray
            jackknife variance estimate for KPM approximation at x
        
        """
        
        if degree is None:
            degree = self.s
        assert degree <= self.s, 'approxiamtion degree exceeds maximum possible degree'

        damp_coeff = np.ones(degree)
        if damping == 'Jackson':
            damp_coeff = self._jackson_weights(degree)

        LOO_est = np.tile(np.zeros_like(x),(self.m,1))
        for i in range(self.m):
            LOO_est[i] = self.σ(x)*self.σ.series(x,damp_coeff*self.μ[i,:degree])

        return np.var(LOO_est,axis=0)/(self.m-1)

    