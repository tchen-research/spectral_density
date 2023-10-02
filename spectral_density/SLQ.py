import numpy as np
import scipy as sp

class SLQ:

    def __init__(self,αβ_list):
        """
        Parameters
        ----------
        αβ_list : list of tuple of ndarray
            Lanczos recurrence coeffs from m runs
        """
        
        self.m = len(αβ_list)
        self.αβ_list = αβ_list

        k_list = [min(len(α),len(β)+1) for α,β in αβ_list]
        k = min(k_list)
        self.k = k

        θ_list = np.full((self.m,k),np.nan)
        w_list = np.full((self.m,k),np.nan)
        for i,(α,β) in enumerate(αβ_list):

            try:
                θ,S = sp.linalg.eigh_tridiagonal(α,β[:k-1])
            except:
                T = sp.sparse.diags([α,β[:k-1],β[:k-1]],[0,1,-1])
                θ,S = np.linalg.eigh(T.A)
                
            θ_list[i] = θ
            w_list[i] = S[0]**2
        
        self.θ = θ_list
        self.w = w_list

    def _K(self,d,width):
        """
        smoothing kernel
        """
        return np.exp(-(d/width)**2/2)/(width*np.sqrt(2*np.pi))

    def __call__(self,x,width=1,k=None):
        """
        evaluate SLQ approximation smoothed by convolving with a Gaussian kernel

        Parameters
        -----
        x : float or ndarray
            points at which to evaluate KPM approximation
        k : int
            maximum size of approximation, defaults to maximum possible degree
        width : float
            width parameter for convolution

        Returns
        -------
        float or ndarray
            smoothed SLQ approximation at x
        
        """

        if k is None:
            θ_list = self.θ
            w_list = self.w
        else:
            assert k <= self.k, 'requested degree exceeds maximum possible degree'
            θ_list = np.full((self.m,k),np.nan)
            w_list = np.full((self.m,k),np.nan)
            for i,(α,β) in enumerate(self.αβ_list):
    
                try:
                    θ,S = sp.linalg.eigh_tridiagonal(α[:k],β[:k-1])
                except:
                    T = sp.sparse.diags([α[:k],β[:k-1],β[:k-1]],[0,1,-1])
                    θ,S = np.linalg.eigh(T.A)
                θ_list[i] = θ
                w_list[i] = S[0]**2
            
        y = np.zeros_like(x)
        for θ,w in zip(θ_list.flatten(),w_list.flatten()):
            y += w*self._K(x-θ,width)
            
        return y/self.m

    def var(self,x,width=1,k=None):
        """
        evaluate jackknife variance of SLQ approximation smoothed by convolving with a Gaussian kernel

        Warning: the jackknife error estimate is an estimate of the error due to the statistical noise in in the Lanczos recurrence coefficients. 
        This is NOT AN ESTIMATE of the overall error in the algorithm.
        
        Parameters
        -----
        x : float or ndarray
            points at which to evaluate KPM approximation
        k : int
            maximum degree of approximation, defaults to maximum possible degree
        width : float
            width parameter for convolution

        Returns
        -------
        float or ndarray
            jackknife variance estimate for SLQ approximation at x
        
        """

        if k is None:
            θ_list = self.θ
            w_list = self.w
        else:
            assert k <= self.k, 'requested degree exceeds maximum possible degree'
            θ_list = np.full((self.m,k),np.nan)
            w_list = np.full((self.m,k),np.nan)
            for i,(α,β) in enumerate(self.αβ_list):
    
                try:
                    θ,S = sp.linalg.eigh_tridiagonal(α[:k],β[:k-1])
                except:
                    T = sp.sparse.diags([α[:k],β[:k-1],β[:k-1]],[0,1,-1])
                    θ,S = np.linalg.eigh(T.A)
                θ_list[i] = θ
                w_list[i] = S[0]**2
                
        LOO_est = np.tile(np.zeros_like(x),(self.m,1))
        for i in range(self.m):
            for θ,w in zip(θ_list[i],w_list[i]):
                LOO_est[i] += w*self._K(x-θ,width)

        return np.var(LOO_est,axis=0)/(self.m-1)

    def specsum(self,f,k=None):
        """
        evaluate SLQ spectral sum approximation

        Warning: the jackknife error estimate is an estimate of the error due to the statistical noise in in the Lanczos recurrence coefficients. 
        This is NOT AN ESTIMATE of the overall error in the algorithm.
        
        Parameters
        -----
        f : function

        Returns
        -------
        float
            SLQ approxiamtion to spectral sum
        float
            jackknife variance estimate
        
        """

        if k is None:
            θ_list = self.θ
            w_list = self.w
        else:
            assert k <= self.k, 'requested degree exceeds maximum possible degree'
            θ_list = np.full((self.m,k),np.nan)
            w_list = np.full((self.m,k),np.nan)
            for i,(α,β) in enumerate(self.αβ_list):
    
                try:
                    θ,S = sp.linalg.eigh_tridiagonal(α[:k],β[:k-1])
                except:
                    T = sp.sparse.diags([α[:k],β[:k-1],β[:k-1]],[0,1,-1])
                    θ,S = np.linalg.eigh(T.A)
                θ_list[i] = θ
                w_list[i] = S[0]**2
                
        LOO_est = np.full(self.m,np.nan)
        for i in range(self.m):
            LOO_est[i] = f(θ_list[i])@w_list[i]

        return np.mean(LOO_est),np.var(LOO_est)/(self.m-1)



