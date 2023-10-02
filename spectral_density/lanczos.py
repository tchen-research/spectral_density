import numpy as np

def lanczos(A,v,k,reorth=True):
    """
    Lanczos algorithm

    Parameters
    ----------
    A : (n,n) matrix-like
        matrix
    v : (n,) ndarray
        starting vector
    k : int 
                maximum iterations
    reoth : bool, default=True
                reorthogonalize or not

    Returns
    -------
    α : (k,) ndarray
        recurrence coefficients
    β : (k,) ndarray
        recurrence coefficients
    """
    
    n = len(v)
    
    α = np.zeros(k,dtype=np.float64)
    β = np.zeros(k,dtype=np.float64)
    if reorth:
        Q = np.zeros((n,k+1),dtype=np.float64)
    
    q = v / np.sqrt(v.T@v)
    if reorth:
        Q[:,0] = q
    for i in range(0,k):

        q__ = np.copy(q)
        q = A@q - β[i-1]*q_ if i>0 else A@q
        q_ = q__
        
        α[i] = q@q_
        q -= α[i]*q_

        # double Gram-Schmidt reorthogonalization
        if reorth:
            q -= Q@(Q.T@q) 
            q -= Q@(Q.T@q) 
        
        β[i] = np.sqrt(q.T@q)
        q /= β[i]

        if reorth:  
            Q[:,i+1] = q
    
    else:
        return (α,β)