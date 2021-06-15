import numpy as np
import itertools
import operator

from numpy.linalg import norm as Norm
from numpy.linalg import solve as Solve
from scipy.linalg import block_diag

"""
A few functions used in parametric PDE-FIND

Samuel Rudy.  2018
"""

def FiniteDiff(u, dx, d):
    """
    Takes dth derivative data using 2nd order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3
    
    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    """
    
    n = u.size
    ux = np.zeros(n, dtype=u.dtype)
    
    if d == 1:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-u[i-1]) / (2*dx)
        
        ux[0] = (-3.0/2*u[0] + 2*u[1] - u[2]/2) / dx
        ux[n-1] = (3.0/2*u[n-1] - 2*u[n-2] + u[n-3]/2) / dx
        return ux
    
    if d == 2:
        for i in range(1,n-1):
            ux[i] = (u[i+1]-2*u[i]+u[i-1]) / dx**2
        
        ux[0] = (2*u[0] - 5*u[1] + 4*u[2] - u[3]) / dx**2
        ux[n-1] = (2*u[n-1] - 5*u[n-2] + 4*u[n-3] - u[n-4]) / dx**2
        return ux
    
    if d == 3:
        for i in range(2,n-2):
            ux[i] = (u[i+2]/2-u[i+1]+u[i-1]-u[i-2]/2) / dx**3
        
        ux[0] = (-2.5*u[0]+9*u[1]-12*u[2]+7*u[3]-1.5*u[4]) / dx**3
        ux[1] = (-2.5*u[1]+9*u[2]-12*u[3]+7*u[4]-1.5*u[5]) / dx**3
        ux[n-1] = (2.5*u[n-1]-9*u[n-2]+12*u[n-3]-7*u[n-4]+1.5*u[n-5]) / dx**3
        ux[n-2] = (2.5*u[n-2]-9*u[n-3]+12*u[n-4]-7*u[n-5]+1.5*u[n-6]) / dx**3
        return ux
    
    if d > 3:
        return FiniteDiff(FiniteDiff(u,dx,3), dx, d-3)

def PolyDiff(u, x, deg = 3, diff = 1, width = 5):
    
    """
    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    width = width of window to fit to polynomial

    This throws out the data close to the edges since the polynomial derivative only works
    well when we're looking at the middle of the points fit.
    """

    u = u.flatten()
    x = x.flatten()

    n = len(x)
    du = np.zeros((n - 2*width,diff))

    # Take the derivatives in the center of the domain
    for j in range(width, n-width):

        points = np.arange(j - width, j + width)

        # Fit to a polynomial
        poly = np.polynomial.chebyshev.Chebyshev.fit(x[points],u[points],deg)

        # Take derivatives
        for d in range(1,diff+1):
            du[j-width, d-1] = poly.deriv(m=d)(x[j])

    return du

def PolyDiffPoint(u, x, deg = 3, diff = 1, index = None):
    
    """
    Same as above but now just looking at a single point

    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    """
    
    n = len(x)
    if index == None: index = int((n-1)/2)

    # Fit to a polynomial
    poly = np.polynomial.chebyshev.Chebyshev.fit(x,u,deg)
    
    # Take derivatives
    derivatives = []
    for d in range(1,diff+1):
        derivatives.append(poly.deriv(m=d)(x[index]))
        
    return derivatives

##################################################################################
##################################################################################
#
# Functions specific to PDE-FIND
#               
##################################################################################
##################################################################################

def build_Theta(data, derivatives, derivatives_description, P, data_description = None):
    """
    builds a matrix with columns representing polynoimials up to degree P of all variables

    This is used when we subsample and take all the derivatives point by point or if there is an 
    extra input (Q in the paper) to put in.

    input:
        data: column 0 is U, and columns 1:end are Q
        derivatives: a bunch of derivatives of U and maybe Q, should start with a column of ones
        derivatives_description: description of what derivatives have been passed in
        P: max power of polynomial function of U to be included in Theta

    returns:
        Theta = Theta(U,Q)
        descr = description of what all the columns in Theta are
    """
    
    n,d = data.shape
    m, d2 = derivatives.shape
    if n != m: raise Exception('dimension error')
    if data_description is not None: 
        if len(data_description) != d: raise Exception('data descrption error')
    
    # Create a list of all polynomials in d variables up to degree P
    rhs_functions = {}
    f = lambda x, y : np.prod(np.power(list(x), list(y)))
    powers = []            
    for p in range(1,P+1):
            size = d + p - 1
            for indices in itertools.combinations(range(size), d-1):
                starts = [0] + [index+1 for index in indices]
                stops = indices + (size,)
                powers.append(tuple(map(operator.sub, stops, starts)))
    for power in powers: rhs_functions[power] = [lambda x, y = power: f(x,y), power]

    # First column of Theta is just ones.
    Theta = np.ones((n,1), dtype=data.dtype)
    descr = ['']
    
    # Add the derivaitves onto Theta
    for D in range(1,derivatives.shape[1]):
        Theta = np.hstack([Theta, derivatives[:,D].reshape(n,1)])
        descr.append(derivatives_description[D])
        
    # Add on derivatives times polynomials
    for D in range(derivatives.shape[1]):
        for k in rhs_functions.keys():
            func = rhs_functions[k][0]
            new_column = np.zeros((n,1), dtype=data.dtype)
            for i in range(n):
                new_column[i] = func(data[i,:])*derivatives[i,D]
            Theta = np.hstack([Theta, new_column])
            if data_description is None: descr.append(str(rhs_functions[k][1]) + derivatives_description[D])
            else:
                function_description = ''
                for j in range(d):
                    if rhs_functions[k][1][j] != 0:
                        if rhs_functions[k][1][j] == 1:
                            function_description = function_description + data_description[j]
                        else:
                            function_description = function_description + data_description[j] + '^' + str(rhs_functions[k][1][j])
                descr.append(function_description + derivatives_description[D])

    return Theta, descr

def build_linear_system(u, dt, dx, D = 3, P = 3,time_diff = 'poly',space_diff = 'poly',lam_t = None,lam_x = None, width_x = None,width_t = None, deg_x = 5,deg_t = None,sigma = 2):
    """
    Constructs a large linear system to use in later regression for finding PDE.  
    This function works when we are not subsampling the data or adding in any forcing.

    Input:
        Required:
            u = data to be fit to a pde
            dt = temporal grid spacing
            dx = spatial grid spacing
        Optional:
            D = max derivative to include in rhs (default = 3)
            P = max power of u to include in rhs (default = 3)
            time_diff = method for taking time derivative
                        options = 'poly', 'FD', 'FDconv','TV'
                        'poly' (default) = interpolation with polynomial 
                        'FD' = standard finite differences
                        'FDconv' = finite differences with convolutional smoothing 
                                   before and after along x-axis at each timestep
                        'Tik' = Tikhonov (not recommended for short simulations)
            space_diff = same as time_diff with added option, 'Fourier' = differentiation via FFT
            lam_t = penalization for L2 norm of second time derivative
                    only applies if time_diff = 'TV'
                    default = 1.0/(number of timesteps)
            lam_x = penalization for L2 norm of (n+1)st spatial derivative
                    default = 1.0/(number of gridpoints)
            width_x = number of points to use in polynomial interpolation for x derivatives
                      or width of convolutional smoother in x direction if using FDconv
            width_t = number of points to use in polynomial interpolation for t derivatives
            deg_x = degree of polynomial to differentiate x
            deg_t = degree of polynomial to differentiate t
            sigma = standard deviation of gaussian smoother
                    only applies if time_diff = 'FDconv'
                    default = 2
    Output:
        ut = column vector of length u.size
        R = matrix with ((D+1)*(P+1)) of column, each as large as ut
        rhs_description = description of what each column in R is
    """

    n, m = u.shape

    if width_x == None: width_x = n/10
    if width_t == None: width_t = m/10
    if deg_t == None: deg_t = deg_x

    # If we're using polynomials to take derviatives, then we toss the data around the edges.
    if time_diff == 'poly': 
        m2 = m-2*width_t
        offset_t = width_t
    else: 
        m2 = m
        offset_t = 0
    if space_diff == 'poly': 
        n2 = n-2*width_x
        offset_x = width_x
    else: 
        n2 = n
        offset_x = 0

    if lam_t == None: lam_t = 1.0/m
    if lam_x == None: lam_x = 1.0/n

    ########################
    # First take the time derivaitve for the left hand side of the equation
    ########################
    ut = np.zeros((n2,m2), dtype=u.dtype)

    if time_diff == 'FDconv':
        Usmooth = np.zeros((n,m), dtype=u.dtype)
        # Smooth across x cross-sections
        for j in range(m):
            Usmooth[:,j] = ConvSmoother(u[:,j],width_t,sigma)
        # Now take finite differences
        for i in range(n2):
            ut[i,:] = FiniteDiff(Usmooth[i + offset_x,:],dt,1)

    elif time_diff == 'poly':
        T= np.linspace(0,(m-1)*dt,m)
        for i in range(n2):
            ut[i,:] = PolyDiff(u[i+offset_x,:],T,diff=1,width=width_t,deg=deg_t)[:,0]

    elif time_diff == 'Tik':
        for i in range(n2):
            ut[i,:] = TikhonovDiff(u[i + offset_x,:], dt, lam_t)

    else:
        for i in range(n2):
            ut[i,:] = FiniteDiff(u[i + offset_x,:],dt,1)
    
    ut = np.reshape(ut, (n2*m2,1), order='F')

    ########################
    # Now form the rhs one column at a time, and record what each one is
    ########################

    u2 = u[offset_x:n-offset_x,offset_t:m-offset_t]
    Theta = np.zeros((n2*m2, (D+1)*(P+1)), dtype=u.dtype)
    ux = np.zeros((n2,m2), dtype=u.dtype)
    rhs_description = ['' for i in range((D+1)*(P+1))]

    if space_diff == 'poly': 
        Du = {}
        for i in range(m2):
            Du[i] = PolyDiff(u[:,i+offset_t],np.linspace(0,(n-1)*dx,n),diff=D,width=width_x,deg=deg_x)
    if space_diff == 'Fourier': ik = 2*np.pi*1j*np.fft.fftfreq(n, d = dx)
        
    for d in range(D+1):

        if d > 0:
            for i in range(m2):
                if space_diff == 'Tik': ux[:,i] = TikhonovDiff(u[:,i+offset_t], dx, lam_x, d=d)
                elif space_diff == 'FDconv':
                    Usmooth = ConvSmoother(u[:,i+offset_t],width_x,sigma)
                    ux[:,i] = FiniteDiff(Usmooth,dx,d)
                elif space_diff == 'FD': ux[:,i] = FiniteDiff(u[:,i+offset_t],dx,d)
                elif space_diff == 'poly': ux[:,i] = Du[i][:,d-1]
                elif space_diff == 'Fourier': ux[:,i] = np.fft.ifft(ik**d*np.fft.fft(u[:,i]))
        else: ux = np.ones((n2,m2), dtype=u.dtype) 
            
        for p in range(P+1):
            Theta[:, d*(P+1)+p] = np.reshape(np.multiply(ux, np.power(u2,p)), (n2*m2), order='F')

            if p == 1: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+'u'
            elif p>1: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+'u^' + str(p)
            if d > 0: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+\
                                                   'u_{' + ''.join(['x' for _ in range(d)]) + '}'

    return ut, Theta, rhs_description

def print_pde(w, rhs_description, ut = 'u_t'):
    pde = ut + ' = '
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + ' + '
            pde = pde + "(%05f %+05fi)" % (w[i].real, w[i].imag) + rhs_description[i] + "\n   "
            first = False
    print(pde)

##################################################################################
##################################################################################
#
# Functions for sparse regression.
#               
##################################################################################
##################################################################################


def Factor(A, rho):

    n,m = A.shape
    
    if n >= m:
        return np.linalg.cholesky(A.T.dot(A) + rho*np.eye(A.shape[1]))
    else:
        return np.linalg.cholesky(1/rho*A.dot(A.T) + np.eye(A.shape[0]))

def ObjectiveGroupLasso(A,b,lam,groups,x):
    """
    Evaluate objective function for group lasso
    """
    obj = 0.5*Norm(A.dot(x)-b)**2
    obj = obj + lam*np.sum([Norm(x[g]) for g in groups])
    return obj

def Shrinkage(x, kappa):

    return (1 - kappa/Norm(x)).clip(0)*x

def GroupLassoADMM(As, bs, lam, groups, rho, alpha, maxiter=1000, abstol=1e-4, reltol=1e-2):
    """
    Solver for group lasso via ADMM that has been taylored for problems with block diagonal design matrix
    passed in as a list of the blocks.  Assumes they all have the same size.
    
    Adapted from MatLab code found here:
    https://web.stanford.edu/~boyd/papers/admm/group_lasso/group_lasso.html
    
    Instead of passing in group sizing, pass in a list of groups, each being a list of columns in that group.
    i.e. for an 8 column matrix groups could be [[1,3,5],[2,4],[6,7,8]]
    """
    
    n,D = As[0].shape
    m = len(As)
    
    Atbs = [A.T.dot(b) for (A,b) in zip(As,bs)]
        
    Ls = [Factor(A,rho) for A in As]
    Us = [L.T for L in Ls]
    
    x = np.zeros((m*D,1))
    z = 1e-5*np.random.randn(m*D,1)
    u = 1e-5*np.random.randn(m*D,1)
    
    # Indices of x for each timestep. x[Ts[t]] is the coefficient vector for time t
    Ts = [j*D + np.arange(D) for j in range(m)]
    
    history = {}
    history['objval'] = []
    history['gl_objval'] = []
    history['r_norm'] = []
    history['s_norm'] = []
    history['eps_pri'] = []
    history['eps_dual'] = []
    
    for k in range(maxiter):
        
        # x update
        for j in range(m):
            q = Atbs[j] + rho*(z[Ts[j]]-u[Ts[j]])

            if n >= D:
                x[Ts[j]] = Solve(Us[j],Solve(Ls[j],q))
            else:
                x[Ts[j]] = q/rho-As[j].T.dot(Solve(Us[j],Solve(Ls[j],As[j].dot(q))))/rho**2
        
        # z update
        zold = np.copy(z)
        x_hat = alpha*x+(1-alpha)*zold
        for g in groups:
            z[g] = Shrinkage(x_hat[g]+u[g], lam/rho)
            
        u = u+(x_hat-z)
        
        # record history
        history['objval'].append(ObjectiveADMM(As,bs,Ts,lam,groups,x,z))
        history['gl_objval'].append(ObjectiveGLASSO_block(As,bs,Ts,lam,groups,x))
        history['r_norm'].append(Norm(x-z))
        history['s_norm'].append(Norm(rho*(z-zold)))
        history['eps_pri'].append(np.sqrt(m)*abstol+reltol*np.max([Norm(x),Norm(z)]))
        history['eps_dual'].append(np.sqrt(m)*abstol+reltol*Norm(rho*u))
        
        # check for termination
        if (history['r_norm'][-1] < history['eps_pri'][-1]) and \
           (history['s_norm'][-1] < history['eps_dual'][-1]):
            break
    
    # Return unbiased sparse predictor
    z = z.reshape(D,m, order = 'F')
    nz_coords = np.where(np.sum(abs(z), axis = 1) != 0)[0]
    if len(nz_coords) != 0: 
        for j in range(m):
            z[nz_coords,j] = np.linalg.lstsq(As[j][:, nz_coords], bs[j])[0][:,0]
    
    return z, history

def TrainGroupLasso(As, bs, groups, num_lambdas = 50, normalize=2):
    """
    Searches over values of lambda to find optimal performance using PDE_FIND_Loss.
    """

    np.random.seed(0) # for consistancy

    m = len(As)
    n,D = As[0].shape

    # Normalize
    if normalize != 0:

        # get norm of each column
        candidate_norms = np.zeros(D)
        for i in range(D):
            candidate_norms[i] = Norm(np.vstack(A[:,i] for A in As), normalize)

        norm_bs = [m*Norm(b, normalize) for b in bs]

        # normalize 
        for i in range(m):
            As[i] = As[i].dot(np.diag(candidate_norms**-1))
            bs[i] = bs[i]/norm_bs[i]

    # parameters for ADMM
    rho = 1e-3
    alpha = 1.5

    # Get array of lambdas to check
    # Looking at KKT conditions for group lasso, lambda higher than lambda_max will result in x=0
    # lambda_min is set arbitrailly to 1e-5 but if the optimal lambda turns out to be 0 or 1e-5, then one
    # could change this to check lower values
    lambda_max = np.max([np.sum([Norm(A[:,g].T.dot(b)) for (A,b) in zip(As,bs)]) for g in range(D)])
    lambda_min = 1e-5*lambda_max
    Lam = [0]+[np.exp(alpha) for alpha in np.linspace(np.log(lambda_min), np.log(lambda_max), num_lambdas)][:-1]

    # Test each value of lambda to find the best
    X = []
    Losses = []
    Histories = []

    for lam in Lam:
        x,history = GroupLassoADMM(As,bs,lam,groups,rho,alpha)
        X.append(x.reshape(D,m, order = 'F'))
        Losses.append(PDE_FIND_Loss(As,bs,x))
        Histories.append(history)

    if normalize != 0:
        for x in X:
            for i in range(D):
                for j in range(m):
                    x[i,j] = x[i,j]/candidate_norms[i]*norm_bs[j]
        for i in range(m):
            As[i] = As[i].dot(np.diag(candidate_norms))
            bs[i] = bs[i]*norm_bs[i]

    return X,Lam,Losses,Histories

def ObjectiveADMM(As, bs, Ts, lam, groups, x, z):
    """
    Evaluate group lasso objective function for ADMM
    """
    
    obj = 0
    for j in range(len(As)):
        obj = obj + 0.5*Norm(As[j].dot(x[Ts[j]])-bs[j])**2

    obj = obj + lam*np.sum([Norm(z[g]) for g in groups])
    return obj

def ObjectiveGLASSO_block(As, bs, Ts, lam, groups, x):
    """
    Evaluate group lasso objective function for ADMM
    """
    
    obj = 0
    for j in range(len(As)):
        obj = obj + 0.5*Norm(As[j].dot(x[Ts[j]])-bs[j])**2

    obj = obj + lam*np.sum([Norm(x[g]) for g in groups])
    return obj

def Ridge(A,b,lam):
    if lam != 0: return np.linalg.solve(A.T.dot(A)+lam*np.eye(A.shape[1]), A.T.dot(b))
    else: return np.linalg.lstsq(A, b)[0]
    
def SGTRidge(Xs, ys, tol, lam = 10**-5, maxit = 5, penalize_noise = False, verbose = False):
    """
    Sequential Threshold Group Ridge
    """
    
    # Make sure the inputs are sensible
    if len(Xs) != len(ys): raise Exception('Number of Xs and ys mismatch')
    if len(set([X.shape[1] for X in Xs])) != 1: 
        raise Exception('Number of coefficients inconsistent across timesteps')
        
    d = Xs[0].shape[1]
    m = len(Xs)
    
    # Get the standard ridge esitmate for each timestep
    W = np.hstack([Ridge(X,y,lam) for [X,y] in zip(Xs,ys)])
        
    num_relevant = d
    biginds = [i for i in range(d) if np.linalg.norm(W[i,:]) > tol]
    
    for j in range(maxit):
        
        # Figure out which items to cut out
        smallinds = [i for i in range(d) if np.linalg.norm(W[i,:]) < tol]
        new_biginds = [i for i in range(d) if i not in smallinds]
            
        # If nothing changes then stop
        if num_relevant == len(new_biginds): j = maxit-1
        else: num_relevant = len(new_biginds)
            
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0 and verbose: 
                print("Tolerance too high - all coefficients set below tolerance")
            break
        biginds = new_biginds
        
        # Otherwise get a new guess
        for i in smallinds:
            W[i,:] = np.zeros(m)
        if j != maxit -1:
            for i in range(m):
                W[biginds,i] = Ridge(Xs[i][:, biginds], ys[i], lam).reshape(len(biginds))
        else: 
            for i in range(m):
                W[biginds,i] = np.linalg.lstsq(Xs[i][:, biginds],ys[i])[0].reshape(len(biginds))
                
    return W

def PDE_FIND_Loss(As,bs,x,epsilon=1e-5):

    D,m = x.shape
    n,_ = As[0].shape
    N = n*m
    rss = np.sum([np.linalg.norm(bs[j] - As[j].dot(x[:,j].reshape(D,1)))**2 for j in range(m)])  
    k = np.count_nonzero(x)/m

    return N*np.log(rss/N+epsilon) + 2*k + (2*k**2+2*k)/(N-k-1)

def TrainSGTRidge(As, bs, num_tols = 50, lam = 1e-5, normalize = 2):
    """
    Searches over values of tol to find optimal performance according to PDE_FIND_Loss.
    """

    np.random.seed(0) # for consistancy

    m = len(As)
    n,D = As[0].shape
    
    # Normalize
    if normalize != 0:

        # get norm of each column
        candidate_norms = np.zeros(D)
        for i in range(D):
            candidate_norms[i] = Norm(np.vstack(A[:,i] for A in As), normalize)

        norm_bs = [m*Norm(b, normalize) for b in bs]

        # normalize 
        for i in range(m):
            As[i] = As[i].dot(np.diag(candidate_norms**-1))
            bs[i] = bs[i]/norm_bs[i]
    
    # Get array of tols to check
    x_ridge = np.hstack([Ridge(A,b,lam) for (A,b) in zip(As, bs)])
    max_tol = np.max([Norm(x_ridge[j,:]) for j in range(x_ridge.shape[0])])
    min_tol = np.min([Norm(x_ridge[j,:]) for j in range(x_ridge.shape[0])])
    Tol = [0]+[np.exp(alpha) for alpha in np.linspace(np.log(min_tol), np.log(max_tol), num_tols)][:-1]

    # Test each value of tol to find the best
    X = []
    Losses = []

    for tol in Tol:
        x = SGTRidge(As,bs,tol)
        X.append(x)
        Losses.append(PDE_FIND_Loss(As, bs, x))

    if normalize != 0:
        for x in X:
            for i in range(D):
                for j in range(m):
                    x[i,j] = x[i,j]/candidate_norms[i]*norm_bs[j]
        for i in range(m):
            As[i] = As[i].dot(np.diag(candidate_norms))
            bs[i] = bs[i]*norm_bs[i]
            
    return X,Tol,Losses













