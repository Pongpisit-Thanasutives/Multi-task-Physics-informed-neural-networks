import numpy as np
from numpy import linalg as LA
from numpy.linalg import norm as Norm
from parametric_pde_diff import TrainSGTRidge
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse import dia_matrix
import itertools
import operator

"""
A few functions used in PDE-FIND

Samuel Rudy.  2016

"""


##################################################################################
##################################################################################
#
# Functions for taking derivatives.
# When in doubt / nice data ===> finite differences
#               \ noisy data ===> polynomials
#             
##################################################################################
##################################################################################

def TikhonovDiff(f, dx, lam, d = 1):
    """
    Tikhonov differentiation.

    return argmin_g \|Ag-f\|_2^2 + lam*\|Dg\|_2^2
    where A is trapezoidal integration and D is finite differences for first dervative

    It looks like it will work well and does for the ODE case but 
    tends to introduce too much bias to work well for PDEs.  If the data is noisy, try using
    polynomials instead.
    """

    # Initialize a few things    
    n = len(f)
    f = np.matrix(f - f[0]).reshape((n,1))

    # Get a trapezoidal approximation to an integral
    A = np.zeros((n,n))
    for i in range(1, n):
        A[i,i] = dx/2
        A[i,0] = dx/2
        for j in range(1,i): A[i,j] = dx
    
    e = np.ones(n-1)
    D = sparse.diags([e, -e], [1, 0], shape=(n-1, n)).todense() / dx
    
    # Invert to find derivative
    g = np.squeeze(np.asarray(np.linalg.lstsq(A.T.dot(A) + lam*D.T.dot(D),A.T.dot(f), rcond=-1)[0]))
    
    if d == 1: return g

    # If looking for a higher order derivative, this one should be smooth so now we can use finite differences
    else: return FiniteDiff(g, dx, d-1)
    
def FiniteDiff(u, dx, d):
    """
    Takes dth derivative data using 2nd order finite difference method (up to d=3)
    Works but with poor accuracy for d > 3
    
    Input:
    u = data to be differentiated
    dx = Grid spacing.  Assumes uniform spacing
    """
    
    n = u.size
    ux = np.zeros(n, dtype=np.complex64)
    
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
    
def ConvSmoother(x, p, sigma):
    """
    Smoother for noisy data
    
    Inpute = x, p, sigma
    x = one dimensional series to be smoothed
    p = width of smoother
    sigma = standard deviation of gaussian smoothing kernel
    """
    
    n = len(x)
    y = np.zeros(n, dtype=np.complex64)
    g = np.exp(-np.power(np.linspace(-p,p,2*p),2)/(2.0*sigma**2))

    for i in range(n):
        a = max([i-p,0])
        b = min([i+p,n])
        c = max([0, p-i])
        d = min([2*p,p+n-i])
        y[i] = np.sum(np.multiply(x[a:b], g[c:d]))/np.sum(g[c:d])
        
    return y

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

        # Fit to a Chebyshev polynomial
        # this is the same as any polynomial since we're on a fixed grid but it's better conditioned :)
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
    if index == None: index = (n-1)/2

    # Fit to a Chebyshev polynomial
    # better conditioned than normal polynomials
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
    Theta = np.ones((n,1), dtype=np.complex64)
    descr = ['']
    
    # Add the derivaitves onto Theta
    for D in range(1,derivatives.shape[1]):
        Theta = np.hstack([Theta, derivatives[:,D].reshape(n,1)])
        descr.append(derivatives_description[D])
        
    # Add on derivatives times polynomials
    for D in range(derivatives.shape[1]):
        for k in rhs_functions.keys():
            func = rhs_functions[k][0]
            new_column = np.zeros((n,1), dtype=np.complex64)
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
                        'Tik' = Tikhonov (takes very long time)
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
    ut = np.zeros((n2,m2), dtype=np.complex64)

    if time_diff == 'FDconv':
        Usmooth = np.zeros((n,m), dtype=np.complex64)
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
    Theta = np.zeros((n2*m2, (D+1)*(P+1)), dtype=np.complex64)
    ux = np.zeros((n2,m2), dtype=np.complex64)
    rhs_description = ['' for i in range((D+1)*(P+1))]

    if space_diff == 'poly': 
        Du = {}
        for i in range(m2):
            Du[i] = PolyDiff(u[:,i+offset_t],np.linspace(0,(n-1)*dx,n),diff=D,width=width_x,deg=deg_x)
    if space_diff == 'Fourier': ik = 1j*np.fft.fftfreq(n)*n
        
    for d in range(D+1):

        if d > 0:
            for i in range(m2):
                if space_diff == 'Tik': ux[:,i] = TikhonovDiff(u[:,i+offset_t], dx, lam_x, d=d)
                elif space_diff == 'FDconv':
                    Usmooth = ConvSmoother(u[:,i+offset_t],width_x,sigma)
                    ux[:,i] = FiniteDiff(Usmooth,dx,d)
                elif space_diff == 'FD': ux[:,i] = FiniteDiff(u[:,i+offset_t],dx,d)
                elif space_diff == 'poly': ux[:,i] = Du[i][:,d-1]
                elif space_diff == 'Fourier': ux[:,i] = np.fft.ifft(ik**d*np.fft.fft(ux[:,i]))
        else: ux = np.ones((n2,m2), dtype=np.complex64) 
            
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
# Functions for Schrodinger equation.
#               
##################################################################################
##################################################################################

def build_schsystem(u, n, m, potential, dt, dx):
    # Build Library of Candidate Terms
    ut = np.zeros((m,n), dtype=np.complex64)
    ux = np.zeros((m,n), dtype=np.complex64)
    uxx = np.zeros((m,n), dtype=np.complex64)
    uxxx = np.zeros((m,n), dtype=np.complex64)

    for i in range(n):
       ut[:,i] = FiniteDiff(u[:,i], dt, 1)
    for i in range(m):
       ux[i,:] = FiniteDiff(u[i,:], dx, 1)
       uxx[i,:] = FiniteDiff(u[i,:], dx, 2)
       uxxx[i,:] = FiniteDiff(u[i,:], dx, 3)
    
    ut = np.reshape(ut, (n*m,1), order='F')
    ux = np.reshape(ux, (n*m,1), order='F')
    uxx = np.reshape(uxx, (n*m,1), order='F')
    uxxx = np.reshape(uxxx, (n*m,1), order='F')
    X_ders = np.hstack([np.ones((n*m,1)),ux,uxx,uxxx])
    X_data = np.hstack([np.reshape(u, (n*m,1), order='F'), 
                    np.reshape(abs(u), (n*m,1), order='F'), 
                    np.reshape(potential, (n*m,1), order='F')])
    derivatives_description = ['','u_{x}','u_{xx}', 'u_{xxx}']

    X, descr = build_Theta(X_data, X_ders, derivatives_description, 2, data_description = ['u','|u|','V'])

    return X, ut, descr

def build_noise_schsystem(un, n, m, potential, dt, dx):
    width_x = 10
    width_t = 10
    deg = 6

    m2 = m-2*width_t
    n2 = n-2*width_x

    utn = np.zeros((m2,n2), dtype=np.complex64)
    uxn = np.zeros((m2,n2), dtype=np.complex64)
    uxxn = np.zeros((m2,n2), dtype=np.complex64)
    uxxxn = np.zeros((m2,n2), dtype=np.complex64)

    for i in range(n2):
       utn[:,i] = PolyDiff(np.real(un[:,i+width_x]), dt*np.arange(m), deg = deg, width = width_t)[:,0]
       utn[:,i] = utn[:,i]+1j*PolyDiff(np.imag(un[:,i+width_x]), dt*np.arange(m), deg = deg, width = width_t)[:,0]

    for i in range(m2):
       x_derivatives = PolyDiff(np.real(un[i+width_t,:]), dx*np.arange(n), deg = deg, diff = 3, width = width_x)
       x_derivatives = x_derivatives+1j*PolyDiff(np.imag(un[i+width_t,:]), dx*np.arange(n), deg = deg, diff = 3, width = width_x)
       uxn[i,:] = x_derivatives[:,0]
       uxxn[i,:] = x_derivatives[:,1]
       uxxxn[i,:] = x_derivatives[:,2]

    utn = np.reshape(utn, (n2*m2,1), order='F')
    uxn = np.reshape(uxn, (n2*m2,1), order='F')
    uxxn = np.reshape(uxxn, (n2*m2,1), order='F')
    uxxxn = np.reshape(uxxxn, (n2*m2,1), order='F')
    Xn_ders = np.hstack([np.ones((n2*m2,1)),uxn,uxxn,uxxxn])
    Xn_data = np.hstack([np.reshape(un[width_t:m-width_t,width_x:n-width_x], (n2*m2,1), order='F'),
                     np.reshape(abs(un[width_t:m-width_t,width_x:n-width_x]), (n2*m2,1), order='F'),
                     np.reshape(potential[width_t:m-width_t,width_x:n-width_x], (n2*m2,1), order='F')])
    derivatives_description = ['','u_{x}','u_{xx}', 'u_{xxx}']


    Xn, descr= build_Theta(Xn_data, Xn_ders, derivatives_description, 2)

    return Xn, utn, descr



###########################################################################################################
#################for NLS equation

def build_nlssystem(u, n, m, dt, dx):

    ut = np.zeros((m,n), dtype=np.complex64)
    ux = np.zeros((m,n), dtype=np.complex64)
    uxx = np.zeros((m,n), dtype=np.complex64)
    uxxx = np.zeros((m,n), dtype=np.complex64)

    for i in range(n):
       ut[:,i] = FiniteDiff(u[:,i], dt, 1)
    for i in range(m):
       ux[i,:] = FiniteDiff(u[i,:], dx, 1)
       uxx[i,:] = FiniteDiff(u[i,:], dx, 2)
       uxxx[i,:] = FiniteDiff(u[i,:], dx, 3)
    
    ut = np.reshape(ut, (n*m,1), order='F')
    ux = np.reshape(ux, (n*m,1), order='F')
    uxx = np.reshape(uxx, (n*m,1), order='F')
    uxxx = np.reshape(uxxx, (n*m,1), order='F')
    X_ders = np.hstack([np.ones((n*m,1)),ux,uxx,uxxx])
    X_data = np.hstack([np.reshape(u, (n*m,1), order='F'), np.reshape(abs(u), (n*m,1), order='F')])
    derivatives_description = ['','u_{x}','u_{xx}', 'u_{xxx}']

    X, rhs_des = build_Theta(X_data, X_ders, derivatives_description, 3, data_description = ['u','|u|'])

    return ut, X, rhs_des

def build_noise_nlssystem(un, n, m, dt, dx):
    width_x = 10
    width_t = 10
    deg = 4

    m2 = m-2*width_t
    n2 = n-2*width_x

    utn = np.zeros((m2,n2), dtype=np.complex64)
    uxn = np.zeros((m2,n2), dtype=np.complex64)
    uxxn = np.zeros((m2,n2), dtype=np.complex64)
    uxxxn = np.zeros((m2,n2), dtype=np.complex64)

    for i in range(n2):
       utn[:,i] = PolyDiff(np.real(un[:,i+width_x]), dt*np.arange(m), deg = deg, width = width_t)[:,0]
       utn[:,i] = utn[:,i]+1j*PolyDiff(np.imag(un[:,i+width_x]), dt*np.arange(m), deg = deg, width = width_t)[:,0]

    for i in range(m2):
       x_derivatives = PolyDiff(np.real(un[i+width_t,:]), dx*np.arange(n), deg = deg, diff = 3, width = width_x)
       x_derivatives = x_derivatives+1j*PolyDiff(np.imag(un[i+width_t,:]), dx*np.arange(n), deg = deg, diff = 3, width = width_x)
       uxn[i,:] = x_derivatives[:,0]
       uxxn[i,:] = x_derivatives[:,1]
       uxxxn[i,:] = x_derivatives[:,2]

    utn = np.reshape(utn, (n2*m2,1), order='F')
    uxn = np.reshape(uxn, (n2*m2,1), order='F')
    uxxn = np.reshape(uxxn, (n2*m2,1), order='F')
    uxxxn = np.reshape(uxxxn, (n2*m2,1), order='F')
    Xn_ders = np.hstack([np.ones((n2*m2,1)),uxn,uxxn,uxxxn])
    Xn_data = np.hstack([np.reshape(un[width_t:m-width_t,width_x:n-width_x], (n2*m2,1), order='F'),
                     np.reshape(abs(un[width_t:m-width_t,width_x:n-width_x]), (n2*m2,1), order='F')])
    derivatives_description = ['','u_{x}','u_{xx}', 'u_{xxx}']

    Xn, rhs_des = build_Theta(Xn_data, Xn_ders, derivatives_description, 3, data_description = ['u','|u|'])

    return utn, Xn, rhs_des
##################################################################################
##################################################################################
#
# Functions for sparse regression.
#               
##################################################################################
##################################################################################
# Some additional docs
# The data could be in the shape of (time_dims, spatial_dims) or (spatial_dims, time_dims)
# Choose one and use it consistently in your code as well
def RobustPCA(U, lam_2 = 1e-3):
    print("Please ensure that the shape of U is correct.")
    Y1         = U
    norm_two  = np.linalg.norm(Y1.ravel(), 2)
    norm_inf  = np.linalg.norm(Y1.ravel(), np.inf) / lam_2
    dual_norm = np.max([norm_two, norm_inf])
    Y1        = Y1 /dual_norm
    Z         = np.zeros(Y1.shape)
    E         = np.zeros(Y1.shape)
    dnorm     = np.linalg.norm(Y1, 'fro')
    eta1       = 1.25 / norm_two
    rho       = 1.1
    sv        = 30.
    n         = Y1.shape[1]
    iter_print = 50
    tol       = 1e-5
    maxIter   = 1e4
    iter      = 0
    err       = 10**5
  
    while iter< maxIter and err > tol:
        iter += 1
        # update E
        tempE = U - Z + (1/eta1) * Y1
        E     = shrink(tempE, lam_2 / eta1)
        # update Z
        tempZ = U - E + (1/eta1) * Y1
        Z, nc_norm = pcasvd_threshold(tempZ, eta1, n, sv)
        # update Lafrange multiplier Y1 and eta1  
        Err   = U - Z - E
        Y1    = Y1 + eta1 * Err
        eta1   = np.min([eta1 * rho, 1e8])
        err   = np.linalg.norm(Err, 'fro')/dnorm

        if (iter % iter_print) == 0 or iter == 1 or iter > maxIter or err < tol:
           print('iteration:{0}, err:{1}, nc_norm:{2} eta1:{3}'.format(iter, err, nc_norm, eta1))

    return Z, E


def DLrSR(R, Ut, xi_true, lam_1 = 1e-5, lam_3 = 0.1, lam_4 = 1e-5, d_tol = 30):
    nx, nt    = Ut.shape[0], Ut.shape[1]
    # for robust low-rank PCA
    Y2        = Ut
    norm_two  = np.linalg.norm(Y2.ravel(), 2)
    norm_inf  = np.linalg.norm(Y2.ravel(), np.inf) / lam_3
    dual_norm = np.max([norm_two, norm_inf])
    Y2        = Y2 /dual_norm
    X         = np.zeros(Y2.shape)
    E2        = np.zeros(Y2.shape)
    Rxmatrix  = np.zeros(Y2.shape)
    #x         = np.zeros((R.shape[1],1))
    dnorm     = np.linalg.norm(Y2, 'fro')

    eta2       = 1.25 / norm_two
    rho       = 1.2
    sv        = 30.
    n         = Y2.shape[1]
    iter_print = 50
    tol       = 1e-5
    maxIter   = 1e4
    iter      = 0
    err       = 10**5

    start_num = 201 
  
    while iter< maxIter and err > tol:
        iter += 1
        if iter < start_num:
           # update E
           tempE2 = Ut - X + (1/eta2) * Y2
           E2  = shrink(tempE2, lam_3 / eta2)
           # update A
           tempX = Ut - E2 + (1/eta2) * Y2
           X, nc_norm = pcasvd_threshold(tempX, eta2, n, sv)  
        else: 
           # update x
           tempX = (eta2/(lam_4+eta2))*(Ut - E2 + (1 / eta2) * Y2) + (lam_4/(lam_4+eta2))*Rxmatrix
           X, nc_norm = pcasvd_threshold(tempX, eta2+lam_4, n, sv)
           vectorX    = np.reshape(X,(nx*nt,1))
           X_grouped = [vectorX[j*nx:(j+1)*nx] for j in range(nt)]
           Xi,Tol,Losses =  TrainSGTRidge(R,X_grouped,lam_1, d_tol)
           xi = Xi[np.argmin(Losses)]
           tempX = np.hstack([tempR.dot(tempxi) for [tempR,tempxi] in zip(R,xi.T)])
           X     = np.reshape(tempX,(nx,nt))
           #vX_grouped = [vectorX[j*nx:(j+1)*nx] for j in range(nt)]
        # update Lafrange multiplier Q and eta  
        Err = Ut - X - E2
        Y2 = Y2 + eta2 * Err
        eta2 = np.min([eta2 * rho, 1e7])
        err = np.linalg.norm(Err, 'fro')/dnorm

        if (iter % iter_print) == 0 or iter == 1 or iter > maxIter or err < tol:
           print('iteration:{0}, err:{1}, nc_norm:{2} eta2:{3}'.format(iter, err, nc_norm, eta2))

    if iter < start_num:
       print("IALM Finished at iteration %d" % (iter))
       vectorX    = np.reshape(X,(nx*nt,1))
       X_grouped = [vectorX[j*nx:(j+1)*nx] for j in range(nt)]
       Xi,Tol,Losses =  TrainSGTRidge(R,X_grouped,lam_1, d_tol)
       xi = Xi[np.argmin(Losses)] 

    return xi, X, E2


def Robust_LRSTR(R, Ut, rhs_des, lam_1 = 1e-5, lam_3 = 0.1, lam_4 = 1e-5, d_tol = 30):
    Ut = np.reshape(Ut, R.shape)
    nx, nt    = Ut.shape[0], Ut.shape[1]
    # for robust low-rank PCA
    Y2        = Ut
    norm_two  = np.linalg.norm(Y2.ravel(), 2)
    norm_inf  = np.linalg.norm(Y2.ravel(), np.inf) / lam_3
    dual_norm = np.max([norm_two, norm_inf])
    Y2        = Y2 /dual_norm
    X         = np.zeros(Y2.shape)
    E2        = np.zeros(Y2.shape)
    Rxmatrix  = np.zeros(Y2.shape)
    x         = np.zeros((R.shape[1],1))
    dnorm     = np.linalg.norm(Y2, 'fro')

    eta2       = 1.25 / norm_two
    rho       = 1.2
    sv        = 30.
    n         = Y2.shape[1]
    iter_print = 50
    tol       = 1e-5
    maxIter   = 1e4
    iter      = 0
    err       = 10**5

    start_num = 20 
  
    while iter< maxIter and err > tol:
        iter += 1
        if iter < start_num:
           # update E
           tempE2 = Ut - X + (1/eta2) * Y2
           E2  = shrink(tempE2, lam_3 / eta2)
           # update A
           tempX = Ut - E2 + (1/eta2) * Y2
           X, nc_norm = pcasvd_threshold(tempX, eta2, n, sv)  
        else: 
           # update x
           tempX = (eta2/(lam_4+eta2))*(Ut - E2 + (1 / eta2) * Y2) + (lam_4/(lam_4+eta2))*Rxmatrix
           X, nc_norm = pcasvd_threshold(tempX, eta2+lam_4, n, sv)
           vectorX    = np.reshape(X,(nx*nt,1))
           x        = TrainSTRidge(R, vectorX, lam_1, d_tol)  
           Rxmatrix = np.reshape(R.dot(x), (nx,nt))
           print_pde(x, rhs_des)

        # update Lafrange multiplier Q and eta  
        Err = Ut - X - E2
        Y2 = Y2 + eta2 * Err
        eta2 = np.min([eta2 * rho, 1e7])
        err = np.linalg.norm(Err, 'fro')/dnorm

        if (iter % iter_print) == 0 or iter == 1 or iter > maxIter or err < tol:
           print('iteration:{0}, err:{1}, nc_norm:{2} eta2:{3}'.format(iter, err, nc_norm, eta2))
           print_pde(x, rhs_des)
    if iter < start_num:
       print("IALM Finished at iteration %d" % (iter))
       vectorX   = np.reshape(X,(nx*nt,1))
       x         = TrainSTRidge(R, vectorX, lam_1, d_tol)  
       print_pde(x, rhs_des)

    return x, X, E2

######################################################################################################
def DLrSR_para(R, Ut, rhs_des, lam_1 = 1e-5, lam_3 = 0.1, lam_4 = 1e-5, d_tol = 30):
    nx, nt    = Ut.shape[0], Ut.shape[1]
    # for robust low-rank PCA
    Y2        = Ut
    norm_two  = np.linalg.norm(Y2.ravel(), 2)
    norm_inf  = np.linalg.norm(Y2.ravel(), np.inf) / lam_3
    dual_norm = np.max([norm_two, norm_inf])
    Y2        = Y2 /dual_norm
    X         = np.zeros(Y2.shape)
    E2        = np.zeros(Y2.shape)
    Rxmatrix  = np.zeros(Y2.shape)
    x         = np.zeros((R.shape[1],1))
    dnorm     = np.linalg.norm(Y2, 'fro')

    eta2       = 1.25 / norm_two
    rho       = 1.2
    sv        = 30.
    n         = Y2.shape[1]
    iter_print = 50
    tol       = 1e-5
    maxIter   = 1e4
    iter      = 0
    err       = 10**5

    start_num = 20 
    allerr = []
    allnc_norm = []
    while iter< maxIter and err > tol:
        iter += 1
        if iter < start_num:
           # update E
           tempE2 = Ut - X + (1/eta2) * Y2
           E2  = shrink(tempE2, lam_3 / eta2)
           # update A
           tempX = Ut - E2 + (1/eta2) * Y2
           X, nc_norm = pcasvd_threshold(tempX, eta2, n, sv)  
        else: 
           # update x
           tempX = (eta2/(lam_4+eta2))*(Ut - E2 + (1 / eta2) * Y2) + (lam_4/(lam_4+eta2))*Rxmatrix
           X, nc_norm = pcasvd_threshold(tempX, eta2+lam_4, n, sv)
           vectorX    = np.reshape(X,(nx*nt,1))
           x        = TrainSTRidge(R, vectorX, lam_1, d_tol)  
           Rxmatrix = np.reshape(R.dot(x), (nx,nt))
           print_pde(x, rhs_des)

        # update Lafrange multiplier Q and eta  
        Err = Ut - X - E2    
        Y2 = Y2 + eta2 * Err
        eta2 = np.min([eta2 * rho, 1e7])
        err = np.linalg.norm(Err, 'fro')/dnorm
        err_E2 = np.linalg.norm(E2,ord=1, keepdims = True)
        err_xi = np.linalg.norm(np.real(x),ord=1, keepdims = True)
        allerr_temp = err + nc_norm+ lam_3*err_E2 + lam_1*np.linalg.cond(R)*err_xi
        allerr.append(allerr_temp)
        allnc_norm.append(nc_norm)

        if (iter % iter_print) == 0 or iter == 1 or iter > maxIter or err < tol:
           print('iteration:{0}, err:{1}, nc_norm:{2} eta2:{3}'.format(iter, err, nc_norm, eta2))
           print_pde(x, rhs_des)
    if iter < start_num:
       print("IALM Finished at iteration %d" % (iter))
       vectorX   = np.reshape(X,(nx*nt,1))
       x         = TrainSTRidge(R, vectorX, lam_1, d_tol)  
       print_pde(x, rhs_des)

    return x, X, E2, allerr, allnc_norm

###########################################################################################################
def frobenius_norm(M):
    return np.linalg.norm(M, ord='fro')

def shrink(M, tau):
    return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape)) 
   
def hardshrink(M, tau):
    return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))  

def svd_threshold(M, tau):
    U, S, V = np.linalg.svd(M, full_matrices=False)
    eig = shrink(S, tau)
    return np.dot(U, np.dot(np.diag(shrink(S, tau)), V)), eig

def pcasvd_threshold(M, mu, n, sv):
    U, S, V = np.linalg.svd(M, full_matrices=False)
    svp = (S > 1 / mu).shape[0]
    if svp < sv:
       sv = np.min([svp + 1, n])
    else:
       sv = np.min([svp + round(0.05 * n), n])
    A = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
    nc_norm = np.sum(S)
    return A, nc_norm

def compute_err(xi, xi_true):
    allerr=xi-xi_true
    return np.linalg.norm(allerr,ord=1, keepdims = True)*100/np.linalg.norm(xi_true,ord=1, keepdims = True)

####################################################################################################
####################################################################################################

def TrainSTRidge(R, Ut, lam, d_tol, maxit = 25, STR_iters = 10, l0_penalty = None, normalize = 2, split = 0.8, print_best_tol = False):
    """
    This function trains a predictor using STRidge.

    It runs over different values of tolerance and trains predictors on a training set, then evaluates them 
    using a loss function on a holdout set.
    """

    # Split data into 80% training and 20% test, then search for the best tolderance.
    np.random.seed(0) # for consistancy
    n,_ = R.shape
    train = np.random.choice(n, int(n*split), replace = False)
    test = [i for i in np.arange(n) if i not in train]
    TrainR = R[train,:]
    TestR = R[test,:]
    TrainY = Ut[train,:]
    TestY = Ut[test,:]
    D = TrainR.shape[1]       

    # Set up the initial tolerance and l0 penalty
    d_tol = float(d_tol)
    tol = d_tol
    if l0_penalty == None: l0_penalty = 0.001*np.linalg.cond(R)

    # Get the standard least squares estimator
    w = np.zeros((D,1))
    w_best = np.linalg.lstsq(TrainR, TrainY, rcond=-1)[0]
    err_best = np.linalg.norm(TestY - TestR.dot(w_best), 2) + l0_penalty*np.count_nonzero(w_best)
    tol_best = 0

    # Now increase tolerance until test performance decreases
    for iter in range(maxit):

        # Get a set of coefficients and error
        w = STRidge(R,Ut,lam,STR_iters,tol,normalize = normalize)
        err = np.linalg.norm(TestY - TestR.dot(w), 2) + l0_penalty*np.count_nonzero(w)

        # Has the accuracy improved?
        if err <= err_best:
            err_best = err
            w_best = w
            tol_best = tol
            tol = tol + d_tol
        else:
            tol = max([0,tol - 2*d_tol])
            d_tol  = 2*d_tol / (maxit - iter)
            tol = tol + d_tol

    if print_best_tol: print("Optimal tolerance:", tol_best)

    return w_best
    
def STRidge(X0, y, lam, maxit, tol, normalize = 0, print_results = False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse 
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column
    """

    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0
    
    # Get the standard ridge esitmate
    if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y), rcond=-1)[0]
    else: w = np.linalg.lstsq(X,y, rcond=-1)[0]
    num_relevant = d
    biginds = np.where( abs(w) > tol)[0]

    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where( abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
            
        # If nothing changes then stop
        if num_relevant == len(new_biginds): break
        else: num_relevant = len(new_biginds)
            
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0: 
                #if print_results: print "Tolerance too high - all coefficients set below tolerance"
                return w
            else: break
        biginds = new_biginds
        
        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y), rcond=-1)[0]
        else: w[biginds] = np.linalg.lstsq(X[:, biginds],y, rcond=-1)[0]

    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y, rcond=-1)[0]

    if normalize != 0: return np.multiply(Mreg,w)
    else: return w
