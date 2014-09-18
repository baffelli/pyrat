# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:28:04 2014

@author: baffelli
Various functions for polarimetric processing
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import corefun
import matrices
def cloude_pottier_histogram(*args,**kwargs):
    """
    This function computes and displays the Cloude-Pottier histogram of a coherency matrix.
    
    Parameters
    ----------
    T : coherencyMatrix
        the matrix to be visualized.
    **kwargs
        additional arguments to be passed to the `plt.hist2d` function.
        May contain the "mask" keyword that will be used to produce
        the histogram from a subsection of the image.
    """

    if len(args) is 1 and type(args[0]) is corefun.coherencyMatrix:
        H, anisotropy, alpha, beta,p = args[0].cloude_pottier()
    else:
        H = args[0]
        alpha = args[1]
    if kwargs.has_key('mask'):
        mask = kwargs.pop('mask')
        H = H[mask]
        alpha = alpha[mask]
    cmap = matplotlib.cm.jet
    cmap.set_bad('w')
    w = np.ones_like(H) / H.size * 100
    print H.shape
    print alpha.shape
    plt.hist2d(H.flatten(),np.rad2deg(alpha).flatten(), range = [[0,1],[0,90]], cmap = cmap, cmin = 1e-5, weights = w, **kwargs)
    plt.xlabel(r'$Entropy$')
    plt.ylabel(r'$Mean\quad\alpha$')
    return w
    


    
    
def coherence(im1,im2,win):
    """
    This function computes the interferometric
    coherence between two images.
    
    Parameters
    ----------
    im1 : ndarray
        first image
    im2 : ndarray
        second image
    win : iterable, optional
        Window size for the processing.
    Returns
    -------
    array_like
        Coherency image
    """
    return corefun.smooth(im1 * im2.conj(),win)/np.sqrt(corefun.smooth(im1 * im1.conj(),win)*corefun.smooth(im2 * im2.conj(),win))

    

def temporal_coherency_matrix(stack,basis='pauli'):
    T = np.zeros_like(stack[0].shape[0:2] + (3,))
    for image in stack:
        T = image.to_coherency_matrix(basis = basis) + T
    T = T / len(stack)
    return T

def otd(T1, T2):
    """
    This function is the orthogonal transformation detector.
    
    Parameters
    ----------
    T1 : coherencyMatrix
        the first image.
    T2 : coherencyMatrix
        the second image to process.
    Returns
    -------
    ndarray
        The distance image as a float array.
    Notes
    -----
    This function is equivalent to `gpd` when the parameter `RedR` is set to 1
    """
    
    shp2 = T2.shape
    #Compute eigendiagonalization
    l, u = np.linalg.eigh(T2)
    if len(shp2) is 4:
        u_inv = np.conj(u.transpose([0,1,3,2]))
        norm_l = np.linalg.norm(l, axis = 2)
    else:
         u_inv = np.conj(u.transpose())
         norm_l = np.linalg.norm(l)
    #Transform
    T_transf = corefun.transform(u_inv,T1, u)
    #Exctract vector
    if T_transf.ndim is 4:
        mu_vector = T_transf.diagonal(axis1 = 2, axis2 = 3)
    else:
        mu_vector = T_transf.diagonal()
    #Need to separate cases depending on the shapes
    mu_c = mu_vector.conj()
    if mu_c.ndim is 3 and l.ndim is 1:
        sum_string = "xyi,...i"
        norm_mu  =np.linalg.norm(T1,axis=(2,3),ord= 'fro')
    elif mu_c.ndim is 3 and l.ndim is 3:
        sum_string = "...i,...i"
        norm_mu = np.linalg.norm(T1,axis=(2,3),ord='fro')
    elif mu_c.ndim is 1 and l.ndim is 1:
        sum_string = "i,i"
        norm_mu = np.linalg.norm(T1,ord='fro')
    dots = np.einsum(sum_string,mu_c,l)
    dots1 = dots / (norm_mu * norm_l)
    dots1 = np.abs(dots1)
    dots2 = dots1.view(np.ndarray)
    return dots2

def eigenvalue_detector(T1,T2):
    """
    This function implements a polarimetric detector
    based on the eigenvalue feature vectors.
    
    Parameters
    ----------
    T1 : coherencyMatrix
        the first image.
    T2 : coherencyMatrix
        the second image to process.
    Returns
    -------
    ndarray
        The distance image as a float array.
    Notes
    -----
    This function results in a very poor detector, it
    is mainly used as a comparison with the other polarimetric detectors
    """
    l1, w = np.linalg.eigh(T1)
    l2, w = np.linalg.eigh(T2)
    dot = lambda i1,i2: np.einsum('...i,...i',i1,i2)
    detector = dot(l1,l2) / (np.sqrt(dot(l1,l1)) * np.sqrt(dot(l2,l2)))
    return detector
    
def modified_eigenvalue_detector(T1,T2):
    l1,w = np.linalg.eigh(T1)
    w_inv = w.transpose([0,1,3,2]).conj()
    M_tilde = corefun.transform(w_inv,T2,w)
    mu_tilde = M_tilde.diagonal(axis1 = 2, axis2 = 3)
    dot = lambda i1,i2: np.einsum('...i,...i',i1,i2)
    detector = dot(l1,mu_tilde) / (np.sqrt(dot(l1,l1)) * np.sqrt(dot(mu_tilde,mu_tilde)))
    return detector

    

def wishart(T1,T2, n_looks):
    """
    This function is the Wishart distance classification detector.
    
    Parameters
    ----------
    T1 : coherencyMatrix
        the first image.
    T2 : coherencyMatrix
        the second image to process.
    Returns
    -------
    ndarray
        The distance image as a float array.
    Notes
    -----
    This implements the Wishart distance assuming a knonwn class mean.
    """
    T_transf = corefun.transform(np.linalg.inv(T2),T1,np.eye(3))
#    T_transf_1 = T2.transform(np.linalg.inv(T1),np.eye(3))
    if T_transf.ndim is 4:
        tr = np.trace(T_transf, axis1= 2, axis2 = 3)
    elif T_transf.ndim is 3:
        tr = np.trace(T_transf, axis1= 1, axis2 = 2)
    elif T_transf.ndim is 2:
        tr = np.trace(T_transf, axis1= 0, axis2 = 1)
    return np.real(frac_det(T1,T2)**n_looks *  np.exp(- (n_looks* tr + 3)))

def gpd(T1,T2,RedR=1):
    """
    This function is the geometrical perturbation filter detector.
    
    Parameters
    ----------
    T1 : coherencyMatrix
        the first image.
    T2 : coherencyMatrix
        the second image to process.
    RedR : double, optional.
        The RedR parameter
    Returns
    -------
    ndarray
        The distance image as a float array.
    Notes
    -----
    This function is equivalent to `otd` when the parameter `RedR` is set to 1
    """
    v1 = T1.vectorize()
    v2 = T2.vectorize()
    if T1.ndim == 4 and T2.ndim == 4:
        v2_norm = v2 / np.linalg.norm(v2,axis = 2)[:,:,np.newaxis]
        ptot = np.linalg.norm(v1, axis = 2, ord = 2)**2
        pt = np.abs(np.einsum("...i,...i",v1.conj(),v2_norm))**2
    elif T1.ndim == 2 and T2.ndim == 2:
        v2_norm = v2 / np.linalg.norm(v2)
        ptot = np.linalg.norm(v1, ord = 2)**2
        pt = np.abs(np.einsum("i,i",v1.conj(),v2_norm))**2
    elif T1.ndim == 4 and T2.ndim == 2:
        v2_norm = v2 / np.linalg.norm(v2)
        ptot = np.linalg.norm(v1, axis = 2, ord = 2)**2
        pt = np.abs(np.einsum("...i,...i",v1.conj(),v2_norm))**2
    frac = ptot / pt - 1
    deter = 1 / np.sqrt(1 + RedR *frac)
    return deter

def extract_single_pol(T):
    if type(T) is coherencyMatrix:
        if not T.basis is 'lexicographic':
            C1 = T.pauli_to_lexicographic()
        else:
            C1 = T
        if T.ndim is 4:
            indices1 =  (Ellipsis, Ellipsis, 0,0)
        if T.ndim is 2:
            indices1 = (0,0)
        HH1 = C1[indices1]
        return np.asarray(np.abs(HH1), dtype=np.float64)
    else:
        return T

def S_di(angle):
 return  matrices.scatteringMatrix(np.array([[np.cos(2*angle), np.sin(2*angle)], [np.sin(2*angle), -np.cos(2*angle)]]))



def sp(T1,T2,n):
    HH1 = extract_single_pol(T1)
    HH2 = extract_single_pol(T2)
    return np.asarray(HH1**n / HH2**n * np.exp(-n*(1/HH2*HH1 - 1)), dtype=np.float64)

def spcd(T1,T2,m,n):
    HH1 = extract_single_pol(T1)
    HH2 = extract_single_pol(T2)
    cd = (n+m)**(n+m)/(n**n*m**m)*(HH1**n * HH2**n) / (HH1 + HH2)**(2*n)
    return cd


def frac_det(T,T1):
    s, logdet = np.linalg.slogdet(T)
    s1, logdet1 = np.linalg.slogdet(T1)
    d = s * s1 * np.exp(logdet - logdet1)
    return d
    
def new_det(T):
     s, logdet = np.linalg.slogdet(T)
     return s * np.exp(logdet)

def wishart_cd(T1,T2):
    T_mean = (T1 + T2) / 2
    det = np.sqrt(new_det(T1)*new_det(T2))/new_det(T_mean)
    return np.real(det)

def classifier(T,classes,distance_function, threshold):
    n_classes = len(classes)
    distances = np.ndarray(T.shape[0:2]+(n_classes,))
    for index,value in enumerate(classes):
        dist_temp = distance_function(T,value)
        distances[:,:,index] = dist_temp
    classified = np.argmax(distances, axis = 2) + 1
    classified[np.all(distances  < threshold , axis = 2)] = 0
    return classified, distances

def u2_tau(th):
    return np.array([[np.cos(th), 1j*np.sin(th) ],[1j * np.sin(th), np.cos(th)]])
    
def u2_phi(ph):
    return np.array([[np.cos(ph), -np.sin(ph) ],[np.sin(ph), np.cos(ph)]])
    
def u3_tau(th):
    return np.array([[np.cos(2*th), np.zeros_like(th), 1j* np.sin(2*th)],\
                     [np.zeros_like(th), np.zeros_like(th) + 1, np.zeros_like(th)                    ],\
                     [1j*np.sin(2*th), np.zeros_like(th), np.cos(2*th)]])

def u3_phi(ph):
    return np.array([[1 + np.zeros_like(ph), np.zeros_like(ph), np.zeros_like(ph) ],\
                     [np.zeros_like(ph), np.cos(2*ph), np.sin(2*ph)],\
                     [np.zeros_like(ph), -np.sin(2*ph) , np.cos(2*ph)]])


def pol_synthesis(S, phi, tau):
    if type(S) is matrices.scatteringMatrix:
        M1 = u2_phi(phi)
        M2 = u2_tau(tau)
        M = corefun.transform(M1,M2,np.eye(2, dtype = np.complex64))
        ST = corefun.transform(M.T,S,M)
    elif type(S) is matrices.coherencyMatrix:
        M1 = u3_phi(phi)
        M2 = u3_tau(tau)
        M = corefun.transform(M1,M2,np.eye(3,dtype = np.complex64))
        if S.basis is 'pauli':
            T = S
        else:
            T = S.lexicographic_to_pauli()
        ST = corefun.transform(M,T,np.linalg.inv(M))
    return ST

                

def pol_signature(S, n_points = 100):
        tilt = np.linspace(-np.deg2rad(90),  np.deg2rad(90), n_points)
        ellipticity = np.linspace(- np.pi / 4, np.pi / 4, n_points)
        phi_m, tau_m = np.meshgrid(tilt, ellipticity, indexing = 'ij')
        if type(S) is matrices.scatteringMatrix:
            M1 = np.transpose(u2_phi(phi_m), [2,3,0,1])
            M2 = np.transpose(u2_tau(tau_m),[2,3,0,1])
            A = corefun.transform(M1,M2,np.eye(2))
            co_sig = corefun.transform(A.transpose([0,1,3,2]),S,A)
            power = lambda m : np.abs(np.array(m)[:,:,0,0])**2
            power_cross = lambda m : np.abs(np.array(m)[:,:,0,1])**2
            co_power = power(co_sig) / S.span()
            cross_power = power_cross(co_sig) / S.span()
            return co_power, cross_power, tilt, ellipticity
        elif type(S) is matrices.coherencyMatrix:
            if S.basis is 'pauli':
                T = S
                print 'here'
            else:
                T = S.lexicographic_to_pauli()
            M1 = np.transpose(u3_phi(phi_m), [2,3,0,1])
            M2 = np.transpose(u3_tau(tau_m),[2,3,0,1])
            A = corefun.transform(M1,M2,np.eye(3, dtype=  np.complex64))
            T_transf = corefun.transform(A,T,np.linalg.inv(A))
#            power = lambda m : np.abs(np.array(m)[:,:,0,0]) / 2
#            power_cross = lambda m : np.abs(np.array(m)[:,:,1,1])/2
#            co_power = power(T_transf) / T.span()
#            cross_power = power_cross(T_transf) / T.span()
            T1 = np.array(T_transf)
            co_power = (np.abs(T1[:,:,0,0] +  2 * np.real(T1[:,:,0,1]) + T1[:,:,1,1])/2)/T.span()
            cross_power = np.abs(T1[:,:,2,2])/T.span()
            return co_power, cross_power, tilt, ellipticity

def polinsar_matrix(S1,S2, win = [2,2],basis = 'pauli', bistatic = False):
    T1 = S1.to_coherency_matrix(basis = basis)
    T2 = S2.to_coherency_matrix(basis = basis)
    T1 = T1.boxcar_filter(win)
    T2 = T2.boxcar_filter(win)
    omega = matrices.coherencyMatrix(corefun.outer_product(S1.scattering_vector(basis = basis, bistatic = bistatic),S2.scattering_vector(basis = basis, bistatic = bistatic)))
    omega = omega.boxcar_filter(win)
    T1_s = corefun.matrix_root(T1)
    T2_s = corefun.matrix_root(T2)
    return corefun.transform(T1_s,omega,T2_s)