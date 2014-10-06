# -*- coding: utf-8 -*-
"""
Created on Thu May 15 14:20:32 2014

@author: baffelli
"""
import numpy as np
from ..fileutils import gpri_files, other_files  
from . import corefun
from ..visualization import visfun

class gpriImage(np.ndarray):
    def __new__(*args):
        cls = args[0]
        if len(args) is 2:
            if type(args[1]) is list:
                data = ()
                #Load whole stack
                paths = args[1]
                par = gpri_files.load_par(paths[0] + '.par')
                for path in paths:
                    temp_data = gpri_files.load_slc(path)
                    data = data + (temp_data,) 
                data_1 = np.dstack(data)
#                data_1.transpose([1,2,0])
            elif type(args[1]) is str:
                path = args[1]
                data_1= gpri_files.load_slc(path)
                par = gpri_files.load_par(path + '.par')
        obj = data_1.view(cls)
        north = par['GPRI_ref_north'][0]
        east = par['GPRI_ref_east'][0]
        r_min = par['near_range_slc'][0] 
        r_max = par['far_range_slc'][0]
        n_range = par['range_samples'][0]
        az_step = np.deg2rad(par['GPRI_az_angle_step'][0])
        az_min = np.deg2rad(par['GPRI_az_start_angle'][0])
        n_az = par['azimuth_lines'][0] 
        az_max = az_min + n_az * az_step
        az_vec = np.linspace(az_min,az_max,n_az) 
        r_vec = np.linspace(r_min,r_max,n_range)
        #Set attributes
        obj.center = [north, east,par['GPRI_ref_alt'][0]]
        obj.az_vec = az_vec
        obj.r_vec = r_vec
        obj.tx_coord = par['GPRI_tx_coord']
        obj.incidence_angle = par['incidence_angle']
        obj.center_frequency = par['radar_frequency'][0]
        obj.utc = par['utc']
        return obj 
    

    def __array_wrap__(self, out_arr, context=None):
        temp_arr = np.ndarray.__array_wrap__(self, out_arr, context)
        temp_arr = temp_arr.view(gpriImage)
        temp_arr.__dict__.update(self.__dict__)

        return temp_arr
    

        
    def __getitem__(self,sl):
        new_obj_1 = np.array(super(gpriImage, self).__getitem__(sl))
        new_obj_1 = new_obj_1.view(gpriImage)
        new_obj_1.__dict__.update(self.__dict__)
        if type(sl) is slice or type(sl) is tuple:
            r_vec = self.r_vec
            az_vec = self.az_vec
            new_obj_1.__setattr__('r_vec',r_vec[(sl[1])])
            new_obj_1.__setattr__('az_vec',az_vec[(sl[0])])
            return new_obj_1
        return new_obj_1

        
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.az_vec = getattr(obj, 'az_vec','none')
        self.r_vec = getattr(obj, 'r_vec','none')
        self.center = getattr(obj, 'center','none')
        self.utc = getattr(obj, 'utc','none')
        





   

 

class scatteringMatrix(np.ndarray):
    
    pauli_basis = [np.array([[1,0],[0,1]])*1/np.sqrt(2),np.array([[1,0],[0,-1]])*1/np.sqrt(2),np.array([[0,1],[1,0]])*1/np.sqrt(2)]
    lexicographic_basis = [np.array([[1,0],[0,0]])*2,np.array([[0,1],[0,0]])*2*np.sqrt(2),np.array([[0,0],[0,1]])*2]

    def __new__(*args,**kwargs):      
        cls = args[0]
        gpri = kwargs.get('gpri')
        if gpri is True:
            if 'chan' in kwargs:
                chan = kwargs.get('chan')
            H_ant = 'A'
            V_ant = 'B'
            base_path = args[1]
            slc_path_HH = base_path + "_" + H_ant + H_ant + H_ant + chan + ".slc"
            slc_path_VV = base_path + "_" + V_ant + V_ant + V_ant + chan + ".slc"
            slc_path_HV = base_path + "_" + H_ant + V_ant + V_ant + chan + ".slc"
            slc_path_VH = base_path + "_" + V_ant + H_ant + H_ant + chan + ".slc"
            HH = gpriImage(slc_path_HH)
            VV = gpriImage(slc_path_VV)
            HV = gpriImage(slc_path_HV)
            VH = gpriImage(slc_path_VH)
            s_matrix = np.zeros(HH.shape + (2,2), dtype = HH.dtype)
            s_matrix[:,:,0,0] = HH
            s_matrix[:,:,1,1] = VV
            s_matrix[:,:,0,1] = HV
            s_matrix[:,:,1,0] = VH
            obj = np.asarray(s_matrix).view(cls)
            obj.r_vec = HH.r_vec
            obj.az_vec = HH.az_vec
            obj.utc = HH.utc
            obj.center = HH.center
            obj.incidence_angle = HH.incidence_angle
            phase_center = []
            obj.geometry = 'polar'
            obj.center_frequency = HH.center_frequency
            TX_VEC = [0,0.125]
            RX_VEC_U = [0.475,0.6]
            RX_VEC_L = [0.725,0.85]
            phase_center = []
            for polarimetric_channel, idx_tx,idx_rx in zip([HH,HV,VH,VV],[0,0,1,1],[0,1,0,1]):
                if chan is 'l':
                    rx_vec = RX_VEC_U
                else:
                    rx_vec = RX_VEC_L
                phase_center.append((rx_vec[idx_rx] + TX_VEC[idx_tx])/2)#Compute effective phase center
            obj.ant_vec = np.reshape(phase_center,[2,2])
        else:
            if type(args[1]) is np.ndarray:
                s_matrix = args[1]            
            elif type(args[1]) is str:
                args = args[1:None]
                s_matrix = other_files.load_scattering(*args,**kwargs)
            obj = s_matrix.view(scatteringMatrix)
            obj.geometry = 'cartesian'
        return obj
        
    def __array_finalize__(self, obj):
        if obj is None: return
        self.geometry = getattr(obj, 'geometry', None)
        
    
    def from_gpri_to_normal(self):
        self_1 = self[:]
        self_1.geometry = None
        return self_1
        
    def __getitem__(self,sl):
        if type(sl) is str:
            if sl is 'HH':
                base_idx = (0,0)
            elif sl is 'VV':
                base_idx = (1,1)
            elif sl is 'HV':
                base_idx = (0,1)
            elif sl is 'VH':
                base_idx = (1,0)
            if self.ndim is 4:
                sli = (Ellipsis,Ellipsis) + base_idx
            elif self.ndim is 3:
                sli = (Ellipsis,) + base_idx
            else:
                sli = base_idx
                new_obj_1 =  (super(scatteringMatrix,self).__getitem__(sli))
                return new_obj_1
        elif type(sl) is slice or type(sl) is tuple or type(sl) is int or type(sl) is list:
            sli = sl
        #apply slicing
        new_obj_1 =  (super(scatteringMatrix,self).__getitem__(sli))
        
        if  hasattr(self,'geometry') and type(sli) is tuple:
            if hasattr(new_obj_1, '__dict__'):
                new_obj_1.__dict__.update(self.__dict__)
            if self.geometry is 'polar':
                r_vec = self.r_vec[(sli[1])]
                az_vec = self.az_vec[(sli[0])]
                new_obj_1.__setattr__('r_vec',r_vec)
                new_obj_1.__setattr__('az_vec',az_vec)
                if type(sl) is str:
                    chan_phase_center = self.ant_vec[base_idx]
                    new_obj_1.__setattr__('ant_vec',chan_phase_center)
        return new_obj_1
        
        
    def __copy__(self):
        print self.r_vec
        new_obj = scatteringMatrix(np.array(self).__copy__())
        new_obj.__dict__.update(self.__dict__)
        return new_obj
        
    def __setitem__(self,sl,item):
        if type(sl) is str:
            if sl is 'HH':
                base_idx = (0,0)
            elif sl is 'VV':
                base_idx = (1,1)
            elif sl is 'HV':
                base_idx = (0,1)
            elif sl is 'VH':
                base_idx = (1,0)
            if self.ndim is 4:
                sli = (Ellipsis,Ellipsis) + base_idx
            else:
                sli = base_idx
        else:
            sli = sl
        self1 = self.view(np.ndarray)
        self1.__setitem__(sli,item)
        self1 = self1.view(scatteringMatrix)
        self1.__dict__.update(self.__dict__)
        if self.geometry is 'polar':
            r_vec = self.r_vec
            az_vec = self.az_vec
            self1.__setattr__('r_vec',r_vec)
            self1.__setattr__('az_vec',az_vec)
        self = self1 

    
    def __array_wrap__(self, out_arr, context=None):
        temp_arr = np.ndarray.__array_wrap__(self, out_arr, context)
        temp_arr = temp_arr.view(scatteringMatrix)
        temp_arr.__dict__.update(self.__dict__)
        return temp_arr
        
        
    
    def scattering_vector(self,basis = 'pauli',bistatic = True):
        """ 
        This function returns the scattering vector in the pauli or lexicographics basis
    
        Parameters
        ----------
        basis : string
            The basis of interest. Can be either 'pauli' or 'lexicographic'
        bistatic :  bool
            Set to True if the bistatic scattering vector is needed
        Returns
        -------
        sv : ndarray 
            The resulting scattering vector
        """
        #Generate necessary transformation matrices
        if bistatic is True:
            sv = np.zeros(self.shape[0:2] +(4,),dtype = self.dtype)
            if basis is 'pauli':
                sv1 = np.array(self['HH'] + self['VV'])
                sv2 = np.array(self['HH'] - self['VV'])
                sv3 = np.array(self['HV'] + self['VH'])
                sv4 = np.array(1j*(self['HV'] - self['VH']))
                factor =  1/np.sqrt(2)
            elif basis is 'lexicographic':
                sv1 = np.array(self['HH'])
                sv2 = np.array(self['HV'])
                sv3 = np.array(self['VH'])
                sv4 = np.array(self['VV'])
                factor = 1
            if self.ndim is 2:
                sv = factor * np.hstack((sv1,sv2,sv3,sv4))
            else:
                sv = factor * np.dstack((sv1,sv2,sv3,sv4))
        elif bistatic is False:
            sv = np.zeros(self.shape[0:2] + (3,),dtype = self.dtype)
            if basis is 'pauli':
                sv1 = np.array(self['HH'] + self['VV'])
                sv2 = np.array(self['HH'] - self['VV'])
                sv3 = np.array((self['HV']) * 2)
                factor =  1/np.sqrt(2)
            elif basis is 'lexicographic':
                sv1 = np.array(self['HH'])
                sv2 = np.array(np.sqrt(2) * self['HV'])    
                sv3 = np.array(self['VV'])
                factor = 1
            if self.ndim is 2:
                sv = factor * np.hstack((sv1,sv2,sv3))
            else:
                sv = factor * np.dstack((sv1,sv2,sv3))
        return sv

        
    
    def span(self):
        """ 
        This function computes the polarimetric span of the scattering matrix
        Parameters
        ----------
        None
        Returns
        -------
        sp: ndarray
            The resulting span
        """
        v = self.scattering_vector(basis='lexicographic', bistatic = True)
        if self.ndim is 4:
            ax = 2
        else:
            ax = 0
        sp = np.sum(np.abs(v)**2,axis = ax)
        return sp
    
    def symmetrize(self):
        """ 
        This function symmetrizes the scattering matrix
        """
        Z_hv = (self['HV'] + (self['VH'] * np.exp(-1j*(np.angle(self['HV'])-np.angle(self['VH'])))))/2
        self['HV'] = Z_hv
        self['VH'] = Z_hv


        
    def to_coherency_matrix(self,bistatic=False,basis='pauli'):
        """ 
        This function converst the scattering matrix into a coherency matrix
        using the chosen basis. It does not perform any averaging, so the resulting
        matrix has rank 1.
        
        Parameters
        ----------
        bistatic : bool
            If set to true, the bistatci coherency matric is computed
        basis : string
            Allows to choose the basis for the scattering vector
        -----
        Returns
        coherencyMatrix
            the resulting coherency matrix
        """
        k= self.scattering_vector(bistatic=bistatic,basis=basis)
        T = corefun.outer_product(k, k)
        T = T.view(coherencyMatrix)
        T.basis = basis
        T.par = getattr(self, 'par',None)
        T.r_vec = getattr(self, 'r_vec', None)
        T.az_vec = getattr(self, 'az_vec', None)
        T.ant_vec = getattr(self, 'ant_vec', None)
        T.geometry = getattr(self, 'geometry', None)
        T.title = getattr(self, 'title', None)
        return T


    def pauli_image(self,**kwargs):
        if 'basis' in kwargs:
            basis = kwargs.pop('basis')
        else:
            basis = 'pauli'
        k = self.scattering_vector(bistatic = False, basis = basis)
        if basis is 'pauli':
            k = k[:,:,[1,2,0]]
        else:
            pass
        im = visfun.pauli_rgb(np.abs(k)**2,**kwargs)
        return im
        
   

        
        

class coherencyMatrix(np.ndarray):
    
    global U3LP, U3PL
    U3LP = 1/np.sqrt(2) * np.array([[1, 0, 1],[1, 0, -1],[0, np.sqrt(2), 0]])
    U3PL = 1/np.sqrt(2) * np.array([[1, 1, 0],[0, 0, np.sqrt(2)],[1, -1, 0]])
    
    def vectorize(self):
        dim = self.shape
        if dim[-1] is 3: 
                access_vector = [0,4,8,1,2,5,3,6,7]
        elif dim[-1] is 4:
                access_vector = [0,5,10,15,1,2,3,6,7,11,4,8,9,12,13,14]
        else:
                access_vector = range(dim[-1])
        if self.ndim is 4:
            new_self = self.reshape(self.shape[0:2] + (self.shape[-1]**2,))[:,:,access_vector]
        else:
            new_self = self.reshape((self.shape[-1]**2,))[access_vector]
        return new_self
    
  
    def normalize(self):
        span = self.span()
        if self.ndim == 2:
            T_norm = self / span
        elif self.ndim == 4:
            T_norm = self / span[:,:,None,None]
        elif self.ndim == 3:
            T_norm = self / span[:,None,None]
        return T_norm

    def cloude_pottier(self):
        l,w = np.linalg.eigh(self)
        l = np.array(l)
        w = np.array(w)
        if self.ndim is 4:
            l_sum = np.sum(np.abs(l),axis = 2)
            p = l / l_sum[:,:,np.newaxis]
            H = -np.sum( p * np.log10(p) / np.log10(3), axis = 2)
            alpha = np.arccos(np.abs(w[:,:,0,0:None]))
            rotated = w[:,:,1,0:None] * 1/np.sin(alpha)
            beta = np.arccos(np.abs(rotated))
            anisotropy = (l[:,:,1] - l[:,:,2]) / (l[:,:,1] + l[:,:,2])
            alpha_m = np.sum(alpha * p,axis = 2)
            beta_m = np.sum(beta * p,axis = 2)
        if self.ndim is 2:
            l_sum = np.sum(l,axis = 1)
            p = l / l_sum[:,np.newaxis]
            H = -np.sum( p * np.log10(p) / np.log10(3), axis = 1)
            alpha = np.arccos(np.abs(w[:,0,0:None]))
            alpha_m = np.sum(alpha * p, axis = 1)
            anisotropy = (l[1] - l[2]) / (l[1] + l[2])
        return H, anisotropy, alpha_m, beta_m, p, w

    def transform(self,T1,T2):
        out = self.__array_wrap__(np.einsum("...ik,...kl,...lj->...ij",T1,self,T2))
#        out1 = out.view(coherencyMatrix)
        return out

    def rank_image(self, threshold):
        image = self
        l, u = np.linalg.eigh(image)
        dist = lambda x, y: np.abs(x-y) < threshold
        c1 = dist(l[:,:,0],l[:,:,1])
        c2 = dist(l[:,:,0],l[:,:,2])
        c3 = dist(l[:,:,1],l[:,:,2])
        counts = c1.astype(np.int) + c2.astype(np.int) + c3.astype(np.int)
        return counts
    
            
    def __getitem__(self,sl):
        new_obj_1 = np.array(self).__getitem__(sl)
        new_obj_1 = new_obj_1.view(coherencyMatrix)
        if type(sl) is slice or type(sl) is tuple and self.geometry is 'polar':
            r_vec = self.r_vec
            az_vec = self.az_vec
            new_obj_1.__setattr__('r_vec',r_vec[(sl[1])])
            new_obj_1.__setattr__('az_vec',az_vec[(sl[0])])
            new_obj_1.__setattr__('basis',self.basis)
            new_obj_1.__setattr__('geometry',self.geometry)
            new_obj_1.__setattr__('title',self.title)
            return new_obj_1
        return new_obj_1

    def __new__(*args,**kwargs):
        cls = args[0]
        print type(args[1])
        #Get keywords
        coherency = kwargs.get('coherency')
        agrisar = kwargs.get('agrisar')
        polsarpro = kwargs.get('polsarpro')
        gpri = kwargs.get('gpri')
        dim = kwargs.get('dim')
        if 'bistatic' not in kwargs:
            bistatic = False
        else:
            bistatic = kwargs.get('bistatic')
        if 'basis' not in kwargs:
            basis = 'pauli'
        else:
            basis = kwargs.get('basis')
        if type(args[1]) is np.ndarray:
            T = args[1]
#            if corefun.is_hermitian(T):
#                obj = np.asarray(T).view(cls)
#            else:
#                raise np.linalg.LinAlgError("T is not Hermitian")
            obj = np.asarray(T).view(cls)
            obj.basis = 'pauli'
        elif type(args[1]) is str:
            path = args[1]
            if agrisar:
                s_matrix = scatteringMatrix(path,fmt='esar')
                pauli = s_matrix.scattering_vector(bistatic = bistatic, basis = basis)
                T = corefun.outer_product(pauli,pauli)
            elif coherency:
                T = other_files.load_coherency(path,dim)
            elif polsarpro:
                s_matrix = scatteringMatrix(path,dim,fmt='polsarpro')
                #Pauli representation is defaulto
                pauli = s_matrix.scattering_vector(bistatic = bistatic, basis = basis)
                T = corefun.outer_product(pauli,pauli)
            elif gpri:
                if 'chan' in kwargs:
                    chan = kwargs.get('chan')
                else:
                    chan = 'l'
                s_matrix = scatteringMatrix(path,gpri = True, chan = chan)
                pauli = s_matrix.scattering_vector(bistatic = bistatic, basis = basis)
                T = corefun.outer_product(pauli, pauli)
                obj = T.view(cls)
                obj.window = [1,1]
                obj.r_vec = s_matrix.r_vec
                obj.az_vec = s_matrix.az_vec
                obj.geometry = 'polar'
                obj.basis = basis
                obj.ant_coord = s_matrix.ant_coord
                obj.title = s_matrix.title
                return obj
        elif type(args[1]) is gpriImage:    
            stack = args[1]
            T = corefun.outer_product(stack, stack)
        # Finally, we must return the newly created object:
        obj = T.view(cls)
        obj.window = [1,1]
        obj.basis = basis
        obj.geometry = 'cartesian'
        return obj
        
    def __array_finalize__(self, obj):
        if obj is None: return
        self.geometry = getattr(obj, 'geometry', None)
        
    def __array_wrap__(self, out_arr, context=None):
        temp_arr = np.ndarray.__array_wrap__(self, out_arr, context)
        temp_arr = temp_arr.view(coherencyMatrix)
        temp_arr.window = getattr(self, 'window', [1,1])
        temp_arr.geometry = getattr(self, 'geometry', None)
        temp_arr.basis = getattr(self, 'basis',None )
        temp_arr.r_vec = getattr(self, 'r_vec', None)
        temp_arr.az_vec = getattr(self, 'az_vec', None)
        temp_arr.par = getattr(self, 'par', None)
        temp_arr.title = getattr(self, 'title', None)
        return temp_arr
        
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.window = getattr(obj, 'window', [1,1])
        self.geometry = getattr(obj, 'geometry', None)
        self.basis = getattr(obj, 'basis','pauli')
        self.r_vec = getattr(obj, 'r_vec', None)
        self.az_vec = getattr(obj, 'az_vec', None)
        self.title = getattr(obj, 'title', None)

    def boxcar_filter(self,window, discard= False):
        """
        This function applies boxcar averaging on the coherency matrix
        Parameters
        ----------
        window : tuple
            a tuple of window sizes
        discard :  bool, optional
            set to true if only the centra pixel of the window has to be kept
        """
        T = self.__array_wrap__(corefun.smooth(self, window + [1,1]))
        if discard:
            T = T[0:None:window[0],1:None:window[1],:,:]
        return T

    def span(self):
        shp = self.shape
        if len(shp) is 4:
            s = np.trace(self,axis1=2,axis2=3)
        else:
            s = np.trace(self)
        s1 = np.real(s.view(np.ndarray))

        return s1
        

    def pauli_image(self,**kwargs):
        if self.basis is 'pauli':
            k = np.diagonal(self,axis1 = 2, axis2 = 3)[:,:,[1,2,0]]
        else:
            k = np.diagonal(self,axis1 = 2, axis2 = 3)
        im = visfun.pauli_rgb(k,**kwargs)
        return im
       

    def generateRealizations(self, n_real, n_looks):
#        #Generate unit vectors
         n_tot = n_real * n_looks
         k = np.random.multivariate_normal(np.zeros((3)),self,n_tot)
         k = k.transpose()
         outers = np.einsum('i...,j...',k,k.conj())
         outers = np.reshape(outers,(n_real,n_looks,3,3))
         outers = np.mean(outers, axis = 1)
         return outers.view(coherencyMatrix)

    
    def pauli_to_lexicographic(self):
        """
        This function converst the current matrix to the lexicographic basis
        """
        if self.basis is 'pauli':
            C = self.transform((U3PL),np.linalg.inv(U3PL))
            C.basis = 'lexicographic'
        C = self.__array_wrap__(C)
        C.basis = 'lexicographic'
        print 'success'
        return C
        
    def lexicographic_to_pauli(self):
        """
        This function converst the current matrix to the pauli basis
        """        
        if self.basis is 'lexicographic':
            C = self.transform(U3LP,np.linalg.inv(U3LP))
            C.basis = 'pauli'
        C = self.__array_wrap__(C)
        C.basis = 'pauli'
        print 'success'
        return C
 
#TODO fix block processing   
class block_array:
    
    def __init__(*args):
        
       obj = args[0]
       #The array to be split
       A = args[1]
       #Size of desired blocks
       block_size = args[2]
       #Size of the processing window
       window_size = args[3]
       #2D shape of array
       shape_2d = A.shape[0:2]
       #Create object
       obj.bs = block_size
       obj.A = A
       #iterate over block sizes to compute indices for each dim
       obj.rsa = []
       obj.rea = []
       obj.wsa = []
       obj.wea = []
       #Current index for the iterator
       obj.current = 0
       obj.nblocks = []
       for bs, wins, ars in zip(block_size, window_size, shape_2d):
           #Compute number of block
           N_block = np.ceil( (ars - wins + 1) / (bs - wins + 1))
           block_idx = np.arange(N_block)
           rs = (block_idx) * (bs - wins + 1)
           re = rs + bs  - 1
           re[-1] = ars  - 1
           ws = np.zeros(N_block) + (wins -1) / 2
           we = np.zeros(N_block) + bs -(wins +1) / 2
           ws[0] = 0
           re[-1] = ars - 1
           we[-1] = ars -1 - rs[-1]
           obj.rsa.append(rs)
           obj.rea.append(re)
           obj.wsa.append(ws)
           obj.wea.append(we)
           obj.nblocks.append(N_block)
       obj.maxiter = np.prod(obj.nblocks)
 
    def __getitem__(self,sl):
        if len(sl) is 1:
            return self.take_block(sl)
            
    def __setitem__(self,sl,item):
            self.put_block(sl,item)
            
    def __iter__(self):
        return self
        
    def next(self):
        if self.current >= self.maxiter:
            self.current = 0
            raise StopIteration
        else:
            c = self.take_block(self.current)
            self.current += 1
            return c
            
            
    def put_current(self, item):
        self.put_block(self.current, item)

    
    def center_index(self,idx):
        try:
            return np.unravel_index(idx,self.nblocks)
        except:
            return 0,0
    
    def take_block(self,idx):
        if idx < np.prod(self.nblocks):
            i,j = np.unravel_index(idx,self.nblocks)
            block = self.A[self.rsa[0][i]:self.rsa[0][i] + self.rea[0][i], self.rsa[1][j]:self.rsa[1][j] + self.rea[1][j]]
        return block
    
    def put_block(self,idx,block):

        if idx < np.prod(self.nblocks):
            i,j = np.unravel_index(idx,self.nblocks)
            clipped_block = block[self.wsa[0][i]:self.wsa[0][i] + self.wea[0][i] ,self.wsa[1][j]:self.wea[1][j] + self.wsa[1][j]]
            print clipped_block.shape
            start_i = self.rsa[0][i] + self.wsa[0][i]
            start_j = self.rsa[1][j] + self.wsa[1][j]
            self.A[start_i:start_i + clipped_block.shape[0],
                 start_j:start_j + clipped_block.shape[1]] = clipped_block
            

        
        
    