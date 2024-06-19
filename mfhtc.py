#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator
from scipy.signal import butter, filtfilt, lfilter, freqz, argrelextrema
from scipy.optimize import curve_fit
import seaborn as sns
import logging, pickle, os, sys
from opt_einsum import contract
from gmanp import pBasis, Pauli, Boson
from time import time, perf_counter, process_time, sleep
from datetime import datetime, timedelta
from pprint import pprint, pformat
from copy import copy
from scipy.integrate import solve_ivp, quad_vec, RK45
from scipy.linalg import expm
SOLVER = RK45 # Best!
from scipy import constants
import itertools
from scipy.fft import fft, fft2, ifft, ifft2, fftshift, ifftshift # recommended over numpy.fft
try:
    import pretty_traceback
    pretty_traceback.install()
except ModuleNotFoundError:
    # colored traceback not supported
    pass

logger = logging.getLogger(__name__)

sns.set_theme(context='notebook', style='ticks', palette='colorblind6', # 'colorblind' if need more than 6 lines
              rc={'legend.fancybox':False,
                  'text.usetex':True,
                  'text.latex.preamble':r'\usepackage{amsmath}',
                  'figure.dpi':400.0, 
                  'legend.edgecolor':'0.0', # '0.0' for opaque, '1.0' for transparent
                  'legend.borderpad':'0.2',
                  'legend.fontsize':'9',
                  }
              )

class HTC:
    COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
    EV_TO_FS = (constants.hbar/constants.e)*1e15 # convert time in electronvolts to time in fs
    DEFAULT_DIRS = {'data':'./data', 'figures':'./figures'} # output directories
    # N.B. type of parameters used to parse (non-default) input parameters
    DEFAULT_PARAMS = {
            'Q0': 15, # how many modes either side of K0 (or 0 for populations) to include; 2*Q0+1 modes total 
            'Nm': 6001, # Number of molecules
            'Nnu': 1, # Number of vibrational levels for each molecules
            'L': 60.0, # Crystal propagation length, inverse micro meters
            'nr':1.0, # refractive index, sets effective speed of light c/nr
            'omega_c':1.94, # omega_0 = 1.94eV (Fig S4C)
            'epsilon':2.14, # exciton energy, detuning omega_0-epsilon (0.2eV for model I in Xu et al. 2023)
            'gSqrtN':0.15, # light-matter coupling
            'kappa_c':3e-3, # photon loss
            'Gam_z':0.0, # molecular pure dephasing
            'Gam_up':0.0, # molecular pumping
            'Gam_down':1e-7, # molecular loss
            'S':0.0, # Huang-Rhys parameter
            'omega_nu':0.00647, # vibrational energy spacing
            'T':0.026, # k_B T in eV (.0259=300K, .026=302K)
            'gam_nu':0.01, # vibrational damping rate
            'initial_state': 'photonic', # or incoherent
            'A': 0.8, # amplitude of initial wavepacket
            'k_0': 3.0, # central wavenumber of initial wavepacket
            'sig_0': 4.0, # s.d. of initial wavepacket
            #'sig_f':0, # s.d. in microns instead (if specified)
            'atol':1e-9, # solver tolerance
            'rtol':1e-6, # solver tolerance
            'dt': 0.5, # determines interval at which solution is evaluated. Does not effect the accuracy of solution, only the grid at which observables are recorded
            'exciton': True, # if True, initial state is pure exciton; if False, a lower polariton initial state is created
            #'lowpass': 0.01, # used for smoothing oscillations in observable plots
            }

    @classmethod
    def default_params(self):
        return copy(self.DEFAULT_PARAMS)

    def __init__(self, params=None):
        for name, dp in self.DEFAULT_DIRS.items():
            if not os.path.exists(dp):
                os.makedirs(dp)
        self.params = self.parse_params(params)
        self.gp = pBasis(self.params['Nnu'], verbose=False, tests=False)
        self.add_useful_params(self.params) # requires self.gp. Otherwise must be run before all other functions
        self.boson = Boson(self.Nnu)
        self.rates = self.get_rates(self.params)
        self.Ks, self.ks = self.get_modes()
        self.make_state_dic()
        self.ns = np.arange(self.Nk)
        self.rs = self.ns * self.params['delta_r']
        self.wrapit(self.make_coeffs, f'Constructing EoM coefficients...', timeit=False)
        #self.wrapit(self.create_initial_state, f'Creating initial state...', timeit=False)

    def parse_params(self, params):
        if params is None:
            return self.default_params()
        default_params = self.default_params()
        parsed_params = {}
        used_defaults = {}
        unused = {}
        for name, val in default_params.items():
            if name not in params:
                used_defaults[name] = val
                parsed_params[name] = val
                continue
            try:
                parsed_params[name] = type(val)(params[name])
            except TypeError or ValueError:
                logger.warning(f'Param {name} should be {type(val)}, using default {val}')
                used_defaults[name] = val
                unused[name] = params[name]
                parsed_params[name] = val
        for name, val in params.items():
            if name not in default_params:
                unused[name] = val
        if len(used_defaults) > 0:
            logger.info('The following parameters were not specified hence assigned default values:')
            logger.info(pformat(used_defaults, sort_dicts=False))
        if len(unused) > 0:
            logger.warning('The following parameters were not recognised and have not been used:')
            logger.warning(pformat(unused, sort_dicts=False))
        return parsed_params

    def add_useful_params(self, params):
        params['nb'] = self.nb(params['T'])
        self.Q0, self.N0, self.N1 = params['Q0'], self.gp.indices_nums[0], self.gp.indices_nums[1]
        self.L = self.params['L'] * 1e-6 # system length in m
        self.c = constants.c / self.params['nr'] # speed of light in material
        self.wc = self.params['omega_c']
        # Compute prefactor for K in cavity dispersion
        # with this definition K_factor K^2 equiv (1/e) * (hbar^2 k^2 c^2) / (2*m*c^2) has units eV
        self.K_to_eV = (constants.h * self.c) / (constants.e * self.L) # factor 1/e for Joules to eV
        self.K_factor = self.K_to_eV / self.wc
        self.params['mph'] = self.wc * self.params['nr']**2 # effective photon mass
        params['Nk'] = 2 * self.Q0+1 # total number of modes
        self.Nm, self.Nk, self.Nnu = params['Nm'], params['Nk'], params['Nnu']
        self.NE = self.Nm/self.Nk # Number of molecules in an ensemble
        self.dt = params['dt']
        self.params['delta_r'] = params['L']/params['Nk'] # grid spacing in micrometers
        self.params['delta'] = round(params['epsilon'] - self.wc, 5) # detuning
        self.off_diag_indices_Nk = np.where(~np.eye(self.Nk,dtype=bool))
        self.diag_indices_Nk = np.diag_indices(self.Nk)
        
    def get_rates(self, params):
        rates = {}
        for name, val in params.items():
            if 'Gam' or 'gam' in name:
                rates[name] = val
        rates['gam_up'] = rates['gam_nu'] * params['nb']
        rates['gam_down'] = rates['gam_nu'] * (params['nb'] + 1)
        rates['gam_delta'] = rates['gam_up'] - rates['gam_down']
        rates['Gam_delta'] = rates['Gam_up'] - rates['Gam_down']
        return rates

    def get_modes(self):
        Qs = np.array([Q for Q in self.get_Q_range()])
        return Qs, (2*np.pi/self.params['L']) * Qs # integers, inverse microns

    def get_Q_range(self, sign=1, offset=0, reverse=False, start=None):
        if start == None:
            shift = 0
        else:
            shift = start + self.Q0 # e.g. start=0 makes [0, 2Q0] rather than [-Q0,Q0], for indexing arrays
        min_Q = max(-self.Q0, -self.Q0 - sign*offset)
        max_Q = min(self.Q0, self.Q0 - sign*offset)
        if reverse:
            return range(max_Q+shift, min_Q-1+shift, -1)
        # Q in [-Q0, Q0] such that offset + sign*Q is also in this interval
        return range(min_Q+shift, max_Q+1+shift)

    def make_state_dic(self):
        Nk, Nnu = self.Nk, self.Nnu
        slices = {}
        names = ['a', 'lp', 'l0']
        self.state_dic = {
            'a': {'shape': Nk}, 
            'lp': {'shape': (Nnu**2, Nk)}, 
            'l0': {'shape': (2*Nnu**2-1, Nk)}, 
        }
        self.state_split_list, self.state_reshape_list = [], []
        tot = 0
        for name in self.state_dic:
            shape = self.state_dic[name]['shape']
            self.state_reshape_list.append(shape)
            num = np.prod(shape)
            self.state_dic[name]['num'] = num
            self.state_dic[name]['slice'] = slice(tot, tot+num)
            self.state_split_list.append(tot+num)
            tot += num
        self.state_length = np.sum([self.state_dic[name]['num'] for name in self.state_dic]) 
        correct_state_length = Nk + Nk*Nnu**2 + (2*Nnu**2-1)*Nk  
        assert self.state_length == correct_state_length,\
                f'state length is {self.state_length} but should be {correct_state_length}'

    def wrapit(self, meth, msg='', timeit=True):
        if msg:
            logger.info(msg)
        t0 = time()
        ret = meth()
        if timeit:
            logger.info('...done ({:.2f}s)'.format(time()-t0))
        return ret

    def make_coeffs(self):
        '''Dictionary for equation coefficients and constants used in their construction.'''
        coeffs, consts = {}, {}
        gp = self.gp
        params = self.params
        rates = self.rates
        Nm, Nk, Nnu = self.Nm, self.Nk, self.Nnu
        Hvib = Boson(Nnu)
        b, bd, bn, bi = Hvib.b, Hvib.bd, Hvib.n, Hvib.i
        sm, sp, sz, si = Pauli.m, Pauli.p, Pauli.z, Pauli.i
        exciton = params['exciton']
        A = 0.5*params['epsilon']*np.kron(sz, bi) +\
                params['omega_nu']*np.kron(si, bn) +\
                params['omega_nu']*np.sqrt(params['S'])*np.kron(sz, b+bd) + 0j
        kba = False
        if not kba:
            A += 0.25 * (-1j * rates['gam_delta']) * np.kron(si, (bd @ bd - b @ b))
        B = params['gSqrtN'] * np.kron(sp, bi)
        C1 = np.sqrt(rates['Gam_z']) * np.kron(sz, bi)
        if kba:
            sz2 = np.kron(sz, bi)
            S = params['S']
            C2 = np.sqrt(rates['gam_up']) * (np.kron(si, bd) - np.sqrt(S) * sz2)
            C3 = np.sqrt(rates['gam_down']) * (np.kron(si, b) - np.sqrt(S) * sz2)
        else:
            C2 = np.sqrt(rates['gam_up']) * np.kron(si, bd)
            C3 = np.sqrt(rates['gam_down']) * np.kron(si, b)
        Dp = np.sqrt(rates['Gam_up']) * np.kron(sp, bi)
        Dm = np.sqrt(rates['Gam_down']) * np.kron(sm, bi)
        consts['A0'], _discard = gp.get_coefficients(A, sgn=0, eye=True) # discard part proportional to identity
        consts['Bp'] = gp.get_coefficients(B, sgn=1) # N.B. gets i_+ coefficients i.e. traces against lambda_{i_-}
        consts['gam0'] = np.array([gp.get_coefficients(C, sgn=0) for C in [C1, C2, C3]])
        consts['gamp'] = gp.get_coefficients(Dp, sgn=1)
        consts['gamm'] = gp.get_coefficients(Dm, sgn=-1)
        consts['gam00'] = contract('ar,ap->rp', consts['gam0'].conj(), consts['gam0']) # perform sum over mu_0
        consts['gampp'] = contract('i,j->ij', consts['gamp'].conj(), consts['gamp'])
        consts['gammm'] = contract('i,j->ij', consts['gamm'].conj(), consts['gamm'])
        f000 = gp.f_tensor((0,0,0))
        f011 = gp.f_tensor((0,1,1))
        z000 = gp.z_tensor((0,0,0))
        z011 = gp.z_tensor((0,1,1))
        z011_swap = np.swapaxes(z011, 1, 2)
        assert np.allclose(z011, np.conj(z011_swap))
        zm011 = gp.z_tensor((0,-1,-1))
        consts['xi'] = 2 * contract('ipj,p->ij', f000, consts['A0']) \
                + 2 * contract('irq,rp,qpj->ij', f000, consts['gam00'], z000).imag \
                + 2 * contract('ij,aip,bpj->ab', consts['gampp'], f011.conj(), zm011).imag \
                + 2 * contract('ij,aip,bpj->ab', consts['gammm'], f011, z011).imag
        consts['phi0'] = (2/params['Nnu']) * contract('ajq,jq->a', f000, consts['gam00'].imag) \
                + (2/params['Nnu']) * contract('ij,aij->a', consts['gampp'], f011.conj()).imag \
                + (2/params['Nnu']) * contract('ij,aij->a', consts['gammm'], f011).imag
                  # Note gamm00 index order reversed below (conjugate)
        consts['xip'] = - 2 * contract('aij,a->ij', f011, consts['A0']) \
                + 1j * contract('aip,ab,bpj->ij', f011, consts['gam00'], zm011.conj()) \
                - 1j * contract('aip,ba,bpj->ij', f011, consts['gam00'], z011) \
                + 1j * contract('aip,qp,aqj->ij', f011, consts['gammm'], zm011.conj()) \
                - 1j * contract('aip,pq,aqj->ij', f011, consts['gampp'], z011) 
        shifted_Ks = np.fft.ifftshift(self.Ks) # ALL COMPUTATIONS DONE WITH numpy order of modes
        consts['kappa'] = self.kappa(shifted_Ks)
        consts['omega'] = self.omega(shifted_Ks, exciton)
        consts['zetap'] = gp.get_coefficients(np.kron(sp, bi), sgn=1, eye=False)
        consts['zetazeta'] = contract('i,j,aij->a', consts['zetap'],
                             consts['zetap'], z011)
        # CIJ means Jth coeff. of Ith eqn.
        # Written in terms of rescaled a: a_tilda = a / sqrt(Nm)
        # EQ 1 
        coeffs['11_k'] = np.ones(Nk, dtype=complex) 
        for i, K in enumerate(shifted_Ks):
                coeffs['11_k'][i] *= -(1j * self.omega(K, exciton) + 0.5 * self.kappa(K))
        coeffs['12_1'] = 1j * consts['Bp']
        # EQ 2
        coeffs['21_11'] = 1 * consts['xip'] 
        coeffs['22_10'] = 2 * contract('j,aij->ai', consts['Bp'], f011) 
        # EQ 3
        coeffs['31_00'] = 1 * consts['xi'] 
        coeffs['32_0'] = np.broadcast_to(consts['phi0'], (Nk, self.N0)).T # cast phi to a matrix to match dimensions of other variables in equation
        coeffs['33_01'] = 4 * contract('aij,i->aj', f011, consts['Bp']) 
        # Hopfield coefficients 
        consts['zeta_k'] = 0.5 * np.sqrt( (params['epsilon'] - consts['omega'])**2 + 4 * params['gSqrtN']**2 )
        if exciton:
            coeffs['X_k'] = np.zeros_like(shifted_Ks)
            coeffs['Y_k'] = np.ones_like(shifted_Ks)
        else:
            coeffs['X_k'] = np.sqrt(0.5  + 0.5**2 * (params['epsilon'] - consts['omega'])/consts['zeta_k'])
            coeffs['Y_k'] = np.sqrt(0.5  - 0.5**2 * (params['epsilon'] - consts['omega'])/consts['zeta_k'])
        assert np.allclose(coeffs['X_k']**2+coeffs['Y_k']**2, 1.0), 'Hopfield coeffs. not normalised'
        ocoeffs = {}
        # 'pup_l' and 'pup_I' are C^0_{i_0} and D^0 in thesis
        ocoeffs['pup_l'], ocoeffs['pup_I'] = \
                self.gp.get_coefficients(np.kron(Pauli.p1, self.boson.i), sgn=0, eye=True)
        self.consts, self.coeffs, self.ocoeffs = consts, coeffs, ocoeffs  
        
    def omega(self, K, exciton = True):
        # dispersion to match MODEL used by Xu et al. 2023 (self.K_factor set in
        # self.add_useful_params) or no dispersion if initial state is pure exciton
        if exciton:
            return np.zeros_like(K)
        return self.wc * np.sqrt(1 + self.K_factor**2 * K**2)

    def kappa(self, K):
        # uniform
        return self.params['kappa_c'] * np.ones_like(K)

    def nb(self, T):
        if T==0.0:
            return 0.0
        return 1/(np.exp(self.params['omega_nu']/self.params['T'])-1)

    def partition(self, T):
        if T==0.0:
            return 1.0
        if T==np.inf:
            return self.params['Nnu']
        Nnu, omnu = self.params['Nnu'], self.params['omega_nu']
        return (1-np.exp(-Nnu * omnu / T))/(1-np.exp(-omnu/T))

    def thermal_rho_vib(self, T):
        if T==0.0:
            ps = [1.0 if n==0 else 0.0 for n in range(self.Nnu)]
            return np.diag(ps)
        if T==np.inf:
            return (1/self.Nnu) * np.eye(self.Nnu)
        Z = self.partition(T)
        ps = [np.exp(-n * self.params['omega_nu'] / T) for n in range(self.Nnu)]
        return (1/Z) * np.diag(ps)
        
    def initial_state(self):
        """Return initial state on the lower polariton branch as a flattened
        1D array [<a_k>, <lambda_n^i+>, <lambda_n^i0>]"""
        #state = np.zeros(self.state_length, dtype=complex) 
        rho0_vib = self.thermal_rho_vib(self.params['T']) # molecular vibrational density matrix
        shifted_Ks = np.fft.ifftshift(self.Ks) 
        alpha_k = self.params['A']*np.exp(-(shifted_Ks-self.params['k_0'])**2 / (2*self.params['sig_0']**2)) # create gaussian profile at k values
        # build density matrices
        TLS_matrix = np.array([[0.0,0.0],[0.0,1.0]]) # initially in ground state
        a0, lp0, l00 = [], [], [] 
        a0.append(alpha_k*self.coeffs['X_k']) # expectation values of initial a_k (not rescaled)
        beta_n = fft(-alpha_k*self.coeffs['Y_k'], axis=0, norm='ortho')  
        beta_n /= np.sqrt(self.NE) # corrected normalisation
        for n in range(self.Nk):
            U_n = expm(np.array([[0.0, beta_n[n]],[-np.conj(beta_n[n]), 0.0]]))
            U_n_dag = U_n.conj().T
            exciton_matrix_n = U_n @ TLS_matrix @ U_n_dag # initial exciton matrix
            rho0n = np.kron(exciton_matrix_n, rho0_vib) # total density operator
            coeffsp0 = self.gp.get_coefficients(rho0n, sgn=-1, warn=False) # lambda i+
            lp0.append(2*coeffsp0)
            coeffs00 = self.gp.get_coefficients(rho0n, sgn=0, warn=False) # lambda i0
            l00.append(2*coeffs00)
        a0 /= np.sqrt(self.Nm) # rescale initial state (a -> a-tilda)
        # flatten and concatenate to match input state structure of RK (1d array)
        state = np.concatenate((a0, lp0, l00), axis=None)
        assert len(state) == self.state_length, 'Initial state does not match the required dimensions'
        return state

    def split_reshape_return(self, state, check_rescaled=False, copy=False):
        """Return views of variables <a_k>, <lambda_n^i+>, <lambda_n^i0>
        in state reshaped into conventional multidimensional arrays."""
        split = np.split(state, self.state_split_list) # a, lp, l0
        # reshape each array (len+1 as rescale_int removed)
        reshaped = [split[i].reshape(self.state_reshape_list[i]) for i in range(len(self.state_reshape_list))]
        # N.B. np.split, reshape returns VIEWS of original array; BEWARE mutations 
        # - copy if want to change return without mutating original state variable
        if copy:
            reshaped = [np.copy(X) for X in reshaped]
        return reshaped
        
    def eoms(self, t, state):
        """Equations of motion as in mf_eoms_fourier.pdf"""
        #state = self.initial_state()
        C = self.coeffs
        a, lp, l0 = self.split_reshape_return(state) 
        # Calculate DFT
        alpha = ifft(a, axis=0, norm = 'forward')
        # EQ 1 
        pre_c = contract('i,in->n', C['12_1'], lp)   
        post_c = fft(pre_c, axis=0, norm='forward')
        dy_a = C['11_k'] * a + np.conj(post_c)
        # EQ 2
        dy_lp = contract('ij,jn->in', C['21_11'], lp) + contract('ai,n,an->in', C['22_10'], np.conj(alpha), l0)
        # EQ 3
        dy_l0 = contract('aj,jn->an', C['31_00'], l0) + C['32_0'] + contract('aj,n,jn->an', C['33_01'], alpha, lp).real
        # flatten and concatenate to match input state structure of RK (1d array)
        dy_state = np.concatenate((dy_a, dy_lp, dy_l0), axis=None)
        return dy_state

    def quick_integration(self, state, tf, ti = 0.0):
        """Integrates the equations of motion from t = 0 to tf using solve_ivp."""
        assert tf > ti, 'Final time is smaller than initial time'
        #state_i = self.initial_state()
        ivp = solve_ivp(self.eoms, [ti,tf], state, dense_output=True)
        state_f = ivp.y[:,-1]
        #state_t = ivp.t[-1]
        return state_f

    def calculate_n_photon(self, a, kspace = False):
        """Calculates photonic population. Use evolve = True to get correct normalisation if evolving the state in time."""
        if kspace:
            return np.outer(np.conj(a),a)*self.Nm 
        a_r = ifft(a, norm = 'ortho')
        return np.conj(a_r)*a_r*self.Nm
        
    def calculate_n_molecular_real(self, l0): #, kspace = False):
        """Calculates molecular population (in real space)."""
        #zzzl0 = contract('a,an->n', self.consts['zetazeta'], l0) # zeta(i+)zeta(j+)Z(i0i+j+)<lambda(i0)>
        #n_M_r = (zzzl0 + 0.5)*self.NE
        n_M_r = self.NE * (contract('a,an->n', self.ocoeffs['pup_l'], l0) + self.ocoeffs['pup_I'])
        return n_M_r

    def calculate_n_bright_real(self, l0, lp): #, kspace = False):
        """Calculates population of bright exciton state (in real space)."""
        zlp = contract('i,in->n', self.consts['zetap'], lp) # zeta(i+)<lambda(i+)>
        zzlplp = np.outer(zlp, np.conj(zlp)) # zeta(i+)zeta(j+)<lambda(i+)><lambda(j-)>
        zzzl0 = contract('a,an->n', self.consts['zetazeta'], l0)
        n_B_r = (self.NE - 1)*zzlplp + zzzl0 + 0.5
        #print(-(self.NE - 1)*zzlplp + (self.NE - 1)*(zzzl0 + 0.5)/self.NE)
        return n_B_r

    def calculate_upper_lower_polariton(self, a, lp, l0): 
        """Calculate coherences <sigma_k'(+)sigma_k(-)> and <a_k sigma_k(+)>, 
        as well as upper and lower polariton populations, all in k-space."""
        gp = self.gp
        z011 = gp.z_tensor((0,1,1))
        sNm = np.sqrt(self.Nm)
        sNE = np.sqrt(self.NE)
        n_k = self.calculate_n_photon(a, kspace = True) # photonic population, no rescaling (initial state)
        sig_plus = contract('i,in->n', self.consts['zetap'], lp)  # zeta(i+)<lambda(i+)>
        post_sigp = fft(sig_plus, axis = 0, norm = 'ortho') # (in k-space, negative exponents) 
        asig_k = np.outer(a, post_sigp)*sNm*sNE # expectation value <a_k sigma(k'+)>; includes rescaling
        sigsig_k1 = np.outer(post_sigp, np.conj(post_sigp))*self.NE # first term of <sigma(k'+)sigma(k-)>
        sig_abs_sq = sig_plus * sig_plus.conj()  # zeta(i+)zeta(j+)<lambda(i+)><lambda(j-)>
        sigsig_k2 = ifft(sig_abs_sq, axis=0, norm='backward') # (in k-space, positive exponents)
        pre_l0 = contract('a,an->n', self.consts['zetazeta'], l0) # zeta(i+)zeta(j+)Z+(i0i+j+)<lambda(i0)> (in real space)
        post_l0 = ifft(pre_l0, axis = 0, norm = 'backward') # (in k-space, positive exponents)
        sigsig = np.zeros((self.Nk, self.Nk), dtype=complex) # initialise array for <sigma(k'+)sigma(k-)> values
        n_L = np.zeros((self.Nk, self.Nk), dtype=complex) # initialise array for lower polariton population values
        n_U = np.zeros((self.Nk, self.Nk), dtype=complex) # initialise array for upper polariton population values
        delta_k = np.eye(self.Nk) # equivalent to 2*Nnu*zeta(i+)zeta(j+)delta(i+,j+)
        for p, k in itertools.product(range(self.Nk), range(self.Nk)): 
            sigsig[p,k] = sigsig_k1[k,p] - sigsig_k2[k-p] + post_l0[k-p] + 0.5 * delta_k[k,p]
            n_U[p,k] = self.coeffs['X_k'][p] * self.coeffs['X_k'][k] * sigsig[p,k] \
            + self.coeffs['Y_k'][p] * self.coeffs['Y_k'][k] * n_k[p,k] \
            + self.coeffs['X_k'][p] * self.coeffs['Y_k'][k] * asig_k[k,p] \
            + self.coeffs['Y_k'][p] * self.coeffs['X_k'][k] * np.conj(asig_k[p,k])
            n_L[p,k] = self.coeffs['X_k'][p] * self.coeffs['X_k'][k] * n_k[p,k] \
            + self.coeffs['Y_k'][p] * self.coeffs['Y_k'][k] * sigsig[p,k] \
            - self.coeffs['X_k'][p] * self.coeffs['Y_k'][k] * np.conj(asig_k[p,k]) \
            - self.coeffs['Y_k'][p] * self.coeffs['X_k'][k] * asig_k[k,p]
        return n_k, n_L, n_U, sigsig, asig_k
        
    def calculate_observables(self, state, kspace = True):
        """Calculate coherences, polariton, photon and molecular numbers for a given state. n_L, n_U, sigsig, asig_k always in k space; n_M and n_B always in real space, n_k optional in real space.""" 
        a, lp, l0 = self.split_reshape_return(state) 
        n_M = self.calculate_n_molecular_real(l0) # molecular population (real space)
        n_B = self.calculate_n_bright_real(l0, lp) # bright state population (real space)
        n_k, n_L, n_U, sigsig, asig_k = self.calculate_upper_lower_polariton(a, lp, l0) # k-space, initial populations (not evolved)
        if not kspace:
            #n_k = self.calculate_n_photon(a, kspace = False) # real space, equivalent to double fourier transform
            nkft1 = ifft(n_k, axis=0, norm = 'ortho') # double fourier transform
            n_k = fft(nkft1, axis=-1, norm = 'ortho')
            nlft1 = ifft(n_L, axis=0, norm = 'ortho') # double fourier transform
            n_L = fft(nlft1, axis=-1, norm = 'ortho')
            nuft1 = ifft(n_U, axis=0, norm = 'ortho') # double fourier transform
            n_U = fft(nuft1, axis=-1, norm = 'ortho')
            #nbft1 = ifft(sigsig, axis=0, norm = 'ortho') # double fourier transform
            #n_B = fft(nbft1, axis=-1, norm = 'ortho')
        else:
            #nmft1 = fft(n_M, axis=0) # double fourier transform
            n_M = sigsig
            nbft1 = fft(n_B, axis=0, norm = 'ortho') # double fourier transform
            n_B = ifft(nbft1, axis=1, norm = 'ortho')
        n_D = n_M - n_B
        return n_k, n_M, n_L, n_U, sigsig, asig_k, n_B, n_D

    def calculate_diagonal_elements(self, state, kspace):
        n_k, n_M, n_L, n_U, sigsig, asig_k, n_B, n_D = self.calculate_observables(state, kspace)
        assert np.allclose(np.diag(n_M).imag, 0.0), "Molecular population has imaginary components"
        assert np.allclose(np.diag(n_k).imag, 0.0), "Photon population has imaginary components"
        assert np.allclose(np.diag(n_L).imag, 0.0), "Lower polariton population has imaginary components"
        assert np.allclose(np.diag(n_U).imag, 0.0), "Upper polariton population has imaginary components"
        assert np.allclose(np.diag(n_B).imag, 0.0), "Bright exciton population has imaginary components"
        assert np.allclose(np.diag(n_D).imag, 0.0), "Dark exciton population has imaginary components"
        assert np.allclose(np.diag(sigsig).imag, 0.0), "The coherences have imaginary components"
        #assert np.allclose(np.diag(asig_k).imag, 0.0), "The coherences have imaginary components"  # is that possible?
        if not kspace:
            #n_k_diag = fftshift(n_k.real)#fftshift(np.diag(n_k).real) # shift back so that k=0 component is at the center
            n_M_diag = fftshift(n_M.real) # shift back so that k=0 component is at the center        
        else:
            n_M_diag = fftshift(np.diag(n_M).real) # shift back so that k=0 component is at the center  
        n_k_diag = fftshift(np.diag(n_k).real) # shift back so that k=0 component is at the center
        #n_M_diag = fftshift(n_M.real) # shift back so that k=0 component is at the center        
        n_L_diag = fftshift(np.diag(n_L).real) # shift back so that k=0 component is at the center
        n_U_diag = fftshift(np.diag(n_U).real) # shift back so that k=0 component is at the center
        n_B_diag = fftshift(np.diag(n_B).real) # shift back so that k=0 component is at the center
        n_D_diag = fftshift(np.diag(n_D).real) # shift back so that k=0 component is at the center n_M_diag - n_B_diag
        sigsig_diag = fftshift(np.diag(sigsig).real) # shift back so that k=0 component is at the center
        asig_k_diag = fftshift(np.diag(asig_k).real) # shift back so that k=0 component is at the center
        return n_k_diag, n_M_diag, n_L_diag, n_U_diag, n_B_diag, n_D_diag, sigsig_diag, asig_k_diag

    def calculate_evolved_observables(self, tf = None, fixed_position_index = False, kspace = False):
        t_fs = np.arange(0.0, tf, step = self.dt) # create array of integration times 
        state = self.initial_state() # build initial state
        n_k_arr, n_M_arr, n_L_arr, n_U_arr, n_B_arr, n_D_arr, sigsig_arr, asig_k_arr = self.calculate_diagonal_elements(state, kspace) # calculate observables for initial state
        if fixed_position_index != None: # pick values of observables at fixed k/r value
            n_k_arr = n_k_arr[fixed_position_index]
            n_M_arr = n_M_arr[fixed_position_index]
            n_L_arr = n_L_arr[fixed_position_index]
            n_U_arr = n_U_arr[fixed_position_index]
            n_B_arr = n_B_arr[fixed_position_index]
            n_D_arr = n_D_arr[fixed_position_index]
            sigsig_arr = sigsig_arr[fixed_position_index]
            asig_k_arr = asig_k_arr[fixed_position_index]
        for i in range(len(t_fs)-1):
            state = self.quick_integration(state, tf = t_fs[i+1], ti = t_fs[i]) # integrate eoms from ti to tf
            n_k_diag, n_M_diag, n_L_diag, n_U_diag, n_B_diag, n_D_diag, sigsig_diag, asig_k_diag = self.calculate_diagonal_elements(state, kspace) # calculate observables for evolved state
            if fixed_position_index != False: # append with calculated observables at a fixed k/r value at each time step 
                n_k_arr = np.append(n_k_arr, n_k_diag[fixed_position_index])
                n_M_arr = np.append(n_M_arr, n_M_diag[fixed_position_index])
                n_L_arr = np.append(n_L_arr, n_L_diag[fixed_position_index])
                n_U_arr = np.append(n_U_arr, n_U_diag[fixed_position_index])
                n_B_arr = np.append(n_B_arr, n_B_diag[fixed_position_index])
                n_D_arr = np.append(n_D_arr, n_D_diag[fixed_position_index])
                sigsig_arr = np.append(sigsig_arr, sigsig_diag[fixed_position_index])
                asig_k_arr = np.append(asig_k_arr, asig_k_diag[fixed_position_index])
            else: # append with calculated observables for all k/r values at each time step 
                n_k_arr = np.append(n_k_arr, n_k_diag)
                n_M_arr = np.append(n_M_arr, n_M_diag)
                n_L_arr = np.append(n_L_arr, n_L_diag)
                n_U_arr = np.append(n_U_arr, n_U_diag)
                n_B_arr = np.append(n_B_arr, n_B_diag)
                n_D_arr = np.append(n_D_arr, n_D_diag)
                sigsig_arr = np.append(sigsig_arr, sigsig_diag)
                asig_k_arr = np.append(asig_k_arr, asig_k_diag)
        if fixed_position_index == None: # check that arrays of observables have the correct length
            assert len(n_k_arr) == self.Nk*(len(t_fs)), 'Length of evolved photonic population array does not have the required dimensions'
            assert len(n_M_arr) == self.Nk*(len(t_fs)), 'Length of evolved molecular population array does not have the required dimensions'
            assert len(n_L_arr) == self.Nk*(len(t_fs)), 'Length of evolved lower polariton population array does not have the required dimensions'
            assert len(n_U_arr) == self.Nk*(len(t_fs)), 'Length of evolved upper polariton population array does not have the required dimensions'
            assert len(n_B_arr) == self.Nk*(len(t_fs)), 'Length of evolved bright exciton population array does not have the required dimensions'
            assert len(n_D_arr) == self.Nk*(len(t_fs)), 'Length of evolved dark exciton population array does not have the required dimensions'
        return t_fs, n_k_arr, n_M_arr, n_B_arr, n_D_arr, n_L_arr, n_U_arr, sigsig_arr # sigsig_arr, asig_k_arr
                
    def plot_evolution(self, savefig = False, tf = 1.0, fixed_position_index = 6, kspace = False):
        times, n_k_arr, n_M_arr, n_B_arr, n_D_arr, n_L_arr, n_U_arr, sigsig_arr = self.calculate_evolved_observables(tf, fixed_position_index, kspace)
        fig, ax = plt.subplots(1,1,figsize = (6,4))
        ax.plot(times, n_B_arr, label = '$n_{B}$')
        ax.scatter(times, n_B_arr, marker = '.')
        ax.plot(times, n_k_arr, label = '$n_{phot}$')
        ax.scatter(times, n_k_arr, marker = '.')
        #ax.plot(times, n_D_arr, label = '$n_{D}$')
        #x.scatter(times, n_D_arr, marker = '.')
        ax.set_xlabel('time')
        ax.set_ylabel(f'n(k={self.Ks[fixed_position_index]})')
        ax.legend()
        ax.set_title('Photonic and Excitonic Population Evolution over Time')
        if savefig:
            plt.savefig(fname = 'evolution.jpg', format = 'jpg')

    def plot_waterfall(self, n_L = False, n_B = False, n_k = False, savefig = False, tf = 101.1, kspace = False, legend = False, step = 10):
        step = 2*step*self.dt
        slices = np.arange(0.0, tf, step)
        times, n_k_arr, n_M_arr, n_B_arr, n_D_arr, n_L_arr, n_U_arr, sigsig_arr = self.calculate_evolved_observables(tf, kspace)
        fig = plt.figure(figsize=(10,6), layout = 'tight')
        ax = fig.add_subplot(projection='3d')
        colors = plt.cm.coolwarm(np.linspace(0,1,len(slices)))
        if n_B:
            n_arr = n_B_arr
            ax.set_zlabel('$n_{B}(r_n)$')
        if n_k:
            n_arr = n_k_arr
            ax.set_zlabel('$n_{phot}(r_n)$')
        if n_L:
            n_arr = n_L_arr
            ax.set_zlabel('$n_{L}(r_n)$')
        assert isinstance(n_arr, np.ndarray), "Please, specify one of n_L, n_B and n_k"
        for i in range(len(slices)):
            index = np.where(times == slices[i])
            n_i = n_arr[i*self.Nk:(i+1)*self.Nk]
            ax.plot(self.Ks, n_i, label = f't = {slices[i]}', zdir = 'y', zs=slices[i], zorder = (len(slices)-i), color=colors[i])
        if kspace:
            ax.set_xlabel('$k$')
        else:
            ax.set_xlabel('$r_n$')
        ax.set_ylabel('t')
        ax.set_title('Time Snapshots of Wavepacket Evolution')
        if legend:
            ax.legend()
        if savefig:
            plt.savefig(fname = 'plot_waterfall.jpg', format = 'jpg')
        
    def plot_initial_populations(self, savefig = False, kspace = False):
        state_i = self.initial_state()
        n_k_diag, n_M_diag, n_L_diag, n_U_diag, n_B_diag, n_D_diag, sigsig_diag, asig_k_diag = self.calculate_diagonal_elements(state_i, kspace)
        #print(n_B_diag[19])
        fig1, ax1 = plt.subplots(5,1,figsize = (12,10),sharex = True)
        ax1[0].scatter(self.Ks, n_U_diag, marker = '.')
        ax1[0].plot(self.Ks, n_U_diag)
        ax1[1].scatter(self.Ks, n_L_diag, marker = '.')
        ax1[1].plot(self.Ks, n_L_diag)
        ax1[2].scatter(self.Ks, n_k_diag, marker = '.')
        ax1[2].plot(self.Ks, n_k_diag)
        if kspace:
            ax1[0].set_ylabel('$n_U(k)$')
            ax1[1].set_ylabel('$n_L(k)$')
            ax1[2].set_ylabel('$n_k(k)$')
        else:
            ax1[0].set_ylabel('$n_U(r_n)$')
            ax1[1].set_ylabel('$n_L(r_n)$')
            ax1[2].set_ylabel('$n_k(r_n)$')
            ax1[2].set_xlabel('$r_n$')            
        ax1[3].scatter(self.Ks, sigsig_diag, marker = '.')
        ax1[3].plot(self.Ks, sigsig_diag)
        ax1[3].set_ylabel('$< \sigma_{k}^{+} \sigma_k^{-}>$')
        ax1[4].scatter(self.Ks, asig_k_diag, marker = '.')
        ax1[4].plot(self.Ks, asig_k_diag)
        ax1[4].set_xlabel('$k$')
        ax1[4].set_ylabel('$< a_k \sigma_k^{+}>$')
        fig1.suptitle('Initial Populations and Coherences in k-space')
        fig1.tight_layout(h_pad=0.2)
        if savefig:
            plt.savefig(fname = 'state_i.jpg', format = 'jpg')
        fig2, ax2 = plt.subplots(3,1,figsize = (12,6),sharex = True)                
        ax2[0].scatter(self.Ks, n_M_diag, marker = '.')
        ax2[0].plot(self.Ks, n_M_diag)
        ax2[0].set_ylabel('$n_M(r_n)$')
        ax2[0].set_xlabel('$r_n$')
        ax2[1].scatter(self.Ks, n_B_diag, marker = '.')
        ax2[1].plot(self.Ks, n_B_diag)
        ax2[1].set_ylabel('$n_B(r_n)$')
        ax2[1].set_xlabel('$r_n$')
        ax2[2].scatter(self.Ks, n_D_diag, marker = '.')
        ax2[2].plot(self.Ks, n_D_diag)
        ax2[2].set_ylabel('$n_D(r_n)$')
        ax2[2].set_xlabel('$r_n$')
        fig2.suptitle('Initial Excitonic Populations in Real Space')
        fig2.tight_layout(h_pad=0.2)
        if savefig:
            plt.savefig(fname = 'brightdark_populations.jpg', format = 'jpg')
            
if __name__ == '__main__':
    logging.basicConfig(
        format='%(filename)s L%(lineno)s %(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%H:%M')
    params = {
        'Q0': 15, # how many modes either side of K0 (or 0 for populations) to include; 2*Q0+1 modes total 
        'Nm': 6002, # Number of molecules
        'Nnu': 1, # Number of vibrational levels for each molecules
        'L': 10.0, # Crystal propagation length, inverse micro meters
        'nr': 1.0, # refractive index, sets effective speed of light c/nr
        'omega_c': 1.94, # omega_0 = 1.94eV (Fig S4C)
        'epsilon': 0.0, #2.14, # exciton energy, detuning omega_0-epsilon (0.2eV for model I in Xu et al. 2023)
        'gSqrtN': 0.15, # light-matter coupling
        'kappa_c': 3e-3, # photon loss
        'Gam_z': 0.0, # molecular pure dephasing
        'Gam_up': 0.0, # molecular pumping
        'Gam_down': 0.0, #1e-7, # molecular loss
        'S': 0.0, # Huang-Rhys parameter
        'omega_nu': 0.00647, # vibrational energy spacing
        'T': 0.026, # k_B T in eV (.0259=300K, .026=302K)
        'gam_nu': 0.0, #0.01, # vibrational damping rate
        'initial_state': 'incoherent', # or incoherent
        'A': 0.8, # amplitude of initial wavepacket
        'k_0': 0.0, # central wavenumber of initial wavepacket
        'sig_0': 4.0, # s.d. of initial wavepacket
        'atol': 1e-9, # solver tolerance
        'rtol': 1e-6, # solver tolerance
        'dt': 0.5, # determines interval at which solution is evaluated. Does not effect the accuracy of solution, only the grid at which observables are recorded
        'exciton': True, # if True, initial state is pure exciton; if False, a lower polariton initial state is created
        }
    
    htc = HTC(params)
    htc.quick_integration(100)
    #htc.calculate_evolved_observables(tf = 2.1, fixed_position_index = 5)
    htc.plot_evolution(tf = 100.1, savefig = True, fixed_position_index = 16, kspace = False)
    htc.plot_initial_populations(kspace = False)
    htc.plot_waterfall(legend = True, n_L = True)
