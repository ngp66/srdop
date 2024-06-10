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
            'Nnu':1, # Number of vibrational levels for each molecules
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
            'A':1, # amplitude of initial wavepacket
            'k_0':0.01, # central wavenumber of initial wavepacket
            'sig_0':0.1, # s.d. of initial wavepacket
            #'sig_f':0, # s.d. in microns instead (if specified)
            'atol':1e-9, # solver tolerance
            'rtol':1e-6, # solver tolerance
            'dt': 0.5, # determines interval at which solution is evaluated. Does not effect the accuracy of solution, only the grid at which observables are recorded
            #'lowpass': 0.01, # used for smoothing oscillations in observable plots
            #'calculate_lp': False, # compute LP population for each state - adds overhead due to for-loops
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
        #self.labels = self.get_labels()

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
        # dictionary for equation coefficients and constants used in their construction
        coeffs, consts = {}, {}
        gp = self.gp
        params = self.params
        rates = self.rates
        Nm, Nk, Nnu = self.Nm, self.Nk, self.Nnu
        Hvib = Boson(Nnu)
        b, bd, bn, bi = Hvib.b, Hvib.bd, Hvib.n, Hvib.i
        sm, sp, sz, si = Pauli.m, Pauli.p, Pauli.z, Pauli.i
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
        consts['omega'] = self.omega(shifted_Ks)
        # CIJ means Jth coeff. of Ith eqn.
        # Written in terms of rescaled a: a_tilda = a / sqrt(Nm)
        # EQ 1 
        coeffs['11_k'] = np.ones(Nk, dtype=complex) 
        for i, K in enumerate(shifted_Ks):
                coeffs['11_k'][i] *= -(1j * self.omega(K) + 0.5 * self.kappa(K))
        coeffs['12_1'] = 1j * consts['Bp']
        # EQ 2
        coeffs['21_11'] = 1 * consts['xip'] 
        coeffs['22_10'] = 2 * contract('j,aij->ai', consts['Bp'], f011) 
        # EQ 3
        coeffs['31_00'] = 1 * consts['xi'] 
        coeffs['32_0'] = np.broadcast_to(consts['phi0'], (Nk, self.N0)).T   # cast phi to a matrix to match dimensions of other variables in the equation
        coeffs['33_01'] = 4 * contract('aij,i->aj', f011, consts['Bp']) 
        # Hopfield coefficients 
        consts['zeta_k'] = 0.5 * np.sqrt( (params['epsilon'] - consts['omega'])**2 + 4 * params['gSqrtN']**2 )
        coeffs['X_k'] = np.sqrt(0.5  + 0.5**2 * (params['epsilon'] - consts['omega'])/consts['zeta_k'])
        coeffs['Y_k'] = np.sqrt(0.5  - 0.5**2 * (params['epsilon'] - consts['omega'])/consts['zeta_k'])
        assert np.allclose(coeffs['X_k']**2+coeffs['Y_k']**2, 1.0), 'Hopfield coeffs. not normalised'
        self.consts, self.coeffs = consts, coeffs  
        
    def omega(self, K):
        # dispersion to match MODEL used by Xu et al. 2023 (self.K_factor set in
        # self.add_useful_params)
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
        TLS_matrix = np.array([[0,0],[0,1]]) # initially in ground state
        a0, lp0, l00 = [], [], [] 
        a0.append(alpha_k*self.coeffs['X_k']) # expectation values of initial a_k (not rescaled)
        for n in range(self.Nk):
            beta_n = fft(-alpha_k*self.coeffs['Y_k'], axis=0, norm='ortho')[n]  #is the norm truly ortho here? or stick to previous convention?
            beta_n *= np.sqrt(self.Nm) # is the normalisation correct? Nm rather than Nk
            U_n = expm(np.array([[0, beta_n],[np.conj(beta_n), 0]]))
            U_n_dag = U_n.conj().T 
            exciton_matrix_n = U_n @ TLS_matrix @ U_n_dag # initial exciton matrix
            rho0n = np.kron(exciton_matrix_n, rho0_vib) # total density operator
            coeffsp0 = self.gp.get_coefficients(rho0n, sgn=1, warn=False) # lambda i+
            lp0.append(coeffsp0)
            coeffs00 = self.gp.get_coefficients(rho0n, sgn=0, warn=False) # lambda i0
            l00.append(coeffs00)
        # flatten and concatenate to match input state structure of RK (1d array)
        state = np.concatenate((a0, lp0, l00), axis=None)
        assert np.allclose(len(state) - self.state_length, 0.0), 'Initial state does not match the required dimensions'
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
        
    def eoms(self,t, state):
        """Equations of motion as in mf_eoms_fourier.pdf"""
        #state = self.initial_state()
        C = self.coeffs
        a, lp, l0 = self.split_reshape_return(state) 
        a /= np.sqrt(self.Nm) # rescale initial state (a -> a-tilda)
        # Calculate DFT
        alpha = fft(a, axis=0)  
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

    def quick_integration(self, tf):
        """Integrates the equations of motion from t = 0 to tf using solve_ivp."""
        in_state = self.initial_state().real
        ivp = solve_ivp(self.eoms, [0,tf], in_state, dense_output=True)
        last_t_index = len(ivp.t)
        state_f = []
        for i in range(self.state_length):
            state_f.append(ivp.y[i][last_t_index-1])
        a_f, lp_f, l0_f = self.split_reshape_return(state_f) 
        return a_f, lp_f, l0_f


if __name__ == '__main__':
    logging.basicConfig(
        format='%(filename)s L%(lineno)s %(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%H:%M')
    params = {
        'Q0': 2, # how many modes either side of K0 (or 0 for populations) to include; 2*Q0+1 modes total 
        'Nm': 4, # Number of molecules
        'Nnu': 2, # Number of vibrational levels for each molecules
        'L': 10.0, # Crystal propagation length, inverse micro meters
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
        'initial_state': 'incoherent', # or incoherent
        'A':1, # amplitude of initial wavepacket
        'k_0':0.01, # central wavenumber of initial wavepacket
        'sig_0':0.1, # s.d. of initial wavepacket
        #'sig_f':0, # s.d. in microns instead (if specified)
        'atol':1e-9, # solver tolerance
        'rtol':1e-6, # solver tolerance
        #'dt': 0.5, # determines interval at which solution is evaluated. Does not effect the accuracy of solution, only the grid at which observables are recorded
        }
    
    htc = HTC(params)
    print(htc.quick_integration(100))