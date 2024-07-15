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
from time import time
from pprint import pprint, pformat
from copy import copy
from scipy.integrate import solve_ivp, RK45, DOP853
from scipy.linalg import expm
SOLVER = RK45 # or DOP853
from scipy import constants
import itertools
from scipy.fft import fft, fft2, ifft, ifft2, fftshift, ifftshift # recommended over numpy.fft
try:
    import pretty_traceback
    pretty_traceback.install()
except ModuleNotFoundError:
    # colored traceback not supported
    pass
from decimal import Decimal
from matplotlib.ticker import FormatStrFormatter

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
            'Q0': 120, # how many modes either side of K0 (or 0 for populations) to include; 2*Q0+1 modes total 
            'Nm': 6001, # Number of molecules
            'Nnu': 1, # Number of vibrational levels for each molecules
            'L': 60.0, # Crystal propagation length, inverse micro meters
            'nr': 1.0, # refractive index, sets effective speed of light c/nr
            'omega_c': 1.94, # omega_0 = 1.94eV (Fig S4C)
            'epsilon': 2.14, # exciton energy, detuning omega_0-epsilon (0.2eV for model I in Xu et al. 2023)
            'gSqrtN': 0.15, # light-matter coupling
            'kappa_c': 3e-3, # photon loss
            'Gam_z': 0.0, # molecular pure dephasing
            'Gam_up': 0.0, # molecular pumping
            'Gam_down': 1e-7, # molecular loss
            'S': 0.0, # Huang-Rhys parameter
            'omega_nu': 0.00647, # vibrational energy spacing
            'T': 0.026, # k_B T in eV (.0259=300K, .026=302K)
            'gam_nu': 0.015, # vibrational damping rate
            'initial_state': 'photonic', # or incoherent
            'Am': 0.1, # amplitude of initial wavepacket
            'K_0': 50.0, # central wavenumber of initial wavepacket
            'sig_0': 4.0, # s.d. of initial wavepacket
            #'sig_f':0, # s.d. in microns instead (if specified)
            'atol':1e-9, # solver tolerance
            'rtol':1e-6, # solver tolerance
            'dt': 0.5, # determines interval at which solution is evaluated. Does not effect the accuracy of solution, only the grid at which observables are recorded
            'exciton': False, # if True, initial state is pure exciton; if False, a lower polariton initial state is created
            'photon': False, # if True, initial state is pure exciton; if False, a lower polariton initial state is created
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
        '''Creates dictionary RATES with dissipative parameters'''
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
        '''Creates lists of wavenumbers [-K0,K0] and [-k0,k0]. Note capitalised K0 denotes unitless integers here after, 
        while k0 stands for wavenumber values in inverse microns'''
        
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
        '''Creates dictionary for the state parameters and a slicing list for further reshaping procedures (see self.split_reshape_return)'''
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

    def make_coeffs(self, S = None, Gam_z = None):
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
        photon = params['photon']
        if S == None:
            S = params['S']
        if Gam_z == None:
            Gam_z = rates['Gam_z']
        A = 0.5*params['epsilon']*np.kron(sz, bi) +\
                params['omega_nu']*np.kron(si, bn) +\
                params['omega_nu']*np.sqrt(S)*np.kron(sz, b+bd) + 0j
        kba = False
        if not kba:
            A += 0.25 * (-1j * rates['gam_delta']) * np.kron(si, (bd @ bd - b @ b))
        B = params['gSqrtN'] * np.kron(sp, bi)
        C1 = np.sqrt(Gam_z) * np.kron(sz, bi)
        if kba:
            sz2 = np.kron(sz, bi)
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
        consts['zeta_k'] = 0.5 * np.sqrt( (params['epsilon'] - consts['omega'])**2 + 4*params['gSqrtN']**2 )
        if exciton:
            coeffs['X_k'] = np.zeros_like(shifted_Ks)
            coeffs['Y_k'] = np.ones_like(shifted_Ks)
        elif photon:
            coeffs['X_k'] = np.ones_like(shifted_Ks)
            coeffs['Y_k'] = np.zeros_like(shifted_Ks)            
        else:
            coeffs['X_k'] = np.sqrt(0.5  + 0.5**2 * (params['epsilon'] - consts['omega'])/consts['zeta_k'])
            coeffs['Y_k'] = np.sqrt(0.5  - 0.5**2 * (params['epsilon'] - consts['omega'])/consts['zeta_k'])
        assert np.allclose(coeffs['X_k']**2+coeffs['Y_k']**2, 1.0), 'Hopfield coeffs. not normalised'
        self.consts, self.coeffs = consts, coeffs  
        
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
        
    def initial_state(self, K0):
        """Returns initial state as a Gaussian wavepacket on the lower polariton branch.
        
           Outputs: state [array of floats] - flattened 1D array [<a_k>, <lambda_n^i+>, <lambda_n^i0>] that defines the initial state"""
        
        rho0_vib = self.thermal_rho_vib(self.params['T']) # molecular vibrational density matrix
        shifted_Ks = np.fft.ifftshift(self.Ks) 
        alpha_k = self.params['Am']*np.exp(-(shifted_Ks-K0)**2 / (2*self.params['sig_0']**2)) # create gaussian profile at k values
        # build density matrices
        TLS_matrix = np.array([[0.0,0.0],[0.0,1.0]]) # initially in ground state
        a0, lp0, l00 = [], [], [] 
        a0.append(alpha_k*self.coeffs['X_k']) # expectation values of initial a_k (not rescaled)
        beta_n = ifft(-alpha_k*self.coeffs['Y_k'], axis=0, norm='ortho')  
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
        lp0 = np.array(lp0).T # put n index last for einstein summation
        l00 = np.array(l00).T
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
        
    def eoms(self, t, state_arr):
        """Equations of motion as in mf_eoms_fourier.pdf
        
           Inputs: t [float]: time. Variable used by solve_ivp in integration routines
                   state [array of floats]: initial conditions. See self.initial_state() for correct shape and dimensions
           Outputs: dy_state [array of floats]: state evolved by the equations of motion. Same shape as STATE"""
        
        C = self.coeffs
        a_vals, lp_vals, l0_vals = self.split_reshape_return(state_arr) 
        # Calculate DFT
        alpha = ifft(a_vals, axis=0, norm = 'forward')
        # EQ 1 
        pre_c = contract('i,in->n', C['12_1'], lp_vals)   
        post_c = ifft(pre_c, axis=0, norm='backward')
        dy_a = C['11_k'] * a_vals + np.conj(post_c)
        # EQ 2
        dy_lp = contract('ij,jn->in', C['21_11'], lp_vals) + contract('ai,n,an->in', C['22_10'], np.conj(alpha), l0_vals)
        # EQ 3
        dy_l0 = contract('aj,jn->an', C['31_00'], l0_vals) + C['32_0'] + contract('aj,n,jn->an', C['33_01'], alpha, lp_vals).real
        # flatten and concatenate to match input state structure of RK (1d array)
        dy_state = np.concatenate((dy_a, dy_lp, dy_l0), axis=None)
        return dy_state

    def stepwise_integration(self, tf, y0=None, ti=0.0, dt=None):
        """Integrates the equations of motion from state y0 
        at t = 0 to tf using explicit Runge-Kutta
        Evaluates on grid spacing dt (default self.params['dt'])
        Solver is called directly in a loop allowing for calculation 
        of observables on the fly.
        Returns array of times and corresponding values of observables
        [currently just entire state]"""
        assert tf > ti, 'Final time is smaller than initial time'
        if dt is None:
            dt = self.params['dt']
        if y0 is None:
            y0 = self.initial_state()
        t_eval = np.arange(ti, tf+dt/2, dt)
        num_t = len(t_eval)
        num_checkpoints = 6 # checkpoints at 0, 20%,...
        checkpoint_spacing = int(round(num_t/num_checkpoints))
        checkpoints = np.linspace(0, num_t-1, num=num_checkpoints, dtype=int)
        next_check_i = 1
        solver_t = [] # keep track of solver times (not fixed grid)
        last_solver_i = 0
        observables = {'a_k': np.zeros((num_t, self.Nk), dtype=complex),
                       #'etc':  TODO
                       }
        states = []
        tic = time() # time the computation
        solver = SOLVER(self.eoms,
                        t0=0.0,
                        y0=y0,
                        t_bound=tf,
                        rtol=1e-10,
                        atol=1e-10,
                        )
        # Save initial state
        t_index = 0
        assert solver.t == t_eval[t_index], 'Solver initial time incorrect'
        states.append(solver.y)
        solver_t.append(solver.t)
        t_index += 1
        next_t = t_eval[t_index]
        while solver.status == 'running':
            end = False # flag to break integration loop
            solver.step() # perform one step (necessary before call to dense_output())
            solver_t.append(solver.t)
            if solver.t >= next_t: # solver has gone past one (or more) of our grid points, so now evaluate soln
                soln = solver.dense_output() # interpolation function for the last timestep
                while solver.t >= next_t: # until soln has been evaluated at all grid points up to solver time
                    y = soln(next_t)
                    states.append(y)
                    t_index += 1
                    if t_index >= num_t: # reached the end of our grid, stop solver
                        end = True
                        break
                    next_t = t_eval[t_index]
            if next_check_i < num_checkpoints and t_index >= checkpoints[next_check_i]:
                solver_diffs = np.diff(solver_t[last_solver_i:])
                logger.info('Progress {:.0f}% ({:.0f}s)'.format(
                    100*(checkpoints[next_check_i]+1)/num_t, time()-tic))
                logger.info('Avg. solver dt for last part: {:.2g} (grid dt={:.3g})'\
                        .format(np.mean(solver_diffs), dt))
                next_check_i += 1
                last_solver_i = len(solver_t) - 1
            if end:
                break # safety, stop solver if we have already calculated state at self.t[-1]
        toc = time()
        compute_time = toc-tic # ptoc-ptic
        logger.info('...done ({:.0f}s)'.format(compute_time))
        return t_eval, states

    def full_integration(self, tf, y0=None, ti = 0.0, dt = None, K0val = 50.0):
        """Integrates the equations of motion from state y0 
        at t = 0 to tf using solve_ivp.
        Evaluates on grid spacing dt (default self.params['dt'])
        Returns array of times and array of corresponding state variables."""
        assert tf > ti, 'Final time is smaller than initial time'
        if dt is None:
            dt = self.params['dt']
        if y0 is None:
            y0 = self.initial_state(K0 = K0val)
        t_eval = np.arange(ti, tf, dt) # in natural units 
        ivp = solve_ivp(self.eoms, [ti,tf], y0, t_eval=t_eval,
                        method='DOP853',
                        atol=1e-8,
                        rtol=1e-7)
        logger.info(ivp.message)
        return ivp.t, ivp.y

    def quick_integration(self, state, tf, ti = 0.0):
        """Integrates the equations of motion from t = 0 to tf using solve_ivp."""
        assert tf > ti, 'Final time is smaller than initial time'
        ivp = solve_ivp(self.eoms, [ti,tf], state, dense_output=True)
        state_f = ivp.y[:,-1]
        #state_t = ivp.t[-1]
        return state_f

    def calculate_n_photon(self, aval, kspace = False):
        """Calculates photonic population in k space or real space.
        
           Inputs:  aval [array of floats] - values a_k(tilda) for all wavectors self.Ks
           Outputs: n_ks [array of floats] - photon populations at all positions/wavevectors"""
        if kspace:
            return np.outer(np.conj(aval),aval)*self.Nm 
        a_r = ifft(aval, norm = 'ortho')
        n_ks = np.conj(a_r)*a_r*self.Nm
        return n_ks
        
    def calculate_n_molecular_real(self, l0val): 
        """Calculates molecular population in real space.
        
           Inputs: l0val [array of floats] - values of the matrix lambda_0 at all wavectors self.Ks
           Outputs: n_M_r [array of floats] - molecular populations at all positions in real space"""
        
        zzzl0 = contract('a,an->n', self.consts['zetazeta'], l0val) # zeta(i+)zeta(j+)Z(i0i+j+)<lambda(i0)>
        n_M_r = (zzzl0 + 0.5)*self.NE
        return n_M_r

    def calculate_n_bright_real(self, l0val, lpval): #, kspace = False):
        """Calculates population of bright exciton state in real space.
        
           Inputs:  l0val [array of floats] - values of the matrix lambda_0 at all wavectors self.Ks
                    lpval [array of floats] - values of the matrix lambda_+ at all wavectors self.Ks
           Outputs: n_B_r [array of floats] - bright state populations at all positions in real space """

        zlp = contract('i,in->n', self.consts['zetap'], lpval) # zeta(i+)<lambda(i+)>
        zzlplp = np.outer(zlp, np.conj(zlp)) # zeta(i+)zeta(j+)<lambda(i+)><lambda(j-)>
        zzzl0 = contract('a,an->n', self.consts['zetazeta'], l0val)
        n_B_r = (self.NE - 1)*zzlplp + np.diag(zzzl0 + 0.5)
        return n_B_r

    def calculate_upper_lower_polariton(self, a_arr, lp_arr, l0_arr, Julia = False): 
        """Calculates coherences <sigma_k'(+)sigma_k(-)> and <a_k sigma_k(+)>, 
        as well as upper and lower polariton populations, all in k-space.
        
        Inputs:  a_arr [array of floats] - values a_k(tilda) for all wavectors self.Ks
                 lp_arr [array of floats] - values of the matrix lambda_+ at all wavectors self.Ks
                 l0_arr [array of floats] - values of the matrix lambda_0 at all wavectors self.Ks
                 Julia [bool] - if True, return also value of sig_plus required for building initial state in 
                 Julia code (see self.Julia_comparison())
        Outputs: n_k, n_L, n_U, sigsig, asig_k [arrays of floats] - photon, lower polariton, upper polariton, 
                 bright state populations and coherences in k space at all wave vectors k """
        
        gp = self.gp
        z011 = gp.z_tensor((0,1,1))
        sNm = np.sqrt(self.Nm)
        sNE = np.sqrt(self.NE)
        n_k = self.calculate_n_photon(a_arr, kspace = True) # photonic population, no rescaling (initial state)
        sig_plus = contract('i,in->n', self.consts['zetap'], lp_arr)  # zeta(i+)<lambda(i+)>
        post_sigp = ifft(sig_plus, axis = 0, norm = 'ortho') # (in k-space, negative exponents) 
        asig_k = np.outer(a_arr, post_sigp)*sNm*sNE # expectation value <a_k sigma(k'+)>; includes rescaling
        sigsig_k1 = np.outer(post_sigp, np.conj(post_sigp))*self.NE # first term of <sigma(k'+)sigma(k-)>
        sig_abs_sq = sig_plus * sig_plus.conj()  # zeta(i+)zeta(j+)<lambda(i+)><lambda(j-)>
        sigsig_k2 = fft(sig_abs_sq, axis=0, norm='forward') # (in k-space, positive exponents)
        pre_l0 = contract('a,an->n', self.consts['zetazeta'], l0_arr) # zeta(i+)zeta(j+)Z+(i0i+j+)<lambda(i0)> (in real space)
        post_l0 = fft(pre_l0, axis = 0, norm = 'forward') # (in k-space, positive exponents)
        sigsig = np.zeros((self.Nk, self.Nk), dtype=complex) # initialise array for <sigma(k'+)sigma(k-)> values
        n_L = np.zeros((self.Nk, self.Nk), dtype=complex) # initialise array for lower polariton population values
        n_U = np.zeros((self.Nk, self.Nk), dtype=complex) # initialise array for upper polariton population values
        delta_k = np.eye(self.Nk) # equivalent to 2*Nnu*zeta(i+)zeta(j+)delta(i+,j+)
        for p, k in itertools.product(range(self.Nk), range(self.Nk)): 
            sigsig[p,k] = sigsig_k1[p,k] - sigsig_k2[k-p] + post_l0[k-p] + 0.5 * delta_k[p,k]
            n_U[p,k] = self.coeffs['X_k'][p] * self.coeffs['X_k'][k] * sigsig[p,k] \
            + self.coeffs['Y_k'][p] * self.coeffs['Y_k'][k] * n_k[p,k] \
            + self.coeffs['X_k'][p] * self.coeffs['Y_k'][k] * asig_k[k,p] \
            + self.coeffs['Y_k'][p] * self.coeffs['X_k'][k] * np.conj(asig_k[p,k])
            n_L[p,k] = self.coeffs['X_k'][p] * self.coeffs['X_k'][k] * n_k[p,k] \
            + self.coeffs['Y_k'][p] * self.coeffs['Y_k'][k] * sigsig[p,k] \
            - self.coeffs['X_k'][p] * self.coeffs['Y_k'][k] * np.conj(asig_k[p,k]) \
            - self.coeffs['Y_k'][p] * self.coeffs['X_k'][k] * asig_k[k,p]
        if Julia:
            return n_k, n_L, n_U, sigsig, sig_plus, asig_k
        return n_k, n_L, n_U, sigsig, asig_k

    def Julia_comparison(self, K0):
        """Returns elements a_k, sigma_k(+), <sigma_k(+) sigma_k(-)> for building equivalent initial state 
           in Julia code. Note that Julia code model contains single photon mode i.e. Q0 = 0 required.
           
        Inputs: K0 [float] - central wavenumber of the initial state [unitless; k0 = K0*L/(2pi)]
        Outputs: a, sig_plus, sigsig [floats] - values of a_k, sigma_k(+), <sigma_k(+) sigma_k(-)> for self.initial_state()"""
        
        state = self.initial_state(K0 = K0) # build initial state
        a, lp, l0 = self.split_reshape_return(state)
        n_k, n_L, n_U, sigsig, sig_plus, asig_k = self.calculate_upper_lower_polariton(a, lp, l0, Julia = True)
        return a, sig_plus, sigsig
        
    def calculate_observables(self, state, kspace = False):
        """Calculates coherences, polariton, photon and molecular populations for a given state. 
        
        Inputs:  state [array of floats] - array (a, lp, l) of length self.state_length. To get the correct 
                 flattening and dimensions, pass state to self.split_reshape_return()
                 kspace [bool] - if True, calculate observables in kspace. If False, calculate in real space
                 Note that sigsig, asig_k always in k space
        Outputs: n_k_vals, n_M, n_L_vals, n_U_vals, sigsig_vals, asig_k_vals, n_B, n_D [arrays of floats] - arrays of photon, molecular, lower
                 polariton, upper polariton, bright, dark populations, coherences <sigma_k'(+) sigma_k(-)> and 
                 <a_k' sigma_k(+)> respectively for all k and k' values""" 
        
        a_i, lp_i, l0_i = self.split_reshape_return(state) 
        n_M_vals = self.calculate_n_molecular_real(l0_i) # molecular population (real space)
        n_B_vals = self.calculate_n_bright_real(l0_i, lp_i) # bright state population (real space)
        n_k_vals, n_L_vals, n_U_vals, sigsig_vals, asig_k_vals = self.calculate_upper_lower_polariton(a_i, lp_i, l0_i) # k-space, initial populations (not evolved)
        if not kspace:
            asig_r1 = fft(asig_k_vals, axis=0, norm = 'ortho') # double fourier transform <a_m sign_n+>
            asig_r = ifft(asig_r1, axis=-1, norm = 'ortho')
            asig_current = 1j*self.NE*(self.params['gSqrtN']**2)*(asig_r - np.conj(asig_r))
            nkft1 = fft(n_k_vals, axis=0, norm = 'ortho') # double fourier transform
            n_k = ifft(nkft1, axis=-1, norm = 'ortho')
            nlft1 = fft(n_L_vals, axis=0, norm = 'ortho') # double fourier transform
            n_L = ifft(nlft1, axis=-1, norm = 'ortho')
            nuft1 = fft(n_U_vals, axis=0, norm = 'ortho') # double fourier transform
            n_U = ifft(nuft1, axis=-1, norm = 'ortho')
            n_D_vals = n_M_vals - n_B_vals
            return n_k, n_M_vals, n_L, n_U, sigsig_vals, asig_current, n_B_vals, n_D_vals
        else:
            n_M = sigsig_vals
            nbft1 = ifft(n_B_vals, axis=-1, norm = 'ortho') # double fourier transform
            n_B = fft(nbft1, axis=0, norm = 'ortho')
            asig_current = 1j*self.NE*(self.params['gSqrtN']**2)*(asig_k_vals - np.conj(asig_k_vals))
            n_D = n_M - n_B
        return n_k_vals, n_M, n_L_vals, n_U_vals, sigsig_vals, asig_current, n_B, n_D
        
    def calculate_diagonal_elements(self, state, kspace):
        """Calculates diagonal elements of the coherence, polariton, photon and molecular population
           arrays for a given state and asserts that relevant elements have real values. 
        
        Inputs:  state [array of floats] - array (a, lp, l) of length self.state_length. To get the correct 
                 flattening and dimensions, pass state to self.split_reshape_return()
                 kspace [bool] - if True, calculate observables in kspace. If False, calculate in real space
                 Note that sigsig_diag, asig_k_diag always in k space
        Outputs: n_k_diag, n_M_diag, n_L_diag, n_U_diag, n_B_diag, n_D_diag, sigsig_diag, asig_k_diag [arrays of floats] - 
                 arrays of diagonal values of photon, molecular, lower polariton, upper polariton, bright, dark populations, 
                 coherences <sigma_k(+) sigma_k(-)> and <a_k sigma_k(+)> respectively"""
        
        n_ks, n_Ms, n_Ls, n_Us, sigsigs, asig_rs, n_Bs, n_Ds = self.calculate_observables(state, kspace)
        assert np.allclose(np.diag(n_Ms).imag, 0.0), "Molecular population has imaginary components"
        assert np.allclose(np.diag(n_ks).imag, 0.0), "Photon population has imaginary components"
        assert np.allclose(np.diag(n_Ls).imag, 0.0), "Lower polariton population has imaginary components"
        assert np.allclose(np.diag(n_Us).imag, 0.0), "Upper polariton population has imaginary components"
        assert np.allclose(np.diag(n_Bs).imag, 0.0), "Bright exciton population has imaginary components"
        assert np.allclose(np.diag(n_Ds).imag, 0.0), "Dark exciton population has imaginary components"
        assert np.allclose(np.diag(sigsigs).imag, 0.0), "The coherences have imaginary components"
        assert np.allclose(np.diag(asig_rs).imag, 0.0), "The light-matter current has imaginary components"
        if not kspace:
            n_M_diag = np.fft.fftshift(n_Ms.real) # shift back so that k=0 component is at the center   
        else:
            n_M_diag = np.fft.fftshift(np.diag(n_Ms).real) # shift back so that k=0 component is at the center  
        n_k_diag = np.fft.fftshift(np.diag(n_ks).real) # shift back so that k=0 component is at the center
        n_L_diag = np.fft.fftshift(np.diag(n_Ls).real) # shift back so that k=0 component is at the center
        n_U_diag = np.fft.fftshift(np.diag(n_Us).real) # shift back so that k=0 component is at the center
        n_B_diag = np.fft.fftshift(np.diag(n_Bs).real) # shift back so that k=0 component is at the center
        n_D_diag = np.fft.fftshift(np.diag(n_Ds).real) # shift back so that k=0 component is at the center n_M_diag - n_B_diag
        sigsig_diag = np.fft.fftshift(np.diag(sigsigs).real) # shift back so that k=0 component is at the center
        asig_r_diag = np.fft.fftshift(np.diag(asig_rs).real) #fft.fftshift(np.diag(asig_rs).real) # shift back so that k=0 component is at the center
        return n_k_diag, n_M_diag, n_L_diag, n_U_diag, n_B_diag, n_D_diag, sigsig_diag, asig_r_diag
        
    def calculate_evolved_observables_fixed_k(self, tf = 100.0, fixed_position_index = 1, kspace = False, K0val = 80.0):
        """Evolves self.initial_state() from time ti = 0.0 to time tf in time steps self.dt. Calculates 
           diagonal elements of populations for each time step in either real or k space and returns values for FIXED_POSITION_INDEX only.
        
        Inputs:  tf [float] - integration time in seconds
                 fixed_position_index [int] - if specified, evolution is returned only for specific k/r value
                 kspace [bool] - if True, calculate observables in k space. If False, calculate in real space
                 Note that sigsig_arr always returned in k space
        Outputs: t_fs, n_k_arr, n_M_arr, n_B_arr, n_D_arr, n_L_arr, n_U_arr, sigsig_arr [arrays of floats]: arrays of 
                 integration times, photon, molecular, bright, dark, lower and upper polariton populations and 
                 coherences <sigma_k(+) sigma_k(-)> respectively for each time step of the evolution"""
        
        state = self.initial_state(K0 = K0val) # build initial state
        t_fs, y_vals = self.full_integration(tf, state, ti = 0.0)
        y_vals = y_vals.T
        n_k_diag, n_M_diag, n_L_diag, n_U_diag, n_B_diag, n_D_diag, sigsig_diag, asig_r_diag = self.calculate_diagonal_elements(y_vals[0,:], kspace) # calculate observables for initial state
        n_k_arr, n_M_arr, n_L_arr, n_U_arr, n_B_arr, n_D_arr, sigsig_arr, asig_r_arr = [np.zeros_like(t_fs, dtype=float) for _ in range(8)]
        n_k_arr[0] = n_k_diag[fixed_position_index]
        n_M_arr[0] = n_M_diag[fixed_position_index]
        n_L_arr[0] = n_L_diag[fixed_position_index]
        n_U_arr[0] = n_U_diag[fixed_position_index]
        n_B_arr[0] = n_B_diag[fixed_position_index]
        n_D_arr[0] = n_D_diag[fixed_position_index]
        sigsig_arr[0] = sigsig_diag[fixed_position_index]
        asig_r_arr[0] = asig_r_diag[fixed_position_index]
        for i in range(1,len(t_fs)):
            state_e = y_vals[i,:]
            n_k_diag, n_M_diag, n_L_diag, n_U_diag, n_B_diag, n_D_diag, sigsig_diag, asig_r_diag = self.calculate_diagonal_elements(state_e, kspace) # calculate observables for evolved state
            n_k_arr[i] = n_k_diag[fixed_position_index]
            n_M_arr[i] = n_M_diag[fixed_position_index]
            n_L_arr[i] = n_L_diag[fixed_position_index]
            n_U_arr[i] = n_U_diag[fixed_position_index]
            n_B_arr[i] = n_B_diag[fixed_position_index]
            n_D_arr[i] = n_D_diag[fixed_position_index]
            sigsig_arr[i] = sigsig_diag[fixed_position_index]
            asig_r_arr[i] = asig_r_diag[fixed_position_index]
        assert len(n_k_arr) == len(t_fs), 'Length of evolved photonic population array does not have the required dimensions'
        assert len(n_M_arr) == len(t_fs), 'Length of evolved molecular population array does not have the required dimensions'
        assert len(n_L_arr) == len(t_fs), 'Length of evolved lower polariton population array does not have the required dimensions'
        assert len(n_U_arr) == len(t_fs), 'Length of evolved upper polariton population array does not have the required dimensions'
        assert len(n_B_arr) == len(t_fs), 'Length of evolved bright exciton population array does not have the required dimensions'
        assert len(n_D_arr) == len(t_fs), 'Length of evolved dark exciton population array does not have the required dimensions'
        return t_fs, n_k_arr, n_M_arr, n_B_arr, n_D_arr, n_L_arr, n_U_arr, sigsig_arr, asig_r_diag 

    def calculate_evolved_observables_all_k(self, tf = 100.0, kspace = False, K0val = 50.0, plot_all = False, K0s = None):
        """Evolves self.initial_state() from time ti = 0.0 to time tf in time steps self.dt. Calculates 
           diagonal elements of populations for each time step in either real or k space.
        
        Inputs:  tf [float] - integration time in seconds
                 kspace [bool] - if True, calculate observables in k space. If False, calculate in real space
                 Note that sigsig_arr always returned in k space
                 K0 [float] - central wavenumber of the initial state [unitless; k0 = K0*L/(2pi)]
                 plot_all [bool] - if True, evolved observables calculated for a range K0s of K_0 values of the initial population
        Outputs: t_fs, n_k_arr, n_M_arr, n_B_arr, n_D_arr, n_L_arr, n_U_arr, sigsig_arr [arrays of floats]: arrays of 
                 integration times, photon, molecular, bright, dark, lower and upper polariton populations and 
                 coherences <sigma_k(+) sigma_k(-)> respectively for each time step of the evolution"""
        
        if not plot_all:        
            t_fs, y_vals = self.full_integration(tf, ti = 0.0, K0val = K0val)
            y_vals = y_vals.T
            n_k_diag0, n_M_diag0, n_L_diag0, n_U_diag0, n_B_diag0, n_D_diag0, sigsig_diag0, asig_r_diag0 = self.calculate_diagonal_elements(y_vals[0,:], kspace) # calculate observables for initial state
            n_k_arr, n_M_arr, n_L_arr, n_U_arr, n_B_arr, n_D_arr, sigsig_arr, asig_r_arr = [np.zeros((len(t_fs), self.Nk), dtype=float) for _ in range(8)]
            n_k_arr[0,:] = n_k_diag0
            n_M_arr[0,:] = n_M_diag0
            n_L_arr[0,:] = n_L_diag0
            n_U_arr[0,:] = n_U_diag0
            n_B_arr[0,:] = n_B_diag0
            n_D_arr[0,:] = n_D_diag0
            sigsig_arr[0,:] = sigsig_diag0
            asig_r_arr[0,:] = asig_r_diag0
            for i in range(1,len(t_fs)):
                assert len(y_vals[i,:]) == self.state_length, 'Length of evolved state is incorrect'
                n_k_diag1, n_M_diag1, n_L_diag1, n_U_diag1, n_B_diag1, n_D_diag1, sigsig_diag1, asig_r_diag1 = self.calculate_diagonal_elements(y_vals[i,:], kspace) # calculate observables for evolved state 
                n_k_arr[i,:] = n_k_diag1
                n_M_arr[i,:] = n_M_diag1
                n_L_arr[i,:] = n_L_diag1
                n_U_arr[i,:] = n_U_diag1
                n_B_arr[i,:] = n_B_diag1
                n_D_arr[i,:] = n_D_diag1
                sigsig_arr[i,:] = sigsig_diag1
                asig_r_arr[i,:] = asig_r_diag1
        else:
            for K0val in K0s:
                ind = np.where(K0s == K0val)
                t_fs, y_vals = self.full_integration(tf, ti = 0.0, K0val = K0val)
                y_vals = y_vals.T
                n_k_diag0, n_M_diag0, n_L_diag0, n_U_diag0, n_B_diag0, n_D_diag0, sigsig_diag0, asig_r_diag0 = self.calculate_diagonal_elements(y_vals[0,:], kspace) # calculate observables for initial state
                n_k_arr, n_M_arr, n_L_arr, n_U_arr, n_B_arr, n_D_arr, sigsig_arr, asig_r_arr = [np.zeros((len(K0s), len(t_fs), self.Nk), dtype=float) for _ in range(8)]
                n_k_arr[ind,0,:] = n_k_diag0
                n_M_arr[ind,0,:] = n_M_diag0
                n_L_arr[ind,0,:] = n_L_diag0
                n_U_arr[ind,0,:] = n_U_diag0
                n_B_arr[ind,0,:] = n_B_diag0
                n_D_arr[ind,0,:] = n_D_diag0
                sigsig_arr[ind,0,:] = sigsig_diag0
                asig_r_arr[ind,0,:] = asig_r_diag0
                for i in range(1,len(t_fs)):
                    assert len(y_vals[i,:]) == self.state_length, 'Length of evolved state is incorrect'
                    n_k_diag1, n_M_diag1, n_L_diag1, n_U_diag1, n_B_diag1, n_D_diag1, sigsig_diag1, asig_r_diag1 = self.calculate_diagonal_elements(y_vals[i,:], kspace) # calculate observables for evolved state 
                    n_k_arr[ind,i,:] = n_k_diag1
                    n_M_arr[ind,i,:] = n_M_diag1
                    n_L_arr[ind,i,:] = n_L_diag1
                    n_U_arr[ind,i,:] = n_U_diag1
                    n_B_arr[ind,i,:] = n_B_diag1
                    n_D_arr[ind,i,:] = n_D_diag1
                    sigsig_arr[ind,i,:] = sigsig_diag1
                    asig_r_arr[ind,i,:] = asig_r_diag1            
        return t_fs, n_k_arr, n_M_arr, n_B_arr, n_D_arr, n_L_arr, n_U_arr, sigsig_arr, asig_r_arr
        
    def cavity_velocity(self, K):
        """Calculates cavity group velocity in units of micrometer/fs.
        
        Inputs:  K[array of floats] - wavenumbers K = k*2pi/L in units inverse micrometers
        Outputs: v_c [array of floats] - cavity velocity at each K value in units micrometer/fs"""
        
        v_c = (1e6 * 1e-15) * self.c * self.K_factor * K / np.sqrt(1 + self.K_factor**2 * K**2) # in micrometer / fs (self.c in m/s; * self.K_factor * K in eV)
        return v_c
        
    def group_velocity_expected(self, Ks_auto = True, all_Ks = None):
        """Calculates theoretical group velocity as gradient of lower/upper polariton population
           in units of micrometer/fs.
        
        Outputs: v_c, v_L, v_U [arrays of floats] - arrays of cavity lower and upper polariton 
                 velocities at each K value in units micrometer/fs"""
        
        exciton = self.params['exciton']
        if Ks_auto == True:
            all_Ks = np.linspace(-1.5*np.abs(self.Ks[0]), 1.5*np.abs(self.Ks[-1]), 250)        
        v_c = self.cavity_velocity(all_Ks) # in micrometer / fs
        omega_k = self.omega(all_Ks, exciton) #self.consts['omega']
        epsilon = self.params['epsilon']
        ep_omegas =  epsilon - omega_k
        zeta_k = 0.5 * np.sqrt(ep_omegas**2 + 4*self.params['gSqrtN']**2) #self.consts['zeta_k'] 
        v_L = 0.5*v_c*(1 + 0.5*ep_omegas/zeta_k) #EV_TO_FS = (constants.hbar/constants.e)*1e15 # convert time in electronvolts to time in fs
        v_U = 0.5*v_c*(1 - 0.5*ep_omegas/zeta_k)
        return v_c, v_L, v_U

    def plot_all(self, tf, kspace = False, fixed_position_index = None, savefigs = False):
        """"Routine for plotting plots that depend on a single value of K_0 in one go. That includes plots of the expected group velocities
            for the lower polariton dispersion, the dependence of the velocity of the lower polariton on the values of S and Gamma_z, the 
            initial state populations for the given K_0 (in params), the motion of the n_k, n_L, n_B and n_D mean positions over time [0, TF], 
            the interconversion of n_B into n_D over, the light-matter current and (optionally) the evolution of the wavepackets at some 
            specified position given by FIXED_POSITION_INDEX.
        
            Inputs:  tf [float] - final time for the evolution from the initial state
                     kspace [bool] - if True, plots are produced in kspace. If False, they are transformed to 
                     real space
                     fixed_position_index [int] - if given, a plot of the evolution of n_B, n_D, n_k AT THE GIVEN INDEX
                     is provided
                     savefigs [bool] - if True, all plots saved the names outlined in their respective plotting functions"""
        
        K0val = self.params['K_0']
        self.plot_group_velocities(savefig = savefigs)
        self.plot_wrt_S(Svals = np.arange(0.0,10.1,1.0), Gam_z = np.arange(0,0.04,0.002), tf = 20.1, all_plots = True, savefig = savefigs)
        times_arr, n_k_arr, n_M_arr, n_B_arr, n_D_arr, n_L_arr, n_U_arr, sigsig_arr, asig_r_arr = self.calculate_evolved_observables_all_k(tf, kspace = kspace, K0val = K0val, plot_all = False)
        times_arr *= self.EV_TO_FS # convert to femtoseconds for plotting
        self.plot_initial_populations(savefig = savefigs, kspace = kspace, K0val = K0val)
        self.plot_msd_motion(times = times_arr, n_arr_B = n_B_arr, n_arr_D = n_D_arr, n_arr_L = n_L_arr, n_arr_k = n_k_arr, tf = None, K0 = K0val, savefig = savefigs)
        self.plot_waterfall(times = times_arr, n_k_arr = n_k_arr, n_B_arr = None, n_D_arr = None, n_L_arr = None, n_L = False, n_B = False, n_D = False, n_k = True, savefig = savefigs, tf = None, kspace = False, legend = True, K0vals = np.array([K0val]))
        self.plot_waterfall(times = times_arr, n_k_arr = None, n_B_arr = None, n_D_arr = None, n_L_arr = n_L_arr, n_L = True, n_B = False, n_D = False, n_k = False, savefig = savefigs, tf = None, kspace = False, legend = True, K0vals = np.array([K0val]))
        self.plot_waterfall(times = times_arr, n_k_arr = None, n_B_arr = n_B_arr, n_D_arr = None, n_L_arr = None, n_L = False, n_B = True, n_D = False, n_k = False, savefig = savefigs, tf = None, kspace = False, legend = True, K0vals = np.array([K0val]))
        self.plot_waterfall(times = times_arr, n_k_arr = None, n_B_arr = None, n_D_arr = n_D_arr, n_L_arr = None, n_L = False, n_B = False, n_D = True, n_k = False, savefig = savefigs, tf = None, kspace = False, legend = True, K0vals = np.array([K0val]))
        self.plot_total_dark_population_growth(times = times_arr, n_B_arr = n_B_arr, n_D_arr = n_D_arr, tf = None, K0val = K0val, savefig = savefigs)
        self.plot_photon_exciton_current(times = times_arr, asig_arr = asig_r_arr, tf = None, K0val = K0val, savefig = savefigs)
        if fixed_position_index != None:
            self.plot_evolution(times = times_arr, n_k_arr = n_k_arr[:,fixed_position_index], n_B_arr = n_B_arr[:,fixed_position_index], n_D_arr = n_D_arr[:,fixed_position_index], kspace = False, savefig = False)
        #return times_arr, n_k_arr, n_M_arr, n_B_arr, n_D_arr, n_L_arr, n_U_arr, sigsig_arr, asig_r_arr
                 
    def plot_group_velocities(self, savefig = False):
        """Plots cavity velocity and theoretical group velocities of upper and lower polariton as a function of wavenumber K = k*2*pi/L.

        Inputs:  savefig [bool] - if True, saves plot as 'dispersion.jpg'"""
        
        all_Ks = np.linspace(-1.5*np.abs(self.Ks[0]), 1.5*np.abs(self.Ks[-1]), 250)        
        vc, vgL, vgU = self.group_velocity_expected()
        fig, ax = plt.subplots(1,1,figsize = (6,4), layout = 'tight')
        ax.plot(all_Ks, vgL, label = '$v^{L}_{g}$', color = 'black')
        ax.plot(all_Ks, vgU, label = '$v^{U}_{g}$', color = 'blue')
        ax.plot(all_Ks, vc, label = '$v_{cav}$', color = 'purple')        
        ax.set_ylabel('$v [\mu m/ fs]$')
        ax.set_xlabel('$K_0 [\mu m^{-1}]$')
        ax.set_title('Theoretical group velocities along the upper and lower polariton branches')
        ax.legend()
        if savefig:
            plt.savefig(fname = 'dispersion.jpg', format = 'jpg')
        plt.show()
        
    def plot_evolution(self, times = None, n_k_arr = None, n_B_arr = None, n_D_arr = None, tf = None, fixed_position_index = 100, kspace = False, savefig = False):
        """Plots evolution of photonic, bright and dark state populations over time TF at given position FIXED_POSITION_INDEX.

        Inputs:  times [float] - array of times corresponding to the evolved arrays n_k_arr, n_B_arr, n_D_arr. If None, evolution calculated
                 by the function over time period [0,tf]
                 n_k_arr [float] - array of values of the photonic population at times TIMES
                 n_B_arr [float] - array of values of the bright state population at times TIMES
                 n_D_arr [float] - array of values of the dark state population at times TIMES
                 Note: Must provide ALL THREE of N_K_ARR, N_B_ARR, N_D_ARR
                 tf [float] - final time of the evolution
                 fixed_position_index [int] - index of the r/k array at which the evolution is evaluated
                 kspace [bool] - if True, plot in k space. If False, plot in real space
                 savefig [bool] - if True, saves plot as 'evolution.jpg' """

        if tf != None:
            times, n_k_arr, n_M_arr, n_B_arr, n_D_arr, n_L_arr, n_U_arr, sigsig_arr, asig_r_arr = self.calculate_evolved_observables_fixed_k(tf, fixed_position_index, kspace = kspace)
            times *= self.EV_TO_FS # convert to femtoseconds for plotting
        fig, ax = plt.subplots(1,1,figsize = (6,4))
        ax.plot(times, n_B_arr, label = '$n_{B}$')
        ax.scatter(times, n_B_arr, marker = '.')
        ax.plot(times, n_k_arr, label = '$n_{phot}$')
        ax.scatter(times, n_k_arr, marker = '.')
        ax.plot(times, n_D_arr, label = '$n_{D}$')
        ax.scatter(times, n_D_arr, marker = '.')
        ax.set_xlabel('time [$fs$]')
        ax.set_ylabel(f'n($K_0$={self.Ks[fixed_position_index]})')
        ax.legend()
        ax.set_title('Population evolution and interconversion over time')
        if savefig:
            plt.savefig(fname = 'evolution.jpg', format = 'jpg')
        plt.show()
            
    def plot_waterfall(self, times = None, n_k_arr = None, n_B_arr = None, n_D_arr = None, n_L_arr = None, n_L = False, n_B = False, n_D = False, n_k = False, savefig = False, tf = None, kspace = False, legend = False, K0vals = np.array([50.0]), num_slices = 8):
        """Plots selected time snapshot of the evolution of either the lower polariton, the bright or the photon population as a waterfall plot.

        Inputs:  n_L [bool] - if True, plot lower polariton population
                 n_B [bool] - if True, plot bright population
                 n_k [bool] - if True, plot photon population
                 Note only one of n_L, n_B, n_k can be specified at a time.
                 savefig [bool] - if True, saves plot as 'plot_waterfall.jpg'
                 tf [float] - final time for the evolution of the wavepacket in seconds (transformation to femtoseconds performed internally by code)
                 kspace [bool] - if True, time snapshots plotted over the array of K-values self.Ks. If False, plot in real space
                 legend [bool] - if True, include legend with times of the time snapshots
                 by 2*step*self.dt are plotted
                 K0vals [float] - array of central wavenumbers of the intial populations [unitless; k0 = K0*L/(2pi)] 
                 num_slices [int] - number of time snapshots plotted (default is 8). Be careful to turn LEGEND = FALSE if plotting a 
                 very large number of slices"""

        fig, ax = plt.subplots(1,1,figsize=(10,6), layout = 'tight')
        fig2, ax2 = plt.subplots(1,1,figsize=(10,6), layout = 'tight')
        for K0val in K0vals:
            if tf != None:
                times, n_k_arr, n_M_arr, n_B_arr, n_D_arr, n_L_arr, n_U_arr, sigsig_arr, asig_arr = self.calculate_evolved_observables_all_k(tf, kspace = kspace, K0val = K0val)
                times *= self.EV_TO_FS # convert to femtoseconds for plotting
            slices = np.linspace(0, len(times)-1, num_slices)
            L = self.params['L']
            rs = np.linspace(-L/2,L/2,len(self.Ks)) #self.params['delta_r']*self.Ks
            colors = plt.cm.coolwarm(np.linspace(0,1,len(slices)))
            if n_B:
                n_arr = n_B_arr
                ax.set_ylabel('$n_{B}(r_n)$')       
            if n_D:
                n_arr = n_D_arr
                ax.set_ylabel('$n_{D}(r_n)$')
            if n_k:
                n_arr = n_k_arr
                ax.set_ylabel('$n_{k}(r_n)$')
            if n_L:
                n_arr = n_L_arr
                ax.set_ylabel('$n_{L}(r_n)$')
            offset = 0.1 * np.max(n_arr)
            for i in range(num_slices):
                index = int(slices[i])
                n_i = n_arr[index,:]
                #var = (rs - np.ones_like(rs)*rs[np.where(n_i == np.max(n_i))[0]])**2
                #std = np.sqrt(np.sum(var)/len(rs)) #std(n_i)
                K0_ind = np.where(K0vals == K0val)[0]
                #ax2.scatter(times[index], std, color = 'blue', marker = '.')
                if len(K0_ind) == 0:
                    continue
                else:
                    K0_ind = K0_ind[0]  
                if kspace:
                    ax.plot(self.Ks+np.max(self.Ks)*K0_ind, n_i + i * offset, label = f't = {times[index]:.2E}', color=colors[i])
                else:
                    if K0_ind != 0:
                        ax.plot(rs+np.max(self.Ks)*K0_ind, n_i + i * offset, zorder = (len(slices)-i), color=colors[i])
                    else:
                        ax.plot(rs, n_i + i * offset, label = f't = {times[index]:.2E}', zorder = (len(slices)-i), color=colors[i])
        if kspace:
            ax.set_xlabel('$k [\mu m^{-1}]$')
        else:
            ax.set_xlabel('$r_n [\mu m]$')
        ax.set_title('Time Snapshots of Wavepacket Evolution')
        ax.minorticks_on()
        ax.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=13)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.3)
        ax.grid(alpha = 0.2)   
        if legend:
            ax.legend()
        if savefig:
            plt.savefig(fname = 'plot_waterfall.jpg', format = 'jpg')
        plt.show()

    def plot_total_dark_population_growth(self, times = None, n_B_arr = None, n_D_arr = None, tf = None, K0val = None, savefig = False):
        """Plots photon-exciton current igN_E(<a_m sig_n+> - <a_m+ sig_n->) over time.

        Inputs:  times [float] - array of times corresponding to the evolved arrays N_B_ARR and N_D_ARR. If None, then tf 
                 must be specified, so that the evolution is calculated manually over time span [0,tf].
                 n_B_arr [float] - array of bright population values at times TIMES. 
                 n_D_arr [float] - array of dark population values at times TIMES
                 tf [float] - final time for the evolution of the wavepacket in physical units if TIMES, N_B_ARR and N_D_ARR are not specified
                              (transformation to femtoseconds performed internally by code)
                 K0val [float] - central wavenumber of the intial population [unitless; k0 = K0*L/(2pi)] 
                 savefig [bool] - if True, saves plot as 'dark_state_growth_rate.jpg'"""

        if K0val == None:
            K0val = self.params['K_0']
        if tf != None:
            times, n_k_arr, n_M_arr, n_B_arr, n_D_arr, n_L_arr, n_U_arr, sigsig_arr, asig_arr = self.calculate_evolved_observables_all_k(tf, kspace = False, K0val = K0val)
            times *= self.EV_TO_FS # convert to femtoseconds for plotting
        fig, ax = plt.subplots(1,1,figsize=(10,6), layout = 'tight')

        S = self.params['S']
        Gz = self.params['Gam_z']
        n_Bs = np.ones_like(times)
        n_Ds = np.ones_like(times)
        for i in range(len(times)):
            n_Bs[i] = n_Bs[i]*np.sum(n_B_arr[i,:])
            n_Ds[i] = n_Ds[i]*np.sum(n_D_arr[i,:])
        ax.plot(times, n_Bs, color = 'black', label = 'n_B', marker = '.', ls = '--')
        ax.plot(times, n_Ds, color = 'blue', label = 'n_D', marker = '.', ls = '--')
        ax.set_title(f'Bright-dark state interconversion (S = {S}, $\Gamma_z$ = {Gz}, $K_0$ = {K0val})')
        ax.minorticks_on()
        ax.set_xlabel('$t$ $[fs]$')
        ax.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=13)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.3)
        ax.grid(alpha = 0.2)   
        ax.legend()
        if savefig:
            plt.savefig(fname = f'dark_state_growth_rate_K0_{K0val}.jpg', format = 'jpg')
        plt.show()
        
    def plot_msd_motion(self, times = None, n_arr_B = None, n_arr_D = None, n_arr_L = None, n_arr_k = None, tf = None, K0 = None, savefig = False):
        """Plots displacement of peak of either photonic, lower polariton, bright or dark state population.
        
        Inputs:  times [float] - array of time values corresponding to evolved array N_ARR. If None, tf must be specified, so that
                 the evolution of is performed inside the function
                 n_B_arr [float] - array of bright population values at times TIMES. 
                 n_D_arr [float] - array of dark population values at times TIMES
                 n_L_arr [float] - array of lower polariton population values at times TIMES. 
                 n_k_arr [float] - array of photonic population values at times TIMES
                 tf [float] - integration time in physical units. Must be specified if TIMES and N_ARR are not provided
                 (conversion to femtoseconds performed internally)
                 kspace [bool] - if True, time snapshots plotted over the array of K-values self.Ks. If False, plot in real space
                 K0 [float] - central wavenumber of the initial population [unitless; k0 = K0*L/(2pi)]
                 savefig [bool] - if True, saves plots"""
        
        if K0 == None:
            K0 = self.params['K_0']
        if tf != None:
            times, n_arr = self.calculate_evolved_n_L_all_k(tf, kspace = False, K0val = K0)
            times *= self.EV_TO_FS # rescale for plotting
        n_0_B = n_arr_B[0,:]
        msd_arr_B, msd_arr_D, msd_arr_L, msd_arr_k = [np.zeros((len(times)), dtype=float) for _ in range(4)]
        rs = np.linspace(0,self.params['L'],len(self.Ks))
        msd_val_B_0 = np.average(rs, weights=n_0_B) # position of weighted mean (i.e ~ position of peak)  
        msd_arr_B[0] = 0.0 # shift coordinate system, so that initial position is at 0.0
        n_0_D = n_arr_D[0,:]
        msd_val_D_0 = np.average(rs, weights=n_0_D) # position of weighted mean (i.e ~ position of peak)
        msd_arr_D[0] = 0.0
        n_0_L = n_arr_L[0,:]
        msd_val_L_0 = np.average(rs, weights=n_0_L) # position of weighted mean (i.e ~ position of peak)
        msd_arr_L[0] = 0.0
        n_0_k = n_arr_k[0,:]
        msd_val_k_0 = np.average(rs, weights=n_0_k) # position of weighted mean (i.e ~ position of peak)
        msd_arr_k[0] = 0.0
        for i in range(1, len(times)):
            n_i_B = n_arr_B[i,:]
            msd_val_B = np.average(rs, weights=n_i_B) # weighted mean
            msd_arr_B[i] = msd_val_B-msd_val_B_0
            n_i_D = n_arr_D[i,:]
            msd_val_D = np.average(rs, weights=n_i_D) # weighted mean
            msd_arr_D[i] = msd_val_D-msd_val_D_0
            n_i_L = n_arr_L[i,:]
            msd_val_L = np.average(rs, weights=n_i_L) # weighted mean
            msd_arr_L[i] = msd_val_L-msd_val_L_0
            n_i_k = n_arr_k[i,:]
            msd_val_k = np.average(rs, weights=n_i_k) # weighted mean
            msd_arr_k[i] = msd_val_k-msd_val_k_0
            
        fig, ax = plt.subplots(1,1,figsize=(6,4), layout = 'tight')
        ax.set_xlabel('time [$fs$]', fontsize=14)
        ax.set_ylabel('md [$\mu m$]', fontsize=14)
        ax.minorticks_on()
        ax.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=13)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.3)
        ax.grid(alpha = 0.2)
        fig.suptitle('Motion of MD of different populations', fontsize=16)
        ax.plot(times, msd_arr_B, label = 'n_B', marker = '.', color = 'black') 
        ax.plot(times, msd_arr_D, label = 'n_D', marker = '.', color = 'blue') 
        ax.plot(times, msd_arr_L, label = 'n_L', marker = '.', color = 'purple') 
        ax.plot(times, msd_arr_k, label = 'n_k', marker = '.', color = 'orange') 
        ax.legend()    
        if savefig:
            plt.savefig(fname = 'msd_motion.jpg', format = 'jpg')
        plt.show()
        
    def plot_photon_exciton_current(self, times = None, asig_arr = None, tf = None, K0val = 50.0, savefig = False):
        """Plots photon-exciton current igN_E(<a_m sig_n+> - <a_m+ sig_n->) over time.

        Inputs:  times [float] - array of times corresponding to the evolved arrays N_B_ARR and N_D_ARR. If None, then tf 
                 must be specified, so that the evolution is calculated manually over time span [0,tf].
                 asig_arr [float] - array of photon-exciton current values at times TIMES. 
                 tf [float] - final time for the evolution of the wavepacket in physical units if TIMES, N_B_ARR and N_D_ARR are not specified
                              (transformation to femtoseconds performed internally by code)
                 K0val [float] - central wavenumber of the intial population [unitless; k0 = K0*L/(2pi)] 
                 savefig [bool] - if True, saves plot as 'asig_current.jpg'"""
        
        if K0val == None:
            K0val = self.params['K_0']        
        if tf != None:
            times, n_k_arr, n_M_arr, n_B_arr, n_D_arr, n_L_arr, n_U_arr, sigsig_arr, asig_arr = self.calculate_evolved_observables_all_k(tf, kspace = False, K0val = K0val)
            times *= self.EV_TO_FS # convert to femtoseconds for plotting
        fig, ax = plt.subplots(1,1,figsize=(10,6), layout = 'tight')
        S = self.params['S']
        Gz = self.params['Gam_z']
        asigs = np.ones_like(times)
        for i in range(len(times)):
            asigs[i] = asigs[i]*np.sum(asig_arr[i,:])
        ax.plot(times, asigs, color = 'black', label = f'S = {S}, $\Gamma_z$ = {Gz}, $K_0$ = {K0val}', marker = '.', ls = '--')
        ax.set_title('Photon-exciton current')
        ax.minorticks_on()
        ax.set_xlabel('$t$ $[fs]$')
        ax.set_ylabel('$igN_E(<a_m \sigma_n^+> - <a_m^+ \sigma_n^->)$')
        ax.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=13)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.3)
        ax.grid(alpha = 0.2)   
        ax.legend()
        if savefig:
            plt.savefig(fname = f'asig_current_K0_{K0val}.jpg', format = 'jpg')
        plt.show()
        
    def plot_n_D_wrt_k(self, times = None, n_D_arr = None, tf = 10.1, K0vals = np.array([50.0]), concentrations = False, savefig = False):
        """Plots total dark state population as a function of central wavenumber K0.

        Inputs:  times [float] - array of times corresponding to the evolved array N_D_ARR. If None, then tf 
                 must be specified, so that the evolution is calculated manually over time span [0,tf].
                 n_D_arr [float] - array of photon-exciton current values at times TIMES. 
                 tf [float] - final time for the evolution of the wavepacket in physical units if 
                               TIMES, N_D_ARR not specified. Note that tf must be set to NONE if plot_all is used.
                              (transformation to femtoseconds performed internally by code)
                 K0vals [float] - array of central wavenumbers of the intial population [unitless; k0 = K0*L/(2pi)]
                 concentrations [bool] - if True, plot w.r.t. photonic concentration |X_k|^2. If False, x-axis is K0vals.
                 savefig [bool] - if True, saves plot as 'dark_state_wrt_k.jpg'"""
        
        fig, ax = plt.subplots(1,1,figsize=(10,6), layout = 'tight')
        S = self.params['S']
        Gz = self.params['Gam_z']
        if K0vals == None:
            K0vals = np.array([self.params['K_0']])        
        n_vals = np.ones_like(K0vals, dtype = float)
        p_weights = np.ones_like(K0vals, dtype = float)
        if tf != None:
            times, n_k_arr, n_M_arr, n_B_arr, n_D_arr, n_L_arr, n_U_arr, sigsig_arr, asig_arr = self.calculate_evolved_observables_all_k(tf = tf, kspace = False, K0val = None, plot_all = True, K0s = K0vals)
            times *= self.EV_TO_FS # convert to femtoseconds for plotting
        for K0val in K0vals:
            k_index = np.where(self.Ks == K0val) #self.params['K_0'])
            p_weight = (np.fft.fftshift(self.coeffs['X_k'])[k_index])**2
            e_weight = (np.fft.fftshift(self.coeffs['Y_k'])[k_index])**2
            n_arr = n_D_arr
            m = np.where(K0vals == K0val)
            ax.set_ylabel('$n_{D}$')
            n_i = np.sum(n_arr[m,-1,:]) # last time
            n_vals[m] *= n_i
            p_weights[m] *= p_weight
            print('Current K0val =', m[0])

        if concentrations:
            ax.plot(p_weights, n_vals, marker = '.', ls = '--', color = 'blue', label = 'n_D')
        else:
            ax.plot(K0vals, n_vals, marker = '.', ls = '--', color = 'blue', label = 'n_D')
    
        ax.set_xlabel('$X_k^2$')
        ax.set_title(f'Dark state population after {times[-1]:.2E} fs (S = {S}, $\Gamma_z$ = {Gz})')
        ax.minorticks_on()
        ax.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=13)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.3)
        ax.grid(alpha = 0.2)   
        ax.legend()
        if savefig:
            plt.savefig(fname = f'dark_state_wrt_k_S_{S}_Gz_{Gz}.jpg', format = 'jpg')
        plt.show()
        
    def plot_initial_populations(self, savefig = False, kspace = False, K0val = 40.0):
        """Plots upper, lower polariton and photonic populations on one figure in either k space or real space. 
           Plots molecular, bright and dark populations in real space on another figure.

           Inputs:  savefig [bool] - if True, saves plots as 'state_i.jpg' and 'brightdark_populations.jpg'
                    kspace [bool] - if True, plot first figure in k space. If False, plot figure in real space 
                    Note that second figure is always in real space
                    K0 [float] - central wavenumber of the wavepacket [unitless; k0 = 2pi/L K0]"""

        if K0val == None:
            K0val = self.params['K_0']
        state_i = self.initial_state(K0 = K0val)
        n_k_diag, n_M_diag, n_L_diag, n_U_diag, n_B_diag, n_D_diag, sigsig_diag, asig_k_diag = self.calculate_diagonal_elements(state_i, kspace)
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
        ax1[3].set_ylabel(r'$< \sigma_{k}^{+} \sigma_k^{-}>$')
        ax1[4].scatter(self.Ks, asig_k_diag, marker = '.')
        ax1[4].plot(self.Ks, asig_k_diag)
        ax1[4].set_xlabel(r'$k$')
        ax1[4].set_ylabel(r'$< a_k \sigma_k^{+}>$')
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
        plt.show()

    def julia_comparison(julia_fp=None):
        params = {'Q0':0,
                  'Nm':int(1e5),
                  'Nnu':1,
                  'gSqrtN':0.45,
                  'kappa_c':0.05,
                  'omega_c':0.2,
                  'epsilon':0.1,
                  'Gam_up':0.06, # lasing phase 
                  'Gam_down':0.05,
                  'Gam_z':0.01,
                  'k_0': 0.0,
                  'A': 0.2,
                  'exciton':False,
                  'dt': 0.5,
                  }
        htc = HTC(params)
        tf = 300.0
        y0 = htc.initial_state()
        a, lp, l0 = htc.split_reshape_return(y0) 
        a *= np.sqrt(params['Nm']) # rescale!
        sp = np.kron(Pauli.p, htc.boson.i)
        p1 = np.kron(Pauli.p1, htc.boson.i)
        sp_coeffs = htc.gp.get_coefficients(sp, sgn=1, eye=False)
        p1_coeffs, p1_eye = htc.gp.get_coefficients(p1, sgn=0, eye=True)
        sp_val = htc.gp.get_expectation(lp, sp_coeffs)
        p1_val = htc.gp.get_expectation(l0, p1_coeffs, eye=p1_eye)
        print(f'    <a>   = {a[0]}')
        print(f'<sigma^+> = {sp_val}')
        print(f'   <p^up> = {p1_val}')
        print(f'       tf = {tf}')
        if julia_fp is None:
            return
        ts, ys = htc.stepwise_integration(tf)
        ns = np.abs([y[0] for y in ys])**2
        fig, ax = plt.subplots(figsize=(4,4), constrained_layout=True)
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$n/N$')
        ax.plot(ts, ns, label=r'\rm{mfhtc.py}') # scaled photon number
        julia_data = np.loadtxt(julia_fp)
        ax.plot(julia_data[0], julia_data[1]/params['Nm'], label=r'\rm{QuantumCumulants.jl}', ls='--')
        ax.set_title(r'$g\sqrt{{N_m}}={gSqrtN},\ \Gamma_z={Gam_z}$'.format(**params))
        ax.legend()
        fig.savefig('figures/julia_comparison.png', dpi=350, bbox_inches='tight')
        plt.show()

    ################################# Lighter version for calculating working only with n_L populations #################################
    def calculate_n_L(self, state, kspace = True):
        """Calculates only lower polariton population for a given state. Use instead of calculate_observables() 
        for faster code execution when only n_L is required.
        
        Inputs:  state [array of floats] - array (a, lp, l) of length self.state_length. To get the correct 
                 flattening and dimensions, pass state to self.split_reshape_return()
                 kspace [bool] - if True, calculate n_L in kspace. If False, calculate in real space
        Outputs: n_L [array of floats] - lower polariton population in either real or k space""" 
        
        a_val, lp_val, l0_val = self.split_reshape_return(state) 
        n_k, n_L, n_U, sigsig, asig_k = self.calculate_upper_lower_polariton(a_val, lp_val, l0_val) # k-space, initial populations (not evolved)
        if not kspace:
            nlft1 = ifft(n_L, axis=-1, norm = 'ortho') # double fourier transform
            return fft(nlft1, axis=0, norm = 'ortho')
        return n_L
        
    def calculate_diagonal_n_L(self, state, kspace):
        """Calculates diagonal elements of lower polariton population array for a given state STATE 
        and asserts that relevant elements have real values. Use instead of calculate_diagonal_elements() 
        for faster code execution when only n_L is required.
        
        Inputs:  state [array of floats] - array (a, lp, l) of length self.state_length. To get the correct 
                 flattening and dimensions, pass state to self.split_reshape_return()
                 kspace [bool] - if True, calculate observables in kspace. If False, calculate in real space
                 Note that sigsig_diag, asig_k_diag always in k space
        Outputs: n_k_diag, n_M_diag, n_L_diag, n_U_diag, n_B_diag, n_D_diag, sigsig_diag, asig_k_diag [arrays of floats] - 
                 arrays of diagonal values of photon, molecular, lower polariton, upper polariton, bright, dark populations, 
                 coherences <sigma_k(+) sigma_k(-)> and <a_k sigma_k(+)> respectively"""
        
        n_L = self.calculate_n_L(state, kspace)
        assert np.allclose(np.diag(n_L).imag, 0.0), "Lower polariton population has imaginary components"
        n_L_diag = fftshift(np.diag(n_L).real) # shift back so that k=0 component is at the center
        return n_L_diag
        
    def calculate_evolved_n_L_all_k(self, tf = 100.0, kspace = False, K0val = 50.0):
        """Evolves self.initial_state() from time ti = 0.0 to time tf in time steps self.dt. Calculates 
           diagonal elements of populations for each time step in either real or k space.
        
        Inputs:  tf [float] - integration time in physical units
                 kspace [bool] - if True, calculate observables in k space. If False, calculate in real space
                 Note that sigsig_arr always returned in k space
                 K0 [float] - central wavenumber of the initial population [unitless; k0 = K0*L/(2pi)]
        Outputs: t_fs, n_k_arr, n_M_arr, n_B_arr, n_D_arr, n_L_arr, n_U_arr, sigsig_arr [arrays of floats]: arrays of 
                 integration times, photon, molecular, bright, dark, lower and upper polariton populations and 
                 coherences <sigma_k(+) sigma_k(-)> respectively for each time step of the evolution"""
        
        state_i = self.initial_state(K0 = K0val) # build initial state
        t_fs, y_vals = self.full_integration(tf, state_i, ti = 0.0)
        y_vals = y_vals.T
        n_L = self.calculate_diagonal_n_L(state_i, kspace) # calculate observables for initial state
        n_L_arr = np.zeros((len(t_fs), self.Nk), dtype=float)
        n_L_arr[0,:] = n_L
        for i in range(1,len(t_fs)):
            state_e = y_vals[i,:] 
            n_L_diag = self.calculate_diagonal_n_L(state_e, kspace) # calculate observables for evolved state 
            n_L_arr[i,:] = n_L_diag
        return t_fs, n_L_arr

    def plot_waterfall_n_L(self, savefig = False, tf = 101.1, legend = False, step = 100, K0val = None):
        """Plots selected time snapshots of the evolution of the lower polariton population as a waterfall plot in real space.

        Inputs:  savefig [bool] - if True, saves plot as 'plot_waterfall.jpg'
                 tf [float] - final time for the evolution of the wavepacket in physical units (transformation to 
                 femtoseconds performed internally by code)
                 legend [bool] - if True, include legend with times of the time snapshots
                 step [float] - time step that defines array of snapshot times. Although full evolution is calculated,
                 only snapshots at times separated by step are plotted
                 K0 [float] - central wavenumber of the initial population [unitless; k0 = K0*L/(2pi)]
        Outputs: r_of_nmax [array of floats] - locations of the peak at each snapshot time"""

        if K0val == None:
            K0val = self.params['K_0']
        slices = np.arange(0.0, tf, step)
        times, n_array = self.calculate_evolved_n_L_all_k(tf, kspace = False, K0val = K0val)
        times *= self.EV_TO_FS # convert to femtoseconds for plotting
        slices *= self.EV_TO_FS # convert to femtoseconds for plotting
        fig = plt.figure(figsize=(10,6), layout = 'tight')
        ax = fig.add_subplot()
        colors = plt.cm.coolwarm(np.linspace(0,1,len(slices)))
        assert isinstance(n_array, np.ndarray), "Please, specify one of n_L, n_B and n_k"
        offset = 0.1 * np.max(n_array)
        for i in range(len(slices)):
            index = np.where(times == slices[i])[0]
            if len(index) == 0:
                continue
            else:
                index = index[0]
            n_i_arr = n_array[index,:]
            ax.plot(self.Ks*self.params['delta_r'], n_i_arr + i * offset, label = f't = {slices[i]:.2E}', zorder = (len(slices)-i), color=colors[i])
        ax.set_xlabel('$r_n [\mu m]$')
        ax.set_ylabel('$n_{L}(r_n)$')
        ax.set_title('Time Snapshots of Wavepacket Evolution')
        ax.minorticks_on()
        ax.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=13)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.3)
        ax.grid(alpha = 0.2)        
        if legend:
            ax.legend()
        if savefig:
            plt.savefig(fname = 'plot_waterfall_n_L.jpg', format = 'jpg')
        plt.show()

    def butter_lowpass_filter(self, data, cutoff, fs, order=5, axis=-1):
        data = np.array(data)
        b, a = butter(order, cutoff, fs=fs, btype='low')
        padlen = 3 * max(len(a), len(b)) # default filtfilt padlength
        if padlen >= data.shape[-1] - 1:#but must be less than data.shape[-1]-1
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html#scipy.signal.filtfilt
            padlen = max(0, data.shape[-1]-2) 
        return filtfilt(b, a, data, axis=axis, padlen=padlen)
    
    def plot_n_L_peak_velocity(self, tf = 1.1, npgradient = True, plot_hopfield = False, suppress_plots = False, K0 = 50.0, savefig = False, filename = 'v_rmsd.jpg'):
        """Plots displacement and velocity of peak of lower polariton population.
        
        Inputs:  savefig [bool] - if True, saves plots as 'filename.jpg'
                 filename [string] - default is 'v_rmsd.jpg'
                 tf [float] - integration time in physical units (conversion to femtoseconds performed internally)
                 kspace [bool] - if True, time snapshots plotted over the array of K-values self.Ks. If False, plot in real space
                 K0 [float] - central wavenumber of the initial population [unitless; k0 = K0*L/(2pi)]
                 plot_hopfield [bool] - if True, function plots Hopfiled coefficients for full range of K0 values in the problem
                 npgradient [bool] - if True, velocities calculated both through a curve fit and by applying np.gradient() to the trajectory of the
                 wavepacket, with both estimates plotted on the same figure. If False, only curve_fit procedure is applied
                 suppress_plots [bool] - if True, plots of trajectory and velocity over time generated by self.plot_n_L_peak_velocity
        Outputs: p_weight [float] - photonic Hopfield coefficient corresponding to K0 
                 e_weight [float] - photonic Hopfield coefficient corresponding to K0
                 avg_v [float] - if npgradient, then avg_v is the average velocity for the npgradient fit. If not npgradient, avg_v defaults to 0
                 fit_v_of_msd[int(np.round(len(times)/2))] [float] - velocity fit of the curve_fit procedure. Note that an entry far from the first
                 time step is considered as the point ti = 0.0 results in a starting fitting velocity v = 0.0. This is also taken into account in the 
                 plots which start from fit[1]. Note that this velocity estimate is only accurate for trajectories that are (approximately) linear. If 
                 in doubt whether the motion is ballistic, always consult the trajectory plots before utilizing the fit velocity.
                 popt[1] - estimate of the diffusion coefficient. Again, only accurate for ballistic motion."""

        k_index = np.where(self.Ks == K0) #self.params['K_0'])
        p_weight = (np.fft.fftshift(self.coeffs['X_k'])[k_index])**2
        e_weight = (np.fft.fftshift(self.coeffs['Y_k'])[k_index])**2
        print(K0)
        print('Photonic weight, X_k^2 =', p_weight, 'Molecular weight, Y_k^2 =', e_weight)
        if plot_hopfield:
            fig1, ax1 = plt.subplots(1,1,figsize = (2,2))
            ax1.plot(self.ks, np.fft.fftshift(self.coeffs['X_k']), label = 'X_k')
            ax1.plot(self.ks, np.fft.fftshift(self.coeffs['Y_k']), label = 'Y_k')
            ax1.set_title('Hopfield coefficients')
            ax1.set_xlabel('k [$\mu m$]')
            ax1.legend()    

        times, n_arr = self.calculate_evolved_n_L_all_k(tf, kspace = False, K0val = K0)
        times *= self.EV_TO_FS # rescale for plotting
        rs = np.linspace(0,self.params['L'], len(self.Ks))
        n_0 = n_arr[0,:]
        msd_val_0 = np.average(rs, weights=n_0) # position of weighted mean (i.e ~ position of peak)
        msd_arr = np.zeros((len(times)), dtype=float) 
        for i in range(1,len(times)):
            n_i = n_arr[i,:]
            msd_val = np.average(rs, weights=n_i) # weighted mean
            msd_arr[i] = msd_val-msd_val_0 # shift coordinate system so that initial population starts at r = 0.0
        
        def f(t,D):
            return D*t + msd_arr[0] 
        def v(t,D):
            return D*np.ones(len(t))  
            
        popt, pcov = curve_fit(f, times, msd_arr, bounds=([0,1]))  
        fit_data = f(times, popt[0])
        
        avg_v = 0
        if not suppress_plots:
            fig, ax = plt.subplots(1,2,figsize = (9,4), layout = 'tight')
            ax[0].plot(times, msd_arr, label = 'data', color = 'black') 
            ax[0].plot(times, fit_data, label = 'fit', ls = '--', color = 'blue') 
            ax[0].set_xlabel('time [$fs$]', fontsize=14)
            ax[0].set_ylabel('md [$\mu m$]', fontsize=14)
            if npgradient:
                dt = times[1] - times[0]
                v_of_msd = np.gradient(msd_arr, dt)
                ax[1].plot(times, v_of_msd, label = 'np.gradient', color = 'black')
                ax[1].plot(times, v(times, popt[0]), label = 'curve_fit', ls = '--', color = 'blue') 
                avg_v = np.mean(v_of_msd)
            else:
                if np.round(popt[0],2) == 1.0:  # set y tick spacing manually when v = const.; automatic setting makes units hard to read
                    ax[1].set_yticks(np.arange(0.5*fit_v_of_msd[int(np.round(len(times)/2))], 1.5*fit_v_of_msd[int(np.round(len(times)/2))], 0.2*fit_v_of_msd[int(np.round(len(times)/2))]))
                    ax[1].ticklabel_format(axis = 'y', useOffset=False)
                    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                ax[1].plot(times, v(times, popt[0]), label = 'curve_fit', color = 'blue')     
            ax[1].set_xlabel('time [$fs$]', fontsize=14)
            ax[1].set_ylabel('$v_{md}$ [$\mu m$ $fs^{-1}$]', fontsize=14)
            for i in range(2):
                ax[i].minorticks_on()
                ax[i].tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=13)
                for axis in ['top','bottom','left','right']:
                    ax[i].spines[axis].set_linewidth(1.3)
                ax[i].grid(alpha = 0.2)
                ax[i].legend()    
            fig.suptitle('Position and speed of MD of lower polariton distrubution', fontsize=16)
            if savefig:
                plt.savefig(fname = filename, format = 'jpg')
            plt.show()
        return p_weight, e_weight, avg_v, v(times, popt[0])[int(np.round(len(times)/2))], popt[0], np.sqrt(np.diag(pcov))

    def plot_v_wrt_k(self, K0s = None, tf = 1.1, npgradient = False, suppress_plots = False, relative = False, concentrations = False):
        """Calculates and plots the velocity of the mean position of the wavepacket by estimating the gradient of its trajectory
           through fitting a curve to it and/or by using np.gradient().
           Inputs:  K0s [float] - array of central wavenumbers of the initial population [unitless; k0s = K0s*L/(2pi)] 
                    tf [float] - integration time in physical units (conversion to femtoseconds performed internally)
                    npgradient [bool] - if True, plot velocity both as the derivative of a fit to the trajectory
                    and as a direct np.gradient. Note that the np.gradient routine might result in a more jittery pattern.
                    suppress_plots [bool] - if True, plots of trajectory and velocity over time generated by self.plot_n_L_peak_velocity
                    are suppressed
                    concentrations [bool] - if True, plots with respect to photonic concentration. If False, plots with respect to K0
                    relative [float] - if True, function plots v_obs/v_k0. If False, both v_obs and v_k0 are plotted on the same plot
           Outputs: p_weights, e_weights, vvals, vvalsnp, p0s [floats] - respectively, arrays of polariton and exciton 
                    concentrations, wavepacket velocities extracted through curve fitting and np.gradient routines 
                    and powers of the curve fit [used to estimate whether transport is diffusive/ballistic]"""

        if K0s == None:
            K0s = np.array([self.params['K_0']])
        p_weights = np.ones_like(K0s)
        e_weights = np.ones_like(K0s)
        vvalsnp = np.ones_like(K0s)
        vvals = np.ones_like(K0s)
        p0s = np.ones_like(K0s)

        S = self.params['S']
        Gz = self.params['Gam_z']
        ind = 0
        for i in K0s:
            p_weight, e_weight, vvalnp, vval, p0, pcov = self.plot_n_L_peak_velocity(savefig = suppress_plots, suppress_plots = suppress_plots, filename = f'v_at_k0_{i}_S_{S}.jpg', tf = tf, npgradient = npgradient, K0 = i) 
            print('Pcov', pcov, np.shape(pcov))
            p_weights[ind] = p_weights[ind]*p_weight
            e_weights[ind] = e_weights[ind]*e_weight
            vvalsnp[ind] = vvalsnp[ind]*vvalnp
            vvals[ind] = vvals[ind]*vval
            p0s[ind] = p0s[ind]*p0
            ind = ind+1
            assert i != 0.0, "Can't normalise v_obs with respect to v_K0 = 0. Please specify K0s different from 0.0."
            
        v_c, v_L, v_U = self.group_velocity_expected(Ks_auto = False, all_Ks = K0s)

        if npgradient:
            fig, ax = plt.subplots(1,1,figsize = (6,4.5), layout = 'tight')
            if relative:
                if concentrations:
                    ax.plot(p_weights, vvals/v_L, marker = '.', ls = '--', color = 'blue')
                    ax.set_xlabel('$X_k^2$')
                    ax.set_ylabel('$v_{obs}$/$v_{k0}^{L}$')
                else:
                    ax.plot(K0s, vvals/v_L, marker = '.', ls = '--', color = 'blue')
                    ax.set_xlabel('$K_0$')
                    ax.set_ylabel('$v_{obs}$/$v_{k0}^{L}$')
            else:
                if concentrations:
                    ax.plot(p_weights, 1000.0*v_L, marker = '.', label = '$v_{k0}^L$', ls = '--', color = 'black')
                    ax.errorbar(p_weights, 1000.0*vvals, yerr=1000.0*pcov[0], marker='.', label = '$v_{obs}$', ls = '--', color = 'blue')
                    ax.set_xlabel('$X_k^2$')
                else:
                    ax.plot(K0s, 1000.0*v_L, marker = '.', label = '$v_{k0}^L$', ls = '--', color = 'black')
                    ax.errorbar(K0s, 1000.0*vvals, yerr=1000.0*pcov[0], marker='.', label = '$v_{obs}$', ls = '--', color = 'blue')            
                    ax.set_xlabel('$K_0$')                
                ax.set_ylabel('$v_{obs} [\mu m$ $ps^{-1}]$')
                ax.legend()
            ax.minorticks_on()
            ax.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=13)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(1.3)
            ax.grid(alpha = 0.2)
            ax.set_title(f'Lower polariton group velocities (S = {S}, $\Gamma_z$ = {Gz})')
            plt.savefig(fname = f'velocities_np_gradient_S_{S}_Gz_{Gz}.jpg', format = 'jpg')

        fig1, ax1 = plt.subplots(1,1,figsize = (6,4.5), layout = 'tight')
        if relative:
            if concentrations:
                ax1.plot(p_weights, vvals/v_L, marker = '.', ls = '--', color = 'blue')
                ax1.set_xlabel('$X_k^2$')
                ax1.set_ylabel('$v_{obs}$/$v_{k0}^{L}$')
            else:
                ax1.plot(K0s, vvals/v_L, marker = '.', ls = '--', color = 'blue')
                ax1.set_xlabel('$K_0$')
                ax1.set_ylabel('$v_{obs}$/$v_{k0}^{L}$')
        else:
            if concentrations:
                ax1.plot(p_weights, 1000.0*v_L, marker = '.', label = '$v_{k0}^L$', ls = '--', color = 'black')
                ax1.errorbar(p_weights, 1000.0*vvals, yerr=1000.0*pcov[0], marker='.', label = '$v_{obs}$', ls = '--', color = 'blue')
                ax1.set_xlabel('$X_k^2$')
            else:
                ax1.plot(K0s, 1000.0*v_L, marker = '.', label = '$v_{k0}^L$', ls = '--', color = 'black')
                ax1.errorbar(K0s, 1000.0*vvals, yerr=1000.0*pcov[0], marker='.', label = '$v_{obs}$', ls = '--', color = 'blue')
                ax1.set_xlabel('$K_0$')                
            ax1.set_ylabel('$v_{obs} [\mu m$ $ps^{-1}]$')
            ax1.legend()
        ax1.minorticks_on()
        ax1.tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=13)
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(1.3)
        ax1.grid(alpha = 0.2)
        ax1.set_title(f'Lower polariton group velocities (S = {S}, $\Gamma_z$ = {Gz})')
        plt.savefig(fname = f'velocities_curvefit_dephasing_S_{S}_Gz_{Gz}.jpg', format = 'jpg')
        plt.show()      
        return p_weights, e_weights, vvals, vvalsnp, p0s

    def plot_wrt_S(self, Svals = np.arange(0,10,0.5), Gam_z = np.arange(0,0.01,0.001), tf = 1.1, savefig = False, all_plots = False):
        """Calculates and plots the velocity of the mean position of the lower polariton population for different values SVALS of the Huang-Rhys parameter
           S and the dephasing constant GAM_Z.
           Inputs:  Svals [float] - array of Huang-Rhys parameter values 
                    tf [float] - integration time in physical units (conversion to femtoseconds performed internally)
                    Gam_z [float] - array of dephasing constant values
                    savefig [bool] - if True, the final plot of v wrt Svals and Gam_z is stored under the name 'plot_v_as_function_of_S_Gam_z.jpg'
                    all_plots [bool] - if True, plots of the trajectory and the velocity estimate of the wavepacket are produced for each S and Gamma_z
           Outputs: p_weights, e_weights, vvals, vvalsnp, p0s [floats] - respectively, arrays of polariton and exciton 
                    concentrations, wavepacket velocities extracted through curve fitting and np.gradient routines 
                    and powers of the curve fit [used to estimate whether transport is diffusive/ballistic]
        """
        
        avg_vs = np.ones_like(Svals, dtype = float)
        fit_v_of_msd = np.ones_like(Svals, dtype = float)
        fig, ax = plt.subplots(1,2,figsize = (9,4.5), layout = 'tight')
        K0 = self.params['K_0']
        v_c, v_L, v_U = self.group_velocity_expected(Ks_auto = False, all_Ks = np.array([K0]))
        def f(t,D):
            return D*t + msd_arr[0] 
        def v(t,D):
            return D*np.ones(len(t)) 
        for s in Svals:
            self.make_coeffs(S = s, Gam_z = np.array([0.0]))
            k_index = np.where(self.Ks == K0) 
            p_weight = (np.fft.fftshift(self.coeffs['X_k'])[k_index])**2
            e_weight = (np.fft.fftshift(self.coeffs['Y_k'])[k_index])**2
            print('Current S = ', s)
            times, n_arr= self.calculate_evolved_n_L_all_k(tf, kspace = False, K0val = K0) #different values for each s
            times *= self.EV_TO_FS # rescale for plotting
            n_0 = n_arr[0,:]
            msd_arr = np.zeros((len(times)), dtype=float) 
            msd_val = np.average(self.params['delta_r']*self.Ks, weights=n_0) # position of weighted mean (i.e ~ position of peak)
            msd_arr[0] = msd_val
            for i in range(1, len(times)):
                n_i = n_arr[i,:]
                msd_val = np.average(self.params['delta_r']*self.Ks, weights=n_i) # weighted mean
                msd_arr[i] = msd_val
                    
            popt, pcov = curve_fit(f, times, msd_arr, bounds=([0,1]))  
            fit_data = f(times, popt[0])
            fit_v_of_msd[np.where(Svals==s)] *= v(times, popt[0])[int(np.round(len(times)/2))] 
    
            dt = times[1] - times[0]
            v_of_msd = np.gradient(msd_arr, dt)
            avg_v = np.mean(v_of_msd)
            avg_vs[np.where(Svals==s)] *= avg_v 

            if all_plots:
                fig1, ax1 = plt.subplots(1,2,figsize = (9,4), layout = 'tight')
                ax1[0].plot(times, msd_arr, label = 'data') 
                ax1[0].plot(times, fit_data, label = 'fit', ls = '--') 
                ax1[0].set_xlabel('time [$fs$]', fontsize=14)
                ax1[0].set_ylabel('md [$\mu m$]', fontsize=14)
                ax1[1].plot(times, v(times, popt[0]), label = 'curve_fit', marker = '.')     
        ax[0].plot(Svals, fit_v_of_msd/v_L, marker = '.', ls = '--', color = 'blue')      
        ax[0].set_xlabel('$S$', fontsize=14)
        avg_vs = np.ones_like(Gam_z, dtype = float)
        fit_v_of_msd = np.ones_like(Gam_z, dtype = float)
        
        for Gz in Gam_z:
            self.make_coeffs(S = np.array([0.0]), Gam_z = Gz)
            k_index = np.where(self.Ks == K0)
            p_weight = (np.fft.fftshift(self.coeffs['X_k'])[k_index])**2
            e_weight = (np.fft.fftshift(self.coeffs['Y_k'])[k_index])**2
            print('Gam_z = ', Gz)
            times, n_arr= self.calculate_evolved_n_L_all_k(tf, kspace = False, K0val = K0) # different values for each Gz
            times *= self.EV_TO_FS # rescale for plotting
            n_0 = n_arr[0,:]
            msd_val = np.average(self.params['delta_r']*self.Ks, weights=n_0) # position of weighted mean (i.e ~ position of peak)
            msd_arr = np.array([msd_val])
            for i in range(1, len(times)):
                n_i = n_arr[i,:]
                msd_val = np.average(self.params['delta_r']*self.Ks, weights=n_i) # weighted mean
                msd_arr = np.append(msd_arr, msd_val)
                    
            popt, pcov = curve_fit(f, times, msd_arr, bounds=([0,1])) 
            fit_data = f(times, popt[0]) 
            fit_v_of_msd[np.where(Gam_z==Gz)] *= v(times, popt[0])[int(np.round(len(times)/2))]
            dt = times[1] - times[0]
            v_of_msd = np.gradient(msd_arr, dt)
            avg_v = np.mean(v_of_msd)
            avg_vs[np.where(Gam_z==Gz)] *= avg_v 
            if all_plots:
                fig2, ax2 = plt.subplots(1,2,figsize = (9,4), layout = 'tight')
                ax2[0].plot(times, msd_arr, label = 'data') 
                ax2[0].plot(times, fit_data, label = 'fit', ls = '--') 
                ax2[0].set_xlabel('time [$fs$]', fontsize=14)
                ax2[0].set_ylabel('md [$\mu m$]', fontsize=14)
                ax2[1].plot(times, v(times, popt[0]), label = 'curve_fit', marker = '.')         
        ax[1].plot(Gam_z, fit_v_of_msd/v_L, marker = '.', ls = '--', color = 'blue')  
        ax[1].set_xlabel('$\Gamma_z$', fontsize=14)

        for i in range(2):
            ax[i].minorticks_on()
            ax[i].tick_params(axis="both", direction="in", which="both", right=True, top=True, labelsize=13)
            for axis in ['top','bottom','left','right']:
                ax[i].spines[axis].set_linewidth(1.3)
                ax[i].grid(alpha = 0.2)
            ax[i].set_title('$v_{obs}/v_{k0}^{L}$', fontsize=16)
        if savefig:
            plt.savefig(fname = 'plot_v_as_function_of_S_Gam_z.jpg', format = 'jpg')
        plt.show()
        
if __name__ == '__main__':
    logging.basicConfig(
        format='%(filename)s L%(lineno)s %(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%H:%M')
    params = {
        'Q0': 150, # how many modes either side of K0 (or 0 for populations) to include; 2*Q0+1 modes total 
        'Nm': 6001, # Number of molecules
        'Nnu': 4, # Number of vibrational levels for each molecules
        'L': 180.0, # Crystal propagation length, inverse micro meters
        'nr': 1.0, # refractive index, sets effective speed of light c/nr
        'omega_c': 1.94, # omega_0 = 1.94eV (Fig S4C)
        'epsilon': 2.14, # exciton energy, detuning omega_0-epsilon (0.2eV for model I in Xu et al. 2023)
        'gSqrtN': 0.15, # light-matter coupling
        'kappa_c': 5e-3, # photon loss
        'Gam_z': 0.0, # molecular pure dephasing
        'Gam_up': 0.0, # molecular pumping
        'Gam_down': 5e-7, # molecular loss
        'S': 1.5, #1.5, # Huang-Rhys parameter
        'omega_nu': 0.00647, # vibrational energy spacing
        'T': 0.026, # k_B T in eV (.0259=300K, .026=302K)
        'gam_nu': 0.01, # vibrational damping rate
        'Am': 0.1, # amplitude of initial wavepacket
        'K_0': 80.0, # central wavenumber of initial wavepacket
        'sig_0': 4.0, # s.d. of initial wavepacket
        'atol': 1e-7, # solver tolerance
        'rtol': 1e-7, # solver tolerance
        'dt': 5, # determines interval at which solution is evaluated. Does not effect the accuracy of solution, only the grid at which observables are recorded
        'exciton': False, # if True, initial state is pure exciton; if False, a lower polariton initial state is created
        'photon': False, # if True, initial state is pure photon; if False, a lower polariton initial state is created
        }
    
    #julia_comparison('data/julia/gn0.45N1e5Z1.csv') # Gam_z = 0.01 # Julia comparison 2024-06-20
    htc = HTC(params)
    #htc.plot_initial_populations(kspace = False)
    #htc.plot_waterfall(n_L = True, tf = 100, step = 10, kspace = False, legend = True)
    #htc.plot_n_D_fixed_k(savefig = True, tfs = np.array([20.1]), K0vals = np.arange(0,120.1,5))
    #htc.plot_wrt_S(Svals = np.arange(0.0,1.1,1.0), Gam_z = np.arange(0,0.04,0.0051), tf = 20.1, all_plots = True)
    #htc.plot_total_population_growth(tf = 10.1, K0val = 80.0, savefig = True)
    htc.plot_all(tf = 1600.1, kspace = False, fixed_position_index = None, savefigs = True)
