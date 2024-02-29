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
SOLVER = RK45 # Best!
#from scipy.integrate import RK23, DOP853, Radau, BDF, LSODA # Alternative solvers (testing)
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

sns.set_theme(context='notebook', # paper notebook talk poster (mainly scales font and linewidth)
              style='ticks', # default 'darkgrid', 'tick' definitely best
              palette='colorblind6', # 'colorblind' if need more than 6 lines
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
            'initial_state': 'incoherent', # or photonic
            'pex':0.01, # mean initial molecular population (for initial_state 'incoherent')
            'sigma_0':0.1, # s.d. of initial incoherent population as fraction of L
            'sigma_f':0, # s.d. in microns instead (if specified)
            'atol':1e-9, # solver tolerance
            'rtol':1e-6, # solver tolerance
            'dt': 0.5, # determines interval at which solution is evaluated. Does not effect the accuracy of solution, only the grid at which observables are recorded
            'rescale':1, # see self.rescale_int. N.B. not all parts of code use this function, and coefficients must be adjusted according to scaling in self.make_coeffs
            'lowpass': 0.01, # used for smoothing oscillations in observable plots
            'calculate_lp': False, # compute LP population for each state - adds overhead due to for-loops
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
        self.wrapit(self.create_initial_state, f'Creating initial state...', timeit=False)
        self.labels = self.get_labels()

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
        self.rescale_int = params['rescale']
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
        names = ['ada', 'l', 'al', 'll']
        self.state_dic = {
            'ada': {'shape': (Nk, Nk)}, 
            'l': {'shape': (2*Nnu**2-1, Nk)}, 
            'al': {'shape': (Nnu**2, Nk, Nk)}, 
            'll': {'shape': (Nnu**2, Nnu**2, Nk, Nk)}, 
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
        #self.state_split_list.pop() 
        self.added_state_length = 1# EXTRA entry appending to state to indicate if has been rescaled or not
        self.state_length = np.sum([self.state_dic[name]['num'] for name in self.state_dic]) + self.added_state_length
        correct_state_length = Nk**2 + (2*Nnu**2-1)*Nk + Nnu**2*Nk**2 + Nnu**4*Nk**2 + self.added_state_length
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
            # Kristin doesn't have this correction
            A += 0.25 * (-1j * rates['gam_delta']) * np.kron(si, (bd @ bd - b @ b))
        B = params['gSqrtN'] * np.kron(sp, bi)
        C1 = np.sqrt(rates['Gam_z']) * np.kron(sz, bi)
        if kba:
            # and has different thermalisation
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
        consts['xim'] = - 2 * contract('aij,a->ij', f011.conj(), consts['A0']) \
                + 1j * contract('aip,ab,bpj->ij', f011.conj(), consts['gam00'], z011.conj()) \
                - 1j * contract('aip,ba,bpj->ij', f011.conj(), consts['gam00'], zm011) \
                + 1j * contract('aip,qp,aqj->ij', f011.conj(), consts['gampp'], z011.conj()) \
                - 1j * contract('aip,pq,aqj->ij', f011.conj(), consts['gammm'], zm011)
        shifted_Ks = np.fft.ifftshift(self.Ks) # ALL COMPUTATIONS DONE WITH numpy order of modes
        #rolled_Ks = np.roll(self.Ks, -self.Q0) # equivalent to shifted_Ks
        consts['kappa'] = self.kappa(shifted_Ks)
        consts['omega'] = self.omega(shifted_Ks)
        # CIJ means Jth coeff. of Ith eqn.
        def expikr(K, n):
            return np.exp(2j * np.pi / Nk * K * n)
        #coeffs['expi_kn'] = np.zeros((Nk, Nk), dtype=complex)
        coeffs['expi_kn'] = np.fromfunction(np.vectorize(expikr), (Nk, Nk)) # because of periodicity, this is sufficient
        # EQ 1 , here p->k'
        coeffs['11_pk'] = np.zeros((Nk, Nk), dtype=complex) # But MUST construct this manually!
        #b = - ( 1j * -(np.vectorize(self.omega_diff)(shifted_Ks, shifted_Ks)) # minus due to index order
        #                  + 0.5 * np.vectorize(self.kappa_sum)(shifted_Ks, shifted_Ks))
        for i, K in enumerate(shifted_Ks):
            #for j, n in enumerate(self.ns):
            #    coeffs['expi_kn'][i,j] = expikr(K, n)
            for j, P in enumerate(shifted_Ks):
                # EDIT 2023-10-26: Fixed sign of omega term!
                coeffs['11_pk'][j,i] = (1j * self.omega_diff(P,K) - 0.5 * self.kappa_sum(P,K))
        coeffs['12_1'] = 1j * consts['Bp']
        coeffs['13_1'] = coeffs['12_1'].conj()
        # EQ 2
        coeffs['21_00'] = 1 * consts['xi'] # (make a copy)
        # initially 1-d array of length N0, broadcast to N0xNk array by copying the original array into 
        # separate rows, then take transpose to get correct index order (in other words, this constant 
        # is the same at each ensemble n):
        coeffs['22_0n'] = np.broadcast_to(consts['phi0'], (Nk, self.N0)).T 
        coeffs['23_01'] = 4 * contract('i,aij->aj', consts['Bp'], f011)
        # EQ 3
        coeffs['31_11k'] = contract('ij,k->ijk', consts['xip'], np.ones(Nk)) \
                - contract('ij,k->ijk', np.eye(gp.indices_nums[1]),
                           1j * consts['omega'] + 0.5 * consts['kappa'])
        coeffs['32_1'] = coeffs['12_1'].conj()
        coeffs['33_1kn'] = 1j * contract('i,kn->ikn', consts['Bp'].conj(), coeffs['expi_kn']) / Nm
        coeffs['34_1kn0'] = - contract('jkn,aij->ikna', coeffs['33_1kn'], z011)
        coeffs['35_1kn'] = - coeffs['33_1kn'] / Nnu
        coeffs['36_01'] = 2 * contract('aij,j->ai', f011, consts['Bp'].conj())
        # EQ 4
        coeffs['41_11'] = 1 * consts['xip']
        coeffs['42_11'] = 1 * consts['xim']
        coeffs['43_01'] = coeffs['36_01'].conj() 
        coeffs['44_01'] = 1 * coeffs['36_01']
        if self.rescale_int==1:
            coeffs['23_01'] *= (1/Nm)
            coeffs['34_1kn0'] *= Nm
            coeffs['35_1kn'] *= Nm
        elif self.rescale_int==2:
            sNm = np.sqrt(Nm)
            coeffs['23_01'] *= (1/Nm) * (1/sNm)
            #coeffs['32_1'] *= sNm
            #coeffs['33_1kn'] *= sNm
            coeffs['34_1kn0'] *= Nm * sNm
            coeffs['35_1kn'] *= Nm * sNm
            #coeffs['43_01'] /= sNm
            #coeffs['44_01'] /= sNm
        # HOPFIELD coefficients (in shifted basis i.e. K=0,1,2,...,Q0,-Q0,-Q0+1,....-1
        consts['zeta_k'] = 0.5 * np.sqrt( (params['epsilon'] - consts['omega'])**2 + 4 * params['gSqrtN']**2 )
        coeffs['X_k'] = np.sqrt(0.5  + 0.5**2 * (params['epsilon'] - consts['omega'])/consts['zeta_k'])
        coeffs['Y_k'] = np.sqrt(0.5  - 0.5**2 * (params['epsilon'] - consts['omega'])/consts['zeta_k'])
        assert np.allclose(coeffs['X_k']**2+coeffs['Y_k']**2, 1.0), 'Hopfield coeffs. not normalised'
        consts['vsigma'] = gp.get_coefficients(np.kron(sp, bi), sgn=1, eye=False)
        assert np.allclose(consts['vsigma'].imag, 0.0)
        consts['vvsigma'] = self.Nnu/2
        assert np.allclose(consts['vvsigma'], contract('i,i', consts['vsigma'], consts['vsigma']))
        assert np.allclose(contract('i,inm->nm', consts['vsigma'], gp.basis[gp.indices[1]]), np.kron(sp,bi))
        # CHECKING n_M part of n_B
        #C0, D0 = gp.get_coefficients(np.kron(np.matmul(sp,sp.T),bi), sgn=0, eye=True)
        #print(np.isclose(contract('i,i',consts['varsigma'],np.conj(consts['varsigma']))/self.Nnu, D0))
        #print(np.allclose(C0, contract('i,j,aij',consts['varsigma'], np.conj(consts['varsigma']), z011)))
        #sys.exit()
        # COefficients used to calculate observables
        ocoeffs = {}
        # 'pup_l' and 'pup_I' are C^0_{i_0} and D^0 in thesis
        ocoeffs['pup_l'], ocoeffs['pup_I'] = \
                self.gp.get_coefficients(np.kron(Pauli.p1, self.boson.i), sgn=0, eye=True)
        assert np.allclose(contract('a,anm->nm', ocoeffs['pup_l'], gp.basis[gp.indices[0]]),
                           np.kron(Pauli.p1, bi)-ocoeffs['pup_I']*np.eye(2*self.Nnu))
        assert np.allclose(ocoeffs['pup_I'], 0.5)
        ocoeffs['sp_l'] = consts['vsigma']
        ocoeffs['rn'] = params['delta_r'] * self.ns
        ocoeffs['rn2'] = ocoeffs['rn']**2
        ocoeffs['msrn'] = (ocoeffs['rn'] - 0.5 * params['L'])**2
        # assign to instance variables
        self.consts, self.coeffs, self.ocoeffs = consts, coeffs, ocoeffs

    def omega(self, K):
        # dispersion to match MODEL used by Xu et al. 2023 (self.K_factor set in
        # self.add_useful_params)
        return self.wc * np.sqrt(1 + self.K_factor**2 * K**2)

    def kappa(self, K):
        # uniform
        return self.params['kappa_c'] * np.ones_like(K)

    def deltak(self, k=0):
        return np.array([1.0 if q==k else 0.0 for q in self.get_Q_range()])

    def cg(self, K):
        return self.c * self.K_factor * K / np.sqrt(1 + self.K_factor**2 * K**2)

    def velocity(self, t, D, alpha):
        return alpha * D * t**(alpha-1)

    def omega_diff(self, K, P):
        return self.omega(K) - self.omega(P)

    def kappa_sum(self, K, P):
        return self.kappa(K) + self.kappa(P)

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

    def nb(self, T):
        if T==0.0:
            return 0.0
        return 1/(np.exp(self.params['omega_nu']/self.params['T'])-1)

    def create_initial_state(self):
        form = self.params['initial_state']
        if form=='incoherent':
            self.initial_state = self.incoherent_state()
        elif form=='photonic':
            raise NotImplemented
        else:
            print(f'State form {form} unknown') 
            sys.exit(1)

    def incoherent_state(self):
        if self.params['sigma_f'] != 0:
            # absolute width specification (microns) of initial profile width
            self.params['sigma_0'] = self.params['sigma_f']/self.params['L']
        self.plot_initial_profile()
        state = np.zeros(self.state_length, dtype=complex)
        mid_index = self.Nk//2
        if self.params['sigma_0'] == 0:
            pex = self.params['pex'] * np.ones(self.Nk)
        else:
            pex = self.params['pex'] * \
                    np.exp(-(self.ns-mid_index)**2/(2*(self.Nk*self.params['sigma_0'])**2))
        rho0_vib = self.thermal_rho_vib(self.params['T']) # molecular vibrational density matrix
        l, self.all_eye0s = [], [] # all_eye0s only needed if want to recreate d.m. on a site
        for n in range(self.Nk):
            rho0n = np.kron(np.diag([pex[n],1-pex[n]]), rho0_vib) # total molecule density operator
            coeffs0, eye0 = self.gp.get_coefficients(rho0n, sgn=0, eye=True, warn=False)
            l.append(2 * coeffs0)
            self.all_eye0s.append(eye0)
        l = np.real(l).T # i_0 index first, then ensemble index n
        state[self.state_dic['l']['slice']] = l.flatten()
        state[-1] = -1 # indicates state has NOT been rescaled
        return state
    
    def evolve(self, tf=None, tf_fs=None):
        """Integrate second-order cumulants equations of motion from a state of incoherent excitons
        at t=0 to a time tf (in natural units) or tf_fs (femptoseconds)"""
        assert np.sum(np.array([tf, tf_fs]) == None) == 1,\
                'Exactly one of parameters \'tf\' and \'tf_fs\' must be given'
        dt = self.params['dt']
        if tf is not None:
            tf_fs = tf * self.EV_TO_FS
            self.t = np.arange(0.0, tf, step=dt)
            self.t_fs = self.t * self.EV_TO_FS
        else:
            tf = tf_fs / self.EV_TO_FS
            self.t_fs = np.arange(0.0, tf_fs, step=dt)
            self.t = self.t_fs / self.EV_TO_FS
        self.num_t = len(self.t)
        state_MB = sys.getsizeof(self.initial_state) / 1024**2
        logger.info(f'State length {len(self.initial_state):.2e} requiring {state_MB:.0f} MB')
        logger.info(f'Integrating 2nd-order EoMs to tf={self.t_fs[-1]:.0f}fs with interpolation'\
                f' to fixed grid of spacing dt={dt:.3g}')
        self.select_t_fs = [0, 0.25, 50, 100, 200] # record large selection of variables in k-space at these times only
        #self.select_t_fs = list(np.arange(45,55,step=dt))
        self.select_t = [t / self.EV_TO_FS for t in self.select_t_fs]
        self.select_t_index = 0 
        self.setup_observable_storage() # creates self.observables data dictionary
        #
        t_index = 0 # indicates current position in output grid of times
        num_checkpoints = 11 # checkpoints at 0, 10%, 20%,...
        checkpoint_spacing = int(round(self.num_t/num_checkpoints))
        checkpoints = np.linspace(0, self.num_t-1, num=num_checkpoints, dtype=int)
        next_check_i = 1
        last_solver_i = 0
        solver_t = [] # keep track of solver times too (not fixed grid)
        tic = time() # time the computation
        rk45 = SOLVER(self.eoms,
                    t0=0.0,
                    y0=self.initial_state,
                    t_bound=tf,
                    rtol=self.params['rtol'],
                    atol=self.params['atol'],
                    # N.B. max_step makes little difference in terms of computation, maybe more accurate (?)
                    #max_step=dt,
                    )
        # Save initial state
        assert rk45.t == self.t[t_index], 'Solver initial time incorrect'
        self.record_observables(t_index, rk45.y) # record physical observables for initial state
        solver_t.append(rk45.t)
        t_index += 1
        next_t = self.t[t_index]
        while rk45.status == 'running':
            end = False # flag to break integration loop
            rk45.step() # perform one step (necessary before call to dense_output())
            solver_t.append(rk45.t)
            if rk45.t >= next_t: # solver has gone past one (or more) of our grid points, so now evaluate soln
                soln = rk45.dense_output() # interpolation function for the last timestep
                while rk45.t >= next_t: # until soln has been evaluated at all grid points up to solver time
                    y = soln(next_t)
                    self.record_observables(t_index, y) # extract relevant observables from state y 
                    t_index += 1
                    if t_index >= self.num_t: # reached the end of our grid, stop solver
                        end = True
                        break
                    next_t = self.t[t_index]
            if next_check_i < num_checkpoints and t_index >= checkpoints[next_check_i]:
                solver_diffs = np.diff(solver_t[last_solver_i:])
                logger.info('Progress {:.0f}% ({:.0f}s)'.format(100*(checkpoints[next_check_i]+1)/self.num_t, time()-tic))
                logger.info('Avg. solver dt for last part: {:.2g} (grid dt={:.3g}; {:.3g}fs)'\
                        .format(np.mean(solver_diffs), dt, dt*self.EV_TO_FS))
                #logger.info('Avg. solver dt over last interval: {:.2g} (unscaled units) '\
                #        ' compared to set dt={} (considering adjusting if large difference)'.format(
                #            np.mean(solver_diffs), dt))
                # If solver time step far larger, can increase dt to save memory and computation time
                # If solver time step far smaller, may be missing physical information
                next_check_i += 1
                last_solver_i = len(solver_t) - 1
            if end:
                break # safety, stop solver if we have already calculated state at self.t[-1]
        toc = time()
        self.compute_time = toc-tic # ptoc-ptic
        logger.info('...done ({:.0f}s)'.format(self.compute_time))

    def setup_observable_storage(self):
        """Prepare dictionary self.observables to store values of relevant observables
        These arrays (or arrays in dictionaries) are zero initialised and then assigned
        non-zero values in place by self.record_observables during the computation
        """
        Ns = self.num_t
        ns = np.zeros((Ns, self.Nk), dtype=float)
        #ph_dic, mol_dic, coh_dic = 3*[self.blank_density_dic()] # DO NOT USE - creates list of 3 references to the same object! (same for e.g. x,y = 2*[np.array([1])])!
        ph_dic, mol_dic, coh_dic = self.blank_density_dic(), self.blank_density_dic(), self.blank_density_dic(dtype=complex)
        if self.params['calculate_lp']:
            nALs, nBLs, nCLs, nLPs = [np.zeros((Ns, self.Nk), dtype=float) for i in range(4)]
        else:
            nALs, nBLs, nCLs, nLPs = 4 * [None]
        num_select_t = len(self.select_t)
        select_ts = np.empty((num_select_t,), dtype=float)
        select_ts[:] = np.nan # so we know haven't been assigned yet
        adaks = np.zeros((num_select_t, self.Nk, self.Nk), dtype=complex)
        lqs = np.zeros((num_select_t, self.Nk), dtype=complex)
        akls = np.zeros((num_select_t, self.Nk, self.Nk), dtype=complex)
        llks = np.zeros((num_select_t, self.Nk, self.Nk), dtype=complex)
        nBs = np.zeros((Ns, self.Nk), dtype=float)
        nDs = np.zeros((Ns, self.Nk), dtype=float)
        self.observables = {'params': self.params,
                            't': self.t,
                            't_fs': self.t_fs,
                            'n': ns,
                            'nB': nBs,
                            'nD': nDs,
                            'ph_dic': ph_dic,
                            'mol_dic': mol_dic,
                            'coh_dic': coh_dic,
                            'select_data': {'select_t': select_ts,
                                            'adak': adaks,
                                            'lq': lqs,
                                            'alk': akls,
                                            'llk': llks,
                                            },
                            'LP': {'nLP': nLPs,
                                   'nAL': nALs,
                                   'nBL': nBLs,
                                   'nCL': nCLs,
                                   }
                            }

    def blank_density_dic(self, dtype=float):
        Ns = self.num_t
        return {'vals': np.zeros((Ns, self.Nk), dtype=dtype),
                'mean': np.zeros((Ns,), dtype=dtype),
                'var': np.zeros((Ns,), dtype=dtype),
                'msd': np.zeros((Ns,), dtype=dtype),
                }

    def record_observables(self, t_index, y):
        """Calculates and saves observable values from state y at timestep t_index
        To add additional observables, add a key-empty array to self.observables e.g.
        self.observables['my_obs'] in self.setup_storage_observables and then write a
        function to take state, calculate value of observable and assign to
        self.observables['my_obs'][t_index]
        """
        # This is only copy of entire state we make. Essential -otherwise we would be modifying solver's state!
        # To avoid this copy overhead, don't rescale until AFTER calculating each observable (significant rewrite)
        state = y.copy()
        t = self.t[t_index]
        self.rescale_state(state) # correct scale of variables to calculate physical quantities (state modified in-place)
        ada, l, al, ll = self.split_reshape_return(state, check_rescaled=True) # VIEWS of original state i.e. modifying tate will change ada, l, al and ll and vice versa
        # The following directly update the instance variable self.observables which is a
        # dictionary containing numpy arrays of fixed length
        self.calculate_n(t_index, ada)
        if self.params['calculate_lp']:
            self.calculate_lp_contributions(t_index, ada, l, al, ll)
        self.calculate_densities(t_index, ada, l, al)
        self.calculate_bright_dark(t_index, ll)
        if self.select_t_index < len(self.select_t):
            if t >= self.select_t[self.select_t_index]:
                self.calculate_k_observables(t_index, ada, l, al, ll)
                self.select_t_index += 1

    def rescale_states(self, ys):
        """Rescale each state in ys before calculating physical observables
        ys an array where each column corresponds to a state i.e. y.t[0] is
        initial state. Currently unused (now always work with one state at a time)"""
        ada, l, al, ll, rescaled_arr = np.hsplit(ys, self.state_split_list)
        if not np.all(rescaled_arr==-1):
            # could handle this e.g. loop from states and only rescale those
            # which have not already been rescaled
            logger.critical('some states have already been rescaled!')
            sys.exit(1)
        # N.b. split returns a view so following change state directly!
        if self.rescale_int == 0:
            ada *= self.Nm
            al *= np.sqrt(self.Nm)
        elif self.rescale_int == 1:
            al /= np.sqrt(self.Nm)
            ll /= self.Nm
        elif self.rescale_int == 2:
            al /= self.Nm
            ll /= (self.Nm * np.sqrt(self.Nm))
            ada /= np.sqrt(self.Nm)
        # finite value indicates state has been rescaled
        rescaled_arr[:] = self.rescale_int # updates all entries (whether y contains 1 state or hundreds)

    def rescale_state(self, state):
        """Rescale state before calculating physical observables
        N.B. state is modified in place and so does not need to be returned"""
        assert len(state.shape) == 1, 'state must be a 1-d array (single state)'
        ada, l, al, ll, rescaled_arr = np.split(state, self.state_split_list)
        #print('Max ada | l | al | ll = {:.0e} | {:.0e} | {:.0e} | {:.0e}'.format(*[np.max(np.abs(X)) for X in [ada, l, al, ll]]))
        assert rescaled_arr[0]==-1, 'State has already been rescaled!'
        if self.rescale_int == 0:
            ada *= self.Nm
            al *= np.sqrt(self.Nm)
        elif self.rescale_int == 1:
            al /= np.sqrt(self.Nm)
            ll /= self.Nm
        elif self.rescale_int == 2:
            al /= self.Nm
            ll /= (self.Nm * np.sqrt(self.Nm))
            ada /= np.sqrt(self.Nm)
        rescaled_arr[0] = self.rescale_int  # non-negative value indicates state has been rescaled

    def split_reshape_return(self, state, check_rescaled=False, copy=False):
        """Return views of variables <a^dag_k'a_k>, <lambda_n^i>, <a_k lambda_n^i>, <lambda_n^i lamnbda_m^j>
        in state reshaped into conventional multidimensional arrays."""
        split = np.split(state, self.state_split_list) # ada, l, al, ll, rescaled_arr as flattened arrays
        if check_rescaled:
            assert not np.isnan(split[-1][0]), 'State must be rescaled'
        # reshape each array except the 'rescaled_arr' (contains int indicating rescaling factor used)
        reshaped = [split[i].reshape(self.state_reshape_list[i]) for i in range(len(self.state_reshape_list))]
        # N.B. np.split, reshape returns VIEWS of original array; BEWARE mutations 
        # - copy if want to change return without mutating original state variable
        if copy:
            reshaped = [np.copy(X) for X in reshaped]
        return reshaped

    def eoms(self, t, state):
        """Equations of motion as in cumulant_in_code.pdf"""
        C = self.coeffs
        ada, l, al, ll = self.split_reshape_return(state)
        # Calculate DFTs
        alpha = ifft(ada, axis=0, norm='forward')
        d = fft(al, axis=1, norm='backward') # backward default
        # EQ 1 # N.B. kp->pk ordering
        #t0 = time()
        #c = fft(al, axis=2, norm='forward')
        #dy_ada = C['11_pk'] * ada + contract('i,ikp->pk', C['12_1'], c) \
        #                          + contract('i,ipk->pk', C['13_1'], c.conj())
        #t1 = time()-t0
        pre_c = contract('i,ikn->kn', C['12_1'], al)
        post_c = fft(pre_c, axis=1, norm='forward')
        dy_ada2 = C['11_pk'] * ada + np.transpose(post_c) + np.conj(post_c)
        #t2 = time()-t0
        #assert np.allclose(dy_ada, dy_ada2)
        #print('Fastest {:.2g}s, Saved {:.2g}s'.format(t2, 2*t1-t2))
        # EQ 2
        dy_l = contract('ab,bn->an', C['21_00'], l) + C['22_0n'] + contract('aj,jnn->an', C['23_01'], d).real
        # EQ 3
        #t0 = time()
        #beta = ifft(ll, axis=-1, norm='backward') # backward default
        #dy_al = contract('ijk,jkn->ikn', C['31_11k'], al) \
        #        + contract('j,ijnk->ikn', C['32_1'], beta) \
        #        + contract('jkn,ijnn->ikn', C['33_1kn'], ll) \
        #        + contract('ikna,an->ikn', C['34_1kn0'], l) \
        #        + C['35_1kn'] \
        #        + contract('ai,an,nk->ikn', C['36_01'], l, alpha) 
        #t1 = time()-t0
        pre_beta = contract('j,ijnm->imn', C['32_1'], ll) # N.B. swapped axes
        post_beta = ifft(pre_beta, axis=-2, norm='backward') # See Eqs.
        dy_al2 = contract('ijk,jkn->ikn', C['31_11k'], al) \
                + post_beta \
                + contract('jkn,ijnn->ikn', C['33_1kn'], ll) \
                + contract('ikna,an->ikn', C['34_1kn0'], l) \
                + C['35_1kn'] \
                + contract('ai,an,nk->ikn', C['36_01'], l, alpha) 
        #t2 = time()-t0
        #assert np.allclose(dy_al, dy_al2)
        #print('Fastest {:.2g}s, Saved {:.2g}s'.format(t2, 2*t1-t2))
        dy_ll = contract('ip,pjnm->ijnm', C['41_11'], ll) \
                + contract('jp,ipnm->ijnm', C['42_11'], ll) \
                + contract('aj,am,imn->ijnm', C['43_01'], l, d) \
                + contract('ai,an,jnm->ijnm', C['44_01'], l, d.conj())
        dy_rescale_int = np.zeros(1)
        # flatten and concatenate to match input state structure (1d array)
        dy_state = np.concatenate((dy_ada2, dy_l, dy_al2, dy_ll, dy_rescale_int), axis=None)
        return dy_state
        # could instead initialise dy_state = np.zeros_like(state) at the top
        # and assign the results (e.g. dy_l.ravel() or dy_l.reshape(-1); del dy_l)
        # as go along, but not way to avoid some copying unless work without reshaping
        # (or write state in such a way that a single .reshape(-1) gives the correct order)

    WARN_REAL = {}
    def check_real(self, step, arr, name):
        if name not in self.WARN_REAL:
            self.WARN_REAL[name] = True
        if not self.WARN_REAL[name]:
            return
        if not np.allclose(np.imag(arr), 0.0):
            t = self.t[step]
            logger.warning(f'{name} at t={t} has non-zero imaginary part (further warnings suppressed)')
            self.WARN_REAL[name] = False

    def calculate_n(self, t_index, ada):
        # ada and l must have already been rescaled
        ns = fftshift(np.diag(ada))
        self.check_real(ns, t_index, 'Photon numbers')
        self.observables['n'][t_index] = np.real(ns)

    def calculate_k_observables(self, t_index, ada, l, al, ll):
        z011 = self.gp.z_tensor((0,1,1))
        t = self.t[t_index]
        self.observables['select_data']['select_t'][self.select_t_index] = t
        self.observables['select_data']['adak'][self.select_t_index] = ada
        ps = contract('a,an->n', self.ocoeffs['pup_l'], l) + self.ocoeffs['pup_I']
        self.check_real(t_index, ps, 'Exciton populations (all)')
        self.observables['select_data']['lq'][self.select_t_index] = fftshift(fft(ps)) # -ve exponent (arbitrary choice)
        # AL - > a sigma^+ coherences -> FFT (correct sign)
        asp_kn = contract('ikn,i->kn', al, self.ocoeffs['sp_l'])
        alk = np.diag(fftshift(fft(asp_kn, norm='forward', axis=-1))) # N.B. OLD CODE ONLY fftshift over axes=[-1]???
        self.observables['select_data']['alk'][self.select_t_index] = alk
        # LL -> sigma^+ sigma^- coherences for n \neq m, then add n=m component from l
        # Then double FFT
        #spsm = self.NE**2 * contract('ijnm,i,j->nm', ll, sp_coeffs, sm_coeffs) # note diagonal entries will be 0
        # Old code - does not look right
        din = self.diag_indices_Nk
        #ll_all = np.copy(ll) # N.B. avoid modifying original array!
        #ll_all[:,:,*din] *= (self.NE-1)/self.NE
        #ll_all[:,:,*din] += contract('aij,an->ijn', z011, l)
        #ll_all[:,:,*din] += (1/self.Nnu) * contract('ij,n->ijn', np.eye(self.N1), np.ones(self.Nk))
        #spsm = contract('ijnm,i,j->nm', ll_all, self.ocoeffs['sp_l'], self.ocoeffs['sp_l'])
        #spsm_k = fft(spsm, axis=0, norm='forward')
        #spsm_kp = ifft(spsm_k, axis=-1, norm='forward')
        #spsm_kk = np.diag(fftshift(spsm_kp))
        #self.check_real(t_index, spsm_kk, '<sig^+sig^->[k,k]')
        #self.observables['select_data']['llk'][self.select_t_index] = spsm_kp #np.real(spsm_kk)
        # Contribution from populations
        l0 = contract('a,an->n', self.ocoeffs['pup_l'], l) + self.ocoeffs['pup_I']
        l02 = contract('n,nm->nm', l0, np.eye(self.Nk))
        # Coherences - scale by NE, except for diagonal elements which need (NE-1)
        llA = contract('ijnm,i,j->nm', ll, self.consts['vsigma'], self.consts['vsigma'])
        llA2 = self.NE * llA
        llA2[din] *= (self.NE-1)/self.NE
        llB2 = llA2 + l02 # add contributions
        #llD = contract('n,nm->nm', llA[din], np.eye(self.Nk))
        #assert np.allclose(self.NE * llA - llD, llA2) # True
        #llB = self.NE * llA - llD + l02
        #spsm2 = ifft(fft(llB, axis=0), axis=1)
        spsm3 = ifft(fft(llB2, axis=0), axis=1) # take transforms
        self.observables['select_data']['llk'][self.select_t_index] = spsm3 #np.real(spsm_kk)


    def calculate_densities(self, t_index, ada, l, al):
        alpha = ifft(ada, axis=0) # EDIT 2023-11-03: Now including 1/N_k normalisation!
        dft2 = fft(alpha, axis=-1)
        nph = np.diag(dft2) # n(r_n) when n=m
        self.check_real(t_index, nph, 'Photon density')
        self.observables['ph_dic']['vals'][t_index] = nph.real
        # N.B. we do not need to use the coefficients of the initial density
        # matrix for the identity matrix (self.coeff_eyes), that is only
        # relevant if we want to create the density matrix; instead just note if
        # OP = A lambda0 + B lambda1 + C I then <OP> = A <lambda0> + B<lambda1>
        # + C since <I>=1 should always be true, i.e. Tr[rho]==1
        nM = self.NE * (contract('a,an->n', self.ocoeffs['pup_l'], l) + self.ocoeffs['pup_I'])
        # 2023-11-14 EDIT: Bug - forgot parenthesis to NE * constant part"
        self.check_real(t_index, nM, 'Molecular density')
        self.observables['mol_dic']['vals'][t_index] = nM.real
        asp = contract('i,ikn->kn', self.ocoeffs['sp_l'], al)
        dft = fft(asp, axis=0)
        coh = self.NE * np.diag(dft)
        self.observables['coh_dic']['vals'][t_index] = coh
        self.calculate_moments('ph_dic', t_index)
        self.calculate_moments('mol_dic', t_index)
        self.calculate_moments('coh_dic', t_index)

    def calculate_moments(self, name, t_index):
        weights = np.abs(self.observables[name]['vals'][t_index])
        if np.isclose(np.sum(weights), 0.0):
            return 3 * [np.nan]
        avg = lambda x: np.average(x, weights=weights)
        mean = avg(self.ocoeffs['rn'])
        self.observables[name]['mean'][t_index] = mean
        self.observables[name]['var'][t_index] = avg(self.ocoeffs['rn2']) - mean**2
        self.observables[name]['msd'][t_index] = avg(self.ocoeffs['msrn'])

    def calculate_bright_dark(self, t_index, ll):
        diag_ind = self.diag_indices_Nk
        nM = self.observables['mol_dic']['vals'][t_index] # assumes calculate_densities has already been called
        #nM2 = self.NE * (contract('a,an->n', self.ocoeffs['pup_l'], l) + self.ocoeffs['pup_I'])
        #assert np.allclose(nM, nM2), "ERROR IN nM CALCULATION"
        ll_diag = ll[:,:,*diag_ind] # diagonal entries at EACH i,j N.B. returns a new array
        ssll = contract('i,j,ijn->n',
                        self.consts['vsigma'],
                        self.consts['vsigma'],
                        ll_diag)
        nB = (self.NE-1) * ssll + nM/self.NE
        nD = nM-nB
        #print('NE = {:.0f}, 1/NE = {:.3g}, Bright/Tot = {:.3g}'.format(self.NE, 1/self.NE,
        #                                                               np.real(np.sum(nB)/np.sum(nM))))
        #Emol = (1/constants.e) * constants.h * self.c / (self.params['L']*1e-6/self.Nm)
        #numer = np.sqrt((self.params['gSqrtN'] + self.params['epsilon'])**2 - self.params['omega_c']**2)
        #print('Numer = {:.3g}, Emol = {:.3g}, Numer/Emol = {:.3g}'.format(numer, Emol, numer/Emol))
        #print('overall ratio = {:.2g}'.format((numer/Emol) / (np.real(np.sum(nB)/np.sum(nM)))))
        self.check_real(t_index, nB, 'Bright state')
        #self.check_real(t_index, nD, 'Dark state') # must be real if nM, nB checked for real
        self.observables['nB'][t_index] = np.real(nB)
        self.observables['nD'][t_index] = np.real(nD)

    def calculate_lp_contributions(self, t_index, ada, l, al, ll):
        # Hopfield matter Y and optical X coefficients in shifted basis i.e. K=0,1,2,...,Q0,-Q0,-Q0+1,....-1
        X, Y = self.coeffs['X_k'], self.coeffs['Y_k']
        vsig = self.consts['vsigma']
        z011 = self.gp.z_tensor((0,1,1))
        # OPTICAL CONTRIBUTION
        nAL = np.diag(fft(ifft(contract('p,k,pk->pk', X, X, ada), norm='forward', axis=-1), norm='backward', axis=0))
        self.check_real(t_index, nAL, 'nAL(r_n)')
        # COHERENT CONTRIBUTIONS - includes sign and conjugate, necessarily real 
        cp = (self.NE / np.sqrt(self.Nm)) * fft(contract('i,ikn->kn', vsig, al), axis=-1, norm='backward')
        nCL = -2*np.real(np.diag(fft(ifft(contract('p,k,kp->kp',Y,X,cp), norm='forward', axis=0), norm='backward', axis=-1)))
        # EXCITON CONTRIBUTION
        eta = self.NE * fft(ifft(contract('i,j,ijnm->nm', vsig, vsig, ll),
                                axis=-1, norm='backward'),
                                axis=0, norm='backward')
        phi = ifft(contract('i,j,ijnn->n', vsig,  vsig, ll), axis=-1, norm='backward')
        l0 = ifft(contract('i,j,aij,an->n', vsig, vsig, z011, l), axis=-1, norm='backward')
        # Constant term - this could be construct in make coefficients for
        T4 = (1/self.Nnu) * self.consts['vvsigma'] * np.eye(self.Nk)
        # TESTING each term has required symmetry
        #print('t={}'.format(t[step]))
        #assert np.allclose(Y.imag, 0.0), "Y"
        #assert np.allclose(np.conj(np.swapaxes(eta, 0, 1)), eta), "eta"
        #assert np.allclose(np.conj(fftshift(phi)), np.flip(fftshift(phi))), "phi"
        #assert np.allclose(l.imag, 0.0), "l"
        #assert np.allclose(np.conj(fftshift(l0)), np.flip(fftshift(l0))), "l0"
        # Construct matrices manually and double FFT - an order of magnitude faster than manually performing DFTs
        nBL2 = np.zeros(self.Nk, dtype=complex)
        eta2 = np.swapaxes(eta, 0, 1)
        phi2 = np.zeros_like(eta)
        l02 = np.zeros_like(eta)
        for k, p in itertools.product(range(self.Nk), range(self.Nk)):
            phi2[k,p] = phi[k-p]
            l02[k,p] = l0[k-p]
        toDFT = contract('p,k,kp->kp', Y, Y, eta2-phi2+l02+T4)
        DFT = fft(ifft(toDFT,norm='forward',axis=0),norm='backward',axis=1)
        nBL2 = np.diag(DFT)
        self.check_real(t_index, nBL2, 'nBL(r_n)')
        self.observables['LP']['nAL'][t_index] = nAL.real
        self.observables['LP']['nBL'][t_index] = nBL2.real
        self.observables['LP']['nCL'][t_index] = nCL
        self.observables['LP']['nLP'][t_index] = nAL.real+nCL+nBL2.real

    def export_data(self, fp=None):
        if fp is None:
            fp = self.gen_fp()
        if not os.path.exists(os.path.dirname(fp)):
            os.makedirs(os.path.dirname(fp))
        with open(fp, 'wb') as fb:
            pickle.dump(self.observables, fb)
        logger.info(f'Wrote parameters & dynamics data to {fp}')

    def import_data(self, fp=None):
        if fp is None:
            fp = self.generate_fp()
        with open(fp, 'rb') as fb:
            self.observables = pickle.load(fb)
        logger.info('Loaded parameters and dynamics data from {fp}')

    def gen_fp(self):
        fname = 'Nnu{Nnu}/Nk{Nk}/gn{gSqrtN}S{S}Gamz{Gam_z}.pkl'.format(**self.params)
        return os.path.join(self.DEFAULT_DIRS['data'], fname)


    def get_labels(self):
        return {'K': r'\(K\)',
                't': r'\(t\)',
                't_fs': r'\(t\) \rm{(fs)}',
                'rn': r'\(r_n\) \rm{(}\(\mu\)\rm{m)}',
                'ph_rn': r'\(n_{\rm{\text{ph}}}(t, r_n)\)',
                'ph_rn0': r'\(n_{\rm{\text{ph}}}(t, r_n)-n_{\rm{\text{ph}}}(0, r_n) \)',
                'ph_rms': r'\(\sqrt{\text{\rm{MSD}}[n_{\text{ph}}]}\) \rm{(}\(\mu\)\rm{m}\({}^2\)\rm{)}',
                'mol_rn': r'\(n_{M}(t, r_n)\)',
                'mol_rn0': r'\(n_{M}(t, r_n)-n_M(0,r_n)\)',
                'mol_rms': r'\(\sqrt{\text{\rm{MSD}}[n_{M}]}\) \rm{(}\(\mu\)\rm{m}\({}^2\)\rm{)}',
                'ph0nM0': r'\(\Delta n_{\rm{\text{ph}}}+ \Delta n_{M}\)',
                'coh': r'\(\lvert\langle a \sigma^+\rangle\rvert(t,r_n)\)',
                'nB': r'\(n_{\mathcal{B}}(t,r_n)\)',
                'nB0': r'\(n_{\mathcal{B}}(t,r_n)-n_{\mathcal{B}}(0,r_n)\)',
                'nD': r'\(n_{\mathcal{D}}(t,r_n)\)',
                'nD0': r'\(n_{\mathcal{D}}(t,r_n)-n_{\mathcal{B}}(0,r_n)\)',
                'Dph': r'\(\Delta n_{\rm{\text{ph}}}\)',
                'DnM': r'\(\Delta n_{M}\)',
                'DnB': r'\(\Delta n_{\mathcal{B}}\)',
                'DnD': r'\(\Delta n_{\mathcal{D}}\)',
                'D': r'\(\Delta n_X(t) = \sum_{n} \left(n_X(t, r_n) - n_X(0, r_n)\right)\)',
                }
    def plot_all(self):
        fig, axes = plt.subplots(5,2,figsize=(8,18))
        fig.suptitle(r'\texttt{{c2v2 {}}}'.format(datetime.now().strftime('%Y-%m-%d %H:%M')), y=0.925) 
        plt.subplots_adjust(wspace=0.25, hspace=0.35)
        params = self.params
        # PANEL A - initial profile with dispersion inset
        n1, pex1, n2, pex2 = self.plot_initial_profile(data_only=True)
        k1, w1, k2, w2 = self.plot_dispersion(data_only=True)
        axes[0,0].set_title(r'\(p^\uparrow_n(0, r_n)\)', y=1.0)
        axes[0,0].plot(self.params['delta_r'] * n1, pex1, ls='--')
        axes[0,0].scatter(self.params['delta_r'] * n2, pex2, marker='.', c='r', s=75, zorder=2)
        axes[0,0].xaxis.set_major_locator(MaxNLocator(integer=True))
        L = self.params['L']
        axes[0,0].set_xlim([None,L*1.0425])
        axes[0,0].set_xticks([0,L/4,L/2,3*L/4,L])
        axes[0,0].set_xticklabels(['\(0\)','\(L/4\)','\(L/2\)','\(3L/4\)','\(L\)'])
        ax1in = axes[0,0].inset_axes([0.7,0.7,0.25,0.25])
        ax1in.plot(k1, w1, c='orange')
        ax1in.scatter(k2, w2, c='k', s=5, zorder=2)
        #ax1in.set_yticks([L for L in ax1in.get_yticks()])
        #ax1in.set_yticklabels([r'\({:.2f}\)'.format(L) for L in ax1in.get_yticks()])
        # PANEL B - parameters
        #relevant_parameters = \
        size_params = \
            [#r'\rm{Size}',
             r'\(N_m=10^{{{:.0f}}}\qquad L={:.0f}\mu\text{{m}}\)'.format(np.log10(self.Nm), L),
             r'\(N_k={}\)\qquad\(N_\nu={}\)'.format(self.Nk, self.Nnu),
             #r'\(N_\nu={}\)'.format(self.Nnu),
            ]
        sys_params = \
            [
             #'\n',
             r'\rm{System (eV)}',
             r'\(\omega_c={}\), \(\epsilon={}\)'.format(params['omega_c'], params['epsilon']),
             #r'\(\omega_c={}\)'.format(params['omega_c']),
             #r'\(\epsilon={}\)'.format(params['epsilon']),
             r'\(n_r={}\)'.format(params['nr']),
             r'\(g\sqrt{{N_m}}={:.3g}\)'.format(params['gSqrtN']),
            ]
        rate_params = \
            [
             #'\n',
             r'\rm{Rates (eV)}',
             r'\(\kappa={}\)'.format(params['kappa_c']),
             r'\(\Gamma_\uparrow={}\)'.format(params['Gam_up']),
             r'\(\Gamma_\downarrow={}\)'.format(params['Gam_down']),
             r'\(\Gamma_z={}\)'.format(params['Gam_z']),
            ]
        def pow_str(flo, prec=1):
            flo = float(flo)
            r0a = '{:.{prec}e}'.format(flo, prec=prec).split('e')
            r0a[1] = r0a[1].replace('0','')
            return r'\({}\!\times\!10^{{{}}}\)'.format(*r0a)
        bath_params = \
            [
             #'\n',
             r'\rm{Bath (eV)}',
             r'\(S={}\)'.format(params['S']),
             r'\(\omega_\nu={}\)'.format(params['omega_nu']),
             r'\(T={}\)'.format(params['T']),
             #r'\(\gamma_\nu (\gamma_\uparrow, \gamma_\downarrow)={}\ ({:.1e}, {:.1e})\)'.format(
             #    params['gam_nu'], self.rates['gam_up'], self.rates['gam_down']),
             r'\(\gamma_\nu={}\)'.format(params['gam_nu']),
             r'\(\gamma_\uparrow= \) {}'.format(pow_str(self.rates['gam_up'])),
             r'\(\gamma_\downarrow= \) {}'.format(pow_str(self.rates['gam_down'])),
             ]
        numeric_params = \
            [
             #'\n',
             r'\rm{Computation}',
             r'\rm{{atol}} \(=\) {}'.format(pow_str(params['atol'], prec=0)),
             r'\rm{{rtol}} \(=\) {}'.format(pow_str(params['rtol'], prec=0)),
             #r'\rm{{runtime}} \(={:.1f}\)s'.format(self.compute_time),
             r'\rm{{runtime:}} {}'.format(timedelta(seconds=round(self.compute_time))),
             # Scaling string
             ]
        axes[0,1].get_xaxis().set_visible(False)
        axes[0,1].get_yaxis().set_visible(False)
        axes[0,1].text(0.5, 0.975, '\n'.join(size_params),
                       ha='center', va='top', transform=axes[0,1].transAxes, size='small') # axis coords
        axes[0,1].text(0.25, 0.8, '\n'.join(sys_params),
                       ha='center', va='top', transform=axes[0,1].transAxes, size='small') # axis coords
        axes[0,1].text(0.75, 0.8, '\n'.join(rate_params),
                       ha='center', va='top', transform=axes[0,1].transAxes, size='small') # axis coords
        axes[0,1].text(0.25, 0.5, '\n'.join(bath_params),
                       ha='center', va='top', transform=axes[0,1].transAxes, size='small') # axis coords
        axes[0,1].text(0.75, 0.375, '\n'.join(numeric_params),
                       ha='center', va='top', transform=axes[0,1].transAxes, size='small') # axis coords
        # PANEL C and D - PHOTON and EXCITON DYNAMICS
        plot_t = self.observables['t_fs']
        axes[1,0].set_title(r'\(\sum_k  \langle a^\dagger_k a^{\vphantom{\dagger}}_k\rangle\)', y=1.0)
        kap = self.params['kappa_c']
        if np.isclose(kap, 0.0):
            kap_str = r'\(\kappa\sim\infty\)'
        else:
            kap_str = r'\(1/\kappa\sim{:.0f}\) \rm{{fs}}'.format(round((1/self.params['kappa_c'])*self.EV_TO_FS,-2))
        axes[1,0].annotate(\
                kap_str,
                #+'\n'\
                #+r'\(1/\Gamma_{{\downarrow}}\sim{:.0e}\) \rm{{fs}}'.format(round((1/self.params['Gam_down'])*self.EV_TO_FS, -1)),
                           #xy=(0.5,.8),
                           xy=(.625,.875),
                           xycoords='axes fraction',
                           xytext=(0,0),
                           textcoords='offset points',
                           size='small',
                           bbox=dict(boxstyle='Square', fc='white', ec='k'))
        axes[1,0].set_xlabel(self.labels['t_fs'], labelpad=0)
        axes[1,1].set_xlabel(self.labels['t_fs'], labelpad=0)
        axes[1,0].plot(plot_t, np.sum(self.observables['ph_dic']['vals'], axis=1))
        #axes[1,0].plot(plot_t, np.sum(self.observables['n'], axis=1))
        #print(np.allclose(np.sum(self.observables['n'], axis=1), np.sum(self.observables['ph_dic']['vals'], axis=1)))
        nPh, nM = self.observables['ph_dic']['vals'], self.observables['mol_dic']['vals'] 
        nB, nD = self.observables['nB'], self.observables['nD']
        nPh_tots = np.sum(nPh, axis=-1)
        nM_tots = np.sum(nM, axis=-1)
        nB_tots = np.sum(nB, axis=-1)
        nD_tots = np.sum(nD, axis=-1)
        offset=True
        nB_str = r'\(n_{{\mathcal{{B}}}}(0) \sim {:.0g}\)'.format(nB_tots[0])
        nD_str = r'\(n_{{\mathcal{{D}}}}(0) \sim {:.0f}\)'.format(nD_tots[0])
        if offset:
            nPh_plt = nPh_tots - nPh_tots[0]
            nM_plt = nM_tots - nM_tots[0]
            nB_plt = nB_tots - nB_tots[0]
            nD_plt = nD_tots - nD_tots[0]
        else:
            nPh_plt = nPh_tots
            nM_plt = nM_tots
            nB_plt = nB_tots
            nD_plt = nD_tots
        axes[1,1].plot(nPh_plt, label=self.labels['Dph'])
        axes[1,1].plot(nM_plt, label=self.labels['DnM'], ls='-')
        axes[1,1].plot(nB_plt, label=self.labels['DnB'], ls='--')
        axes[1,1].plot(nD_plt, label=self.labels['DnD'])
        axes[1,1].plot(nPh_plt+nM_plt, label=self.labels['ph0nM0'], c='k')
        #axes[1,1].plot(nPh_tots+nM_tots, label=self.labels['ph0nM0'], c='k',ls='--')
        axes[1,1].set_title(self.labels['D'])
        axes[1,1].legend(loc='upper left')
        axes[1,1].annotate(nB_str + '\n' + nD_str,
                           xy=(.665,.825),
                           xycoords='axes fraction',
                           xytext=(0,0),
                           textcoords='offset points',
                           size='small',
                           bbox=dict(boxstyle='Square', fc='white', ec='k'))
        # PANEL E and F + I - DENSITIES
        #plot_x, plot_t, ph_dic, mol_dic = self.plot_densities(t, y, data_only=True)
        plot_x = self.rs
        thres = -1e-3
        mol_vals_masked = np.ma.masked_less(nM_tots, thres)
        mol_title_2 = self.labels['mol_rn0'] + r'\geq{:.0e}\)'.format(thres)
        axes[2,0].set_ylabel(self.labels['t'], rotation=0, labelpad=20)
        axes[2,0].set_xlabel(self.labels['rn'])
        axes[2,1].set_xlabel(self.labels['rn'])
        axes[4,0].set_xlabel(self.labels['rn'])
        axes[2,0].set_title(self.labels['ph_rn'])
        axes[2,1].set_title(self.labels['mol_rn0'])
        axes[4,0].set_title(self.labels['nB0'])
        extent = [plot_x[0], self.params['L'], plot_t[0], plot_t[-1]]
        cm = colormaps['coolwarm'] 
        my_im = lambda axis, vals: axis.imshow(vals, origin='lower', aspect='auto',
                                           interpolation='none', extent=extent, cmap=cm)
        im0 = my_im(axes[2,0], nPh)
        im1 = my_im(axes[2,1], nM-nM[0])
        nB_centered = nB - self.observables['nB'][0] # subtract t=0 row from each t>0 row
        im2 = my_im(axes[4,0], nB_centered)
        cbar0 = fig.colorbar(im0, ax=axes[2,0], aspect=20)
        cbar1 = fig.colorbar(im1, ax=axes[2,1], aspect=20)
        cbar2 = fig.colorbar(im2, ax=axes[4,0], aspect=20)
        axes[2,0].axvline(L/2, c='k')
        axes[2,1].axvline(L/2, c='k')
        axes[4,0].axvline(L/2, c='k')
        # PANEL G & H + I - MSD
        mol_dic, ph_dic, coh_dic = self.observables['mol_dic'], self.observables['ph_dic'], self.observables['coh_dic']
        avgP, errorP, msdP = ph_dic['mean'], np.sqrt(ph_dic['var']), ph_dic['msd']
        avgM, errorM, msdM = mol_dic['mean'], np.sqrt(mol_dic['var']), mol_dic['msd']
        avgC, errorC, msdC = coh_dic['mean'], np.sqrt(coh_dic['var']), coh_dic['msd']
        #valid_i = next((i for i in range(len(msdP)) if not np.isnan(msdP[i])), None)
        #valid_iC = next((i for i in range(len(msdC)) if not np.isnan(msdC[i])), None)
        valid_i = 1 # ignore first datum
        valid_iC = 1
        plot_tP = plot_t[valid_i:]
        msdP = msdP[valid_i:]
        msdC = msdC[valid_iC:]
        plot_tC = plot_t[valid_iC:]
        axes[3,0].plot(plot_tP, np.sqrt(msdP))
        axes[3,0].set_title(self.labels['ph_rms'])
        axes[3,0].set_xlabel(self.labels['t_fs'])
        axes[4,1].set_xlabel(self.labels['t_fs'])
        fs = 1/(plot_t[1]-plot_t[0])
        cutoffFS = self.params['lowpass'] * fs
        logger.info('Lowpass at {} * {:.2g} = {:.2g} fs^-1'.format(self.params['lowpass'], fs, cutoffFS))
        msdP = self.butter_lowpass_filter(msdP, cutoffFS, fs)
        msdC = self.butter_lowpass_filter(msdC, cutoffFS, fs)
        to_plot = np.sqrt(msdP)
        to_plotC = np.sqrt(msdC)
        axes[2,0].plot(self.params['L']/2-to_plot, plot_tC, c=self.COLORS[1])
        def f(t, D, po):
            return D * t**po + to_plot[0]
        early_t = 50
        early_i = next((j for j, t in enumerate(plot_tP) if t > early_t / self.EV_TO_FS), None)
        axes[3,0].plot(plot_tP, to_plot, 
                label=r'\rm{{Lowpass (}}\({:.1g} \text{{\rm{{fs}}}}^{{-1}}\)\rm{{)}}'.format(cutoffFS))
        if len(plot_tP) > 10: # need reasonable number of points
            popt, pcov = curve_fit(f, plot_tP[:early_i], to_plot[:early_i], bounds=([0,1],[np.inf, np.inf]))
            fit_data = f(plot_tP, *popt)
            axes[3,0].plot(plot_tP, fit_data,
                           ls='--',
                           label=r'\(D t^{{\alpha}}\), \(D={:.2g}\), \(\alpha={:.2g}\)'.format(*popt) +\
                                   '\n' +\
                                   r'\rm{{fit to (0,{})}}'.format(early_t)
                           )
            if fit_data[-1] > 1.5 * to_plot[-1]:
                axes[3,0].set_ylim([None, 1.5 * to_plot[-1]])
        else:
            logger.warning('Too few data points to generate fit')
        axes[3,0].legend()
        # VELOCITY FIT PANEL
        self.velocity_plot(ax=axes[3,1], rmsd=True)
        self.velocity_wavefront_plot(ax=axes[4,1], ph_ax=axes[2,0], mol_ax=axes[2,1], msd_ax=axes[3,0])
        fp = os.path.join(self.DEFAULT_DIRS['figures'],
                          f'Nnu{self.Nnu}Nk{self.Nk}S{self.params["S"]}Gamz{self.params["Gam_z"]}.png')
        #fig.savefig(fp, bbox_inches='tight')
        fp2 = os.path.join(self.DEFAULT_DIRS['figures'], 'last.png')
        fig.savefig(fp2, bbox_inches='tight')
        logger.info(f'Combined plot saved to {fp2}')

    def velocity_wavefront_plot(self, ax=None, ph_ax=None, mol_ax=None, msd_ax=None):
        P_CUT_PER = 4
        P_CUT = round(P_CUT_PER/100, 6)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,4))
            savefig = True
        else:
            savefig = False
        dph = self.observables['ph_dic']['vals'][1:]
        dmol = self.observables['mol_dic']['vals'][1:]
        ts = self.observables['t_fs'][1:]
        xphs = []
        xmols = []
        dr = self.params['delta_r']
        for i, t in enumerate(ts):
            ph_vals = dph[i]
            cph_vals = np.cumsum(ph_vals)
            cph_vals /= cph_vals[-1] # normalise by total
            pi_ph = next((j for j, p in enumerate(cph_vals) if p >= P_CUT))
            xphs.append(dr * pi_ph)
            mol_vals = dmol[i]
            cmol_vals = np.cumsum(mol_vals)
            cmol_vals /= cmol_vals[-1] # normalise by total
            pi_mol = next((j for j, p in enumerate(cmol_vals) if p >= P_CUT))
            xmols.append(dr * pi_mol)
        dt = ts[1]-ts[0]
        #cutoffFS = 0.01 * self.params['lowpass'] * (1/dt)
        cutoffFS = 0.1*self.params['lowpass'] * (1/dt)
        xphs_filt = self.butter_lowpass_filter(xphs, cutoffFS, dt)
        xmols_filt = self.butter_lowpass_filter(xmols, cutoffFS, dt)
        if ph_ax is not None:
            #ph_ax.plot(xphs, ts, c='lime')
            #ph_ax.plot(xphs_filt, ts, c='lime')
            ph_ax.plot(xphs_filt, ts, c=self.COLORS[3])
        if mol_ax is not None:
            mol_ax.plot(xmols, ts, c='lime')      
        if msd_ax is not None:
            msd_line = msd_ax.lines[0]
            first_val = msd_line.get_ydata()[0]
            ph_centered = np.array(xphs_filt) - self.params['L']
            abs_ph_centered = np.abs(ph_centered)
            abs_ph_centered -= abs_ph_centered[0]
            abs_ph_centered += first_val
            msd_ax.plot(ts, abs_ph_centered, label=r'\rm{Wavefront (offset)}')
            msd_ax.legend()
        dxphs = np.diff(xphs_filt)
        dxmols = np.diff(xmols_filt)
        cfac = 1e-6 * 1e15 / constants.c # express as fraction of c
        vphs = np.abs(dxphs/dt)
        vmols = np.abs(dxmols/dt)
        logger.info('Max | mean Ph wavefront velocity {:.1g} | {:.1g} mu.m/fs'.format(np.max(vphs), np.mean(vphs)))
        vphs *= cfac
        vmols *= cfac
        ax.set_ylabel(r'\(v/c\)', rotation=0, labelpad=10)
        ax.set_xlabel(r'\(t\) \rm{(fs)}')
        ax.set_title(r'\rm{{Wavefront speed (}}\(P_{{\text{{\rm{{cut}}}}}}={}\)\rm{{\%)}}'.format(P_CUT_PER))
        ax.plot(ts[1:], vphs, label=r'\(n_{\text{ph}}\)')
        ax.plot(ts[1:], vmols, label=r'\(n_{\text{M}}\)')
        ax.legend()
        if savefig:
            fig.savefig('figures/velocity_wavefront.png', bbox_inches='tight')
            plt.close(fig)


    def velocity_plot(self, ax=None, adak_cutoff=0.1, rmsd=False, dominant_rmsd=True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,4))
            savefig = True
        else:
            savefig = False
        Ks = self.Ks[self.Q0:] # non-negative Ks only
        cgs = self.cg(Ks)
        ax.set_ylabel(r'\(v/c\)', rotation=0, labelpad=10)
        ax.set_xlabel(r'\(K\)')
        #ax.set_title(r'\(c_g(K)\) vs. velocity fit')
        #ax.set_title(r'Photon velocities') # Velocities fit to MSD
        # get dominant wavector
        chosen_time_fs = 50
        msd_str = r'\(\sqrt{\text{MSD}[n_{\text{ph}}]}\)' if rmsd else r'\(\text{MSD}[n_{\text{ph}}]\)'
        ax.set_title(r'\rm{{Velocities at}} \(t={}\)\rm{{fs from}} {}'.format(chosen_time_fs, msd_str))
        chosen_time = chosen_time_fs / self.EV_TO_FS
        chosen_i = next((i for i, t in enumerate(self.observables['select_data']['select_t']) if t >= chosen_time_fs), None)
        cgs_c = cgs/constants.c
        if chosen_i is None:
            logger.warning(f'<a^dag_K\' a_K> not recorded at {chosen_time_fs} fs')
            dominant_i = 0
        else:
            ada = self.observables['select_data']['adak'][chosen_i]
            ys = fftshift(np.real(np.diag(ada)))
            ys = ys[self.Q0:]
            rmsd = np.sqrt(np.average(Ks**2, weights=ys))
            dominant_i = int(rmsd)
            ys *= np.max(cgs_c)/np.max(ys)
            ys = self.butter_lowpass_filter(ys, adak_cutoff, 1)
            order = 8
            if not dominant_rmsd:
                dominant_i = None
                while order > 0:
                    i_maxes = argrelextrema(ys, np.greater, order=order)[-1]
                    if len(i_maxes) > 0:
                        dominant_i = i_maxes[-1]
                        break
                    order -= 1
                if dominant_i is None:
                    dominant_i = np.argmax(ys)
            ax.plot(Ks, ys, alpha=0.6,
                    label=r'\(\langle a^\dagger_K a_K\rangle\) \rm{(filtered)}'
                    #label=r'\(\langle a^\dagger_K a_K\rangle\) '\
                    #        r'\rm{{(}}\(t={}\)\rm{{fs)}}'.format(chosen_time_fs)
                    )
            ax.axvline(dominant_i, alpha=0.6, c='m')
        ax.plot(Ks, cgs_c, label=r'\(c_g\)')
        # make three fits
        #fit_times = [50, 75, 100]
        fit_times = [50]
        msdP = self.observables['ph_dic']['msd'][1:] # ignore first data point where no photons
        ts_fs = self.observables['t_fs'][1:]
        fs = 1/(ts_fs[1]-ts_fs[0])
        cutoffFS = self.params['lowpass'] * fs
        msdP = self.butter_lowpass_filter(msdP, cutoffFS, fs)
        if rmsd:
            msdP = np.sqrt(msdP)
        markers = ['o', '^', 's', '*', 'P', 'D']
        def f(t, D, po):
            return D * t**po + msdP[0]
        for mi, fit_t in enumerate(fit_times):
            ei = next((i for i, t in enumerate(ts_fs) if t >= fit_t), -1)
            # fit made in fs
            popt, pcov = curve_fit(f, ts_fs[:ei], msdP[:ei], bounds=([0,1],[np.inf, np.inf]))
            perr = np.sqrt(np.diag(pcov))
            v = self.velocity(chosen_time_fs, popt[0], popt[1]) # this is in micro meters per fs
            # (micro meters because of how moments were calculated - see plot_densities)
            v *= 1e-6 * 1e15 # convert to ms^-1
            v /= constants.c # then divide by c to get as ratio of soeed f light
            logger.debug('Fit to (0,{}): D, alpha [err] {:.2g}, {:.2g} [{:.1g}, {:.1g}],'\
                    'v/c = {:.2g}'.format(fit_t, *popt, *perr, v))
            ax.scatter(dominant_i, v, marker=markers[mi%len(markers)], 
                       label=r'\rm{{fit to }}\((0,{})\)'.format(fit_t))
        # plot with different markers
        ax.legend()
        if savefig:
            fig.savefig('figures/velocity.png', bbox_inches='tight')
            plt.close(fig)

    @classmethod
    def butter_lowpass_filter(cls, data, cutoff, fs, order=5, axis=-1):
        data = np.array(data)
        b, a = butter(order, cutoff, fs=fs, btype='low')
        padlen = 3 * max(len(a), len(b)) # default filtfilt padlength
        if padlen >= data.shape[-1] - 1:#but must be less than data.shape[-1]-1
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html#scipy.signal.filtfilt
            padlen = max(0, data.shape[-1]-2) 
        return filtfilt(b, a, data, axis=axis, padlen=padlen)


    def plot_initial_profile(self, data_only=False):
        sigma = self.params['sigma_0'] # fraction of L (i.e. Nk) 
        all_ns = np.linspace(0, self.Nk, num=300)
        scaled_sigma = sigma * self.Nk
        mid_index = self.Nk//2
        if sigma == 0:
            all_pex = self.params['pex'] * np.ones(len(all_ns))
            select_pex = self.params['pex'] * np.ones(self.Nk)
        else:
            all_pex = self.params['pex'] * np.exp(- (all_ns-mid_index)**2/(2*scaled_sigma**2))
            select_pex = self.params['pex'] * np.exp(-(self.ns-mid_index)**2/(2*scaled_sigma**2))
        if data_only:
            return all_ns, all_pex, self.ns, select_pex
        fig, ax = plt.subplots(figsize=(6,4))
        ax.set_xlabel(r'\(n\)')
        ax.set_ylabel(r'\(p^\uparrow\)', rotation=0, labelpad=15)
        ax.plot(all_ns, all_pex)
        ax.scatter(self.ns, select_pex, c='r', marker='.')
        ax.set_title(r'\(\sigma_0={}\) \(N_k={}\)'.format(sigma, self.Nk), y=1.02)
        fp = os.path.join(self.DEFAULT_DIRS['figures'], 'initial_profile.png')
        fig.savefig(fp, bbox_inches='tight')
        logger.info(f'Initial molecular excitation plot saved to {fp}.')
        return all_ns, all_pex, self.ns, select_pex

    def plot_dispersion(self, data_only=False):
        fig, ax = plt.subplots(figsize=(6,4))
        ax.set_xlabel(r'\(K\)', x=0.47)
        ax.set_ylabel(r'\hspace*{-.3cm}\(\hbar\omega\)\\[2pt]\rm{(eV)}', rotation=0, labelpad=23)
        all_Ks = np.linspace(-2*np.abs(self.Ks[0]), 2*np.abs(self.Ks[-1])+1, 250)
        all_y = self.omega(all_Ks)
        chosen_y = self.omega(self.Ks)
        if data_only:
            return all_Ks, all_y, self.Ks, chosen_y
        ax.plot(all_Ks, all_y)
        ax.scatter(self.Ks, chosen_y, c='r', s=10, zorder=2)
        fp = os.path.join(self.DEFAULT_DIRS['figures'], 'omega.png')
        fig.savefig(fp, bbox_inches='tight')
        logger.info(f'Dispersion plot saved to {fp}.')
        plt.close(fig)
        return all_Ks, all_y, self.Ks, chosen_y

    def plot_hopfield(self):
        xtype='K' # or 'ek'
        fig, axes = plt.subplots(1,2,figsize=(10,4))
        fig2, ax2 = plt.subplots()
        if xtype=='ek':
            axes[0].set_xlabel(r'\(\hbar k \tilde{c}\) \rm{(eV)}')
            axes[1].set_xlabel(r'\(\hbar k \tilde{c}\) \rm{(eV)}')
        else:
            axes[0].set_xlabel(r'\(K\)')
            axes[1].set_xlabel(r'\(K\)')
            ax2.set_xlabel(r'\(k\)')
        FAC = 1.5
        all_Ks = np.linspace(-FAC*np.abs(self.Ks[0]), FAC*np.abs(self.Ks[-1]), 250)
        eks = self.K_to_eV * all_Ks
        if xtype=='ek':
            xvals = eks
        else:
            xvals = all_Ks
        omegas = self.omega(all_Ks)
        omega_c = self.params['omega_c']
        ep_omegas = self.params['epsilon'] - omegas
        xis = 0.5 * np.sqrt(ep_omegas**2 + self.params['gSqrtN']**2)
        Xks = np.sqrt(0.5 + 0.5**2 * ep_omegas/xis)
        Yks = np.sqrt(0.5 - 0.5**2 * ep_omegas/xis)
        assert np.allclose(Xks**2+Yks**2,1.0)
        epUs = 0.5 * (self.params['epsilon'] + omegas) + xis
        epLs = 0.5 * (self.params['epsilon'] + omegas) - xis
        dKdk = self.params['L']/(2*np.pi)
        dK = all_Ks[1] - all_Ks[0]
        xvals2 = xvals / dKdk
        dk = xvals2[1] - xvals2[0]
        mum_ev_s_to_mum_fs = 1e-15 * (constants.e/constants.hbar)
        omega_diffs = mum_ev_s_to_mum_fs  * np.gradient(omegas, dk)
        cgs = self.cg(all_Ks) * 1e6 * 1e-15
        epU_diffs =  mum_ev_s_to_mum_fs * np.gradient(epUs, dk)
        epL_diffs =  mum_ev_s_to_mum_fs * np.gradient(epLs, dk)
        lw=2.5
        ax2.plot(xvals2, epL_diffs, label=r'\(\partial_K \epsilon^L_k\)', lw=lw)
        ax2.plot(xvals2, epU_diffs, label=r'\(\partial_K \epsilon^U_k\)', lw=lw)
        ax2.plot(xvals2, cgs, label=r'\(c_g\)', lw=lw)
        ax2.plot(xvals2, omega_diffs, label=r'\(\partial_K \omega_k\)', ls=':', lw=lw)
        ax2.legend()
        axes[0].plot(xvals, np.abs(Xks), label=r'\(|X_k|\)', lw=lw)
        axes[0].plot(xvals, np.abs(Yks), label=r'\(|Y_k|\)', lw=lw)
        #axes[0].plot(xvals, np.abs(Xks/Yks)**2, label=r'\(|X_k|^2/|Y_k|^2\)', lw=lw)
        axes[0].legend()
        axes[1].plot(xvals, epLs-omega_c, label=r'\(\epsilon^L_k\)', lw=lw)
        axes[1].plot(xvals, epUs-omega_c, label=r'\(\epsilon^U_k\)', lw=lw)
        axes[1].plot(xvals, omegas-omega_c, label=r'\(\omega_k\)', ls=':', lw=lw)
        #axes[1].annotate('', xy=(eks[0], np.min(epLs)),xytext=(eks[-1],np.min(epLs)),
        #    arrowprops=dict(arrowstyle='<->',color='k', lw=2),
        #    annotation_clip=False) 
        if xtype=='eks':
            axes[0].set_xlim([-1.5,1.5])
            axes[1].set_xlim([-1.5,1.5])
            axes[0].set_xlim([0,2.0])
            axes[1].set_xlim([0,2.0])
        else:
            axes[0].set_xlim([0,100])
            axes[1].set_xlim([0,100])
            ax2.set_xlim([0,xvals2[-1]/FAC])
            ax2.set_ylim([0,0.35])
        axes[1].set_ylim([None,.5])
        axes[1].legend(title=r'\(k_{{\text{{cut}}}} \sim {:.0f}\) \rm{{eV}}'.format(eks[-1]/FAC))
        #axes[1].set_title(r'\(\omega_k-\omega_c\) \rm{eV}')
        axes[1].set_ylabel(r'\(+\omega_c\)')
        fig.savefig('figures/hopfield.png', bbox_inches='tight')
        fig2.savefig('figures/group_velocity.png', bbox_inches='tight')
        logger.info(f'Hopfield and energy plots saved to figures/hopfield.png')
        plt.close(fig)

    def plot_dynamics(self, t, y, save_comparison=False):
        fig, axes = plt.subplots(2, sharex=True, figsize=(6,6))
        axes[0].set_title(r'\(N_m={:.0e}\) \(N_k={}\)'.format(self.Nm, self.Nk), y=1.02)
        axes[0].set_ylabel(r'\(\sum_k n^k\)', rotation=0, labelpad=25)
        axes[1].set_ylabel(r'\(\langle \sigma_+ \sigma_- \rangle\)', rotation=0, labelpad=25)
        #axes[1].set_ylabel(r'\(\langle (\sigma_+ \sigma_-) \rangle_{L/2}\)', rotation=0, labelpad=60)
        axes[1].set_xlabel('\(t\)')
        label=r'\rm{C2}'
        ntots, p1s = self.calculate_dynamics_expectations(y)
        axes[0].plot(t, ntots, label=label)
        #pos=[int(self.Nk//4), int(self.Nk//2), 3*int(self.Nk//4)]
        pos_plus_endpoints = np.linspace(0, self.Nk, num=7, dtype=int)
        pos = pos_plus_endpoints[2:-2]
        for p in pos:
            label = r'\({}\)'.format(p)
            axes[1].plot(t, p1s[:,p], label=label) # nth column is for nth molecule
        #axes[1].legend(title=r'\rm{Position}')
        axes[1].legend(title=r'\(n\) \rm{(ensemble)}')
        #print('Min p1 = {:.2e}'.format(np.min(p1s[:,self.Nk//2])))
        fp = os.path.join(self.DEFAULT_DIRS['figures'], 'dynamics.png')
        fig.savefig(fp, bbox_inches='tight')
        logger.info(f'Dynamics plot saved to {fp}.')
        if save_comparison:
            data_fp = os.path.join(self.DEFAULT_DIRS['data'], 'v2.pkl')
            with open(data_fp, 'wb') as fb:
                pickle.dump({'t':t, 'n':ntots, 'p1': p1s[:,self.Nk//2], 'name':'v2', 'params':self.params}, fb)
        return ntots, p1s

    def report_energies(self, tf_fs):
        tf = tf_fs/self.EV_TO_FS
        Nk = self.Nk
        select_y = self.omega(self.Ks)
        omega_max = select_y[-1]
        #print( self.K_factor * self.Q0**2 )
        #check_omega_max =  self.params['omega_c']*np.sqrt(1 + self.K_factor * self.Q0**2 )
        #print(np.isclose(omega_max, check_omega_max))
        kmax = self.ks[-1]
        dk = 0
        dw = 0
        if Nk > 1:
            dk = kmax - self.ks[-2]
            dw = omega_max - select_y[-2]
        omega_max -= self.params['omega_c']
        gn = self.params['gSqrtN']
        mph = self.params['mph']
        nr = self.params['nr']
        delta = self.params['delta']
        dr = self.params['delta_r']
        NE = self.NE
        L =  self.params['L']
        print(f'Nk={Nk} modes spacing (inverse microns) dk={dk:.2g}, kmax={kmax:.2g} -> dw={dw:.2g},'
        f'w_max-w_c={omega_max:.2g}')
        print(f'Compare gSqrtN={gn:.2g}, Delta={delta:.2g}, tf={tf:.2g}')
        print(f'Refractive index nr={nr:.2g} gives mph={mph:.2g}')
        print(f'System L={L:.2g} with NE={NE:.0f} per ensemble (Delta r={dr:.2g})')

if __name__ == '__main__':
    logging.basicConfig(
        format='%(filename)s L%(lineno)s %(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%H:%M')
    #params = HTC.default_params() # copy HTC.DEFAULT_PARAMS 
    NE = 5 # molecules per ensemble
    Nk = 31  # number of ensembles = number of photon modes, see red curves in Fig. 6.12 of thesis for memory usage (Nk ~ 301, Nnu=4 about 15-20Gb?
    dt_fs = 0.33 # timestep length in FS
    dt = dt_fs/HTC.EV_TO_FS # timescale length in 'electronvolts' units currently used by code
    params = {
            'Q0': Nk//2, # mode cut-off (integer) -> Nk = 2*Q0+1 modes total 
            'Nm': Nk * NE, # Number of molecules TOTAL
            'Nnu':2, # Number of vibrational levels for each molecule
            'L': 60.0, # system length, inverse micro meters
            'nr':1.0, # refractive index, sets effective speed of light c/nr
            'omega_c':1.94, # omega_0 = 1.94eV, minimum of cavity dispersion
            'epsilon':2.14, # exciton energy, detuning omega_0-epsilon (0.2eV for model I in Xu et al. 2023)
            'gSqrtN':0.15, # light-matter coupling
            'kappa_c':3e-3, # photon loss
            'Gam_z':0.0, # molecular pure dephasing
            'Gam_up':0.0, # molecular pumping
            'Gam_down':1e-7, # molecular loss
            'S':7.11, # Huang-Rhys parameter
            'omega_nu':0.00647, # vibrational energy spacing
            'T':0.026, # k_B T in eV (.0259=300K, .026=302K)
            'gam_nu':0.01, # vibrational damping rate
            'initial_state': 'incoherent', # see self.incoherent_state()
            'pex':0.01, # mean initial molecular population (for initial_state 'incoherent')
            'sigma_0':0.1, # s.d. of initial incoherent population as FRACTION OF L
            'atol':1e-9, # solver tolerance
            'rtol':1e-6, # solver tolerance
            'dt': dt, # determines interval at which solution is evaluated. Does not effect the accuracy of solution, only the grid at which observables are recorded (if solver e.g. makes a step of 2*dt, interpolation is used to calculate inner points. 
            }
    htc = HTC(params)
    htc.plot_dispersion()
    htc.plot_hopfield()
    tf_fs = 100 # simulation time in fs
    htc.evolve(tf_fs=tf_fs)
    results = htc.observables # contains values of observables at each step
    htc.plot_all() # plotter functions used for transport problem
    plt.close('all') # cleanup any open figures
    #htc.export_data() # export observables and parameters as dict to .pkl file

