#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.ticker import MaxNLocator
from scipy.signal import butter, filtfilt, lfilter, freqz, argrelextrema
from scipy.optimize import curve_fit
import logging, pickle, os, sys
from opt_einsum import contract
from gmanp import pBasis, Pauli, Boson
from time import time, perf_counter, process_time, sleep
from datetime import datetime, timedelta
from pprint import pprint, pformat
from copy import copy
from scipy.integrate import solve_ivp, quad_vec, RK45
SOLVER = RK45 # Runge-Kutta 4th order
from mpmath import polylog
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

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size": 11,
    #"lines.linewidth": 2.5,
})


class HTC:
    COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
    EV_TO_FS = (constants.hbar/constants.e)*1e15 # convert time in electronvolts to time in fs
    DEFAULT_DIRS = {'data':'./data', 'figures':'./figures'} # output directories
    DEFAULT_PARAMS = {
            'Q0': 15, # N_k = 2*Q0+1 nanoparticles
            'NE': 2, # Number of emitters per gap
            'w': 1, # Gap width, nm (Emitter spacing Delta_r = 2a+w = 81nm)
            'a': 40, # Nanoparticle radius, nm (Chain length L = N_k * Delta_r = 10.0 nm)
            #'Delta_r': 81, # Emitter spacing, nm
            #'L': 9.963, # Chain length, micrometers 
            'omega_p': 1.88, # Plasmon reaonsnce, eV
            'omega_0': 1.86, # Dye resonance, eV
            'g': 0.095, # Individual light-matter coupling, eV (collective gSqrt(NE)
            'kappa': 1e-2, # photon loss
            'dephase': 0, # Emitter pure dephasing
            'pump_strength': 1e-3, # Emitter pump magnitude
            'pump_width': 500, # Emitter pump spot width, nm
            'decay': 1e-7, # Emitter non-resonant decay
            'Nnu': 2, # Number of vibrational levels for each emitter
            'S': 0.001, # Huang-Rhys parameter
            'omega_nu': 0.19, # Vibrational mode resonance, eV
            'T':0.026, # k_B T in eV (.0259=300K, .026=302K)
            'gam_nu': 0.01, # vibrational damping rate
            'dt': 0.5, # determines interval at which solution is evaluated. Does not effect the accuracy of solution, only the grid at which dynamics are recorded (if solver e.g. makes a step of 2*dt, interpolation is used to calculate inner points. 
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
        params['Nk'] = 2*self.Q0+1 # total number of modes / nanoparticles
        params['delta_r'] = round(2 * params['a'] + params['w'], 5) # grid spacing in micrometers
        params['L'] = round(params['Nk'] * params['delta_r'], 5) # chain length
        self.c = constants.c # speed of light in material
        self.K_to_eV = (constants.h * self.c) / (constants.e * params['L'] * 1e-06) # factor 1/e for Joules to eV
        self.wp = self.params['omega_p']
        self.rescale_int = 1 # rescale variables according to prescription in thesis
        self.NE, self.Nk, self.Nnu = params['NE'], params['Nk'], params['Nnu']
        params['Nm'] = self.NE * self.Nk # Total number of emitters
        self.Nm = params['Nm']
        params['gSqrtNE'] = params['g'] * self.NE
        self.off_diag_indices_Nk = np.where(~np.eye(self.Nk, dtype=bool))
        self.diag_indices_Nk = np.diag_indices(self.Nk)

    def get_rates(self, params):
        rates = {}
        rates['pump_strength'] = params['pump_strength']
        rates['decay'] = params['decay']
        rates['dephase'] = params['dephase']
        rates['gam_nu'] = params['gam_nu']
        rates['gam_up'] = rates['gam_nu'] * params['nb']
        rates['gam_down'] = rates['gam_nu'] * (params['nb'] + 1)
        rates['gam_delta'] = rates['gam_up'] - rates['gam_down']
        params['gam_up'] = rates['gam_up']
        params['gam_down'] = rates['gam_down']
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
        A = 0.5*params['omega_0']*np.kron(sz, bi) +\
                params['omega_nu']*np.kron(si, bn) +\
                params['omega_nu']*np.sqrt(params['S'])*np.kron(sz, b+bd) + 0j
        A += 0.25 * (-1j * rates['gam_delta']) * np.kron(si, (bd @ bd - b @ b))
        B = params['gSqrtNE'] * np.kron(sp, bi)
        A0_base, _discard = gp.get_coefficients(A, sgn=0, eye=True) # discard part proportional to identity
        consts['A0_n'] = np.outer(A0_base, np.ones(Nk)) # currently no spatial dependence 
        consts['Bp'] = gp.get_coefficients(B, sgn=1) # N.B. gets i_+ coefficients i.e. traces against lambda_{i_-}
        C1_base = gp.get_coefficients(np.kron(sz, bi), sgn=0)
        C2_base = gp.get_coefficients(np.kron(si, bd), sgn=0)
        C3_base = gp.get_coefficients(np.kron(si, b), sgn=0)
        Dp_base = gp.get_coefficients(np.kron(sp, bi), sgn=1)
        Dm_base = gp.get_coefficients(np.kron(sm, bi), sgn=-1)
        consts['gam0_n'] = np.array([np.outer(base, np.sqrt(Xs)) for Xs, base in
                                     zip([self.dephase(self.ns), rates['gam_up'] * np.ones(Nk),
                                          rates['gam_down'] * np.ones(Nk)],[C1_base, C2_base, C3_base])])
        consts['gamp_n'] = np.outer(Dp_base, np.sqrt(self.pump(self.ns)))
        consts['gamm_n'] = np.outer(Dm_base, np.sqrt(self.pump(self.ns)))
        #
        consts['gam00_n'] = contract('arn,apn->rpn', consts['gam0_n'].conj(), consts['gam0_n'])
        consts['gampp_n'] = contract('in,jn->ijn', consts['gamp_n'].conj(), consts['gamp_n'])
        consts['gammm_n'] = contract('in,jn->ijn', consts['gamm_n'].conj(), consts['gamm_n'])
        f000 = gp.f_tensor((0,0,0))
        f011 = gp.f_tensor((0,1,1))
        z000 = gp.z_tensor((0,0,0))
        z011 = gp.z_tensor((0,1,1))
        z011_swap = np.swapaxes(z011, 1, 2)
        assert np.allclose(z011, np.conj(z011_swap))
        zm011 = gp.z_tensor((0,-1,-1))
        consts['xi_n'] = 2 * contract('ipj,pn->ijn', f000, consts['A0_n']) \
                + 2 * contract('irq,rpn,qpj->ijn', f000, consts['gam00_n'], z000).imag \
                + 2 * contract('ijn,aip,bpj->abn', consts['gampp_n'], f011.conj(), zm011).imag \
                + 2 * contract('ijn,aip,bpj->abn', consts['gammm_n'], f011, z011).imag
        consts['phi0_n'] = (2/params['Nnu']) * contract('ajq,jqn->an', f000, consts['gam00_n'].imag) \
                + (2/params['Nnu']) * contract('ijn,aij->an', consts['gampp_n'], f011.conj()).imag \
                + (2/params['Nnu']) * contract('ijn,aij->an', consts['gammm_n'], f011).imag
                  # Note gamm00 index order reversed below (conjugate)
        consts['xip_n'] = - 2 * contract('aij,an->ijn', f011, consts['A0_n']) \
                + 1j * contract('aip,abn,bpj->ijn', f011, consts['gam00_n'], zm011.conj()) \
                - 1j * contract('aip,ban,bpj->ijn', f011, consts['gam00_n'], z011) \
                + 1j * contract('aip,qpn,aqj->ijn', f011, consts['gammm_n'], zm011.conj()) \
                - 1j * contract('aip,pqn,aqj->ijn', f011, consts['gampp_n'], z011)
        consts['xim_n'] = - 2 * contract('aij,an->ijn', f011.conj(), consts['A0_n']) \
                + 1j * contract('aip,abn,bpj->ijn', f011.conj(), consts['gam00_n'], z011.conj()) \
                - 1j * contract('aip,ban,bpj->ijn', f011.conj(), consts['gam00_n'], zm011) \
                + 1j * contract('aip,qpn,aqj->ijn', f011.conj(), consts['gampp_n'], z011.conj()) \
                - 1j * contract('aip,pqn,aqj->ijn', f011.conj(), consts['gammm_n'], zm011)
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
        for i, K in enumerate(shifted_Ks):
            for j, P in enumerate(shifted_Ks):
                # EDIT 2023-10-26: Fixed sign of omega term!
                coeffs['11_pk'][j,i] = (1j * self.omega_diff(P,K) - 0.5 * self.kappa_sum(P,K))
        # 2024-04-02 elements now with n dependence: 21_00n, (22_0n,) 31_11kn, 41_11n, 42_11n
        coeffs['12_1'] = 1j * consts['Bp']
        coeffs['13_1'] = coeffs['12_1'].conj()
        # EQ 2
        coeffs['21_00n'] = 1 * consts['xi_n'] # (make a copy)
        coeffs['22_0n'] = 1 * consts['phi0_n']
        coeffs['23_01'] = 4 * contract('i,aij->aj', consts['Bp'], f011)
        # EQ 3
        coeffs['31_11kn'] = contract('ijn,k->ijkn', consts['xip_n'], np.ones(Nk)) \
                - contract('ij,n,k->ijkn', np.eye(gp.indices_nums[1]), np.ones(Nk),
                           1j * consts['omega'] + 0.5 * consts['kappa'])
        coeffs['32_1'] = coeffs['12_1'].conj()
        coeffs['33_1kn'] = 1j * contract('i,kn->ikn', consts['Bp'].conj(), coeffs['expi_kn']) / Nm
        coeffs['34_1kn0'] = - contract('jkn,aij->ikna', coeffs['33_1kn'], z011)
        coeffs['35_1kn'] = - coeffs['33_1kn'] / Nnu
        coeffs['36_01'] = 2 * contract('aij,j->ai', f011, consts['Bp'].conj())
        # EQ 4
        coeffs['41_11n'] = 1 * consts['xip_n']
        coeffs['42_11n'] = 1 * consts['xim_n']
        coeffs['43_01'] = coeffs['36_01'].conj() 
        coeffs['44_01'] = 1 * coeffs['36_01']
        if self.rescale_int==1:
            coeffs['23_01'] *= (1/Nm)
            coeffs['34_1kn0'] *= Nm
            coeffs['35_1kn'] *= Nm
        elif self.rescale_int==2:
            sNm = np.sqrt(Nm)
            coeffs['23_01'] *= (1/Nm) * (1/sNm)
            coeffs['34_1kn0'] *= Nm * sNm
            coeffs['35_1kn'] *= Nm * sNm
        # HOPFIELD coefficients (in shifted basis i.e. K=0,1,2,...,Q0,-Q0,-Q0+1,....-1
        consts['zeta_k'] = 0.5 * np.sqrt( (params['omega_0'] - consts['omega'])**2 + 4 * params['gSqrtNE']**2 )
        coeffs['X_k'] = np.sqrt(0.5  + 0.5**2 * (params['omega_0'] - consts['omega'])/consts['zeta_k'])
        coeffs['Y_k'] = np.sqrt(0.5  - 0.5**2 * (params['omega_0'] - consts['omega'])/consts['zeta_k'])
        assert np.allclose(coeffs['X_k']**2+coeffs['Y_k']**2, 1.0), 'Hopfield coeffs. not normalised'
        consts['vsigma'] = gp.get_coefficients(np.kron(sp, bi), sgn=1, eye=False)
        assert np.allclose(consts['vsigma'].imag, 0.0)
        consts['vvsigma'] = self.Nnu/2
        assert np.allclose(consts['vvsigma'], contract('i,i', consts['vsigma'], consts['vsigma']))
        assert np.allclose(contract('i,inm->nm', consts['vsigma'], gp.basis[gp.indices[1]]), np.kron(sp,bi))
        # Coefficients used to calculate dynamics
        ocoeffs = {}
        # 'pup_l' and 'pup_I' are C^0_{i_0} and D^0 in thesis
        ocoeffs['pup_l'], ocoeffs['pup_I'] = \
                self.gp.get_coefficients(np.kron(Pauli.p1, bi), sgn=0, eye=True)
        assert np.allclose(contract('a,anm->nm', ocoeffs['pup_l'], gp.basis[gp.indices[0]]),
                           np.kron(Pauli.p1, bi)-ocoeffs['pup_I']*np.eye(2*self.Nnu))
        assert np.allclose(ocoeffs['pup_I'], 0.5)
        ocoeffs['sp_l'] = consts['vsigma']
        ocoeffs['rn'] = params['delta_r'] * self.ns
        ocoeffs['rn2'] = ocoeffs['rn']**2
        ocoeffs['msrn'] = (ocoeffs['rn'] - 0.5 * params['L'])**2
        v_ops = [np.kron(si, np.diag([1.0 if i==j else 0.0 for i in range(Nnu)])) for j in range(Nnu)]
        ocoeffs['vpops'] = [self.gp.get_coefficients(v_op, sgn=0, eye=True) for v_op in v_ops]
        # assign to instance variables
        self.consts, self.coeffs, self.ocoeffs = consts, coeffs, ocoeffs

    def omega_single(self, K):
        if K == 0:
            return self.omega_single(1/10)
        if K < 0:
            return self.omega_single(-K)
        k = 2*np.pi*K/self.params['L']
        a = self.params['a'] # nanoparticle radius
        d = self.params['delta_r'] # gap spacing
        c = self.c
        w0 = self.params['omega_p'] # plasmon resonance
        eta = -2 # longitudinal mode
        Omega = (w0/2) * (a/d)**3 # long-range dipolar coupling strength between LSPs
        cutoff = 1/a # momentum cutoff
        EV_TO_HZ = constants.e/constants.hbar
        exp = np.exp(1j*k*d)
        f = eta * float(polylog(3, exp).real+polylog(3,np.conj(exp)).real)
        omega_0 =  w0 * np.sqrt(1 + 2 * (Omega/w0) * f)
        omega_0_HZ_NM = EV_TO_HZ * omega_0 / 1e09
        cutoff_HZ_NM = c/a
        prefactor = (eta/2) * (w0**2/omega_0) * (k**2 * a**3/d) * np.heaviside(cutoff - np.abs(k), 0.0)
        term1 = np.log(cutoff/k)
        term2 = 0.5*(1 + np.sign(eta)*(omega_0_HZ_NM/(c*k))**2)\
        * np.log(np.abs(c**2*k**2-omega_0_HZ_NM**2)/(cutoff_HZ_NM**2-omega_0_HZ_NM**2))
        return omega_0 + prefactor * (term1 + term2)

    def omega(self, Ks):
        if type(Ks) is not np.ndarray:
            Ks = np.array([Ks])
        return np.array([self.omega_single(K) for K in Ks])

    def kappa(self, K):
        return self.params['kappa'] * np.ones_like(K)

    def gaussian(self, n, _max, _width, _offset=0):
        n0 = self.Q0
        #return self.params['pump_strength'] * np.ones_like(n) # uniform
        return _max * np.exp(- 0.5 * ( (n-n0)/_width )**2 )

    def pump(self, n):
        return self.gaussian(n, self.params['pump_strength'], self.params['pump_width'])

    def decay(self, n):
        return self.params['decay'] * np.ones_like(n)

    def dephase(self, n):
        return self.params['dephase'] * np.ones_like(n)

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
        self.initial_state = self.ground_state()

    def ground_state(self):
        logger.info(f'Creating initial state with 0 photons/excitons + thermal vibrational state')
        pex = 0.0 # initial excited state population of emitters
        rho0_ele = np.diag([pex,1.0-pex])
        rho0_vib = self.thermal_rho_vib(self.params['T']) # molecular vibrational density matrix
        rho0 = np.kron(rho0_ele, rho0_vib)
        coeffs0, eye0 = self.gp.get_coefficients(rho0, sgn=0, eye=True, warn=False)
        l = [2 * coeffs0 for n in range(self.Nk)]
        l = np.real(l).T # i_0 index first, then ensemble index n
        self.all_eye0s = [eye0 for n in range(self.Nk)] # only needed if want to recreate density matrix on a site
        state = np.zeros(self.state_length, dtype=complex)
        state[self.state_dic['l']['slice']] = l.flatten()
        state[-1] = -1 # indicates state has NOT been rescaled
        return state
    
    def evolve(self, tend=100.0, atol=1e-8, rtol=1e-6):
        """Integrate second-order cumulants equations of motion from t=0 to  t=tend (femptoseconds)"""
        dt_fs = self.params['dt']
        self.t_fs = np.arange(0.0, tend, step=dt_fs)
        self.t = self.t_fs / self.EV_TO_FS
        dt = dt_fs / self.EV_TO_FS
        self.num_t = len(self.t)
        state_MB = sys.getsizeof(self.initial_state) / 1024**2
        logger.info(f'Number of variables {len(self.initial_state):.2e}, '\
                f'estimate ~{8*state_MB:.0f} MB to propagate dynamics')
        logger.info(f'Integrating 2nd-order EoMs to tend={tend:.0f}fs with interpolation'\
                f' to fixed grid of spacing dt={dt:.3g}fs')
        self.setup_dynamics_storage() # creates self.dynamics data dictionary
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
                    t_bound=self.t[-1],
                    atol=atol,
                    rtol=rtol,
                    )
        # Save initial state
        assert rk45.t == self.t[t_index], 'Solver initial time incorrect'
        self.record_dynamics(t_index, rk45.y) # record physical dynamics for initial state
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
                    self.record_dynamics(t_index, y) # extract relevant dynamics from state y 
                    t_index += 1
                    if t_index >= self.num_t: # reached the end of our grid, stop solver
                        end = True
                        break
                    next_t = self.t[t_index]
            if next_check_i < num_checkpoints and t_index >= checkpoints[next_check_i]:
                solver_diffs = np.diff(solver_t[last_solver_i:])
                logger.info('Progress {:.0f}% ({:.0f}s)'.format(100*(checkpoints[next_check_i]+1)/self.num_t, time()-tic))
                logger.info('Avg. solver dt for last part: {:.2g} (grid dt_ds={:.3g}fs)'\
                        .format(np.mean(solver_diffs) * self.EV_TO_FS, dt_fs))
                next_check_i += 1
                last_solver_i = len(solver_t) - 1
            if end:
                break # safety, stop solver if we have already calculated state at self.t[-1]
        toc = time()
        self.compute_time = toc-tic # ptoc-ptic
        self.solver_info = {'method': 'RK45', 't0': 0.0, 'tend': tend, 'atol': atol, 'rtol': rtol,
                            'runtime': self.compute_time}
        logger.info('...done ({:.0f}s)'.format(self.compute_time))
        self.results = {'parameters': self.params,
                        'dynamics': self.dynamics,
                        'solver_info': self.solver_info,
                        }
        return self.results


    def setup_dynamics_storage(self):
        """Prepare dictionary self.dynamics to store values of relevant dynamics
        These arrays (or arrays in dictionaries) are zero initialised and then assigned
        non-zero values in place by self.record_dynamics during the computation
        """
        Nt = self.num_t
        nPs = np.zeros((Nt, self.Nk), dtype=float)
        nMs = np.zeros((Nt, self.Nk), dtype=float)
        nBs = np.zeros((Nt, self.Nk), dtype=float)
        g1s = np.zeros((Nt, self.Nk), dtype=complex) # real also??
        vpops = np.zeros((Nt, self.Nk, self.Nnu), dtype=float)
        self.dynamics = {'t': self.t_fs,
                         'r': self.rs, 
                         'nP': nPs,
                         'nM': nMs,
                         'nB': nBs,
                         'g1': g1s,
                         'vpop': vpops,
                         }

    def blank_density_dic(self, dtype=float):
        Ns = self.num_t
        return {'vals': np.zeros((Ns, self.Nk), dtype=dtype),
                'mean': np.zeros((Ns,), dtype=dtype),
                'var': np.zeros((Ns,), dtype=dtype),
                'msd': np.zeros((Ns,), dtype=dtype),
                }

    def record_dynamics(self, t_index, y):
        """Calculates and saves observable values from state y at timestep t_index
        To add additional dynamics, add a key-empty array to self.dynamics e.g.
        self.dynamics['my_obs'] in self.setup_storage_dynamics and then write a
        function to take state, calculate value of observable and assign to
        self.dynamics['my_obs'][t_index]
        """
        # This is only copy of entire state we make. Essential -otherwise we would be modifying solver's state!
        # To avoid this copy overhead, don't rescale until AFTER calculating each observable (significant rewrite)
        state = y.copy()
        t = self.t[t_index]
        self.rescale_state(state) # correct scale of variables to calculate physical quantities (state modified in-place)
        ada, l, al, ll = self.split_reshape_return(state, check_rescaled=True) # VIEWS of original state i.e. modifying state will change ada, l, al and ll and vice versa
        # The following directly update the instance variable self.dynamics 
        self.calculate_photonic(t_index, ada) # Photon exciton densities
        self.calculate_electronic(t_index, l, ll) # Electronic and bright state densities 
        self.calculate_vibronic(t_index, l) # vibrational populations for emitters in each gap

    def calculate_photonic(self, t_index, ada):
        alpha = ifft(ada, axis=0) # including 1/N_k normalisation!
        dft2 = fft(alpha, axis=-1)
        nph = np.diag(dft2) # n(r_n) when n=m
        self.check_real(t_index, nph, 'Photon density')
        g1s = np.zeros(self.Nk, dtype=complex)
        mid_n = self.Q0
        for n in self.ns:
            numer = dft2[n, mid_n]
            demon = np.sqrt(np.real(dft2[n,n]) * np.real(dft2[mid_n, mid_n]))
            if np.isclose(demon, 0.0):
                g1s[n] = np.zeros_like(numer)
            else:
                g1s[n] = numer/demon
        # check g<=1 etc.
        #self.check_real(t_index, g1s, 'First-order coherence')
        self.dynamics['nP'][t_index] = np.real(nph)
        self.dynamics['g1'][t_index] = g1s

    def calculate_electronic(self, t_index, l, ll):
        # N.B. we do not need to use the coefficients of the initial density
        # matrix for the identity matrix (self.coeff_eyes), that is only
        # relevant if we want to create the density matrix; instead just note if
        # OP = A lambda0 + B lambda1 + C I then <OP> = A <lambda0> + B<lambda1>
        # + C since <I>=1 should always be true, i.e. Tr[rho]==1
        nM = self.NE * (contract('a,an->n', self.ocoeffs['pup_l'], l) + self.ocoeffs['pup_I'])
        self.check_real(t_index, nM, 'Electronic excitation density')
        self.dynamics['nM'][t_index] = np.real(nM) # Electronic excitation per gap
        # Bright state calculation
        diag_ind = self.diag_indices_Nk
        ll_diag = ll[:,:,*diag_ind] # diagonal entries at EACH i,j N.B. returns a new array
        ssll = contract('i,j,ijn->n',
                        self.consts['vsigma'],
                        self.consts['vsigma'],
                        ll_diag)
        nB = (self.NE-1) * ssll + nM/self.NE
        self.check_real(t_index, nB, 'Bright state')
        self.dynamics['nB'][t_index] = np.real(nB)

    def calculate_vibronic(self, t_index, l):
        vpops = [contract('a,an->n', self.ocoeffs['vpops'][V][0], l) + self.ocoeffs['vpops'][V][1]
                       for V in range(len(self.ocoeffs['vpops']))]
        self.check_real(t_index, vpops, 'Vibrational populations')
        self.dynamics['vpop'][t_index] = np.real(vpops).T # N index first

    def rescale_state(self, state):
        """Rescale state before calculating physical dynamics
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
        # 2024-04-02 elements now with n dependence: 21_00n, 31_11kn, 41_11n, 42_11n
        """Equations of motion as in cumulant_in_code.pdf"""
        C = self.coeffs
        ada, l, al, ll = self.split_reshape_return(state)
        # Calculate DFTs
        alpha = ifft(ada, axis=0, norm='forward')
        d = fft(al, axis=1, norm='backward') # backward default
        # EQ 1 # N.B. kp->pk ordering
        pre_c = contract('i,ikn->kn', C['12_1'], al)
        post_c = fft(pre_c, axis=1, norm='forward')
        dy_ada2 = C['11_pk'] * ada + np.transpose(post_c) + np.conj(post_c)
        # EQ 2
        dy_l = contract('abn,bn->an', C['21_00n'], l) + C['22_0n'] + contract('aj,jnn->an', C['23_01'], d).real
        # EQ 3
        pre_beta = contract('j,ijnm->imn', C['32_1'], ll) # N.B. swapped axes
        post_beta = ifft(pre_beta, axis=-2, norm='backward') # See Eqs.
        dy_al2 = contract('ijkn,jkn->ikn', C['31_11kn'], al) \
                + post_beta \
                + contract('jkn,ijnn->ikn', C['33_1kn'], ll) \
                + contract('ikna,an->ikn', C['34_1kn0'], l) \
                + C['35_1kn'] \
                + contract('ai,an,nk->ikn', C['36_01'], l, alpha) 
        # EQ 4
        dy_ll = contract('ipn,pjnm->ijnm', C['41_11n'], ll) \
                + contract('jpn,ipnm->ijnm', C['42_11n'], ll) \
                + contract('aj,am,imn->ijnm', C['43_01'], l, d) \
                + contract('ai,an,jnm->ijnm', C['44_01'], l, d.conj())
        dy_rescale_int = np.zeros(1)
        # flatten and concatenate to match input state structure (1d array)
        dy_state = np.concatenate((dy_ada2, dy_l, dy_al2, dy_ll, dy_rescale_int), axis=None)
        return dy_state

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


    def export_data(self, fp=None):
        if fp is None:
            fp = self.gen_fp()
        if not os.path.exists(os.path.dirname(fp)):
            os.makedirs(os.path.dirname(fp))
        with open(fp, 'wb') as fb:
            pickle.dump(self.dynamics, fb)
        logger.info(f'Wrote parameters & dynamics data to {fp}')

    def import_data(self, fp=None):
        if fp is None:
            fp = self.generate_fp()
        with open(fp, 'rb') as fb:
            self.dynamics = pickle.load(fb)
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
             r'\(\omega_c={}\), \(\epsilon={}\)'.format(params['omega_c'], params['omega_0']),
             #r'\(\omega_c={}\)'.format(params['omega_c']),
             #r'\(\epsilon={}\)'.format(params['omega_0']),
             r'\(n_r={}\)'.format(params['nr']),
             r'\(g\sqrt{{N_m}}={:.3g}\)'.format(params['gSqrtNE']),
            ]
        rate_params = \
            [
             #'\n',
             r'\rm{Rates (eV)}',
             r'\(\kappa={}\)'.format(params['kappa']),
             r'\(\Gamma_\uparrow={}\)'.format(params['pump_strength']),
             r'\(\Gamma_\downarrow={}\)'.format(params['decay']),
             r'\(\Gamma_z={}\)'.format(params['dephase']),
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
        plot_t = self.dynamics['t_fs']
        axes[1,0].set_title(r'\(\sum_k  \langle a^\dagger_k a^{\vphantom{\dagger}}_k\rangle\)', y=1.0)
        kap = self.params['kappa']
        if np.isclose(kap, 0.0):
            kap_str = r'\(\kappa\sim\infty\)'
        else:
            kap_str = r'\(1/\kappa\sim{:.0f}\) \rm{{fs}}'.format(round((1/self.params['kappa'])*self.EV_TO_FS,-2))
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
        axes[1,0].plot(plot_t, np.sum(self.dynamics['ph_dic']['vals'], axis=1))
        #axes[1,0].plot(plot_t, np.sum(self.dynamics['n'], axis=1))
        #print(np.allclose(np.sum(self.dynamics['n'], axis=1), np.sum(self.dynamics['ph_dic']['vals'], axis=1)))
        nPh, nM = self.dynamics['ph_dic']['vals'], self.dynamics['mol_dic']['vals'] 
        nB, nD = self.dynamics['nB'], self.dynamics['nD']
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
        nB_centered = nB - self.dynamics['nB'][0] # subtract t=0 row from each t>0 row
        im2 = my_im(axes[4,0], nB_centered)
        cbar0 = fig.colorbar(im0, ax=axes[2,0], aspect=20)
        cbar1 = fig.colorbar(im1, ax=axes[2,1], aspect=20)
        cbar2 = fig.colorbar(im2, ax=axes[4,0], aspect=20)
        axes[2,0].axvline(L/2, c='k')
        axes[2,1].axvline(L/2, c='k')
        axes[4,0].axvline(L/2, c='k')
        # PANEL G & H + I - MSD
        mol_dic, ph_dic, coh_dic = self.dynamics['mol_dic'], self.dynamics['ph_dic'], self.dynamics['coh_dic']
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
                          f'Nnu{self.Nnu}Nk{self.Nk}S{self.params["S"]}Gamz{self.params["dephase"]}.png')
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
        dph = self.dynamics['ph_dic']['vals'][1:]
        dmol = self.dynamics['mol_dic']['vals'][1:]
        ts = self.dynamics['t_fs'][1:]
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
        chosen_i = next((i for i, t in enumerate(self.dynamics['select_data']['select_t']) if t >= chosen_time_fs), None)
        cgs_c = cgs/constants.c
        if chosen_i is None:
            logger.warning(f'<a^dag_K\' a_K> not recorded at {chosen_time_fs} fs')
            dominant_i = 0
        else:
            ada = self.dynamics['select_data']['adak'][chosen_i]
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
        msdP = self.dynamics['ph_dic']['msd'][1:] # ignore first data point where no photons
        ts_fs = self.dynamics['t_fs'][1:]
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


    def plot_pump_profile(self, data_only=False):
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

if __name__ == '__main__':
    logging.basicConfig(
        format='%(filename)s L%(lineno)s %(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%H:%M')
    htc = HTC()
    htc.plot_dispersion()
    htc.evolve()
    plt.close('all') # cleanup any open figures
    #htc.export_data() # export dynamics and parameters as dict to .pkl file

