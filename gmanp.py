#!/usr/bin/env python

import os, sys, pickle
import logging
from opt_einsum import contract
import numpy as np
from time import time
import progressbar
import sparse
from pprint import pprint, pformat

logger = logging.getLogger(__name__)


class Pauli:
    """Pauli operators (matrices)"""
    p = np.array([[0,1],[0,0]], dtype=float)
    m = np.array([[0,0],[1,0]], dtype=float)
    p0 = np.array([[0,0],[0,1]], dtype=float)
    p1 = np.array([[1,0],[0,0]], dtype=float)
    x = p + m
    y = 1j * (m - p)
    z = np.array([[1,0],[0,-1]], dtype=float)
    i = np.eye(2, dtype=float)

class Boson:
    """Operators (matrices) of boson with Nnu levels"""
    def __init__(self, Nnu):
        assert Nnu > 0
        self.Nnu = int(Nnu)
        self.b = self.get_destroy()
        self.bd = self.b.T
        self.n = self.b.T @ self.b
        self.i = np.eye(self.Nnu, dtype=float)
    
    def destroy(self):
        return self.b
    def create(self):
        return self.bd
    def number(self):
        return self.n
    def get_destroy(self):
        def Aij(i, j):
            if j == i+1:
                return np.sqrt(j)
            return 0.0
        return np.fromfunction(np.vectorize(Aij), (self.Nnu, self.Nnu))

class pBasis:
    """Matrices, structure factors and helper functions for lambda-basis defined
    in Sec 6.2 of thesis"""
    TENSOR_DATA_DIR = 'data/tensors' # Save structure factors to save having to compute each time

    @staticmethod
    def inner(mat1, mat2, real=False):
        """inner product"""
        res = (mat1 @ mat2.transpose().conjugate()).trace()
        if real:
            if not np.isclose(res.imag, 0.0):
                logger.warning('Inner product Tr[A A^dag] with non-zero imaginary part')
            return res.real
        return res
    @staticmethod
    def comm(mat1, mat2):
        """commutator"""
        return mat1 @ mat2 - mat2 @ mat1
    @staticmethod
    def acomm(mat1, mat2):
        """anti-commutator"""
        return mat1 @ mat2 + mat2 @ mat1

    def __init__(self, Nnu, tests=True, verbose=False):
        self.Nnu = Nnu # number of vibrational levels 
        self.sl = 2 * Nnu # side length of matrices
        self.dim = self.sl**2 - 1 # number of non-trivial basis vectors (matrices)
        # N.B. 
        # number diagonal matrices: d-1 = 2Nnu-1 = len(self.indices[0])
        # num off-diagonal: d^2-1-(d-1) = d(d-1) = 2Nnu(2Nnu-1) 
        # num i_+: d(d-1)/2 = Nnu(2Nnu-1) = len(self.indices[1]) = len(self.indices[-1])
        saved_log_level = logger.level
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO) # disable info messages during object setup
        self.verbose = verbose # log info messages during setup
        self.basis = [] # list of basis vectors not including the identity
        # structure tensors, to be constructed or loaded from file
        self._d_tensors, self._f_tensors, self._z_tensors, self._zm_tensors = {}, {}, {}, {}
        # stores indices in basis according to electronic action (0, +1, -1) and 
        # some other characteristics
        self.indices = {0:[],1:[],-1:[],
                        'symmetric':[],'antisymmetric':[], 'diagonal':[],
                        'UL':[], 'LL':[], 'UR':[],'LR':[], # quadrant
                        }
        self.indices_nums = {}
        if not os.path.exists(self.TENSOR_DATA_DIR):
            os.makedirs(self.TENSOR_DATA_DIR)
        # Basis and structure tensor constructions
        self.wrapit(self.make_basis, f'Constructing {self.sl}x{self.sl} p-basis matrices...')
        self.wrapit(self.construct_all_tensors, f'Getting f,d structure tensors...')
        # Checks of constructed basis and tensors
        if tests:
            self.wrapit(self.perform_checks, f'Performing checks (basis numbers, orthogonality, product rules)...')
        logger.setLevel(saved_log_level)
        if not verbose:
            logger.info(f'Setup {self.sl}x{self.sl} basis for emitters')

    def get_coefficients(self, mat, sgn=None, eye=False, warn=True, real=False):
        """Calculate coefficients of a matrix 'mat' in lambda basis
        if sgn is specified as 0, 1 or -1, return coefficients for lambda matrices of that
        sign only
        if eye return coefficient of identity as well
        if warn log a warning if sgn is specified such that non-zero coefficients are NOT returned
        if real convert coefficients to real before returning
        """
        coeffs = {0:[], 1:[], -1:[]}
        for i in range(-1,2):
            for bmat in self.basis[self.indices[i]]:
                coeffs[i].append(0.5 * self.inner(mat, bmat)) # N.B. inner includes conjugation
        coeffs['eye'] = (1/self.sl) * self.inner(mat, self.eye)
        if real:
            for key, vals in coeffs.items():
                if not np.allclose(np.imag(vals), 0.0):
                    logger.warning(f'Discarding non-trivial imaginary part of coefficients {key}')
                coeffs[key] = np.real(vals)
        if sgn is None:
            # combine all coefficients in correct order and return
            all_indices = self.indices[0] + self.indices[1] + self.indices[-1]
            all_coeffs = coeffs[0] + coeffs[1] + coeffs[-1]
            all_indices, all_coeffs = zip(*sorted(zip(all_indices, all_coeffs)))
            if eye:
                return np.array(all_coeffs), coeffs['eye']
            return np.array(all_coeffs)
        assert sgn in [-1,0,1], f'Sign {sgn} invalid'
        for i in range(-1,2):
            if sgn == i:
                continue
            if warn and not np.allclose(coeffs[i], 0.0):
                logger.warning(f'Selecting coeffs for sign {sgn} matrices but sign {i} coeffs. are also non-zero')
        if not eye and warn and not np.isclose(coeffs['eye'], 0.0):
            logger.warning(f'Discarding non-trivial identity coefficient')
        if eye:
            return np.array(coeffs[sgn]), coeffs['eye']
        return np.array(coeffs[sgn])

    def expectation_from_rho(self, full_rho, op_e, op_vib, real=False):
        expect = np.trace(np.kron(op_e, op_vib) @ full_rho).real
        if real:
            expect.real
        return expect

    def expectations_from_rho(self, full_rhos, op_e, op_vib, real=False):
        expects = []
        for full_rho in full_rhos:
            expects.append(self.expectation_from_rho(full_rho, op_e, op_vib, real=real))
        return np.array(expects)

    def sorted_indices(self, sgns_in):
        return sorted(list(np.array([self.indices[s] for s in sgns_in]).flatten()))

    def pad_trim_vec(self, vec, sgns_in, sgns_out):
        assert len(vec) == sum([self.indices_nums[s] for s in sgns_in])
        # 1 PAD IF REQUIRED
        if set(sgns_in) == set([-1,0,1]):
            full_vec = np.array(vec)
        else:
            full_vec = np.zeros(self.dim, dtype=complex)
            existing_indices = self.sorted_indices(sgns_in)
            full_vec[existing_indices] = vec
        # 2 TRIM IF REQUIRED
        if set(sgns_out) == set([-1,0,1]):
            return full_vec
        target_indices = self.sorted_indices(sgns_out)
        return full_vec[target_indices]

    def get_rho(self, lambda_vals, eye=0.0, sgns=None):
        if sgns is not None:
            lambda_vals = self.pad_trim_vec(lambda_vals, sgns_in=sgns, sgns_out=[-1,0,1])
        return contract('x,xij->ij', np.array(lambda_vals), self.basis) + eye * self.eye

    def get_rhos(self, all_lambda_vals, eye=0.0, sgn=None):
        all_rhos = []
        for lambda_vals in all_lamba_vals:
            all_rhos.append(self.get_rho(lambda_vals, eye=eye, sgn=sgn))
        return np.array(all_rhos)

    def get_expectation(self, lambda_vals, coeffs, real=False, eye=0.0):
        expect = np.sum(np.array(lambda_vals) @ np.array(coeffs)) + eye
        if real:
            return expect.real
        return expect

    def get_expectations(self, all_lambda_vals, coeffs, real=False, eye=0.0):
        all_expectations = []
        for lambda_vals in all_lambda_vals:
            all_expectations.append(get_expectation(lambda_vals, coeffs, real=real, eye=eye))
        return np.array(all_expectations)

    SIG_MAP = {
            # key sig: target sig, need to conjugate tensors
            (0,0,0): ((0,0,0), False),
            (0,1,1): ((0,1,1), False),
            (0,-1,-1): ((0,1,1), True),
            (1,-1,0): ((0,1,1), False), # Note index order, may want to put 0 last again, not sure
            (-1,1,0): ((0,1,1), True), # currently leaving as is in notes
            }

    def check_sig(self, sig):
        assert sig in self.SIG_MAP, f'Signature {sig} invalid'
        # sig is valid, calculate tensors for it if haven't already
        mapped_sig, conj = self.SIG_MAP[sig]
        if mapped_sig not in self._d_tensors:
            self.construct_tensors(mapped_sig)
        # return mapped signature and whether need to conjugate
        return mapped_sig, conj

    def d_tensor(self, sig):
        mapped_sig, conj = self.check_sig(sig)
        if conj:
            return self._d_tensors[mapped_sig].conj()
        return self._d_tensors[mapped_sig]

    def f_tensor(self, sig):
        mapped_sig, conj = self.check_sig(sig)
        if conj:
            return self._f_tensors[mapped_sig].conj()
        return self._f_tensors[mapped_sig]

    def z_tensor(self, sig):
        mapped_sig, conj = self.check_sig(sig)
        if conj:
            # here conj is not exactly conjugate but Z^- tensor (see notes)
            return self._zm_tensors[mapped_sig]
        return self._z_tensors[mapped_sig]

    def i_tensor(self, sig2):
        # signature is a 2-tuple NOT 3, no separate function for tensor construction
        if sig2 not in self._i_tensors:
            d1, d2 = len(self.indices[sig2[0]]), len(self.indices[sig2[1]])
            i = np.eye(d1, dtype=complex)
            if self.sparse:
                i = sp.COO(i)            
            self._i_tensors[sig2] = i
        return self._i_tensors[sig2]

    def d(self, i, j, k):
        # i, j, k refer to indices in full basis
        lambda_i, lambda_j, lambda_k = self.basis[[i,j,k]]
        return (1/4) * self.inner(self.acomm(lambda_i, lambda_j), lambda_k)

    def f(self, i, j, k):
        # i, j, k refer to indices in full basis
        lambda_i, lambda_j, lambda_k = self.basis[[i,j,k]]
        return (-1j/4) * self.inner(self.comm(lambda_i, lambda_j), lambda_k)

    def wrapit(self, meth, msg, timeit=True):
        if msg:
            logger.debug(msg)
        t0 = time()
        meth()
        if timeit:
            logger.debug('...done ({:.2f}s)'.format(time()-t0))

    def perform_checks(self):
        self.check_basis_numbers()
        self.check_orthonormality()
        self.check_structure_factors()

    def get_first_e_quadrant(self, mat):
        row_i, col_i = mat.nonzero() # indices of non-zero elements
        if row_i[0] < self.Nnu:
            # upper half space
            if col_i[0] < self.Nnu:
                # upper left quadrant
                return 0, 'UL'
            else:
                # upper right quadrant
                return 1, 'UR'
        else:
            # in lower half space
            if col_i[0] < self.Nnu:
                # lower left quadrant
                return -1, 'LL'
            else:
                # lower right quadrant
                return 0, 'LR'

    def add_to_basis(self, mat, names=[]):
        index = len(self.basis)
        for name in names:
            self.indices[name].append(index)
        # actually add to basis!
        self.basis.append(mat)

    def diagonal_scale(self, j):
        return np.sqrt(2/((j+1)*(j+2)))

    def make_basis(self):
        """Construct lambda basis"""
        for i in range(self.sl-1):
            for j in range(i+1, self.sl):
                GS = np.zeros((self.sl, self.sl), dtype=float)
                GA = np.zeros((self.sl, self.sl), dtype=complex)
                GS[i,j], GS[j,i] = 1, 1
                GA[i,j], GA[j,i] = -1j, 1j
                GS_e, GS_quadrant = self.get_first_e_quadrant(GS)
                if GS_quadrant in ['UR', 'LL']:
                    # make +1 and -1 matrices
                    up = (1/np.sqrt(2)) * (GS + 1j * GA)
                    down = (1/np.sqrt(2)) * (GS - 1j * GA)
                    self.add_to_basis(up, [1, 'UR'])
                    self.add_to_basis(down, [-1, 'LL'])
                else:
                    self.add_to_basis(GS, [GS_e, GS_quadrant, 'symmetric'])
                    self.add_to_basis(GA, [GS_e, GS_quadrant, 'antisymmetric'])
                if j == i+1:
                    # (d-1) diagonal matrices
                    GD = np.diag([1.0] * (i+1) + [-(i+1)] + [0.0] * (self.sl-2-i))
                    GD *= self.diagonal_scale(i)
                    if i < self.Nnu:
                        self.add_to_basis(GD, [0, 'UL', 'diagonal'])
                    else:
                        self.add_to_basis(GD, [0, 'UL', 'LR', 'diagonal'])
        self.eye = np.eye(self.sl) # N.B. identity NOT rescaled
        self.basis = np.array(self.basis) # list to array so can use advanced numpy slicing
        # convert each index list into an array too - not needed currently
        #for key, lst in self.indices:
            #self.indices[key] = np.array(lst)
        # store each number of matrices of each type for convenience
        for name, inds in self.indices.items():
            self.indices_nums[name] = len(inds)

    def tensor_fp(self):
        return os.path.join(self.TENSOR_DATA_DIR, f'Nnu{self.Nnu}.pkl')

    def construct_all_tensors(self, sigs=[(0,0,0),(0,1,1)], reset=False, save=True):
        tensor_fp = self.tensor_fp()
        if not reset and os.path.exists(tensor_fp):
            loaded_sparse_dic = self.from_sparse_file()
            self._d_tensors, self._f_tensors = loaded_sparse_dic['d'], loaded_sparse_dic['f']
            for sig in self._d_tensors:
                self._z_tensors[sig] = self._d_tensors[sig] + 1j * self._f_tensors[sig]
                self._zm_tensors[sig] = self._d_tensors[sig].conj() + 1j * self._f_tensors[sig].conj()
            logger.info(f'Loaded structure tensors from {tensor_fp}')
        sigs = [sig for sig in sigs if sig not in self._d_tensors]
        num_sigs = len(sigs)
        if num_sigs == 0:
            return
        logger.info(f'Need to construct tensors for signatures {sigs}')
        t0 = time()
        for i, sig in enumerate(sigs):
            tot_time = time()-t0
            logger.info(f'{i} of {num_sigs}... (overall compute time {tot_time:.0f}s)')
            self.construct_tensors(sig, save)


    def construct_tensors(self, sig, save=True):
        logger.info(f'Constructing tensors at {sig}...')
        indices = [self.indices[i] for i in sig]
        nums = [len(inds) for inds in indices]
        d = np.zeros(nums, dtype=complex)
        f = np.zeros(nums, dtype=complex)
        widgets = [' ', progressbar.Percentage(), ' ',  progressbar.Timer()]
        pbar = progressbar.ProgressBar(maxval=np.prod(nums),
                                       widgets=widgets)
        #pbar.start()
        current = 0
        # For now don't assume anything is real or vanishes see e.g. gmanx.py/self.construct_fd_tensors
        # for more efficient code for GGM structure tensors (indices '0' here)
        for i_a, a in enumerate(indices[0]):
            for i_b, b in enumerate(indices[1]):
                for i_c, c in enumerate(indices[2]):
                    current += 1
                    i_tup = (i_a, i_b, i_c)
                    tup = (a, b, c)
                    d[i_tup] = self.d(*tup)
                    f[i_tup] = self.f(*tup)
        #            pbar.update(current)
       # pbar.finish()
        self._d_tensors[sig] = d
        self._f_tensors[sig] = f
        self._z_tensors[sig] = d + 1j*f
        self._zm_tensors[sig] = d.conj() + 1j*f.conj()
        if save:
            # only save f and d
            tensor_dic = {'f': self._f_tensors, 'd': self._d_tensors}
            self.to_sparse_file(tensor_dic)

    def from_sparse_file(self):
        fp = self.tensor_fp()
        with open(fp, 'rb') as fb:
            sparse_dicts = pickle.load(fb)
        dense_dicts = {}
        for fd, dic in sparse_dicts.items():
            dense_dicts[fd] = {}
            for key in dic:
                dense_dicts[fd][key] = dic[key].todense()
        return dense_dicts

    def to_sparse_file(self, outer_dic):
        # convert all tensors (dense) to sparse arrays
        fp = self.tensor_fp()
        sparse_dicts = {}
        for fd, dic in outer_dic.items():
            sparse_dicts[fd] = {}
            for key in dic:
                sparse_dicts[fd][key] = sparse.COO.from_numpy(dic[key])
        with open(fp, 'wb') as fb:
            pickle.dump(sparse_dicts, fb)
        logger.info(f'Saved structure tensors to {fp}')

    def check_basis_numbers(self):
        self.num_0 = 2*self.Nnu**2-1
        self.num_1 = self.Nnu**2
        assert len(self.basis) == self.dim, 'There should be d**2-1={} basis vectors '\
                'but only {} were constructed'.format(self.dim, len(self.basis))
        assert len(self.indices[0]) == self.num_0, 'There should be 2Nnu**2-1={} i_0 '\
                ' matrices but only {} were added'.format(self.num_0, len(self.indices[0]))
        assert len(self.indices[1]) == self.num_1, 'There should be Nnu**2={} i_+ '\
                ' matrices but only {} were added'.format(self.num_0, len(self.indices[1]))
        assert len(self.indices[-1]) == self.num_1, 'There should be Nnu**2={} i_- '\
                ' matrices but only {} were added'.format(self.num_0, len(self.indices[-1]))

    def check_orthonormality(self):
        for i in range(self.dim):
            for j in range(i, self.dim):
                mat1, mat2 = self.basis[[i,j]]
                res = self.inner(mat1, mat2)
                if i==j:
                    assert np.isclose(res, 2.0), f'Matrix {i} not normalised to 2'
                else:
                    assert np.isclose(res, 0.0), f'Matrices {i} and {j} not orthogonal'

    def check_product_rule(self, sig):
        basis = self.basis
        Z = self.z_tensor(sig)
        target = basis[self.indices[sig[-1]]]
        if sig in [(1,-1,0),(-1,1,0)]:
        #if abs(sig[0]) == abs(sig[1]): # don't use - shouldn't include 0-0 !
            RHS = contract('aij,aIJ->ijIJ', Z, target) # IJ are normal matrix indices
        else:
            RHS = contract('ija,aIJ->ijIJ', Z, target)
        # Add SCALED identity where appropriate
        if abs(sig[0])==abs(sig[1]):
            for i in range(len(RHS)):
                RHS[i,i] += (1/self.Nnu) * self.eye
                #for j in range(len(RHS[i])):
                #    if i==j:
                #        RHS[i,j] += (1/self.Nnu) * self.eye
        # matrix multiplication between ith matrix of 1st list and jth matrix in 2nd
        LHS = contract('iIx,jxJ->ijIJ', basis[self.indices[sig[0]]], basis[self.indices[sig[1]]])
        assert np.allclose(LHS, RHS), f'Product rule failure for signature {sig}'

    def check_commutators(self, sig, acomm=False):
        #ind0, ind1, ind2 = [self.indices[i] for i in sig]
        mats1, mats2, mats3 = [self.basis[self.indices[s]] for s in sig]
        if acomm:
            structure_tensors = self.d_tensor(sig)
            prefactor = 2
            comm = self.acomm
        else:
            structure_tensors = self.f_tensor(sig)
            prefactor = 2j
            comm = self.comm
        # tensors have different order of indices relevant for contractions 
        # depending on signature, see e.g. (1.8) in notes
        if sig in [(1,-1,0),(-1,1,0)]:
            # sum over first index
            target = prefactor * contract('aij,aIJ->ijIJ', structure_tensors, mats3)
        else:
            # sum over final index
            target = prefactor * contract('aij,jIJ->aiIJ', structure_tensors, mats3)
        for i, mat1 in enumerate(mats1):
            for j, mat2 in enumerate(mats2):
                LHS = comm(mat1, mat2)
                RHS = target[i,j]
                # add identity contribution for anti commutator when first two 
                # indices are 0 or \pm 1
                if acomm and abs(sig[0]) == abs(sig[1]) and i == j:
                    RHS += (2/self.Nnu) * self.eye
                assert np.allclose(LHS, RHS), f'Commutator (anti={acomm}) fail for sig {sig}, indices '\
                        f'{self.indices[sig[0]][i]}, {self.indices[sig[1]][j]} in full basis.'

    def check_rearrangments(self):
        f000 = self.f_tensor((0,0,0))
        f000r = -contract('abc->cba', f000)
        assert np.allclose(f000, f000r), 'f000 does not have expected antisymmetry'
        d000 = self.d_tensor((0,0,0))
        d000r = + contract('abc->cba', d000)
        assert np.allclose(d000, d000r), 'd000 does not have expected symmetry'
        f011 = self.f_tensor((0,1,1))
        f011c = f011.conj()
        f011r = - contract('aij->aji', f011)
        assert np.allclose(f011c, f011r), 'f011 does not have expected antisymmetry'


    def check_structure_factors(self):
        # tensors for GGM matrices should be real
        assert np.allclose(self.d_tensor((0,0,0)).imag, 0.0), 'd000 has non-trivial imaginary part'
        assert np.allclose(self.f_tensor((0,0,0)).imag, 0.0), 'f000 has non-trivial imaginary part'
        # check product rule for lambda_i0 lambda_j0
        self.check_product_rule((0, 0, 0))
        # check product rule for lambda_i0 lambda_i+
        self.check_product_rule((0, 1, 1))
        # check product rule for lambda_i0 lambda_i-
        self.check_product_rule((0, -1, -1))
        # check product rule for lambda_i+ lambda_i- (unusual index handled by check_product_rule)
        self.check_product_rule((1, -1, 0))
        # check product rule for lambda_i- lambda_i+ (unusual index handled by check_product_rule)
        self.check_product_rule((-1, 1, 0))
        # check (anti-)commutators for each
        self.check_commutators((0,0,0), acomm=False)
        self.check_commutators((0,0,0), acomm=True)
        self.check_commutators((0,1,1), acomm=False)
        self.check_commutators((0,1,1), acomm=True)
        self.check_commutators((0,-1,-1), acomm=False)
        self.check_commutators((0,-1,-1), acomm=True)
        self.check_commutators((1,-1,0), acomm=False)
        self.check_commutators((1,-1,0), acomm=True)
        self.check_commutators((-1,1,0), acomm=False)
        self.check_commutators((-1,1,0), acomm=True)
        # some additional checks of index rearrangement 
        self.check_rearrangments()

if __name__ == '__main__':
    logging.basicConfig(
        format='%(filename)s L%(lineno)s %(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO,
        datefmt='%H:%M:%S')
    p = pBasis(Nnu=3)


