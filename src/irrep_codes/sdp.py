import itertools as it

import numpy as np
from scipy.optimize import OptimizeResult
import picos as pic
import cvxopt as cvx
from joblib import Parallel, delayed

import pysme.integrate as integ
import qinfo as qi
from qinfo import supops

def get_error_tensors(Ls, H, times, solve_ivp_kwargs=None, n_jobs=1):
    integrator = integ.UncondLindbladIntegrator(Ls, H)
    error_tensors = get_process_tensors_from_lindblad(integrator, times,
                                                      solve_ivp_kwargs,
                                                      n_jobs)
    return error_tensors

def get_process_tensors_from_lindblad(integrator, times, solve_ivp_kwargs=None,
        n_jobs=1):
    dim = integrator.basis.dim
    kets = [np.array([1 if j==k else 0 for k in range(dim)])
            for j in range(dim)]
    rho0_jks = [[np.outer(ket1, ket2.conj()) for ket2 in kets]
               for ket1 in kets]
    if n_jobs == 1:
        soln_jks = [[get_non_herm_soln_time_ind(integrator, rho0, times,
                                                solve_ivp_kwargs)
                     for rho0 in rho0_ks]
                    for rho0_ks in rho0_jks]
    else:
        soln_jks = Parallel(n_jobs=n_jobs)(delayed(get_non_herm_soln_time_ind)
                                           (integrator, rho0, times,
                                            solve_ivp_kwargs)
                                           for rho0 in it.chain(*rho0_jks))
        soln_jks = [soln_jks[dim*n:dim*(n+1)] for n in range(dim)]
    rhos_jks = [[soln.get_density_matrices() for soln in soln_ks]
               for soln_ks in soln_jks]
    return [np.array([[rhos[t_idx] for rhos in rhos_ks]
                      for rhos_ks in rhos_jks])
            for t_idx in range(len(times))]



def get_non_herm_soln_time_ind(integrator, rho0, times, solve_ivp_kwargs):
    rho0_vec = integrator.basis.vectorize(rho0, dense=True)
    if np.linalg.norm(integrator.a_fn(times[0], rho0_vec)) < 1e-9:
        print('Derivative is zero.')
        return integ.Solution(np.repeat(rho0_vec[None,:], len(times), axis=0),
                              integrator.basis.basis.todense())
    else:
        return integrator.integrate_non_herm(rho0, times, solve_ivp_kwargs)

def make_encode_process_tensor(kets):
    return np.array([[np.outer(ket1, ket2.conj()) for ket2 in kets]
                     for ket1 in kets])

def get_fidelity_observable(proc_tensor, dim):
    '''Observable whose expectation value w.r.t. the Choi matrix of another
    process is the fidelity of the composed processes.

    '''
    s = proc_tensor.shape
    return np.transpose(proc_tensor, (3, 1, 2, 0)).reshape(s[3]*s[1], s[2]*s[0])/dim**2

def create_proc_fid_opt_SDP(fidelity_observable, dim_in, dim_out):
    '''Create an SDP that optimizes the process fidelity.

    The process we are optimizing over is composed with some other process,
    encoded in the fidelity observable, and we are optimizing the fidelity of
    the combined process with respect to the identity process.

    '''
    assert fidelity_observable.shape[0] == dim_in*dim_out
    # Identity on the output Hilbert space
    I_out = np.eye(dim_out)
    P = pic.Problem()
    # Choi state for the recovery map (chi / dim_in)
    X = P.add_variable('X',(dim_out*dim_in, dim_out*dim_in),'hermitian')
    # Add completely-positive constraint
    P.add_constraint(X >> 0)
    # Add trace-preserving constraint: tr_out(chi/dim_in) = I_in/dim_in
    Y = {}
    for j in range(dim_in):
        for k in range(j, dim_in):
            E_jk = np.zeros((dim_in, dim_in))
            E_jk[j,k] = 1
            Y[j,k] = np.kron(E_jk, I_out)
            P.add_constraint((Y[j,k]|X)
                             == (1 if j==k else 0))

    # Observable for the fidelity of the recovery operation given an
    # encoding and error channel
    C = cvx.matrix(fidelity_observable)

    # Find the Choi state for the recovery map that optimizes fidelity
    P.set_objective('max', C|X)

    return P

def create_decode_optimizing_SDP(error_proc_tensor, encode_choi_mat):
    '''Create an SDP to optimize the decoding for a given code and error.

    '''
    dim_encoded = error_proc_tensor.shape[0]
    dim_logical = encode_choi_mat.shape[0] // dim_encoded
    encode_proc_tensor = supops.choi_mat_to_proc_tensor(encode_choi_mat,
                                                        dim_in=dim_logical)
    encode_error_proc_tensor = supops.proc_tensor_compose(error_proc_tensor,
                                                          encode_proc_tensor)
    fidelity_observable = get_fidelity_observable(encode_error_proc_tensor,
                                                  dim_logical)
    return create_proc_fid_opt_SDP(fidelity_observable, dim_in=dim_encoded,
                                   dim_out=dim_logical)

def create_encode_optimizing_SDP(error_proc_tensor, decode_choi_mat):
    '''Create an SDP to optimize the encoding for a given error and decoding.

    '''
    dim_encoded = error_proc_tensor.shape[0]
    dim_logical = decode_choi_mat.shape[0] // dim_encoded
    decode_proc_tensor = supops.choi_mat_to_proc_tensor(decode_choi_mat,
                                                          dim_in=dim_encoded)
    error_decode_proc_tensor = supops.proc_tensor_compose(decode_proc_tensor,
                                                          error_proc_tensor)
    fidelity_observable = get_fidelity_observable(error_decode_proc_tensor,
                                                  dim_logical)
    return create_proc_fid_opt_SDP(fidelity_observable, dim_in=dim_logical,
                                   dim_out=dim_encoded)

def biSDP_maximize(x0, x_to_y_result, y_to_x_result, args=(), maxiter=200,
                    xatol=1e-4, yatol=None, fatol=1e-4, **options):
    '''Bi-Semi-Definite-Program maximizer.

    Optimizes two matrices, x and y, that feed into an objective function
    f(x,y) where fixing either results in an SDP for the other.

    Starting with an initial value for x, optimal value of y is solved for. Then,
    for that value of y, the optimal value of x is solved for. This process
    iterates until a stopping criterion is met.

    Parameters
    ----------

    x0 : array_like
        The initial value for the first quantity
    x_to_y_result : callable
        Function that takes a value of x and returns an OptimizeResult for the
        solution to the y-optimization SDP
    y_to_x_result : callable
        Function that takes a value of y and returns an OptimizeResult for the
        x-optimization SDP
    maxiter : positive integer
        The maximum number of SDPs to solve iteratively
    xatol : float
        The minimum difference between x values in Frobenius norm that is
        acceptable to have between iterations
    yatol : float
        The minimum difference between y values in Frobenius norm that is
        acceptable to have between iterations
    fatol : float
        The minimum increase in objective-function value that is
        acceptable to have between iterations

    '''
    if yatol is None:
        yatol = xatol
    best_x = x0
    result = x_to_y_result(best_x)
    best_fun = result.fun
    best_y = result.x
    niter = 0

    while niter < maxiter:
        niter += 1
        result = y_to_x_result(best_y)
        fun = result.fun
        x = result.x
        if fun - best_fun < fatol or np.linalg.norm(x - best_x) < xatol:
            break
        best_fun = fun
        best_x = x

        niter +=1
        result = x_to_y_result(best_x)
        fun = result.fun
        y = result.x
        if fun - best_fun < fatol or np.linalg.norm(y - best_y) < yatol:
            break
        best_fun = fun
        best_y = y

    return OptimizeResult(fun=best_fun,
                          x=[best_x, best_y],
                          nit=niter, success=(niter > 1))

def setup_code_biSDP(error_proc_tensor, ketLs_0):
    '''Construct inputs for `biSDP_maximize` to optimize an encoding.

    Parameters
    ----------
    error_proc_tensor : array_like
        Process tensor (according to `qinfo.supops` convention) for the error
        process to be protected against
    ketLs_0 : list of array_like
        Initial logical basis states to use in the optimization

    Returns
    -------
    dict
        The mandatory kwargs for `biSDP_maximize`

    '''
    encode_proc_tensor_0 = make_encode_process_tensor(ketLs_0)
    encode_choi_mat_0 = supops.proc_tensor_to_choi_mat(encode_proc_tensor_0)

    def encode_choi_mat_to_decode_choi_mat_result(encode_choi_mat):
        P = create_decode_optimizing_SDP(error_proc_tensor, encode_choi_mat)
        soln = P.solve()
        return OptimizeResult(fun=soln.value, x=np.array(P.get_valued_variable('X')))

    def decode_choi_mat_to_encode_choi_mat_result(decode_choi_mat):
        P = create_encode_optimizing_SDP(error_proc_tensor, decode_choi_mat)
        soln = P.solve()
        return OptimizeResult(fun=soln.value, x=np.array(P.get_valued_variable('X')))

    return {'x0': encode_choi_mat_0,
            'x_to_y_result': encode_choi_mat_to_decode_choi_mat_result,
            'y_to_x_result': decode_choi_mat_to_encode_choi_mat_result}

def get_encode_isom_proc_tensor(ket0Ls, X_rep, ket0, ket1):
    '''Create process tensor for map from logical 0 to encoding Choi matrix.

    Parameters
    ----------
    ket0Ls : list of array_like
        Orthonormal basis for the space of logical 0s within the irrep
        multiplicity space
    X_rep : array_like
        Representative for logical X in the physical space: i exp(-i pi Jx).
    ket0 : array_like
        Logical 0 in the abstract logical space (usually [1, 0])
    ket1 : array_like
        Logical 1 in the abstract logical space (usually [0, 1])

    Returns
    -------
    array_like
        The process tensor for the map from logical L to the encoding Choi
        matrix

    '''
    multiplicity = len(ket0Ls)
    multiplicity_kets = np.eye(multiplicity)
    V_Id = sum([np.outer(np.kron(ket0, ket0L), np.conj(multiplicity_ket))
                for ket0L, multiplicity_ket
                in zip(ket0Ls, multiplicity_kets)])
    V_X = sum([np.outer(np.kron(ket1, X_rep @ ket0L), np.conj(multiplicity_ket))
               for ket0L, multiplicity_ket
               in zip(ket0Ls, multiplicity_kets)])

    def encode_isometry(rho):
        '''Process that takes the projector onto the logical 0 state to the Choi matrix
        for the encoding process (that is, the unnormalized density matrix for the
        maximally entangled state between the logical input space and the physical
        output space).

        '''
        return (V_Id @ rho @ np.conj(V_Id).T + V_Id @ rho @ np.conj(V_X).T
                + V_X @ rho @ np.conj(V_Id).T + V_X @ rho @ np.conj(V_X).T)

    return supops.process_to_proc_tensor(encode_isometry, 2)

def setup_multiplicity_code_biSDP(error_proc_tensor, ket0Ls, X_rep, ket0L_0):
    '''Construct everything necessary to run biSDP_maximize for qubit irrep multiplicity codes

    '''
    dim_encoded = error_proc_tensor.shape[0]
    ket0 = np.array([1, 0])
    ket1 = np.array([0, 1])

    encode_isom_proc_tensor = get_encode_isom_proc_tensor(ket0Ls, X_rep, ket0, ket1)

    def rho0L_to_decode_choi_mat_result(rho0L):
        encode_choi_mat = supops.act_proc_tensor(rho0L, encode_isom_proc_tensor)
        P = create_decode_optimizing_SDP(error_proc_tensor, encode_choi_mat)
        soln = P.solve()
        return OptimizeResult(fun=soln.value, x=np.array(P.get_valued_variable('X')))

    def decode_choi_mat_to_rho0L_result(decoding_choi_mat):
        decode_proc_tensor = supops.choi_mat_to_proc_tensor(decoding_choi_mat, dim_encoded)
        error_decode_proc_tensor = supops.proc_tensor_compose(decode_proc_tensor, error_proc_tensor)
        full_proc_tensor = supops.proc_tensor_compose(supops.tensor_proc_tensors(
            supops.get_identity_proc_tensor(2), error_decode_proc_tensor), encode_isom_proc_tensor)
        unnorm_max_ent_state = np.kron(ket0, ket0) + np.kron(ket1, ket1)
        fidelity_observable = np.einsum('jkmn,m,n->jk', full_proc_tensor,
                                        np.conj(unnorm_max_ent_state),
                                        unnorm_max_ent_state)/4
        P = create_proc_fid_opt_SDP(fidelity_observable, dim_in=1, dim_out=2)
        soln = P.solve()
        return OptimizeResult(fun=soln.value, x=np.array(P.get_valued_variable('X')))

    rho0L_0 = qi.rho_from_ket(ket0L_0 / np.linalg.norm(ket0L_0))
    return {'x0': rho0L_0,
            'x_to_y_result': rho0L_to_decode_choi_mat_result,
            'y_to_x_result': decode_choi_mat_to_rho0L_result}

def create_encoding_SDPs(error_proc_tensors, kets):
    '''Create SDPs to optimize decoding for a given code and different errors.

    Parameters
    ----------
    error_proc_tensors : list of array_like
        Process tensors for the different error channels
    kets : list of array_likst
        The logical codewords of the code

    '''
    encoding_proc_tensor = make_encode_process_tensor(kets)
    encode_error_proc_tensors = [supops.proc_tensor_compose(
                                    error_proc_tensor, encoding_proc_tensor)
                                 for error_proc_tensor
                                 in error_proc_tensors]
    dim_logical = len(kets)
    dim_encoded = kets[0].shape[0]
    fidelity_observables = [get_fidelity_observable(
                                encode_error_proc_tensor, dim_logical)
                            for encode_error_proc_tensor
                            in encode_error_proc_tensors]

    Ps = [create_proc_fid_opt_SDP(fidelity_observable, dim_encoded,
                                  dim_logical)
          for fidelity_observable in fidelity_observables]

    return Ps

def get_mult_2_code_kets(ket0_a, ket0_b, theta, phi, sigx_rep):
    '''Get superposition of two different basis kets in the 0 subspace.

    '''
    ket0 = np.cos(theta/2)*ket0_a + np.exp(1.j*phi)*np.sin(theta/2)*ket0_b
    ket1 = sigx_rep @ ket0
    return ket0, ket1

def get_mult_2_optimal_recovery_fidelity(error_tensor, ket0_a, ket0_b, theta, phi, sigx_rep):
    '''Optimize the recovery fidelity for a code within a multiplicity irrep

    '''
    SDP = create_encoding_SDPs([error_tensor],
                               get_mult_2_code_kets(ket0_a, ket0_b,
                                                    theta, phi, sigx_rep))[0]
    SDP_soln = SDP.solve()
    return SDP_soln.value
