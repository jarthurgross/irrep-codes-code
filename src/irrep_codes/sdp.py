import itertools as it

import numpy as np
import picos as pic
import cvxopt as cvx
from joblib import Parallel, delayed

import pysme.integrate as integ
import qinfo.supops as supops

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

    return P

def create_decode_optimizing_SDP(error_proc_tensor, encode_choi_mat,
        dim_logical):
    '''Create an SDP to optimize the decoding for a given code and error.

    '''
    dim_encoded = error_proc_tensor.shape[0]
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
    decode_proc_tensor = supops.choi_mat_to_proc_tensor(decode_choi_mat,
                                                          dim_in=dim_encoded)
    dim_logical = decode_proc_tensor.shape[2]
    error_decode_proc_tensor = supops.proc_tensor_compose(decode_proc_tensor,
                                                          error_proc_tensor)
    fidelity_observable = get_fidelity_observable(error_decode_proc_tensor,
                                                  dim_logical)
    return create_proc_fid_opt_SDP(fidelity_observable, dim_in=dim_logical,
                                   dim_out=dim_encoded)

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
    return SDP_soln['obj']
