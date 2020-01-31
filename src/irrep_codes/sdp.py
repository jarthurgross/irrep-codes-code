import numpy as np
import picos as pic
import cvxopt as cvx

import pysme.integrate as integ
import qinfo.supops as supops

def get_error_tensors(Ls, H, times, solve_ivp_kwargs=None):
    integrator = integ.UncondLindbladIntegrator(Ls, H)
    error_tensors = get_process_tensors_from_lindblad(integrator, times, solve_ivp_kwargs)
    return error_tensors

def get_process_tensors_from_lindblad(integrator, times, solve_ivp_kwargs=None):
    dim = integrator.basis.dim
    kets = [np.array([1 if j==k else 0 for k in range(dim)])
            for j in range(dim)]
    rho0_jks = [[np.outer(ket1, ket2.conj()) for ket2 in kets]
               for ket1 in kets]
    soln_jks = [[get_non_herm_soln_time_ind(integrator, rho0, times, solve_ivp_kwargs) for rho0 in rho0_ks]
                for rho0_ks in rho0_jks]
    rhos_jks = [[soln.get_density_matrices() for soln in soln_ks]
               for soln_ks in soln_jks]
    return [np.array([[rhos[t_idx] for rhos in rhos_ks]
                      for rhos_ks in rhos_jks])
            for t_idx in range(len(times))]

def make_encoding_process_tensor(kets):
    return np.array([[np.outer(ket1, ket2.conj()) for ket2 in kets]
                     for ket1 in kets])

def get_fidelity_observable(encode_error_tensor):
    s = encode_error_tensor.shape
    dim_logical = s[0]
    dim_encoded = s[2]
    return (dim_encoded/dim_logical**2)*np.transpose(
        encode_error_tensor, (1, 3, 0, 2)).reshape((s[1]*s[3], s[0]*s[2]))

def create_encoding_SDPs(error_tensors, ket0, ket1):
    encoding_tensor = make_encoding_process_tensor([ket0, ket1])
    encode_error_tensors = [supops.compose_process_tensors(
                                    error_tensor, encoding_tensor)
                            for error_tensor in error_tensors]
    fidelity_observables = [get_fidelity_observable(encode_error_tensor)
                            for encode_error_tensor in encode_error_tensors]
    d_logical = 2
    d_encoded = ket0.shape[0]
    I_logical = np.eye(d_logical)

    Ps = []
    for fidelity_observable in fidelity_observables:
        P = pic.Problem()
        # Choi state for the recovery map
        X = P.add_variable('X',(d_logical*d_encoded, d_logical*d_encoded),'hermitian')
        P.add_constraint(X >> 0)
        for j in range(d_encoded):
            for k in range(j, d_encoded):
                E_jk = np.zeros((d_encoded, d_encoded))
                E_jk[j,k] = 1
                P.add_constraint((np.kron(I_logical, E_jk)|X)
                                 == (1 if j==k else 0)/d_encoded)

        # Observable for the fidelity of the recovery operation given an
        # encoding and error channel
        C = cvx.matrix(fidelity_observable)

        # Find the Choi state for the recovery map that optimizes fidelity
        P.set_objective('max', C|X)
        Ps.append(P)

    return Ps
