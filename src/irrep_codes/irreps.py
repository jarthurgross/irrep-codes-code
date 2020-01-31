import pickle
from functools import reduce
import numpy as np
from scipy.linalg import expm
from scipy.interpolate import interp1d
import sympy as sy
import quaternion
from sympy.physics.quantum.spin import WignerD
import mayavi_spin_wigner as msw
from qinfo import wigner

with open('data/clifford-h-s-decomp.txt', 'r') as f:
    lines = f.read()

def Jm_mat(s):
    dim = int(2*s + 1)
    return np.array([[np.sqrt((s + 1)*(a + b + 1) - (a + 1)*(b + 1))
                      if a==b+1 else 0 for b in range(dim)]
                     for a in range(dim)], dtype=np.complex)

def Jx_mat(s):
    Jm = Jm_mat(s)
    return (Jm.conj().T + Jm)/2

def Jy_mat(s):
    Jm = Jm_mat(s)
    return (Jm.conj().T - Jm)/(2j)

def Jz_mat(s):
    dim = int(2*s + 1)
    return np.array([[s - a if a==b else 0 for b in range(dim)]
                     for a in range(dim)], dtype=np.complex)

def rho_from_ket(ket):
    return np.outer(ket, ket.conj())

qubit_Cliff_decomps = str.splitlines(lines)

def Jm_mat_sym(s):
    dim = int(2*s + 1)
    s = (dim - 1)/sy.S(2)
    return sy.Matrix([[sy.sqrt((s + 1)*(a + b + 1) - (a + 1)*(b + 1))
                       if a==b+1 else 0 for b in range(dim)]
                      for a in range(dim)])

def Jx_mat_sym(s):
    Jm = Jm_mat_sym(s)
    return (Jm.H + Jm)/2

def Jy_mat_sym(s):
    Jm = Jm_mat_sym(s)
    return (Jm.H - Jm)/(2*sy.I)

def Jz_mat_sym(s):
    dim = int(2*s + 1)
    s = (dim - 1)/sy.S(2)
    return sy.Matrix([[s - a if a==b else 0 for b in range(dim)]
                      for a in range(dim)])

def represent_decomp(decomp, H_rep, S_rep):
    e_rep = np.eye(H_rep.shape[0], dtype=np.complex)
    def char_to_mat(char):
        if char == 'H':
            return H_rep
        elif char == 'S':
            return S_rep
        elif char == 'e':
            return e_rep
        else:
            raise ValueError
    return reduce(lambda x, y: char_to_mat(y) @ x, decomp, e_rep)

def quat_to_matrix(quat):
    Id = np.eye(2)
    sigx = 1.j*np.array([[0, 1], [1, 0]], dtype=np.complex)
    sigy = 1.j*np.array([[0, -1.j], [1.j, 0]], dtype=np.complex)
    sigz = 1.j*np.array([[1, 0], [0, -1]], dtype=np.complex)
    quat_norm = np.array(quat) / np.linalg.norm(np.array(quat))
    return sum(quat_comp*basis_el for quat_comp, basis_el in zip(quat_norm, [Id, sigx, sigy, sigz]))

lbls_orders_angles_quats = [('1a', 1, 0, (1, 0, 0, 0)),
                            ('4a', 12, sy.pi/2, (0, 0, 1, 1)),
                            ('3a', 8, 2*sy.pi/3, (-1, 1, 1, 1)),
                            ('4b', 6, sy.pi/2, (0, 1, 0, 0)),
                            ('2a', 1, sy.pi, (-1, 0, 0, 0)),
                            ('8a', 6, sy.pi/4, (1, 1, 0, 0)),
                            ('6a', 8, sy.pi/3, (1, 1, 1, 1)),
                            ('8b', 6, 3*sy.pi/4, (-1, 1, 0, 0))]

def make_restricted_rep(m):
    return [m + 1 if theta==0
            else (-1)**m*(m + 1) if theta==sy.pi
            else sy.simplify(sy.sin((m+1)*theta))/sy.sin(theta)
            for _, _, theta, _ in lbls_orders_angles_quats]

def quat_to_conj_class_idx(quat):
    if quat in [(1, 0, 0, 0)]:
        return 0
    if quat in [(0, 0, 1, 1),
                (0, 1, 0, 1),
                (0, 1, 1, 0),
                (0, 0, 1, -1),
                (0, -1, 0, 1),
                (0, 1, -1, 0),
                (0, 0, -1, 1),
                (0, 1, 0, -1),
                (0, -1, 1, 0),
                (0, 0, -1, -1),
                (0, -1, 0, -1),
                (0, -1, -1, 0)]:
        return 1
    if quat in [(-1, 1, 1, 1),
                (-1, 1, 1, -1),
                (-1, 1, -1, 1),
                (-1, 1, -1, -1),
                (-1, -1, 1, 1),
                (-1, -1, 1, -1),
                (-1, -1, -1, 1),
                (-1, -1, -1, -1)]:
        return 2
    if quat in [(0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
                (0, -1, 0, 0),
                (0, 0, -1, 0),
                (0, 0, 0, -1)]:
        return 3
    if quat in [(-1, 0, 0, 0)]:
        return 4
    if quat in [(1, 1, 0, 0),
                (1, 0, 1, 0),
                (1, 0, 0, 1),
                (1, -1, 0, 0),
                (1, 0, -1, 0),
                (1, 0, 0, -1)]:
        return 5
    if quat in [(1, 1, 1, 1),
                (1, 1, 1, -1),
                (1, 1, -1, 1),
                (1, 1, -1, -1),
                (1, -1, 1, 1),
                (1, -1, 1, -1),
                (1, -1, -1, 1),
                (1, -1, -1, -1)]:
        return 6
    if quat in [(-1, 1, 0, 0),
                (-1, 0, 1, 0),
                (-1, 0, 0, 1),
                (-1, -1, 0, 0),
                (-1, 0, -1, 0),
                (-1, 0, 0, -1)]:
        return 7

A = -sy.sqrt(2)
irreps = [[1, 1, 1, 1, 1, 1, 1, 1],
          [1, -1, 1, 1, 1, -1, 1, -1],
          [2, 0, -1, 2, 2, 0, -1, 0],
          [2, 0, -1, 0, -2, A, 1, -A],
          [2, 0, -1, 0, -2, -A, 1, A],
          [3, 1, 0, -1, 3, -1, 0, -1],
          [3, -1, 0, -1, 3, 1, 0, 1],
          [4, 0, 1, 0, -4, 0, -1, 0]]

def char_inner_prod(rep1, rep2):
    return sum([order*sy.conjugate(char1)*char2
                for (_, order, _, _), char1, char2 in zip(lbls_orders_angles_quats, rep1, rep2)])/48

def irrep_vec(rep):
    return [char_inner_prod(irrep, rep) for irrep in irreps]

def sym_quat_to_euler(quat):
    angles = quaternion.as_euler_angles(np.quaternion(*quat))
    return [sy.pi*int(4*angle/np.pi)/4 for angle in angles]

# Still need to use sympy for the half-integer spins
def rotation_matrix_sym(s, quat):
    alpha, beta, gamma = sym_quat_to_euler(quat)
    s = int(2*s)/sy.S(2)
    dim = int(2*s + 1)
    ms = [s - n for n in range(dim)]
    return sy.Matrix([[WignerD(s, ms[j], ms[k], alpha, beta, gamma).doit() for k in range(dim)]
                      for j in range(dim)])

bin_oct_quat_rep = [(1, 0, 0, 0),
                    (-1, 0, 0, 0),
                    (0, 1, 0, 0),
                    (0, 0, 1, 0),
                    (0, 0, 0, 1),
                    (0, -1, 0, 0),
                    (0, 0, -1, 0),
                    (0, 0, 0, -1),
                    (1, 1, 0, 0),
                    (1, 0, 1, 0),
                    (1, 0, 0, 1),
                    (1, -1, 0, 0),
                    (1, 0, -1, 0),
                    (1, 0, 0, -1),
                    (-1, 1, 0, 0),
                    (-1, 0, 1, 0),
                    (-1, 0, 0, 1),
                    (-1, -1, 0, 0),
                    (-1, 0, -1, 0),
                    (-1, 0, 0, -1),
                    (0, 0, 1, 1),
                    (0, 1, 0, 1),
                    (0, 1, 1, 0),
                    (0, 0, 1, -1),
                    (0, -1, 0, 1),
                    (0, 1, -1, 0),
                    (0, 0, -1, 1),
                    (0, 1, 0, -1),
                    (0, -1, 1, 0),
                    (0, 0, -1, -1),
                    (0, -1, 0, -1),
                    (0, -1, -1, 0),
                    (1, 1, 1, 1),
                    (1, 1, 1, -1),
                    (1, 1, -1, 1),
                    (1, 1, -1, -1),
                    (1, -1, 1, 1),
                    (1, -1, 1, -1),
                    (1, -1, -1, 1),
                    (1, -1, -1, -1),
                    (-1, 1, 1, 1),
                    (-1, 1, 1, -1),
                    (-1, 1, -1, 1),
                    (-1, 1, -1, -1),
                    (-1, -1, 1, 1),
                    (-1, -1, 1, -1),
                    (-1, -1, -1, 1),
                    (-1, -1, -1, -1)]

# This definition might be what makes me confused later on
H_rep_2 = (1.j/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
S_rep_2 = np.array([[np.exp(-1.j*np.pi/4), 0], [0, np.exp(1.j*np.pi/4)]])

quat_matrices = [quat_to_matrix(quat) for quat in bin_oct_quat_rep]

decomp_rep_2 = [represent_decomp(decomp, H_rep_2, S_rep_2) for decomp in qubit_Cliff_decomps]

decomp_quat_idxs = [np.argmin([np.linalg.norm(rep - quat_matrix)
                               for quat_matrix in quat_matrices])
                    for rep in decomp_rep_2]

decomp_conj_classes = [quat_to_conj_class_idx(bin_oct_quat_rep[idx]) for idx in decomp_quat_idxs]

def get_octahedral_group_rep_sym(s):
    s = int(2*s)/sy.S(2)
    return [rotation_matrix_sym(s, quat) for quat in bin_oct_quat_rep]

def make_irrep_proj(rep, irrep):
    c = sy.S(irrep[0])/48
    return c*sum([irrep[quat_to_conj_class_idx(quat)]*D for quat, D in zip(bin_oct_quat_rep, rep)],
                  sy.zeros(rep[0].shape[0]))

def cmplx_exp_matrix_simp(mat):
    rows, cols = mat.shape
    return sy.Matrix([[sy.simplify(mat[j,k].rewrite(sy.sin)) for k in range(cols)]
                      for j in range(rows)])

def get_logical_states(s, irrep):
    rep = get_octahedral_group_rep_sym(s)
    irrep_proj = cmplx_exp_matrix_simp(make_irrep_proj(rep, irrep))
    z = irrep_proj @ rep[7] @ irrep_proj
    z_evects = z.eigenvects()
    z_m = z_evects[1][2][0]
    z_m = z_m/z_m.norm()
    z_p = z_evects[2][2][0]
    z_p = z_p/z_p.norm()
    return z_p, z_m

def create_irrep_kets(s, irrep_char):
    ket0_sym, ket1_sym = get_logical_states(s, irrep_char)
    ket0 = np.array([complex(elem) for elem in ket0_sym])
    ket1 = np.array([complex(elem) for elem in ket1_sym])
    return ket0, ket1

def get_octahedral_rep_np(s):
    dim = int(np.round(2*s + 1))
    Jz = Jz_mat(s)
    Jx = Jx_mat(s)
    S_rep = np.diagflat(np.exp(-1.j*np.pi*np.diag(Jz)/2))
    # Rotation in the opposite direction from what I'd expect...
    H_rep = expm(1.j*np.pi*(Jx + Jz)/np.sqrt(2))
    return [represent_decomp(decomp, H_rep, S_rep) for decomp in qubit_Cliff_decomps]

def make_irrep_proj_np(rep, irrep):
    irrep_np = np.array([complex(char_entry) for char_entry in irrep])
    c = complex(irrep[0])/48
    return c*np.tensordot(irrep_np[decomp_conj_classes], np.array(rep), ([0], [0]))

def get_code_kets(ket0_a, ket0_b, theta, phi, sigx_rep):
    ket0 = np.cos(theta/2)*ket0_a + np.exp(1.j*phi)*np.sin(theta/2)*ket0_b
    ket1 = sigx_rep @ ket0
    return ket0, ket1

def get_rep_stuff(s, irrep_idx):
    oct_rep_s = get_octahedral_rep_np(s)
    oct_rep_s_irrep_proj = make_irrep_proj_np(oct_rep_s, irreps[irrep_idx])
    X_s = expm(-1.j*np.pi*Jx_mat(s))
    Z_s = expm(-1.j*np.pi*Jz_mat(s))
    Z_s_irrep = oct_rep_s_irrep_proj @ Z_s @ oct_rep_s_irrep_proj
    Z_s_irrep_eig = np.linalg.eigh(-1.j*Z_s_irrep)
    return {'oct-rep': oct_rep_s,
            'X': X_s,
            'Z': Z_s,
            'Z-irrep': Z_s_irrep,
            'Z-irrep-eig': Z_s_irrep_eig}

def get_irrep_kets(s, irrep_idx):
    rep_s = get_rep_stuff(s, irrep_idx)
    ket0_s_irrep = rep_s['Z-irrep-eig'][1][:,-1]
    ket1_s_irrep = rep_s['X'] @ ket0_s_irrep
    return ket0_s_irrep, ket1_s_irrep
