import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from irrep_codes import irreps
import jupyter_helper.visualization as viz
from qinfo import wigner, spin
from mayavi import mlab
import qinfo as qi
from scipy.linalg import expm

PRL_COLUMN_WIDTH = 3.404 # Width of a PRL column in inches

def setup_ico_triangulation(s, N):
    ico = viz.Icosahedron(N)
    points, triangles = ico.get_triangulation()
    normalized_points = points / np.linalg.norm(points, axis=1)[:, None]
    Delta_0 = wigner.Delta_nz(s)
    Jz = spin.Jz_mat(s)
    Jx = spin.Jx_mat(s)
    Rx_pi_2 = expm(-1.j*np.pi/2*Jx)
    Deltas = []
    thetas = np.arccos(normalized_points[:,2])
    phis = np.arctan2(normalized_points[:,1], normalized_points[:,0])
    for theta, phi in zip(thetas, phis):
        Deltas.append(wigner.make_Delta(phi, theta, Delta_0, Jz, Rx_pi_2))
    return {'Deltas': np.array(Deltas),
            'points': points,
            'norm-points': normalized_points,
            'thetas': thetas,
            'phis': phis,
            'triangles': triangles
           }

def plot_wigner(op):
    dim = op.shape[0]
    s = (dim - 1)/2
    tri = setup_ico_triangulation(s, 60)
    Ws = wigner.calc_spin_wigner(op, tri['Deltas'])
    surf = viz.plot_real_sph_fn(tri['norm-points'], tri['triangles'], Ws)
    return surf

def plot_code(ket0, ket1, fname):
    '''mlab.screenshot doesn't seem to work with qt, so to use this function I
    execute

        export ETS_TOOLKIT=wx

    and run the script using pythonw, which I got by installing python.app

    This had something to do with needing a framework build of python. See
    https://matplotlib.org/3.1.0/faq/osx_framework.html#conda

    '''
    plt.style.use('~/jupyter/research-notebooks/paper.mplstyle')

    d = ket0.shape[0]
    s = (d - 1)/2
    rho0 = qi.rho_from_ket(ket0)
    rho1 = qi.rho_from_ket(ket1)
    P = rho0 + rho1
    dim = ket0.shape[0]

    global_vmax = -np.inf
    W_arrays = []
    for op in [rho0, rho1, P]:
        tri = setup_ico_triangulation(s, 60)
        Ws = wigner.calc_spin_wigner(op, tri['Deltas'])
        vmax = np.max(np.abs(Ws))
        global_vmax = max(global_vmax, vmax)
        W_arrays.append(Ws)

    imgmaps = []
    for Ws in W_arrays:
        fig = mlab.figure(size=(900, 900))
        mlab.triangular_mesh(*tri['norm-points'].T, tri['triangles'],
                             scalars=Ws, colormap='RdBu',
                             vmin=-global_vmax, vmax=global_vmax, figure=fig)
        cam = fig.scene.camera
        cam.zoom(1.7)
        # Begin fix I don't understand from
        # https://github.com/enthought/mayavi/issues/702#issuecomment-412546819
        f = mlab.gcf()
        f.scene._lift()
        # End fix I don't understand
        imgmaps.append(mlab.screenshot(figure=fig, mode='rgba',
                                       antialiased=True))
        mlab.close(fig)

    fig2 = plt.figure(figsize=(PRL_COLUMN_WIDTH, 1), dpi=900)
    gs  = mpl.gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1/4])
    axs = [plt.subplot(gs[n]) for n in range(4)]
    for imgmap, ax in zip(imgmaps, axs[:3]):
        ax.imshow(imgmap, interpolation='bilinear')
        ax.set_axis_off()
    norm = mpl.colors.Normalize(-global_vmax, global_vmax)
    cmap = mpl.cm.RdBu
    cbar = fig2.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                         cax=axs[-1])
    plt.tight_layout()
    plt.savefig(fname, transparent=True)

def main():
    ket0 = np.array([np.sqrt(3/10), 0, 0, 0, 0, np.sqrt(7/10), 0, 0])
    ket1 = np.array([0, 0, np.sqrt(7/10), 0, 0, 0, 0, -np.sqrt(3/10)])
    plot_code(ket0, ket1, 'icosahedral-7-2.pdf')

if __name__ == '__main__':
    main()
