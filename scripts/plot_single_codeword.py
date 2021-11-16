import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from mayavi import mlab

from irrep_codes import irreps
import qinfo as qi
from qinfo import wigner, spin

from plot_wigner import setup_ico_triangulation


def plot_codeword(ket, fname):
    '''mlab.screenshot doesn't seem to work with qt, so to use this function I
    execute

        export ETS_TOOLKIT=wx

    and run the script using pythonw, which I got by installing python.app

    This had something to do with needing a framework build of python. See
    https://matplotlib.org/3.1.0/faq/osx_framework.html#conda

    '''
    plt.style.use('~/jupyter/research-notebooks/paper.mplstyle')

    d = ket.shape[0]
    s = (d - 1)/2
    rho = qi.rho_from_ket(ket)
    dim = ket.shape[0]

    tri = setup_ico_triangulation(s, 60)
    Ws = wigner.calc_spin_wigner(rho, tri['Deltas'])
    vmax = np.max(np.abs(Ws))

    fig = mlab.figure(size=(900, 900))
    mlab.triangular_mesh(*tri['norm-points'].T, tri['triangles'],
                         scalars=Ws, colormap='RdBu',
                         vmin=-vmax, vmax=vmax, figure=fig)
    cam = fig.scene.camera
    cam.zoom(1.7)
    # Begin fix I don't understand from
    # https://github.com/enthought/mayavi/issues/702#issuecomment-412546819
    f = mlab.gcf()
    f.scene._lift()
    # End fix I don't understand
    imgmap = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
    mlab.close(fig)

    fig2 = plt.figure(figsize=(1, 1), dpi=900)
    ax = fig2.add_subplot(111)
    ax.imshow(imgmap, interpolation='bilinear')
    ax.set_axis_off()
    norm = mpl.colors.Normalize(-vmax, vmax)
    cmap = mpl.cm.RdBu
    plt.tight_layout()
    plt.savefig(fname, transparent=True)

def main():
    ket0 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    ket1 = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    Jx_7_2 = irreps.Jx_mat(7/2)
    Rx_pi_2 = expm(-1.j * np.pi / 2 * Jx_7_2)
    ket_plus = Rx_pi_2 @ (ket0 + ket1)
    plot_codeword(ket_plus, 'maus-7-2.png')

if __name__ == '__main__':
    main()
