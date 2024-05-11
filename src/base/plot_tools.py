import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from functools import partial

class Plot():
    def __init__(self, base_simulation, simulation_result, name_data, t, grid, showable=True):
        self.prop = base_simulation.prop
        self.grid = grid
        self.ws = simulation_result
        self.name = name_data
        self.spins = ['Spin x proj.', 'Spin y proj.', 'Spin z proj.']
        self.colors = ['r', 'g', 'b']
        self.colormaps = ['Reds', 'Greens', 'Blues']
        self.t = t
        self.showable = showable

    def plot_all(self):
        self.set_maxes()
        axes = ['rho', 'theta', 'phi']
        inds0 = [0, 1, 2]
        inds1 = [1, 0, 0]
        inds2 = [2, 2, 1]
        p0 = [self.max_coords[ind0] for ind0 in inds0]
        p1 = [self.max_coords[ind1] for ind1 in inds1]
        p2 = [self.max_coords[ind2] for ind2 in inds2]
        for i in range(3):
            self.plot_spins_along_axis(axes[i], p1[i], p2[i])
            for spin in range(3):
                self.plot_plane_spin(p0[i], axes[i], spin)
            self.plot_3d(i)
        self.spin_t_dep(*self.max_coords, self.t)


    def create_duration(self, start, stop, shape):
        if stop == -1:
            duration = shape[0] - 1
        else:
            duration = stop - start
        return duration

    def help_spins(self, frame, w, x0, ax0):
        ax0.set_ylabel(str(frame))
        wf = w[frame]
        for i in range(3):
            if self.wfs[i]:
                self.wfs[i].remove()
            self.wfs[i] = ax0.scatter(x0, wf[i], color=self.colors[i], label=self.spins[i])
        if frame == 0:
            plt.legend()

    def create_2p_mode(self, mode, p1, p2):
        if mode == 'theta':
            x0 = self.prop.theta
            w = self.ws[:, :, p1, :, p2]
        elif mode == 'phi':
            x0 = self.prop.phi
            w = self.ws[:, :, p1, p2, :]
        else:
            x0 = self.prop.rho
            w = self.ws[:, :, :, p1, p2]
        return x0, w

    def set_title(self, mode, p1, p2=-1):
        if p2 == -1:
            return 'where $\\' + mode + f'$ = {p1}'
        if mode == 'rho':
            return f'where $\\theta$ = {p1} and $\\phi$ = {p2}'
        if mode == 'theta':
            return f'where $\\rho$ = {-self.grid[0]//2+p1} and $\\phi$ = {p2}'
        return f'where $\\rho$ = {-self.grid[0]//2+p1} and $\\theta$ = {p2}'

    def set_labels_2d(self, mode):
        if mode == 'rho':
            return '$\\theta$', '$\\phi$'
        if mode == 'theta':
            return '$\\rho$', '$\\phi$'
        return '$\\rho$', '$\\theta$'

    def plot_spins_along_axis(self, mode, p1, p2, start=0, stop=-1):
        self.wfs = [None, None, None]
        fig0, ax0 = plt.subplots()
        x0, w = self.create_2p_mode(mode, p1, p2)
        duration = self.create_duration(start, stop, w.shape)
        ani = animation.FuncAnimation(fig0, partial(self.help_spins, ax0=ax0, x0=x0, w=w[start:stop]), duration)
        plt.title(self.name + ' along axis ' + self.set_title(mode, p1, p2))
        plt.xlabel('\\$'+mode+'$')
        plt.ylabel(self.name)
        plt.show()
        return ani

    def set_maxes(self):
        x = self.ws[-1]
        sh = x.shape
        d = [(x[i][j][k][m], (j, k, m)) for i in range(3) for j in range(sh[1]) for k in range(sh[2]) for m in range(sh[3])]
        maximum = max(d, key=lambda val: val[0])
        self.max_coords = maximum[1]

    def help_plane(self, frame, ax0, w, x0, y0, spin):
        ax0.set_ylabel(str(frame))
        c = np.reshape(w[frame], (-1))
        if self.wf:
            self.wf.remove()
        self.wf = ax0.scatter(x0, y0, cmap=self.colormaps[spin], c=c, s=10)

    def plot_plane_spin(self, param, mode, spin, start=0, stop=-1):
        self.wf = None
        x0, y0, z0 = self.prop.toDSC()
        fig0, ax0 = plt.subplots()
        if mode == 'theta':
            w = self.ws[:, spin, :, param, :]
            x0 = x0[:, param, :]
            y0 = y0[:, param, :]
        elif mode == 'phi':
            w = self.ws[:, spin, :, :, param]
            x0 = np.sqrt(x0[:, :, param]**2 + y0[:, :, param]**2)
            x0 = np.concatenate((-1*x0[:x0.shape[0]//2, :], x0[x0.shape[0]//2:, :]), axis=1)
            y0 = z0[:, :, 0]
        else:
            return
        duration = self.create_duration(start, stop, self.ws.shape)
        plt.title(self.spins[spin] + ' on plane ' + self.set_title(mode, param))
        xlabel, ylabel = self.set_labels_2d(mode)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        ani = animation.FuncAnimation(fig0, partial(self.help_plane, ax0=ax0, x0=x0, y0=y0, w=w[start:stop], spin=spin),
                                      duration)
        plt.show()
        return ani

    def help_3d(self, x, y, z, ax, fig, spin, frame):
        ax.set_zlabel(str(frame))
        wf = np.reshape(self.ws[frame][spin], (-1))
        if self.wframe:
            self.wframe.remove()
            self.wframe = ax.scatter(x, y, z, cmap=self.colormaps[spin], c=wf)
        else:
            self.wframe = ax.scatter(x, y, z, cmap=self.colormaps[spin], c=wf)
            fig.colorbar(self.wframe)

    def plot_3d(self, spin, start=0, stop=-1):
        self.wframe = None
        x, y, z = self.prop.toDSC()
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection="3d")
        plt.title('3D plot of ' + self.spins[spin])
        ax.set_ylabel('y')
        ax.set_xlabel('x')
        phis = np.ones((1, 1, self.prop.grid[2]))
        z = z*phis
        x = np.reshape(x, (-1))
        y = np.reshape(y, (-1))
        z = np.reshape(z, (-1))
        duration = self.create_duration(start, stop, self.ws.shape)
        upxyz = partial(self.help_3d, x, y, z, ax, fig, spin)
        ani = animation.FuncAnimation(fig, upxyz, duration)
        plt.show()
        return ani

    def plot_spin_t_dep(self, rho, theta, phi, t=1, start=0, stop=-1):
        fig, ax = plt.subplots()
        point_dinam = self.ws[:, :, rho, theta, phi]
        duration = self.create_duration(start, stop, point_dinam.shape)
        timepoints = np.linspace(0, t, duration+1)
        for i in range(3):
            ax.plot(timepoints, point_dinam[:, i], label=self.spins[i], color=self.colors[i])
        plt.xlabel('Time, s')
        plt.ylabel(self.name)
        plt.title(self.name + f' dinamics at point $\\rho$ = {-self.grid[0]//2+rho}, $\\theta$ = {theta}, $\\phi$ = {phi}')
        plt.legend()
        plt.show()

    def plot_param_t_dep(self, p1, p2, mode, spin, transp=False):
        #fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection="3d")
        x, point_dinam = self.create_2p_mode(mode, p1, p2)
        point_dinam = point_dinam[:, spin, :]
        timepoints = np.linspace(0, self.t, point_dinam.shape[0])
        if transp:
            timepoints, x = np.meshgrid(timepoints, x)
            ax.plot_surface(timepoints, x, point_dinam)
            point_dinam.transpose()
        else:
            x, timepoints = np.meshgrid(x, timepoints)
            ax.plot_surface(x, timepoints, point_dinam)
        namemodes = ['\\rho', '\\theta', '\\phi']
        first = False
        for i in range(3):
            if mode not in namemodes[i]:
                if first:
                    namemodes[i] = p1
                    first = False
                else:
                    namemodes[i] = p2
        plt.title(self.name + ', ' + self.spins[spin] + f' at $\\rho={namemodes[0]}$, $\\theta$={namemodes[1]}, $\\phi$={namemodes[2]}')
        ax.set_xlabel('$\\' + mode + '$, mcm')
        ax.set_ylabel('Time, s')
        ax.set_zlabel(self.name)
        plt.show()
