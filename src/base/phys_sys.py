from math import factorial, sqrt, ceil, cos, sin
import numpy as np
from scipy import linalg
import warnings
from numpy.polynomial import hermite

warnings.simplefilter('ignore', np.RankWarning)

class OddRadiusError(Exception):
    def __init__(self):
        self.message = 'Number of radius points should be even to prevent Zero division by central point in future'

class Propagator():
    @staticmethod
    def a_s():
        return 5.29e-9
    @staticmethod
    def h():
        return 6.626070e-34

    @staticmethod
    def aem():
        return 1.660539e-27 #грамм

    @staticmethod
    def gmu():
        return 0.52 * 1.44 * 5.0507837e-27

    @staticmethod
    def sx():
        return 1*np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.complex64)

    @staticmethod
    def sy():
        return 1*np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=np.complex64)

    @staticmethod
    def sz():
        return 1*np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.complex64)

    @staticmethod
    def reshape_spin(sp):
        return np.reshape(sp, (3, 3, 1, 1, 1))
    # attention! coordinate notation (rho, theta, phi)
    # rho = 2R = diameter
    # nrho = number of points per diameter (not per radius)
    # nphi and ntheta = number of points per halfplane from 0 to pi
    # density = % of filled units
    def __init__(self, w=0, rho=0, nrho=0, ntheta=0, nphi=0,
                 b1=0, b2=0, b0=lambda x: 0, m=1, N=0, density=0.005, spin=0,
                 random_coord=False, random_spin=False, random_phase=False, deg=10):
        # nr -- сколько единичных отрезков входит в радиус
        self.spin_matrixes = [self.reshape_spin(self.sx()), self.reshape_spin(self.sy()), self.reshape_spin(self.sz())]
        if N == 0:
            N = density * nphi * ntheta * nrho
        self.N_points = N
        self.m = self.aem() * m  # float64
        self.dr = [rho / nrho, np.pi / ntheta, np.pi / nphi]
        if nrho % 2 == 1:
            raise OddRadiusError
        if nphi == 1:
            self.phi = np.zeros(1)
        else:
            self.phi = np.linspace(self.dr[2] / 2, np.pi - self.dr[2] / 2, nphi)
        self.phi = np.reshape(self.phi, (1, 1, -1))
        self.rho = np.reshape(np.linspace(-rho, rho, nrho), (-1, 1, 1))
        if ntheta == 1:
            self.theta = np.full(1, np.pi / 2)
        else:
            self.theta = np.linspace(self.dr[1] / 2, np.pi - self.dr[1] / 2, ntheta)
        self.theta = np.reshape(self.theta, (1, -1, 1))
        self.grid = np.array([nrho, ntheta, nphi])
        self.gridsize = nrho * ntheta * nphi
        self.psishape = tuple([3] + list(self.grid))
        rng = np.random.default_rng()
        if random_coord:
            if random_spin:
                self.psi = rng.random((3, nrho, ntheta, nphi)) + 1j * 0
                if random_phase:
                    self.psi += 1j * rng.random((3, nrho, ntheta, nphi))
            else:
                zero_psi = self.create_zerovect(np.complex64)
                if spin == 0:
                    psi_x = rng.random(self.grid) + 1j * 0
                    psi_y = rng.random(self.grid) + 1j * 0
                    if random_phase:
                        psi_x += 1j * rng.random(self.grid)
                        psi_y += 1j * rng.random(self.grid)
                    psi_z = zero_psi[2]
                else:
                    psi_x = zero_psi[0]
                    psi_y = zero_psi[1]
                    psi_z = spin * rng.random(self.grid) + 1j * 0
                self.psi = np.array([psi_x, psi_y, psi_z])
        else:
            self.fac = factorial(nrho)
            psispin = np.array(
                [[[self.solve_binom3d(r) for _ in range(nphi)] for _ in range(ntheta)] for r in range(nrho)])
            if random_spin:
                spins = 2*rng.random((3, 1, 1, 1)) - 1
                self.psi = psispin*spins
            else:
                zero_psi = self.create_zerovect(np.complex64)
                self.psi = np.array([psispin if i == spin else zero_psi[i] for i in range(3)])
            if random_phase:
                phase = 2*np.pi*rng.random()
                self.psi *= cos(phase) + 1j * sin(phase)
        self.normalize()
        self.w = w
        self.set_intersectors()
        self.x, self.y, self.z = self.toDSC()
        self.herm_deg = deg
        self.set_herm()
        self.b1 = b1
        self.b2 = b2
        self.b0 = b0

    def set_intersectors(self):
        rho = self.rho
        ntheta = self.grid[1]
        nphi = self.grid[2]
        self.volume = 4 * np.pi / 3 * np.abs(rho ** 3) # interior volume till exterior boarder
        difference = np.zeros(rho.shape)
        for i in range(1, rho.shape[0]):
            difference[i][0][0] = difference[i-1][0][0] + self.volume[i-1][0][0] # interior volume till interior boarder
        self.intersectors = self.volume - difference # volume bitween boarders
        self.intersectors /= nphi * ntheta * 4 # volume of one sector with coordinates (rho, theta, phi)
        self.sigma = 4*np.pi*self.h()**2*self.a_s()/self.m/self.intersectors

    def create_zerovect(self, dtype=np.float64):
        return np.zeros((3, self.grid[0], self.grid[1], self.grid[2]), dtype=dtype)

    def set_magfield(self, t):
        h = self.create_zerovect()
        x, y, z = self.x, self.y, self.z
        h[0] = self.b1*x
        h[1] = self.b1*y
        h[2] = self.b2*z+self.b0(t)
        return h

    def solve_binom3d(self, r):
        n = self.grid[0] - 1
        return 0 * 1j + sqrt(self.fac / (factorial(n - r) * factorial(r)))

    def eval_norm(self):
        norm = np.sum(np.square(np.absolute(self.psi)))
        assert norm > 0
        return norm

    def eval_interaction(self):
        inter = self.sigma*np.square(np.absolute(self.psi))
        zero = np.zeros(self.grid)
        inter_mat = np.array([[inter[0], zero, zero], [zero, inter[1], zero], [zero, zero, inter[2]]]) + 1j*0
        return inter_mat

    def eval_mag_pot(self, t):
        H = self.set_magfield(t)
        b_x = np.array([[H[0]]])
        b_y = np.array([[H[1]]])
        b_z = np.array([[H[2]]])
        mag_mat = b_x*self.spin_matrixes[0] + b_y*self.spin_matrixes[1] + b_z*self.spin_matrixes[2]
        return self.gmu()*mag_mat

    def propagate_pot(self, t, dt):
        mag_mat = self.eval_mag_pot(t)
        inter_mat = self.eval_interaction()
        swapped_pot_matrix = np.transpose(mag_mat+inter_mat, (2, 3, 4, 0, 1)) # (3, 3', nr, ntheta, nphi) -> (nr, ntheta, nphi, 3, 3')
        swapped_pot_matrix.__repr__()
        exp_mat = linalg.expm(-1j*dt*swapped_pot_matrix/self.h()) # last 2 axes matrix multiplication
        exp_mat = np.transpose(exp_mat, (3, 4, 0, 1, 2)) # vise versa
        psi = self.psi
        psi_x = exp_mat[0][0] * psi[0] + exp_mat[0][1] * psi[1] + exp_mat[0][2] * psi[2]
        psi_y = exp_mat[1][0] * psi[0] + exp_mat[1][1] * psi[1] + exp_mat[1][2] * psi[2]
        psi_z = exp_mat[2][0] * psi[0] + exp_mat[2][1] * psi[1] + exp_mat[2][2] * psi[2]
        self.psi = np.array([psi_x, psi_y, psi_z])

    def propagate_kin(self, dt):
        spec_psi = self.turn_real2spec(self.psi)
        spec_psi_dt = np.reshape(np.exp(-1j*dt*self.h()/self.m/2*self.lambdas), (-1, 1, 1, 1))*spec_psi
        self.psi = self.turn_spec2real(spec_psi_dt)

    def set_herm(self):
        gamma = sqrt(self.w*self.m/self.h())
        self.rho_sq = np.squeeze(self.rho)*gamma/sqrt(2)
        self.lambdas = gamma**2*np.array([1+2*i for i in range(self.herm_deg+1)])

    def turn_real2spec(self, psi):
        psi = np.transpose(psi, (1, 0, 2, 3))
        psi = np.reshape(psi, (self.grid[0], -1))
        spec_psi = hermite.hermfit(self.rho_sq, psi, deg=self.herm_deg)
        spec_psi = np.reshape(spec_psi, (self.herm_deg+1, 3, self.grid[1], self.grid[2]))
        return spec_psi

    def turn_spec2real(self, psi):
        real_psi = hermite.hermval(self.rho_sq, psi)
        real_psi = np.transpose(real_psi, (0, 3, 1, 2))
        return real_psi

    def normalize(self):
        self.psi *= sqrt(self.N_points / self.eval_norm())

    def __call__(self, t, dt):
        self.propagate_kin(dt/2)
        self.normalize()
        self.propagate_pot(t, dt)
        self.normalize()
        self.propagate_kin(dt/2)
        self.normalize()

    def toDSC(self):
        rho = self.rho
        theta = self.theta
        phi = self.phi
        sin_theta = np.sin(theta)
        x = rho * sin_theta * np.cos(phi)
        y = rho * sin_theta * np.sin(phi)
        z = rho * np.cos(theta)
        return x, y, z

class Simulation():
    def __init__(self, p, verb_level=1):
        self.verb = verb_level
        if verb_level > 0:
            print(f"Data: {p}")
        self.prop = Propagator(**p)
        self.N_points = self.prop.gridsize
        self.solutions = []
        self.t_evals = []

    def execute_propagation(self, t, dt, dt_eval=0):
        if dt_eval == 0:
            dt_eval = dt
        steps = ceil(t / dt)
        eval_steps = ceil(t / dt_eval)
        counter = 0
        timepoints = np.linspace(0, t, steps + 1)
        solution = np.empty([eval_steps] + list(self.prop.psishape), dtype=np.complex64)
        solution[counter] = self.prop.psi
        for i in range(steps):
            self.prop(timepoints[i + 1], dt)
            if timepoints[i] / dt_eval >= counter:
                counter += 1
                solution[counter] = self.prop.psi
        eval_points = np.linspace(0, t, eval_steps)
        self.t_evals.append(eval_points)
        self.solutions.append(solution)

    def eval_vorticity(self, psi):
        norm = np.sqrt(np.sum(np.absolute(psi) ** 2, axis=1, keepdims=True))
        psi = np.divide(np.absolute(psi), norm, out=np.zeros_like(psi), where=norm!=0)
        velocity = np.gradient(psi, axis=0)
        x, y, z = self.prop.toDSC()
        vel_x = velocity[:, 0, :, :, :]
        vel_y = velocity[:, 1, :, :, :]
        vel_z = velocity[:, 2, :, :, :]
        def grad(a, b, n):
            aa = np.array([np.gradient(a, axis=i) for i in range(1, n+1)])
            bb = np.expand_dims(np.array([np.gradient(b, axis=i) for i in range(n)]), axis=1)
            c = np.divide(aa, bb, out=np.zeros_like(aa), where=bb!=0)
            return np.sum(c, axis=0)
        vort_x = grad(vel_z, y, 3) - grad(vel_y, z, 2)
        vort_y = grad(vel_x, z, 2) - grad(vel_z, x, 3)
        vort_z = grad(vel_y, x, 3) - grad(vel_x, y, 3)
        vorticity = np.array([vort_x, vort_y, vort_z])
        return np.transpose(vorticity, (1, 0, 2, 3, 4))

class Magnetic_variable():
    def __init__(self, b0, w):
        self.b0 = b0
        self.v = -b0*w/20
    def update(self, t):
        return (self.b0 + t*self.v)