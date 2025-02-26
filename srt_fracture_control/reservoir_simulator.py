import numpy as np
import pandas as pd


class ReservoirSimulator:
    def __init__(self, dt, usePressureDependentPermeability=True):
        ## Set physical parameters
        self.k = 100 * 1e-15  # base permeability [m^2]
        self.phi = 0.3  # porosity [frac]
        self.mu = 0.00032  # fluid viscosity [Pa.s]
        self.C = 8.6e-10  # total compressibility [1/Pa]
        self.B = 1.03  # water formation volume factor [m^3/stm^3]
        self.alpha = self.k / (self.mu * self.phi * self.C)  # diffusivity coefficient
        self.p_res = 25000000.0  # far-field reservoir pressure [Pa]
        self.ri = 0.1  # wellbore radius [m]
        self.ro = 10000  # reservoir outer boundary [m]
        self.h = 15  # reservoir height [m]
        self.S = 0  # skin factor [-]
        self.Q_inj = 0  # injection rate [m^3/s]
        self.xf = 5  # fracture half-length [m]
        self.C_wb = 1.5e-7  # wellbore storage [m^3/Pa]
        self.fracture_opening_pressure = 380e5  # fracture opening pressure [Pa]

        ## Set simulation parameters
        self.T = 150 * 3600  # total simulation time [s]
        self.dt = dt  # time step of outer loop [s]
        self.dt_i = 1.0  # time step of inner loop used inside solver, only for pressure-dependent permeability case [s]
        self.t = np.arange(0, self.T + self.dt, self.dt)  # time array assuming uniform time step

        self.Nt = len(self.t)  # number of time points
        self.Nx = 100  # number of grid points

        self.useUniformGrid = False
        self.usePressureDependentPermeability = usePressureDependentPermeability
        self.useRateLimitation = False  # if set to True, apply rate limitation to get smoother numerical solution
        self.outerBoundaryCondition = 1  # 0 = no-flow (closed circle) outer boundary, 1 = constant pressure outer boundary

        if self.useUniformGrid:
            self.x = np.transpose(np.linspace(self.ri, self.ro, self.Nx))
            self.dx = (self.ro - self.ri) / (self.Nx - 1)  # grid spacing
        else:
            self.x = np.transpose(np.logspace(np.log10(self.ri), np.log10(self.ro), self.Nx))
            self.dx = self.x[1] - self.x[0]  # finest grid spacing

        self.idx_frac = next(
            x for x, val in enumerate(self.x) if val > self.xf)  # index of fracture half-length in spatial grid

        ## Initialize states
        self.P = np.full([self.Nx, self.Nt], np.nan)  # [Pa] reservoir pressure
        self.P_bh = np.ones((self.Nt, 1)) * np.nan  # [Pa] well bottom-hole pressure
        self.Q = np.ones((self.Nt, 1)) * np.nan  # [m^3/s] rate history

        self.P[:, 0] = np.ones((self.Nx, 1)).reshape(-1) * self.p_res
        self.P_bh[0] = self.p_res
        self.Q[0] = 0

    # Permeability as a function of pressure expressed as a multiplier for the
    # base permeability
    def permeability_function(self, p):
        # inputs: pressure, Pa
        # output: permeability multiplier, dimensionless
        k_ratio_min = 1.0
        k_ratio_max = 4.0
        a = 0.008
        y = np.maximum(np.minimum(np.exp(a * (p - self.fracture_opening_pressure) / 1e5), k_ratio_max), k_ratio_min)
        return y

    def solve_tdma(self, a, b, c, d):
        # inputs: NumPy arrays or Python lists a, b, c, d
        # output: the solution array x
        # The tridiagonal matrix algorithm (TDMA) solves the tridiagonal system of equations
        # a_i*x_{i-1} + b_i*x_i + c_i*x_{i+1} = d_i
        # http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

        n = len(d)  # number of equations to solve
        a_prime, b_prime, c_prime, d_prime = map(np.array, (a, b, c, d))
        for i in range(1, n):  # forward elimination
            m = a_prime[i] / b_prime[i - 1]
            b_prime[i] = b_prime[i] - m * c_prime[i - 1]
            d_prime[i] = d_prime[i] - m * d_prime[i - 1]

        x = b_prime.copy()
        x[-1] = d_prime[-1] / b_prime[-1]

        for j in range(n - 2, -1, -1):  # backward substitution
            x[j] = (d_prime[j] - c_prime[j] * x[j + 1]) / b_prime[j]

        return x

    def step(self, P0, dt):
        if self.usePressureDependentPermeability:
            return self.solve_one_step_pressure_dependent_permeability(P0, dt)
        else:
            return self.solve_one_step_constant_permeability(P0, dt)

    def solve_one_step_constant_permeability(self, P0, dt):
        ## Parse input
        Pres = P0

        A = np.zeros((self.Nx, self.Nx))
        for i in range(1, self.Nx - 1):
            A[i, i - 1] = 0.5 * self.alpha / self.x[i] / (self.x[i + 1] - self.x[i - 1]) - self.alpha / (
                    self.x[i] - self.x[i - 1]) / (self.x[i + 1] - self.x[i - 1])
            A[i, i] = 1.0 / dt + self.alpha / (self.x[i + 1] - self.x[i]) / (self.x[i] - self.x[i - 1])
            A[i, i + 1] = - 0.5 * self.alpha / self.x[i] / (self.x[i + 1] - self.x[i - 1]) - self.alpha / (
                    self.x[i + 1] - self.x[i]) / (self.x[i + 1] - self.x[i - 1])
        A[self.Nx - 1, self.Nx - 1] = 1

        if self.outerBoundaryCondition == 0:
            A[self.Nx - 1, self.Nx - 2] = - 1

        A[0, 0] = - 1 - self.dx * self.mu * self.C_wb * (1 - self.S * self.ri / self.dx) / (
                2 * np.pi * self.ri * self.h * self.k * dt)
        A[0, 1] = 1 - self.mu * self.C_wb * self.S / (2 * np.pi * self.h * self.k * dt)

        b = np.zeros((self.Nx, 1))
        b[0] = self.dx * self.mu / (2 * np.pi * self.ri * self.h * self.k * dt) * (
                self.Q_inj * self.B * dt - self.C_wb * (Pres[0] + self.S * self.ri / self.dx * (Pres[1] - Pres[0])))

        if self.outerBoundaryCondition == 1:
            b[self.Nx - 1] = self.p_res

        for i in range(1, self.Nx - 1):
            b[i] = (1.0 / dt - self.alpha / (self.x[i + 1] - self.x[i]) / (self.x[i] - self.x[i - 1])) * Pres[i] - A[
                i, i + 1] * Pres[i + 1] - A[i, i - 1] * Pres[i - 1]

        A_p = [A[i, i] for i in range(self.Nx)]
        A_w = [A[i, i - 1] for i in range(1, self.Nx)]
        A_e = [A[i, i + 1] for i in range(0, self.Nx - 1)]
        A_w = np.concatenate([[0], A_w])
        A_e = np.concatenate([A_e, [0]])
        Pres = self.solve_tdma(A_w, A_p, A_e, b)

        ## Parse output
        P = Pres.reshape(-1)
        P_bh = P[0] - self.S * b[0] * self.ri / self.dx

        return P, P_bh

    def solve_one_step_pressure_dependent_permeability(self, P0, dt_o):
        ## Parse input
        Pres = P0

        Nt = max(int(np.floor(dt_o / self.dt_i)), 1)
        dt = dt_o / Nt
        A = np.zeros((self.Nx, self.Nx))
        kP = np.ones(Pres.shape) * self.k

        # loop in time
        for n in range(Nt):
            kP[:self.idx_frac] = self.k * self.permeability_function(Pres[:self.idx_frac])

            for i in range(1, self.Nx - 1):
                A[i, i - 1] = 0.5 * kP[i] / (self.mu * self.phi * self.C) / self.x[i] / (
                        self.x[i + 1] - self.x[i - 1]) + 0.5 * (
                                      kP[i + 1] - kP[i - 1]) / (self.mu * self.phi * self.C) / (
                                      self.x[i + 1] - self.x[i - 1]) / (
                                      self.x[i + 1] - self.x[i - 1]) - kP[i] / (self.mu * self.phi * self.C) / (
                                      self.x[i] - self.x[i - 1]) / (self.x[i + 1] - self.x[i - 1])
                A[i, i] = 1.0 / dt + kP[i] / (self.mu * self.phi * self.C) / (self.x[i + 1] - self.x[i]) / (
                        self.x[i] - self.x[i - 1])
                A[i, i + 1] = - 0.5 * kP[i] / (self.mu * self.phi * self.C) / self.x[i] / (
                        self.x[i + 1] - self.x[i - 1]) - 0.5 * (kP[i + 1] - kP[i - 1]) / (
                                      self.mu * self.phi * self.C) / (
                                      self.x[i + 1] - self.x[i - 1]) / (self.x[i + 1] - self.x[i - 1]) - kP[i] / (
                                      self.mu * self.phi * self.C) / (self.x[i + 1] - self.x[i]) / (
                                      self.x[i + 1] - self.x[i - 1])
            A[self.Nx - 1, self.Nx - 1] = 1

            if self.outerBoundaryCondition == 0:
                A[self.Nx - 1, self.Nx - 2] = - 1

            A[0, 0] = - 1 - self.dx * self.mu * self.C_wb * (1 - self.S * self.ri / self.dx) / (
                    2 * np.pi * self.ri * self.h * kP[0] * dt)
            A[0, 1] = 1 - self.mu * self.C_wb * self.S / (2 * np.pi * self.h * kP[0] * dt)
            b = np.zeros((self.Nx, 1))
            b[0] = self.dx * self.mu / (2 * np.pi * self.ri * self.h * kP[0] * dt) * (
                    self.Q_inj * self.B * dt - self.C_wb * (
                    Pres[0] + self.S * self.ri / self.dx * (Pres[1] - Pres[0])))

            if self.outerBoundaryCondition == 1:
                b[self.Nx - 1] = self.p_res

            for i in range(1, self.Nx - 1):
                b[i] = (1.0 / dt - kP[i] / (self.mu * self.phi * self.C) / (self.x[i + 1] - self.x[i]) / (
                        self.x[i] - self.x[i - 1])) * Pres[i] - A[i, i + 1] * Pres[i + 1] - A[i, i - 1] * Pres[i - 1]

            A_p = [A[i, i] for i in range(self.Nx)]
            A_w = [A[i, i - 1] for i in range(1, self.Nx)]
            A_e = [A[i, i + 1] for i in range(0, self.Nx - 1)]
            A_w = np.concatenate([[0], A_w])
            A_e = np.concatenate([A_e, [0]])
            Pres = self.solve_tdma(A_w, A_p, A_e, b)

        ## Parse output
        P = Pres.reshape(-1)
        P_bh = P[0] - self.S * b[0] * self.ri / self.dx

        return P, P_bh

    def export_pressure_and_rate_to_csv(self, name):
        df = pd.DataFrame(
            {"Time (hr)": np.array(self.t) / 3600, "Pressure (bar)": np.array(self.P_bh.reshape(-1)) / 1e5,
             "Rate (m3/D)": np.array(self.Q.reshape(-1)) * 3600 * 24}).dropna()
        df.to_csv(f'{name}.csv', index=False)

    def export_pressure_and_rate_to_excel(self, name):
        idx_nan_values = np.argwhere(np.isnan(self.P_bh))
        df = pd.DataFrame(
            {"Time (hr)": np.array(self.t) / 3600, "Pressure (bar)": np.array(self.P_bh.reshape(-1)) / 1e5,
             "Rate (m3/D)": np.array(self.Q.reshape(-1)) * 3600 * 24}).dropna()
        df = df.replace('', np.NaN)

        writer = pd.ExcelWriter(f'{name}.xlsx', engine='openpyxl')
        df.to_excel(writer, index=False)
        writer.save()
