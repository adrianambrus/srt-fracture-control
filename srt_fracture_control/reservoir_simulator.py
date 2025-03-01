import numpy as np
import pandas as pd


class ReservoirSimulator:
    """
    Create ReservoirSimulator class for transient, single-phase reservoir model with pressure-dependent permeability.

    The class provides methods for solving the pressure diffusivity equation for both constant and pressure-dependent
    permeability.

    Attributes
    --------
    k: float
        The reservoir matrix permeability in m^2.
    phi: float
        The reservoir porosity, defined as a fraction.
    mu: float
        The fluid viscosity in Pa.s.
    c: float
        The combined compressibility of the rock and fluid in Pa^-1.
    b: float
        The formation volume factor in m^3/stm^3.
    alpha: float
        The diffusivity coefficient in m^2/s.
    p_res: float
        The far-field reservoir pressure in Pa.
    ri: float
        The wellbore radius in m.
    ro: float
        The radial distance to the reservoir outer boundary in m.
    h: float
        The reservoir thickness in m.
    s: float
        The dimensionless skin factor.
    q: float
        The rate in m^3/s. Sign convention is negative for injection and positive for production.
    c_wb: float
        The wellbore storage constant in m^3/Pa.
    fracture_opening_pressure: float
        The fracture opening pressure in Pa.
    xf: float
        The fracture half-length in m.
    t_max: float
        The maximum simulation time in seconds.
    dt: float
        The time step of the solver outer loop in seconds.
    dt_i: float
        The time step of the solver inner loop in seconds.
    t: np.array
        1d array of simulation time values in seconds, where a constant time step equal to dt is used.
    nt: int
        The size of the simulation time array.
    nx: int
        The size of the simulation spatial grid.
    use_uniform_grid: boolean
        If this is True, a grid with uniform spacing is used, otherwise a logarithmic spacing is used.
    use_pressure_dependent_permeability: boolean
        If this is True, the simulator uses a pressure-dependent permeability function, otherwise uses constant permeability.
    use_rate_limitation: boolean
        If this is True, a low pass filter is applied on the rate command to get smoother numerical solution during rate changes.
    rate_low_pass_filter_constant: float
        The rate low-pass filter time constant in seconds.
    outer_boundary_condition: int
        The boundary condition type at the outer boundary: 0 = no-flow (closed circle), 1 = constant pressure (default).
    x: np.array
        1d array of the simulation spatial grid in m.
    dx: float
        Grid spacing in m. For logarithmic grid, this is the spacing near the wellbore.
    idx_frac: int
        Index of fracture half-length in the spatial grid.
    p: np.array
        2d array (space and time) of the simulated reservoir pressure in Pa.
    p_bh: np.array
        1d array of the simulated well bottom-hole pressure in Pa.
    q_hist: np.array
        1d array of the rate history in m^3/s. Sign convention is negative for injection and positive for production.

    Methods
    --------
    permeability_function():
        Compute the pressure-dependent permeability.
    solve_tdma():
        Solve a tridiagonal system of equations using the TDMA algorithm.
    step():
        Solve the pressure diffusivity equation for one time step.
    solve_one_step_constant_permeability():
        Solve the radial pressure diffusivity equation with constant permeability for one time step.
    solve_one_step_pressure_dependent_permeability():
        Solve the radial pressure diffusivity equation with pressure-dependent permeability for one time step.
    export_pressure_and_rate_to_csv():
        Export simulated well bottom-hole pressure and rate history to csv format.
    export_pressure_and_rate_to_excel():
        Export simulated well bottom-hole pressure and rate history to Excel format.
    """

    def __init__(self,
                 dt=10,
                 t_max=150 * 3600,
                 use_pressure_dependent_permeability=True
                 ):
        ## Set physical parameters
        self.k = 100 * 1e-15
        self.phi = 0.3
        self.mu = 0.00032
        self.c = 8.6e-10
        self.b = 1.03
        self.alpha = self.k / (self.mu * self.phi * self.c)
        self.p_res = 250e5
        self.ri = 0.1
        self.ro = 10000
        self.h = 15
        self.s = 0
        self.q = 0
        self.c_wb = 1.5e-7
        self.fracture_opening_pressure = 380e5
        self.xf = 5

        ## Set simulation parameters
        self.t_max = t_max
        self.dt = dt
        self.dt_i = 1.0
        self.t = np.arange(0, self.t_max + self.dt, self.dt)
        self.nt = len(self.t)
        self.nx = 100

        self.use_uniform_grid = False
        self.use_pressure_dependent_permeability = use_pressure_dependent_permeability
        self.use_rate_limitation = False
        self.rate_low_pass_filter_constant = self.dt / 20
        self.outer_boundary_condition = 1

        if self.use_uniform_grid:
            self.x = np.transpose(np.linspace(self.ri, self.ro, self.nx))
            self.dx = (self.ro - self.ri) / (self.nx - 1)  # constant grid spacing
        else:
            self.x = np.transpose(np.logspace(np.log10(self.ri), np.log10(self.ro), self.nx))
            self.dx = self.x[1] - self.x[0]  # grid spacing near the wellbore

        self.idx_frac = next(
            x for x, val in enumerate(self.x) if val > self.xf)

        self.p = np.full([self.nx, self.nt], np.nan)
        self.p_bh = np.full(self.nt, np.nan)
        self.q_hist = np.full(self.nt, np.nan)

        ## Initialize states
        self.p[:, 0] = np.ones((self.nx, 1)).reshape(-1) * self.p_res
        self.p_bh[0] = self.p_res
        self.q_hist[0] = 0

    def permeability_function(self, p):
        """
        Compute the pressure-dependent permeability.

        This function computes the reservoir permeability as a function of reservoir pressure, as a proxy-model
        for fracture permeability. It is defined as an exponential function with saturation, such that where the
        reservoir pressure is below a threshold, representing the fracture opening pressure, the permeability is equal to
        the matrix permeability. When it is above the fracture opening pressure, it increases exponentially, until some
        maximum ratio of fracture permeability to matrix permeability, which is defined inside this function.
        The output of this function is an array of permeability values with the same size as the pressure array.

        Parameters
        --------
        p: np.array
            1d array of the reservoir pressure for one simulation step in Pa.

        Returns
        --------
        kp: np.array
            1d array of reservoir permeability in m^2.
        """
        k_ratio_min = 1.0  # minimum permeability ratio
        k_ratio_max = 4.0  # maximum permeability ratio
        a = 0.008  # the exponent of the exponential function
        kp = self.k * np.maximum(np.minimum(np.exp(a * (p - self.fracture_opening_pressure) / 1e5), k_ratio_max),
                                 k_ratio_min)

        return kp

    def solve_tdma(self, a, b, c, d):
        """
        Solve a tridiagonal system of equations using the TDMA algorithm.

        This function uses the tridiagonal matrix algorithm (TDMA) to solve the system of linear equations:
            A*x=d

        where:
        - x is the solution vector.
        - A is the tridiagonal matrix split into the vectors a (lower diagonal), b (main diagonal) and c (upper diagonal).
        - d is the right-hand side vector.
        (http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm)

        Parameters
        --------
        a: np.array
            1d array containing the elements of the matrix lower diagonal.
        b: np.array
            1d array containing the elements of the matrix main diagonal.
        c: np.array
            1d array containing the elements of the matrix upper diagonal.
        d: np.array
            1d array of the right-hand side elements.

        Returns
        --------
        x: np.array
            1d array containing the solution of the system of linear equations.
        """
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

    def step(self, p0, q_c):
        """
        Solve the pressure diffusivity equation for one step.

        This function solves the pressure diffusivity equation for one time step. It takes the rate command as input and
        passes it to the solver. If the use_rate_limitation parameter is True, it applies a low-pass filter on the rate
        command before passing it to the solver. The function chooses which solver to use based on the value of the
        use_pressure_dependent_permeability parameter, and proceeds into the corresponding solver.

        Parameters
        --------
        p0: np.array
            1d array of the reservoir pressure from the previous time step in Pa.
        q_c: float
            The rate command in m^3/s. Sign convention is negative for injection and positive for production.

        """
        if self.use_rate_limitation:
            self.q = self.q * (1 - self.rate_low_pass_filter_constant) + self.rate_low_pass_filter_constant * q_c
        else:
            self.q = q_c

        if self.use_pressure_dependent_permeability:
            return self.solve_one_step_pressure_dependent_permeability(p0)
        else:
            return self.solve_one_step_constant_permeability(p0)

    def solve_one_step_constant_permeability(self, p0):
        """
        Solve the radial pressure diffusivity equation with constant permeability for one time step.

        This function solves the radial pressure diffusivity equation for one time step for constant reservoir permeability.
        The diffusivity equation is discretized using the Crank-Nicolson finite difference method.
        (https://en.wikipedia.org/wiki/Crank-Nicolson_method)

        This formulation results in a tridiagonal system of equations which is solved with the solve_tdma() method.

        Parameters
        --------
        p0: np.array
            1d array of the reservoir pressure from the previous time step in Pa.

        Returns
        --------
        p: np.array
            1d array of the reservoir pressure for the current time step in Pa.
        p_bh: float
            Well bottom-hole pressure for the current time step in Pa.
        """
        # Parse input
        p = p0

        # Define internal solver time step
        Nt = max(int(np.floor(self.dt / self.dt_i)), 1)
        dt = self.dt / Nt

        a_matrix = np.zeros((self.nx, self.nx))
        # Assemble matrix from finite difference discretization
        for i in range(1, self.nx - 1):
            a_matrix[i, i - 1] = 0.5 * self.alpha / self.x[i] / (self.x[i + 1] - self.x[i - 1]) - self.alpha / (
                    self.x[i] - self.x[i - 1]) / (self.x[i + 1] - self.x[i - 1])
            a_matrix[i, i] = 1.0 / dt + self.alpha / (self.x[i + 1] - self.x[i]) / (self.x[i] - self.x[i - 1])
            a_matrix[i, i + 1] = - 0.5 * self.alpha / self.x[i] / (self.x[i + 1] - self.x[i - 1]) - self.alpha / (
                    self.x[i + 1] - self.x[i]) / (self.x[i + 1] - self.x[i - 1])
        a_matrix[self.nx - 1, self.nx - 1] = 1

        if self.outer_boundary_condition == 0:
            a_matrix[self.nx - 1, self.nx - 2] = - 1

        a_matrix[0, 0] = - 1 - self.dx * self.mu * self.c_wb * (1 - self.s * self.ri / self.dx) / (
                2 * np.pi * self.ri * self.h * self.k * dt)
        a_matrix[0, 1] = 1 - self.mu * self.c_wb * self.s / (2 * np.pi * self.h * self.k * dt)

        # Solver inner loop
        for n in range(Nt):
            # assemble right-hand side terms
            b = np.zeros((self.nx, 1)).reshape(-1)
            b[0] = self.dx * self.mu / (2 * np.pi * self.ri * self.h * self.k * dt) * (
                    self.q * self.b * dt - self.c_wb * (p[0] + self.s * self.ri / self.dx * (p[1] - p[0])))

            if self.outer_boundary_condition == 1:
                b[self.nx - 1] = self.p_res

            for i in range(1, self.nx - 1):
                b[i] = (1.0 / dt - self.alpha / (self.x[i + 1] - self.x[i]) / (self.x[i] - self.x[i - 1])) * p[i] - \
                       a_matrix[
                           i, i + 1] * p[i + 1] - a_matrix[i, i - 1] * p[i - 1]

            # collect main, lower and upper diagonal from the system matrix
            a_p = np.array([a_matrix[i, i] for i in range(self.nx)])  # main diagonal
            a_w = [a_matrix[i, i - 1] for i in range(1, self.nx)]  # lower diagonal
            a_e = [a_matrix[i, i + 1] for i in range(0, self.nx - 1)]  # upper diagonal
            a_w = np.concatenate([[0], a_w])
            a_e = np.concatenate([a_e, [0]])
            p = self.solve_tdma(a_w, a_p, a_e, b)

        ## Parse output
        p = p.reshape(-1)
        p_bh = p[0] - self.s * b[0] * self.ri / self.dx

        return p, p_bh

    def solve_one_step_pressure_dependent_permeability(self, p0):
        """
        Solve the radial pressure diffusivity equation with pressure-dependent permeability for one time step.

        This function solves the radial pressure diffusivity equation for one time step for pressure-dependent reservoir
        permeability. The diffusivity equation is solved using the Crank-Nicolson finite difference method.
        (https://en.wikipedia.org/wiki/Crank-Nicolson_method)

        The permeability for grid cells in the fractured zone defined by the fracture half-length (xf) parameter is
        updated for each internal solver time step, while for the remaining grid cells, the matrix permeability is used.
        This formulation results in a tridiagonal system of equations which is solved with the solve_tdma() method.

        Parameters
        --------
        p0: np.array
            1d array of the reservoir pressure from the previous time step in Pa.

        Returns
        --------
        p: np.array
            1d array of the reservoir pressure for the current time step in Pa.
        p_bh: float
            Well bottom-hole pressure for the current time step in Pa.
        """
        # Parse input
        p = p0

        # Define internal solver time step
        Nt = max(int(np.floor(self.dt / self.dt_i)), 1)
        dt = self.dt / Nt

        a_matrix = np.zeros((self.nx, self.nx))
        kp = np.ones(p.shape) * self.k

        # Solver inner loop
        for n in range(Nt):
            # update permeability for grid cells in the fractured zone
            kp[:self.idx_frac] = self.permeability_function(p[:self.idx_frac])

            # assemble matrix from finite difference discretization
            for i in range(1, self.nx - 1):
                a_matrix[i, i - 1] = 0.5 * kp[i] / (self.mu * self.phi * self.c) / self.x[i] / (
                        self.x[i + 1] - self.x[i - 1]) + 0.5 * (
                                             kp[i + 1] - kp[i - 1]) / (self.mu * self.phi * self.c) / (
                                             self.x[i + 1] - self.x[i - 1]) / (
                                             self.x[i + 1] - self.x[i - 1]) - kp[i] / (self.mu * self.phi * self.c) / (
                                             self.x[i] - self.x[i - 1]) / (self.x[i + 1] - self.x[i - 1])
                a_matrix[i, i] = 1.0 / dt + kp[i] / (self.mu * self.phi * self.c) / (self.x[i + 1] - self.x[i]) / (
                        self.x[i] - self.x[i - 1])
                a_matrix[i, i + 1] = - 0.5 * kp[i] / (self.mu * self.phi * self.c) / self.x[i] / (
                        self.x[i + 1] - self.x[i - 1]) - 0.5 * (kp[i + 1] - kp[i - 1]) / (
                                             self.mu * self.phi * self.c) / (
                                             self.x[i + 1] - self.x[i - 1]) / (self.x[i + 1] - self.x[i - 1]) - kp[
                                         i] / (
                                             self.mu * self.phi * self.c) / (self.x[i + 1] - self.x[i]) / (
                                             self.x[i + 1] - self.x[i - 1])
            a_matrix[self.nx - 1, self.nx - 1] = 1

            if self.outer_boundary_condition == 0:
                a_matrix[self.nx - 1, self.nx - 2] = - 1

            a_matrix[0, 0] = - 1 - self.dx * self.mu * self.c_wb * (1 - self.s * self.ri / self.dx) / (
                    2 * np.pi * self.ri * self.h * kp[0] * dt)
            a_matrix[0, 1] = 1 - self.mu * self.c_wb * self.s / (2 * np.pi * self.h * kp[0] * dt)

            # assemble right-hand side terms
            b = np.zeros((self.nx, 1)).reshape(-1)
            b[0] = self.dx * self.mu / (2 * np.pi * self.ri * self.h * kp[0] * dt) * (
                    self.q * self.b * dt - self.c_wb * (
                    p[0] + self.s * self.ri / self.dx * (p[1] - p[0])))

            if self.outer_boundary_condition == 1:
                b[self.nx - 1] = self.p_res

            for i in range(1, self.nx - 1):
                b[i] = (1.0 / dt - kp[i] / (self.mu * self.phi * self.c) / (self.x[i + 1] - self.x[i]) / (
                        self.x[i] - self.x[i - 1])) * p[i] - a_matrix[i, i + 1] * p[i + 1] - a_matrix[i, i - 1] * p[
                           i - 1]

            # collect main, lower and upper diagonal from the system matrix
            a_p = np.array([a_matrix[i, i] for i in range(self.nx)])  # main diagonal
            a_w = [a_matrix[i, i - 1] for i in range(1, self.nx)]  # lower diagonal
            a_e = [a_matrix[i, i + 1] for i in range(0, self.nx - 1)]  # upper diagonal
            a_w = np.concatenate([[0], a_w])
            a_e = np.concatenate([a_e, [0]])
            p = self.solve_tdma(a_w, a_p, a_e, b)

        ## Parse output
        p = p.reshape(-1)
        p_bh = p[0] - self.s * b[0] * self.ri / self.dx

        return p, p_bh

    def export_pressure_and_rate_to_csv(self, name):
        """
        Export simulated well bottom-hole pressure and rate history to csv format.

        This function exports to a csv file the simulated well bottom-hole pressure and the rate history for
        each simulation time step.

        Parameters
        --------
        name: str
            File name without the format extension.
        """
        df = pd.DataFrame(
            {"Time (hr)": np.array(self.t) / 3600, "Pressure (bar)": np.array(self.p_bh.reshape(-1)) / 1e5,
             "Rate (m3/D)": np.array(self.q_hist.reshape(-1)) * 3600 * 24}).dropna()
        df.to_csv(f'{name}.csv', index=False)

    def export_pressure_and_rate_to_excel(self, name):
        """
        Export simulated well bottom-hole pressure and rate history to Excel format.

        This function exports to an Excel file the simulated well bottom-hole pressure and the rate history for
        each simulation time step.

        Parameters
        --------
        name: str
            File name without the format extension.
        """
        df = pd.DataFrame(
            {"Time (hr)": np.array(self.t) / 3600, "Pressure (bar)": np.array(self.p_bh.reshape(-1)) / 1e5,
             "Rate (m3/D)": np.array(self.q_hist.reshape(-1)) * 3600 * 24}).dropna()
        df = df.replace('', np.NaN)

        writer = pd.ExcelWriter(f'{name}.xlsx', engine='openpyxl')
        df.to_excel(writer, index=False)
        writer.save()
