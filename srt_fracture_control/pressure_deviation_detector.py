import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator as interp_func


class PressureDeviationDetector:
    """
    Create PressureDeviationDetector class for monitoring of induced fractures and fracture opening in a step rate test.

    The class provides methods for storing pressure and rate data from each transient in the step rate test,
    calculation of Bourdet derivative, constructing a safe operating envelope based on the pressure transient history,
    and monitoring for deviations from the safe operating envelope.

    Attributes
    --------
    t_start_all_steps: list
        List of start times for each transient in hours, relative to the simulation start time.
    q_all_steps: list
        List of rates for each transient in m^3/s, to be used for superposition calculations.
    q_all_steps_pq: list
        List of rates for each transient in m^3/s, to be used for p-q plot.
    p_all_steps_pq: list
        List of final pressures for each transient in bars, to be used for p-q plot.
    t_all_steps_loglog_plot: list
        List of time arrays for each transient in hours, relative to the step start time.
    dp_all_steps: list
        List of normalized pressure transient arrays for each step, in bars.
    dp_der_all_steps: list
        List of pressure derivative arrays for each transient, in bars.
    dp_der_ref: np.array
        1d array of pressure derivative for the reference transient, in bars.
    dp_der_lower_bound: np.array
        1d array of pressure derivative lower bound, in bars.
    dp_der_upper_bound: np.array
        1d array of pressure derivative upper bound, in bars.
    dp_der_t_ref:  np.array
        1d array of time values for the reference transient, in hours.
    dp_der_t_soe: np.array
        1d array of time values for the safe operating envelope, in hours.
    detection_interval_start: float
        Time interval relative to the start of a transient when detection of deviations should start, in hours.
    deviation_check_period: float
        Time period for checking deviations in hours.
    last_deviation_check_time: float
        Time relative to the start of a transient when the last deviation check occurred, in hours.
    max_number_of_steps_for_reference_transient: int
        Maximum number of steps in the step rate test used for computing the reference transient derivative.
    weight_for_averaging: float
        Weight used for averaging the pressure derivative for the latest transient with the reference derivative.
    detection_margin: float
        Log-scale margin used for deviation detection, relative to the reference transient.
    q_ref: float
        Reference rate in m^3/s used for normalization of pressure transients. Sign convention is negative for injection.
    q_previous_step: float
        Rate at the previous step in m^3/s.
    deviation_detected: boolean
        Flag indicating if a deviation from the safe operating envelope has been detected.
    superposition_step_index: int
        The current step index for the superposition time calculation.

    Methods
    --------
    update_pressure_rate_history():
        Store pressure and rate data from the current transient in the step rate test.
    calculate_bourdet_derivative():
        Calculate the Bourdet pressure derivative for the current transient in the step rate test.
    update_safe_operating_envelope():
        Update the safe operating envelope based on the reference transient or from user-defined input.
    check_for_deviations():
        Check for pressure derivative deviations from the safe operating envelope.
    export_pressure_and_derivative_to_excel():
        Export pressure and derivative data to Excel format.
    """

    def __init__(
            self,
            q_ref=-50.0 / 3600
    ):
        self.t_start_all_steps = []
        self.q_all_steps = []
        self.q_all_steps_pq = []
        self.p_all_steps_pq = []
        self.t_all_steps_loglog_plot = []
        self.dp_all_steps = []
        self.dp_der_all_steps = []
        self.dp_der_ref = []
        self.dp_der_lower_bound = []
        self.dp_der_upper_bound = []
        self.dp_der_t_ref = []
        self.dp_der_t_soe = []
        self.detection_interval_start = 1.0  # hours
        self.deviation_check_period = 0.1  # hours
        self.last_deviation_check_time = 0  # hours
        self.max_number_of_steps_for_reference_transient = 3
        self.weight_for_averaging = 0.5
        self.detection_margin = 0.2
        self.q_ref = q_ref
        self.q_previous_step = 0
        self.deviation_detected = False
        self.superposition_step_index = 0

    def update_pressure_rate_history(self, t, q, p_bh):
        """
        Store pressure and rate data from the current transient in the step rate test.

        This function updates the lists containing the rate and final well bottom-hole pressure for each transient in
        the step rate test, as long as no deviation in the pressure derivative was found in that transient.
        The time at the start of the step is also recorded, and the superposition step index is incremented by 1.
        This function should be called only when a step is completed.

        Parameters
        --------
        t: np.array
            1d array of time values for the current step, in hours.
        q: float
            Rate in m^3/s for the current step.
        p_bh: np.array
            1d array of well bottom-hole pressure for the current step, in bars.
        """
        if self.deviation_detected == False:
            self.t_start_all_steps.append(t[0])
            self.q_all_steps_pq.append(q)
            self.p_all_steps_pq.append(p_bh[-1])

        self.q_previous_step = q
        self.q_all_steps.append(q)
        self.superposition_step_index += 1

    def calculate_bourdet_derivative(self, superposition_step_index, t, q, p_bh, store_results=False):
        """
        Calculate the Bourdet pressure derivative for the current transient in the step rate test.

        This function first computes the superposition time based on the logarithmic time and the entire rate history.
        It then finds the pressure difference dp, normalized by rate.

        The local derivative is computed using a weighted average of pressure differences over the superposition time:
            dp_der = ((dp1 / dx1 * dx2 + dp2 / dx2 * dx1) / (dx1 + dx2))
        where:
        - dx1, dx2 are superposition time differences for the current and next interval.
        - dp1, dp2 are pressure differences for the current and next interval.

        The array of Bourdet derivatives is constructed for the entire transient, and if the store_results parameter
        is True, the array is stored together with the pressure difference and time points used in the calculation.

        Parameters
        --------
        superposition_step_index: int
            The current step index for the superposition time calculation.
        t: np.array
            1d array of time values for the current step, in hours.
        q: float
            Rate in m^3/s for the current step.
        p_bh: np.array
            1d array of well bottom-hole pressure for the current step, in bars.
        store_results: boolean
            Flag indicating whether to store the calculation results (default is False).

        Returns
        --------
        dp_der: np.array
            1d array of Bourdet pressure derivative in bars.
        """
        q_old = q
        dp_der_vals = []
        t_vals = []
        t_s = np.zeros(t.shape)
        min_dx = 1.0e-3  # minimum difference between two consecutive superposition times

        # Compute normalized pressure for the current step
        if superposition_step_index == 1:
            dq = q
        elif len(self.q_all_steps) < superposition_step_index:
            dq = q - self.q_all_steps[-1]
        else:
            dq = q - self.q_all_steps[-2]

        dp = np.abs((p_bh - p_bh[0]) / dq * self.q_ref).reshape(-1)

        # Compute superposition time
        if dq > 0:
            if len(self.q_all_steps) < superposition_step_index:
                q = self.q_all_steps[-1]
            else:
                q = self.q_all_steps[-2]

        for step_index in range(1, superposition_step_index + 1):
            if step_index == 1:
                t_s = np.add(t_s, (self.q_all_steps[step_index - 1]) / q * np.log10(t, where=t > 0))
            elif step_index == superposition_step_index:
                t_s = np.add(t_s,
                             (q_old - self.q_all_steps[step_index - 2]) / q * np.log10(t - t[0], where=t - t[0] > 0))
            else:
                t_s = np.add(t_s, (self.q_all_steps[step_index - 1] - self.q_all_steps[step_index - 2]) / q * np.log10(
                    t - self.t_start_all_steps[step_index - 1], where=t - self.t_start_all_steps[step_index - 1] > 0))

        # Compute derivative
        for i in range(1, len(p_bh) - 2):
            dt = t - t[0]
            # check if the first superposition time interval is longer than 0.1 log time units
            j = np.where(dt > dt[i] * np.exp(0.1))[0]
            if len(j) > 0:
                j = j[0]
                dp1 = dp[j] - dp[i]
                dx1 = t_s[j] - t_s[i]
            else:
                break

            # check if the second superposition time interval is longer than 0.1 log time units
            k = np.where(dt > dt[j] * np.exp(0.1))[0]
            if len(k) > 0:
                k = k[0]
                dp2 = dp[k] - dp[j]
                dx2 = t_s[k] - t_s[j]
            else:
                break

            if np.abs(dx1) > min_dx and np.abs(dx2) > min_dx:
                dp_der = (dp1 / dx1 * dx2 + dp2 / dx2 * dx1) / (dx1 + dx2)
            else:
                break

            dp_der = dq / q * dp_der / np.log(10)

            if not (np.isnan(dp_der)) and not ((t[j] - t[0]) in t_vals):
                t_vals.append(t[j] - t[0])
                dp_der_vals.append(dp_der)

        # From the calculated time points interpolate the pressure derivative for the entire time array
        if len(t_vals) > 0:
            dp_der_func = interp_func(t_vals, dp_der_vals, extrapolate=False)
            dp_der = dp_der_func(t - t[0])
        else:
            dp_der = np.ones((len(dp))) * np.nan

        # If no deviations have been detected for the current transient, store the time, pressure difference and derivative array
        if self.deviation_detected == False and store_results:
            self.t_all_steps_loglog_plot.append(t - t[0])
            self.dp_all_steps.append(dp)
            self.dp_der_all_steps.append(dp_der)

        return dp_der

    def update_safe_operating_envelope(self, dp_der_t_soe=None, dp_der_lower_bound=None, dp_der_upper_bound=None):
        """
        Update the safe operating envelope based on the reference transient or from user-defined input.

        This function first updates the reference transient pressure derivative based on the pressure transient history.
        The first transient in the step rate test is used as the initial reference transient, together with the
        corresponding pressure derivative and time points. The derivative is processed to remove NaN values.
        For subsequent steps, the reference derivative is recomputed by averaging the pressure derivative values
        for that step and the previous reference derivative. This is only done if no deviation from the safe operating
        envelope is found for that particular transient, and for a limited number of steps.
        Once the reference derivative has been updated, the detection margin is added symmetrically around it to obtain
        the lower and upper bounds of the safe operating envelope. The function allows the option to provide these
        bounds directly by the user (with the corresponding time points), in which case they will be used instead of
        the ones computed from the reference derivative.

        Parameters
        --------
        dp_der_t_soe : np.array
            1d array of time values for the safe operating envelope, in hours.
        dp_der_lower_bound : np.array
            1d array of pressure derivative lower bound, in bars.
        dp_der_upper_bound : np.array
            1d array of pressure derivative upper bound, in bars.
        """
        if len(self.dp_der_all_steps) == 0:
            raise AttributeError(
                'No pressure derivatives have been computed. Call calculate_bourdet_derivative() first.')
        if len(self.dp_der_all_steps) == 1:
            self.dp_der_ref = self.dp_der_all_steps[0]  # reference transient pressure derivative
            self.dp_der_t_ref = self.t_all_steps_loglog_plot[0]  # time array for reference transient
            idx_nan = np.isnan(self.dp_der_ref)
            self.dp_der_ref = self.dp_der_ref[~idx_nan]
            self.dp_der_t_ref = self.dp_der_t_ref[~idx_nan]
        elif len(
                self.dp_der_all_steps) <= self.max_number_of_steps_for_reference_transient and self.deviation_detected == False:
            idx_nan = np.isnan(self.dp_der_all_steps[-1])
            dp_der_func = interp_func(self.t_all_steps_loglog_plot[-1][~idx_nan],
                                      self.dp_der_all_steps[-1][~idx_nan],
                                      extrapolate=False)
            for k in range(len(self.dp_der_ref)):
                if not (np.isnan(dp_der_func(self.dp_der_t_ref[k]))):
                    self.dp_der_ref[k] = self.dp_der_ref[k] * (1 - self.weight_for_averaging) + dp_der_func(
                        self.dp_der_t_ref[k]) * self.weight_for_averaging

        if (dp_der_t_soe is None) or (dp_der_lower_bound is None) or (dp_der_upper_bound is None):
            if len(self.dp_der_ref) > 0:
                self.dp_der_lower_bound = np.exp(-self.detection_margin) * self.dp_der_ref
                self.dp_der_upper_bound = np.exp(self.detection_margin) * self.dp_der_ref
                self.dp_der_t_soe = self.dp_der_t_ref
        else:
            self.dp_der_lower_bound = dp_der_lower_bound
            self.dp_der_upper_bound = dp_der_upper_bound
            self.dp_der_t_soe = dp_der_t_soe

        self.deviation_detected = False  # if a deviation was detected previously, we set the flag back to False here

    def check_for_deviations(self, step_index, t, p_bh, q):
        """
        Check for pressure derivative deviations from the safe operating envelope.

        This function computes the Bourdet derivative array for the current pressure transient and compares each
        derivative after the detection start time to the upper and lower bounds of the safe operating envelope. If the
        derivative is outside those bounds for any value in the interval investigated, the deviation_detected flag is
        set to True. The pressure and derivative up to the time when the deviation is detected are stored in the lists
        containing the pressure transient data.

        Parameters
        --------
        step_index: int
            The current step index in the step rate test.
        t: np.array
            1d array of time values for the current step, in hours.
        q: float
            Rate in m^3/s for the current step.
        p_bh: np.array
            1d array of well bottom-hole pressure for the current step, in bars.

        Returns
        --------
        deviation_detected: boolean
            Flag indicating if a deviation from the safe operating envelope has been detected.
        """
        if step_index > 1:
            if len(self.dp_der_t_soe) > 1:
                # set up interpolation functions for the safe operating envelope bounds, since they may be evaluated at
                # different time points than the current pressure transient
                dp_der_lower_bound_func = interp_func(self.dp_der_t_soe, self.dp_der_lower_bound, extrapolate=False)
                dp_der_upper_bound_func = interp_func(self.dp_der_t_soe, self.dp_der_upper_bound, extrapolate=False)

                # continue only if the time elapsed since the previous check is longer than the detection check period
                if len(t) > 0 and t[-1] - self.deviation_check_period > self.last_deviation_check_time:
                    t0 = t[0]
                    self.last_deviation_check_time = t[-1]
                    t_loglog = t - t0

                    # check if the time interval is longer than the detection start time
                    if t_loglog[-1] > self.detection_interval_start:
                        dp_der = self.calculate_bourdet_derivative(self.superposition_step_index + 1, t_loglog + t0, q,
                                                                   p_bh)
                        j = np.where(t_loglog > self.detection_interval_start)[0][0]  # interval start index
                        k = np.where(dp_der > 0)[0][-1]  # find index of last valid pressure derivative
                        dq = q - self.q_previous_step
                        dp = np.abs((p_bh - p_bh[0]) / dq * self.q_ref)  # normalized pressure difference
                        index_interval_end = min(k, len(t_loglog))
                        index_interval_start = j

                        for i in range(index_interval_start, index_interval_end):
                            if (dp_der[i] < dp_der_lower_bound_func(t_loglog[i]) or dp_der[i] > dp_der_upper_bound_func(
                                    t_loglog[i])):
                                print(f'Deviation from safe operating envelope detected at injection step {step_index},'
                                      f' pressure={np.round(p_bh[i], 2)} bar')

                                # Update pressure and rate data from the current transient
                                self.t_start_all_steps.append(t[0])
                                self.q_previous_step = q
                                self.q_all_steps.append(q)
                                self.q_all_steps_pq.append(q)
                                self.p_all_steps_pq.append(p_bh[-1])
                                self.t_all_steps_loglog_plot.append(t - t[0])
                                self.dp_all_steps.append(dp)
                                self.dp_der_all_steps.append(dp_der)
                                self.superposition_step_index += 1
                                self.t_start_all_steps.append(t[-1])
                                self.deviation_detected = True

                                return self.deviation_detected
            else:
                raise AttributeError(
                    'No safe operating envelope has been defined. Call update_safe_operating_envelope() first.')

        return self.deviation_detected

    def export_pressure_and_derivative_to_excel(self, name):
        """
        Export pressure and derivative data to Excel format.

        This function exports to an Excel file the normalized pressure and derivative together with the time points
        used in the derivative calculation for each transient in the step rate test.

        Parameters
        --------
        name: str
            File name without the format extension.
        """
        writer = pd.ExcelWriter(f'{name}.xlsx', engine='openpyxl')

        for i in range(len(self.t_all_steps_loglog_plot)):
            df = pd.DataFrame({"Elapsed Time (hr)": self.t_all_steps_loglog_plot[i].reshape(-1),
                               "Pressure (bar)": self.dp_all_steps[i].reshape(-1),
                               "Pressure derivative (bar)": self.dp_der_all_steps[i].reshape(-1)}).dropna()
            sheet_name = 'Step_' + str(i + 1)
            df.to_excel(writer, sheet_name=f'{sheet_name}', index=False)
        writer.save()
