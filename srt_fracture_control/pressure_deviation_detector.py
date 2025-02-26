import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator as interpFunc


class PressureDeviationDetector:
    def __init__(self, Q_ref=-50.0 / 3600):
        self.t_start_allSteps = []
        self.Q_allSteps = []
        self.Q_allSteps_PQ = []
        self.P_allSteps_PQ = []
        self.P_allSteps_transient = []
        self.t_allSteps_loglog_plot = []
        self.dP_allSteps = []
        self.dP_der_allSteps = []
        self.dP_der_ref = []
        self.dP_der_lower_bound = []
        self.dP_der_upper_bound = []
        self.dP_der_t_ref = []
        self.dP_der_t_soe = []
        self.index_step_start = 0
        self.detection_interval_start = 1.0  # hours
        self.deviation_check_frequency = 0.1  # hours
        self.last_deviation_check_time = 0  # hours
        self.max_number_of_steps_for_averaging = 3
        self.weight_for_averaging = 0.5
        self.detection_margin = 0.2
        self.Q_ref = Q_ref
        self.Q_previous_step = 0
        self.deviation_detected = False
        self.detection_index = 0
        self.superposition_step_index = 0

    def update_pressure_rate_history(self, step, t, Q, P_bh):
        if self.deviation_detected == False:
            self.t_start_allSteps.append(t[0])
            self.Q_allSteps_PQ.append(np.float64(Q))
            self.P_allSteps_PQ.append(np.float64(P_bh[-1]))

        t = t - t[0]
        dQ = Q - self.Q_previous_step
        self.Q_previous_step = Q
        self.Q_allSteps.append(np.float64(Q))
        dP = np.abs((P_bh - P_bh[0]) / dQ * self.Q_ref)
        self.P_allSteps_transient.append(dP)
        self.superposition_step_index += 1

    def calculate_Bourdet_derivative(self, step, t, Q, P_bh, storeResults=False):

        Q_old = Q
        dP_der_vals = []
        t_vals = []
        T_s = np.zeros(t.shape)
        min_dx = 1.0e-3
        undefined_derivative = False

        # compute normalized pressure for the current step
        if step == 1:
            dQ = Q
        elif len(self.Q_allSteps) < step:
            dQ = Q - self.Q_allSteps[-1]
        else:
            dQ = Q - self.Q_allSteps[-2]

        dP = np.abs((P_bh - P_bh[0]) / dQ * self.Q_ref).reshape(-1)

        # compute superposition time
        if dQ > 0:
            if len(self.Q_allSteps) < step:
                Q = self.Q_allSteps[-1]
            else:
                Q = self.Q_allSteps[-2]

        for step_index in range(1, step + 1):
            if step_index == 1:
                T_s = np.add(T_s, (self.Q_allSteps[step_index - 1]) / Q * np.log10(t, where=t > 0))
            elif step_index == step:
                T_s = np.add(T_s,
                             (Q_old - self.Q_allSteps[step_index - 2]) / Q * np.log10(t - t[0], where=t - t[0] > 0))
            else:
                T_s = np.add(T_s, (self.Q_allSteps[step_index - 1] - self.Q_allSteps[step_index - 2]) / Q * np.log10(
                    t - self.t_start_allSteps[step_index - 1], where=t - self.t_start_allSteps[step_index - 1] > 0))

        # compute derivative
        for i in range(1, len(P_bh) - 2):
            dt = t - t[0]
            j = np.where(dt > dt[i] * np.exp(0.1))[0]
            if len(j) > 0:
                j = j[0]
                dP1 = dP[j] - dP[i]
                dX1 = T_s[j] - T_s[i]
            else:
                break

            k = np.where(dt > dt[j] * np.exp(0.1))[0]
            if len(k) > 0:
                k = k[0]
                dP2 = dP[k] - dP[j]
                dX2 = T_s[k] - T_s[j]
            else:
                break

            if np.abs(dX1) > min_dx and np.abs(dX2) > min_dx:
                dP_der = (dP1 / dX1 * dX2 + dP2 / dX2 * dX1) / (dX1 + dX2)
            else:
                undefined_derivative = True
                break

            dP_der = dQ / Q * dP_der / np.log(10)

            if not (np.isnan(dP_der)) and not ((t[j] - t[0]) in t_vals):
                t_vals.append(t[j] - t[0])
                dP_der_vals.append(dP_der)

        if len(t_vals) > 0:
            dP_der_func = interpFunc(t_vals, dP_der_vals, extrapolate=False)
            dP_der = dP_der_func(t - t[0])
        else:
            dP_der = np.ones((len(dP))) * np.nan

        if self.deviation_detected == False and storeResults:
            self.t_allSteps_loglog_plot.append(t - t[0])
            self.dP_allSteps.append(dP)
            self.dP_der_allSteps.append(dP_der)

            if step == 1:
                self.dP_der_ref = np.array(dP_der_vals)  # reference transient pressure derivative
                self.dP_der_t_ref = np.array(t_vals)  # time array for reference transient
            else:
                if undefined_derivative == False and step <= self.max_number_of_steps_for_averaging:
                    for k in range(len(self.dP_der_ref)):
                        if not (np.isnan(dP_der_func(self.dP_der_t_ref[k]))):
                            self.dP_der_ref[k] = self.dP_der_ref[k] * (1 - self.weight_for_averaging) + dP_der_func(
                                self.dP_der_t_ref[k]) * self.weight_for_averaging

        return dP_der

    def update_safe_operating_envelope(self, soe_loglog_time=None, soe_lower_bound=None, soe_upper_bound=None):
        if not (soe_loglog_time is None) and not (soe_lower_bound is None) and not (soe_upper_bound is None):
            self.dP_der_lower_bound = soe_lower_bound
            self.dP_der_upper_bound = soe_upper_bound
            self.dP_der_t_soe = soe_loglog_time
        else:
            if len(self.dP_der_ref) > 0:
                self.dP_der_lower_bound = np.exp(-self.detection_margin) * self.dP_der_ref
                self.dP_der_upper_bound = np.exp(self.detection_margin) * self.dP_der_ref
                self.dP_der_t_soe = self.dP_der_t_ref

        self.deviation_detected = False  # if a deviation was detected previously, we set the flag back to False here

    def check_for_deviations(self, step, t, P_bh, Q):

        if step > 1 and len(self.dP_der_t_soe) > 1:
            dP_der_lower_bound_func = interpFunc(self.dP_der_t_soe, self.dP_der_lower_bound, extrapolate=False)
            dP_der_upper_bound_func = interpFunc(self.dP_der_t_soe, self.dP_der_upper_bound, extrapolate=False)

            if len(t) > 0 and t[-1] - self.deviation_check_frequency > self.last_deviation_check_time:
                t0 = t[0]
                self.last_deviation_check_time = t[-1]
                t_loglog = t - t0
                if t_loglog[-1] > self.detection_interval_start:
                    dP_der = self.calculate_Bourdet_derivative(self.superposition_step_index + 1, t_loglog + t0, Q,
                                                               P_bh)
                    j = np.where(t_loglog > self.detection_interval_start)[0][0]
                    k = np.where(dP_der > 0)[0][-1]
                    dQ = Q - self.Q_previous_step
                    dP = np.abs((P_bh - P_bh[0]) / dQ * self.Q_ref)
                    index_interval_end = min(k, len(t_loglog))
                    index_interval_start = j

                    for i in range(index_interval_start, index_interval_end):
                        if (dP_der[i] < dP_der_lower_bound_func(t_loglog[i]) or dP_der[i] > dP_der_upper_bound_func(
                                t_loglog[i])):
                            print(
                                'Deviation from reference detected at injection step ' + str(step) + ',pressure=' + str(
                                    np.round(P_bh[i], 2)) + ' bar')
                            self.t_start_allSteps.append(t[0])
                            self.Q_previous_step = Q
                            self.Q_allSteps.append(np.float64(Q))
                            self.Q_allSteps_PQ.append(np.float64(Q))
                            self.P_allSteps_PQ.append(np.float64(P_bh[-1]))
                            self.t_allSteps_loglog_plot.append(t - t[0])
                            self.dP_allSteps.append(dP)
                            self.dP_der_allSteps.append(dP_der)

                            self.detection_index = i
                            self.superposition_step_index += 1
                            self.t_start_allSteps.append(t[-1])
                            self.deviation_detected = True

                            return True

        return False

    def export_loglog_pressure_and_derivative_to_excel(self, name):
        writer = pd.ExcelWriter(f'{name}.xlsx', engine='openpyxl')

        for i in range(len(self.t_allSteps_loglog_plot)):
            df = pd.DataFrame({"Elapsed Time (hr)": self.t_allSteps_loglog_plot[i].reshape(-1),
                               "Pressure (bar)": self.dP_allSteps[i].reshape(-1),
                               "Pressure derivative (bar)": self.dP_der_allSteps[i].reshape(-1)}).dropna()
            sheet_name = 'Step_' + str(i + 1)
            df.to_excel(writer, sheet_name=f'{sheet_name}', index=False)
        writer.save()
