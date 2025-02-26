import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from srt_fracture_control import ReservoirSimulator, PressureDeviationDetector, RateController
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import PchipInterpolator as interpFunc
from sklearn.linear_model import LinearRegression


def main():
    # Define step rate test parameters
    dQ = - 960.0 / 24 / 3600  # injection rate step, m3/s
    dT = 10  # step duration, h
    N = 10  # number of steps in step rate test
    dt = 10  # time step of main loop, s

    sim = ReservoirSimulator(dt, usePressureDependentPermeability=True)
    controller = RateController(dQ, dT * 3600, N, min_dQ=-120.0 / 3600 / 24)
    detector = PressureDeviationDetector(dQ)

    showPlot = True
    plt.ion()
    plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))

    # main loop for the simulation
    for k in range(sim.Nt):
        if controller.return_to_previous_injection_step == False and detector.check_for_deviations(
                controller.step_index, sim.t[
                                       detector.index_step_start:k] / 3600.0,
                sim.P_bh[
                detector.index_step_start:k] / 1e5,
                sim.Q[k - 1]):
            controller.react_to_deviation(sim.t[k])

        if sim.t[k] >= controller.step_end_time:
            # update pressure response for detection of deviations
            detector.update_pressure_rate_history(controller.step_index, sim.t[detector.index_step_start:k] / 3600.0,
                                                  sim.Q[k - 1], np.array(sim.P_bh[detector.index_step_start:k] / 1e5))

            detector.calculate_Bourdet_derivative(detector.superposition_step_index,
                                                  sim.t[detector.index_step_start:k] / 3600.0,
                                                  sim.Q[k - 1], np.array(sim.P_bh[detector.index_step_start:k] / 1e5),
                                                  storeResults=True)

            detector.update_safe_operating_envelope()
            controller.perform_new_injection_step()

            Qs = []
            Ps = []
            for i in range(len(detector.Q_allSteps_PQ)):
                Qs.append(np.abs(detector.Q_allSteps_PQ[i]) * 3600 * 24)
                Ps.append(np.abs(detector.P_allSteps_PQ[i]))

            # fit a linear regression to the pressure-rate (p-Q) plot
            if len(detector.Q_allSteps_PQ) == 2:
                reg = LinearRegression().fit(np.array(Qs).reshape(-1, 1), np.array(Ps))
            if len(detector.Q_allSteps_PQ) >= 2:
                pQ_linear_reg = reg.predict(np.array(Qs).reshape(-1, 1))

            if showPlot:
                for i in range(ax.shape[0]):
                    ax[i, 1].cla()

                for i in range(len(detector.dP_allSteps)):
                    c = next(ax[1, 1]._get_lines.prop_cycler)['color']
                    dt = detector.t_allSteps_loglog_plot[i]
                    dP = detector.dP_allSteps[i]
                    dP_der = detector.dP_der_allSteps[i]
                    idx_nan = np.argwhere(np.isnan(detector.dP_der_allSteps[0]))

                    ax[1, 1].loglog(dt[idx_nan[1][0] + 1:idx_nan[2][0]], dP[idx_nan[1][0] + 1:idx_nan[2][0]], color=c,
                                    linestyle='None', marker='x', label='Step_' + str(i + 1))
                    ax[1, 1].loglog(dt[idx_nan[1][0] + 1:idx_nan[2][0]], dP_der[idx_nan[1][0] + 1:idx_nan[2][0]],
                                    color=c, linestyle='None', marker='o', label='Step_' + str(i + 1))

                    ax[0, 1].plot(np.abs(detector.Q_allSteps_PQ[i]) * 3600 * 24, detector.P_allSteps_PQ[i], color=c,
                                  linestyle='None', marker='o', label='Step_' + str(i + 1))

                ax[1, 1].plot(detector.dP_der_t_soe, detector.dP_der_upper_bound, 'k--', label='SOE Limits')
                ax[1, 1].plot(detector.dP_der_t_soe, detector.dP_der_lower_bound, 'k--')
                ax[1, 1].plot(detector.dP_der_t_ref, detector.dP_der_ref, 'k-', linewidth=2.0, label='Reference')

                h, l = ax[1, 1].get_legend_handles_labels()
                ph = [plt.plot([], marker="", ls="")[0]] * 2

                handles = ph[:1] + h[::2] + ph[1:] + h[1::2]
                labels = ["Pressure"] + l[::2] + ["Derivative"] + l[1::2]
                leg = ax[1, 1].legend(handles, labels, ncol=2, fontsize=16)

                ax[1, 1].set_xlabel('Time [hr]')
                ax[1, 1].set_ylabel('Pressure and Derivative [bar]')
                ax[1, 1].set_ylim([1, 40])
                ax[1, 1].set_xlim([0.001, 10])
                ax[1, 1].grid(which='minor')
                ax[1, 1].grid(which='major')
                formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
                ax[1, 1].xaxis.set_major_formatter(formatter)
                ax[1, 1].yaxis.set_major_formatter(formatter)

                if len(Qs) > 1:
                    ax[0, 1].plot(Qs, pQ_linear_reg, 'k')
                ax[0, 1].set_xlabel('Rate [$m^3$/D]')
                ax[0, 1].set_ylabel('Pressure [bar]')
                ax[0, 1].legend(fontsize=16)
                ax[0, 1].grid(visible=True)

                c1 = next(ax[0, 0]._get_lines.prop_cycler)['color']
                ax[1, 0].plot(sim.t[detector.index_step_start:] / 3600, sim.Q[detector.index_step_start:] * 3600 * 24,
                              color=c1, label='Step_' + str(i + 1))
                ax[1, 0].set_ylabel('Rate [$m^3$/D]')
                ax[1, 0].set_xlabel('Elapsed Time [hr]')
                ax[1, 0].legend(fontsize=16)
                ax[1, 0].grid(visible=True)

                ax[0, 0].plot(sim.t[detector.index_step_start:] / 3600, sim.P_bh[detector.index_step_start:] / 1e5,
                              color=c1, label='Step_' + str(i + 1))
                ax[0, 0].set_ylabel('Pressure [bar]')
                ax[0, 0].set_xlabel('Time [hr]')
                ax[0, 0].legend(fontsize=16)
                ax[0, 0].grid(visible=True)
                fig.canvas.draw()
                fig.tight_layout()
                fig.canvas.flush_events()

            detector.index_step_start = k

            if controller.step_index > controller.N_steps:
                # write simulation results to files and exit main loop
                sim.export_pressure_and_rate_to_csv('PressureAndRateHistory')
                sim.export_pressure_and_rate_to_excel('PressureAndRateHistory')
                detector.export_loglog_pressure_and_derivative_to_excel('PressureAndDerivative')
                break

        if sim.useRateLimitation:
            sim.Q_inj = sim.Q_inj + sim.dt / 10 * (controller.Q_inj - sim.Q_inj)
        else:
            sim.Q_inj = controller.Q_inj

        sim.Q[k + 1] = sim.Q_inj
        # solve for the next time step
        sim.P[:, k + 1], sim.P_bh[k + 1] = sim.step(sim.P[:, k], dt=sim.dt)

    fig, ax = plt.subplots(2, 1, figsize=(18, 12))
    ax[0].plot(sim.t / 3600, sim.P_bh / 1e5, 'b')
    ax[1].plot(sim.t / 3600, sim.Q * 3600 * 24, 'b')
    ax[0].set_ylabel('Pressure (bar)')
    ax[1].set_ylabel('Rate ($m^3$/D)')
    ax[1].set_xlabel('Time (h)')


if __name__ == "__main__":
    main()
