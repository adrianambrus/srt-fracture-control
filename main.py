import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle

from srt_fracture_control import ReservoirSimulator, PressureDeviationDetector, RateController
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression


def main():
    """
    Main function for testing the induced fracture monitoring and control algorithms with the control-oriented reservoir simulator.

    """
    # Define step rate test parameters
    step_dq = - 960.0 / 24 / 3600  # injection rate step, m3/s
    step_dt = 10 * 3600  # step duration, s
    n_steps = 10  # number of steps in step rate test
    min_dq = - 120.0 / 3600 / 24  # minimum rate step, m3/s

    # Set simulation maximum time and time step. The maximum simulation time should be at least twice the step rate test
    # total duration, to allow for extended steps when fractures are detected
    t_max = step_dt * n_steps * 2.0  # total simulation time, s
    dt = 10  # time step of main loop, s

    # Create ReservoirSimulator, RateController and PressureDeviationDetector objects
    sim = ReservoirSimulator(dt, t_max, use_pressure_dependent_permeability=True)
    controller = RateController(step_dq, step_dt, n_steps, min_dq)
    detector = PressureDeviationDetector(step_dq)

    # Plot settings
    show_plot = True
    plt.ion()
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))
    colors_linear_plots = cycle(ax[0, 0]._get_lines._cycler_items)

    # Main loop for the simulation
    index_step_start = 0
    for k in range(sim.nt):
        if controller.return_to_previous_injection_step == False and detector.check_for_deviations(
                controller.step_index, sim.t[index_step_start:k] / 3600.0,
                sim.p_bh[index_step_start:k] / 1e5,
                sim.q):
            controller.react_to_deviation(sim.t[k])

        if sim.t[k] >= controller.step_end_time:
            # when the step has been completed, update the detector internal variables
            detector.update_pressure_rate_history(sim.t[index_step_start:k] / 3600.0,
                                                  sim.q, sim.p_bh[index_step_start:k] / 1e5)

            detector.calculate_bourdet_derivative(detector.superposition_step_index,
                                                  sim.t[index_step_start:k] / 3600.0,
                                                  sim.q, sim.p_bh[index_step_start:k] / 1e5,
                                                  store_results=True)

            detector.update_safe_operating_envelope()
            # update control parameters
            controller.update_parameters_for_next_step()

            qs = []
            ps = []
            for i in range(len(detector.q_all_steps_pq)):
                qs.append(np.abs(detector.q_all_steps_pq[i]) * 3600 * 24)
                ps.append(np.abs(detector.p_all_steps_pq[i]))

            # fit a linear regression to the pressure-rate (p-q) plot
            if len(detector.q_all_steps_pq) == 2:
                reg = LinearRegression().fit(np.array(qs).reshape(-1, 1), np.array(ps))
            if len(detector.q_all_steps_pq) >= 2:
                pq_linear_reg = reg.predict(np.array(qs).reshape(-1, 1))

            if show_plot:
                colors_loglog_plot = cycle(ax[1, 1]._get_lines._cycler_items)
                for i in range(ax.shape[0]):
                    ax[i, 1].cla()

                for i in range(len(detector.dp_all_steps)):
                    c = next(colors_loglog_plot)['color']
                    t_loglog = detector.t_all_steps_loglog_plot[i]
                    dp = detector.dp_all_steps[i]
                    dp_der = detector.dp_der_all_steps[i]
                    idx_nan = np.argwhere(np.isnan(detector.dp_der_all_steps[0]))

                    ax[1, 1].loglog(t_loglog[idx_nan[1][0] + 1:idx_nan[2][0]], dp[idx_nan[1][0] + 1:idx_nan[2][0]],
                                    color=c,
                                    linestyle='None', marker='x', label='Step_' + str(i + 1))
                    ax[1, 1].loglog(t_loglog[idx_nan[1][0] + 1:idx_nan[2][0]], dp_der[idx_nan[1][0] + 1:idx_nan[2][0]],
                                    color=c, linestyle='None', marker='o', label='Step_' + str(i + 1))

                    ax[0, 1].plot(np.abs(detector.q_all_steps_pq[i]) * 3600 * 24, detector.p_all_steps_pq[i], color=c,
                                  linestyle='None', marker='o', label='Step_' + str(i + 1))

                ax[1, 1].plot(detector.dp_der_t_soe, detector.dp_der_upper_bound, 'k--', label='SOE Limits')
                ax[1, 1].plot(detector.dp_der_t_soe, detector.dp_der_lower_bound, 'k--')
                ax[1, 1].plot(detector.dp_der_t_ref, detector.dp_der_ref, 'k-', linewidth=2.0, label='Reference')

                h, l = ax[1, 1].get_legend_handles_labels()
                ph = [plt.plot([], marker="", ls="")[0]] * 2

                handles = ph[:1] + h[::2] + ph[1:] + h[1::2]
                labels = ["Pressure"] + l[::2] + ["Derivative"] + l[1::2]
                ax[1, 1].legend(handles, labels, ncol=2, fontsize=14)

                ax[1, 1].set_xlabel('Time [hr]')
                ax[1, 1].set_ylabel('Pressure and Derivative [bar]')
                ax[1, 1].set_ylim([1, 40])
                ax[1, 1].set_xlim([0.001, 10])
                ax[1, 1].grid(which='minor')
                ax[1, 1].grid(which='major')
                formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
                ax[1, 1].xaxis.set_major_formatter(formatter)
                ax[1, 1].yaxis.set_major_formatter(formatter)

                if len(qs) > 1:
                    ax[0, 1].plot(qs, pq_linear_reg, 'k')
                ax[0, 1].set_xlabel('Rate [$m^3$/D]')
                ax[0, 1].set_ylabel('Pressure [bar]')
                ax[0, 1].legend(fontsize=14)
                ax[0, 1].grid(visible=True)

                c1 = next(colors_linear_plots)['color']
                ax[1, 0].plot(sim.t[index_step_start:] / 3600, sim.q_hist[index_step_start:] * 3600 * 24,
                              color=c1, label='Step_' + str(len(detector.dp_all_steps)))
                ax[1, 0].set_ylabel('Rate [$m^3$/D]')
                ax[1, 0].set_xlabel('Elapsed Time [hr]')
                ax[1, 0].legend(fontsize=14)
                ax[1, 0].grid(visible=True)

                ax[0, 0].plot(sim.t[index_step_start:] / 3600, sim.p_bh[index_step_start:] / 1e5,
                              color=c1, label='Step_' + str(len(detector.dp_all_steps)))
                ax[0, 0].set_ylabel('Pressure [bar]')
                ax[0, 0].set_xlabel('Time [hr]')
                ax[0, 0].legend(fontsize=14)
                ax[0, 0].grid(visible=True)
                fig.canvas.draw()
                fig.tight_layout()
                fig.canvas.flush_events()

            index_step_start = k

            if controller.step_index > controller.n_steps:
                # write simulation results to files and exit main loop
                sim.export_pressure_and_rate_to_csv('PressureAndRateHistory')
                sim.export_pressure_and_rate_to_excel('PressureAndRateHistory')
                detector.export_pressure_and_derivative_to_excel('PressureAndDerivative')
                break

        # solve for the next time step
        sim.p[:, k + 1], sim.p_bh[k + 1] = sim.step(sim.p[:, k], controller.q_inj)
        sim.q_hist[k + 1] = sim.q

    # Plot the entire pressure and rate history from the simulation
    fig, ax = plt.subplots(2, 1, figsize=(18, 12))
    ax[0].plot(sim.t / 3600, sim.p_bh / 1e5, 'b')
    ax[1].plot(sim.t / 3600, sim.q_hist * 3600 * 24, 'b')
    ax[0].set_ylabel('Pressure (bar)')
    ax[1].set_ylabel('Rate ($m^3$/D)')
    ax[1].set_xlabel('Time (h)')
    ax[0].grid(visible=True)
    ax[1].grid(visible=True)


if __name__ == "__main__":
    main()
