# srt-fracture-control

A Python library for automated monitoring and control of induced fracturing and fracture opening in injection wells using step rate tests and pressure transient analysis (PTA). The library provides methods for detection of induced fracturing and fracture opening based on PTA and safe operating envelope (SOE), and for control of injection rate to respond to detected deviations from the SOE. The library also contains a transient single-phase reservoir flow simulator with proxy modeling of fracture permeability, which can be used to test the fracture monitoring and control methods in synthetic well test scenarios. The library is based on the methodology described in the paper: [Automated Rate Control to Prevent Induced Fracturing and Fracture Opening Monitored with Step Rate Tests](https://doi.org/10.2118/220016-MS).

Usage examples provided in:

<a href="https://colab.research.google.com/drive/1ecY9KGa0Ndlg-13B04Ec6X9tE9UDHO_X?authuser=0#scrollTo=zmF804yggfjf"> Synthetic step rate test example <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a> <br>
<a href="https://colab.research.google.com/drive/1TJzfjieYJOt6c9E-krnyiutNYRJmGS8Q#scrollTo=AMap_Xh0nbax"> Synthetic step rate test example with induced fracture monitoring and control <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a> <br>

## Installation

Install the package using pip:

```bash
pip install srt-fracture-control
```

## Usage

### Reservoir simulator

- Solves the radial diffusivity equation for single-phase flow using the Crank-Nicolson finite difference method
- Takes the rate as input and returns reservoir pressure and well bottom-hole pressure as the output
- Models with constant and pressure-dependent reservoir permeability
- Results can be exported to Excel and CSV formats


```python
from srt_fracture_control import ReservoirSimulator
import matplotlib.pyplot as plt

# Set simulation maximum time and time step
t_max = 1e5 # total simulation time, s
dt = 10 # time step of main loop, s

# Create ReservoirSimulator object using default physical parameters
sim = ReservoirSimulator(dt, t_max)

# or with user-defined physical parameters (in SI units)
sim = ReservoirSimulator(dt, t_max, permeability, porosity, viscosity, compressibility, formation_volume_factor, reservoir_pressure, well_radius, outer_radius, thickness, skin, wellbore_storage, fracture_opening_pressure, fracture_half_length, use_pressure_dependent_permeability=True)

# Main loop for the simulation
for k in range(sim.nt):
    # Solve the radial diffusivity equation for the next time step with the input rate q
    sim.p[:, k + 1], sim.p_bh[k + 1] = sim.step(sim.p[:, k], q)
    # Update the rate history
    sim.q_hist[k + 1] = sim.q

# Plot results
fig, ax = plt.subplots(2, 1, figsize=(18, 12))
ax[0].plot(sim.t / 3600, sim.p_bh / 1e5, 'b')
ax[1].plot(sim.t / 3600, sim.q_hist * 3600 * 24, 'b')
ax[0].set_ylabel('Pressure (bar)')
ax[1].set_ylabel('Rate ($m^3$/D)')
ax[1].set_xlabel('Time (h)')

# Export results to csv file
sim.export_pressure_and_rate_to_csv('PressureAndRateHistory')
# or to Excel file
sim.export_pressure_and_rate_to_excel('PressureAndRateHistory')
```

### Pressure deviation detector
- Takes as input the well bottom-hole pressure and rate from each transient in the step rate test
- Calculates the Bourdet derivative for each pressure transient
- Assembles a safe operating envelope (SOE) based on a reference transient or from user-defined input
- Determines if the derivative falls outside the SOE
- Pressure and derivative for each transient can be exported to Excel format

```python
from srt_fracture_control import PressureDeviationDetector
import matplotlib.pyplot as plt

q_ref = - 960.0 / 24 / 3600  # reference rate, m^3/s
max_steps_reference_transient = 3
detection_interval_start = 1.0 # time in log-log plot when detection of deviations should start, hours
detection_margin = 0.2 # log scale margin for detection of pressure derivative deviation

# Create PressureDeviationDetector object
detector = PressureDeviationDetector(q_ref,max_steps_reference_transient,detection_interval_start,detection_margin)

# Analyze a pressure transient with time array t, rate q and pressure array p_bh
detector.update_pressure_rate_history(t / 3600.0, q, p_bh / 1e5)
detector.calculate_bourdet_derivative(detector.superposition_step_index, t / 3600.0, q, p_bh / 1e5, store_results=True)

# Update safe operating envelope using reference transient
detector.update_safe_operating_envelope()
# or from user-defined data with time array t_soe, lower_bound and upper_bound derivative arrays
detector.update_safe_operating_envelope(t_soe, lower_bound, upper_bound)

# Check for deviations of pressure derivative from safe operating envelope
deviation_detected = detector.check_for_deviations(step_index, t / 3600.0, p_bh / 1e5, q)

# Plot pressure and derivative in log-log plot
fig, ax = plt.subplots(1, 1, figsize=(18, 12))
for i in range(len(detector.dp_all_steps)):
    t_loglog = detector.t_all_steps_loglog_plot[i]
    dp = detector.dp_all_steps[i]
    dp_der = detector.dp_der_all_steps[i]
    ax.loglog(t_loglog, dp, linestyle='None', marker='x', label='Step_' + str(i + 1))
    ax.loglog(t_loglog, dp_der,linestyle='None', marker='o', label='Step_' + str(i + 1))
ax.set_xlabel('Time [hr]')
ax.set_ylabel('Pressure and Derivative [bar]')

# Export pressure and derivative results to Excel file
detector.export_pressure_and_derivative_to_excel('PressureAndDerivative')
```
### Rate controller
- Adjusts injection rate in a step rate test
- For normal case (derivative inside SOE), the rate is increased
- When fracturing is detected (derivative outside SOE), the rate is decreased

```python
from srt_fracture_control import RateController

# Define step rate test parameters
step_dq = - 960.0 / 24 / 3600  # injection rate step, m^3/s
step_dt = 10 * 3600  # step duration, s
n_steps = 10  # number of steps in step rate test
min_dq = - 120.0 / 3600 / 24  # minimum rate step, m^3/s

# Create RateController object
controller = RateController(step_dq, step_dt, n_steps, min_dq)

# Main loop for the simulation
for k in range(sim.nt):

    # if a deviation is detected, adjust the rate accordingly
    if controller.return_to_previous_injection_step == False and deviation_detected:
        controller.react_to_deviation(sim.t[k])

    # when the step has been completed, update the control parameters
    if sim.t[k] >= controller.step_end_time:
        controller.update_parameters_for_next_step()

        # when we reach the maximum number of steps in the step rate test, exit the main loop
        if controller.step_index > controller.n_steps:
            break
```

## Citation

If you use this library in your research, please cite:

```
@conference{ambrus2024automated,
  title={Automated Rate Control to Prevent Induced Fracturing and Fracture Opening Monitored with Step Rate Tests},
  author={Ambrus, A and Mugisha, J and Shchipanov, A and Aarsnes, UJF and {\O}verland, AM},
  booktitle={SPE Europec featured at EAGE Conference and Exhibition?},
  pages={D031S021R001},
  year={2024},
  organization={SPE}
}
```
## Acknowledgements

This research code was developed within the AutoWell research and development project funded by the Research Council of Norway and the industry partners including ConocoPhillips Skandinavia, Sumitomo Corporation Europe Norway Branch, Harbour Energy and Aker BP (grant no. 326580, PETROMAKS2 programme)​.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

