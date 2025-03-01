class RateController:
    """
    Create RateController class for injection rate control.

    The class provides methods for controlling the injection rate in a step rate test.

    Attributes
    --------
    step_dq: float
        The rate step in m^3/s. Sign convention is negative for injection.
    step_dt: float
        The step duration in seconds, assumed constant for all steps, but the control logic can also handle variable step duration.
    n_steps: int
        The number of steps in step rate test.
    min_dq: float
        The minimum rate step (in absolute value) in m^3/s.
    max_dq: float
        The maximum rate step (in absolute value) in m^3/s.
    q_inj: float
        The injection rate at the current step in m^3/s. Sign convention is negative for injection.
    q_inj_previous_step: float
        The injection rate at the previous step in m^3/s. Sign convention is negative for injection.
    step_start_time: float
        The current step start time in seconds, relative to the simulation start time.
    step_end_time: float
        The current step end time in seconds, relative to the simulation start time.
    step_index: int
        The current step index.
    return_to_previous_injection_step: boolean
        If this flag is True, the controller will use the injection rate from the previous step.

    Methods
    --------
    reduce_rate_step():
        Reduce the rate step by a multiplication factor.
    increase_rate_step():
        Increase the rate step by a multiplication factor.
    react_to_deviation():
        React to a deviation from the safe operating envelope.
    update_parameters_for_next_step():
        Update controller parameters for the next injection rate step.
    """

    def __init__(
            self,
            dq,
            dt,
            n_steps,
            min_dq=-400.0 / 3600 / 24,
            max_dq=-5000.0 / 3600 / 24
    ):
        self.step_dq = dq
        self.step_dt = dt
        self.n_steps = n_steps
        self.min_dq = min_dq
        self.max_dq = max_dq
        self.q_inj = dq
        self.q_inj_previous_step = 0
        self.step_start_time = 0
        self.step_end_time = dt
        self.step_index = 1
        self.return_to_previous_injection_step = False

    def reduce_rate_step(self, factor=0.5):
        """
        Reduce the rate step by a multiplication factor.

        This function applies a multiplication factor between 0 and 1 to reduce the current rate step.
        The rate step is limited to a pre-defined minimum value.

        Parameters
        --------
        factor: float
            Multiplication factor, should be between 0 and 1 (default is 0.5).
        """
        if self.step_dq < 0:
            self.step_dq = min(factor * self.step_dq, self.min_dq)
        else:
            self.step_dq = max(factor * self.step_dq, self.min_dq)

    def increase_rate_step(self, factor=1.5):
        """
        React to a deviation from the safe operating envelope.

        This function applies a multiplication factor greater than 1 to reduce the current rate step.
        The rate step is limited to a pre-defined maximum value.

        Parameters
        --------
        factor: float
            Multiplication factor, should be greater than 1 (default is 1.5).
        """
        if self.step_dq < 0:
            self.step_dq = max(factor * self.step_dq, self.max_dq)
        else:
            self.step_dq = min(factor * self.step_dq, self.max_dq)

    def react_to_deviation(self, t_deviation):
        """
        React to a deviation from the safe operating envelope.

        This function applies the following actions when a deviation from the safe operating envelope is detected:
            1. the injection rate is set to the rate from the previous step;
            2. the step start time is set to the time when the deviation is detected;
            3. the step end time is set to the new start time plus the step duration.

        Parameters
        --------
        t_deviation: float
            The time at which the deviation is detected, in seconds (relative to the simulation start time).
        """
        self.return_to_previous_injection_step = True
        self.q_inj = self.q_inj_previous_step
        self.step_start_time = t_deviation
        self.step_end_time = self.step_start_time + self.step_dt

    def update_parameters_for_next_step(self):
        """
        Update controller parameters for the next injection rate step.

        This function applies the following actions when a rate step is completed:
            1. the previous step injection rate is updated;
            2. the step index is incremented by 1;
            3. the step start and end times are updated for the next step;
            4. if the controller returned to the previous step as a result of deviation, the rate step and the
               injection rate are both reduced; otherwise (normal condition) the injection rate is increased.
        """
        self.q_inj_previous_step = self.q_inj
        self.step_index += 1
        self.step_start_time = self.step_end_time
        self.step_end_time = self.step_start_time + self.step_dt

        if self.return_to_previous_injection_step:
            self.return_to_previous_injection_step = False
            self.reduce_rate_step()
            self.q_inj -= self.step_dq
        else:
            self.q_inj += self.step_dq
