class RateController:
    def __init__(self, dQ, dT, N_steps, min_dQ=-400.0 / 3600 / 24, max_dQ=-5000.0 / 3600 / 24):
        self.step_dQ = dQ
        self.step_dT = dT
        self.N_steps = N_steps
        self.Q_inj_previous_step = 0
        self.step_start_time = 0
        self.step_end_time = dT
        self.step_index = 1
        self.return_to_previous_injection_step = False
        self.min_dQ = min_dQ
        self.max_dQ = max_dQ
        self.Q_inj = dQ

    def reduce_rate_step(self):
        # change the step in the injection rate
        if self.step_dQ < 0:
            self.step_dQ = min(0.5 * self.step_dQ, self.min_dQ)
        else:
            self.step_dQ = max(0.5 * self.step_dQ, self.min_dQ)

        return self.step_dQ

    def increase_rate_step(self):
        # change the step in the injection rate
        self.step_dQ = max(1.5 * self.step_dQ, self.max_dQ)

        return self.step_dQ

    def increase_injection_rate(self, step):
        self.Q_inj = self.Q_inj + step

    def decrease_injection_rate(self, step):
        self.Q_inj = self.Q_inj - step

    def react_to_deviation(self, t_deviation):
        self.return_to_previous_injection_step = True
        self.Q_inj = self.Q_inj_previous_step
        self.step_start_time = t_deviation
        self.step_end_time = self.step_start_time + self.step_dT

    def perform_new_injection_step(self):
        self.Q_inj_previous_step = self.Q_inj
        self.step_index += 1
        self.step_start_time = self.step_end_time
        self.step_end_time = self.step_start_time + self.step_dT

        if self.return_to_previous_injection_step:
            self.return_to_previous_injection_step = False
            self.reduce_rate_step()
            self.decrease_injection_rate(self.step_dQ)
        else:
            self.increase_injection_rate(self.step_dQ)
