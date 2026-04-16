import numpy as np

_G       = 9.81
_L       = 1.0
_M       = 1.0
_DAMPING = 0.25
_MAX_TAU = 15.0


class ActiveInferenceController:
    def __init__(self, pi_s_theta: float = 30.0, pi_s_omega: float = 8.0, pi_p_theta: float = 12.0, pi_p_omega: float = 4.0, kappa_a_theta: float = 35.0, kappa_a_omega: float = 7.0, kappa_mu: float = 30.0, dt: float = 1.0 / 60.0, n_belief_substeps: int  = 10):
        self.pi_s = np.array([pi_s_theta, pi_s_omega])
        self.pi_p = np.array([pi_p_theta, pi_p_omega])
        self.kappa_a = np.array([kappa_a_theta, kappa_a_omega])
        self.kappa_mu = kappa_mu
        self.dt = dt
        self.n_substeps = n_belief_substeps
        self._sub_dt = dt / n_belief_substeps
        self.eta = np.array([0.0, 0.0])   # upright & still
        self.mu = np.array([0.0, 0.0])
	self._torque = 0.0
        self._eps_s = np.zeros(2)
        self._eps_p = np.zeros(2)
        self._eps_m = np.zeros(2)  # kept for HUD compatibility
        self._F = 0.0

    def _posterior(self, y: np.ndarray) -> np.ndarray:

        return (self.pi_s * y + self.pi_p * self.eta) / (self.pi_s + self.pi_p)

    def _update_beliefs_gd(self, y: np.ndarray):
        for _ in range(self.n_substeps):
            eps_s = y - self.mu
            eps_p = self.mu - self.eta
            dF_dmu = -self.pi_s * eps_s + self.pi_p * eps_p
            self.mu += (-self.kappa_mu * dF_dmu) * self._sub_dt

        # Wrappinf angle to [-π, π]
        self.mu[0] = (self.mu[0] + np.pi) % (2 * np.pi) - np.pi

    def _errors(self, y: np.ndarray, mu: np.ndarray):
        eps_s = y - mu           # sensory: observation − belief
        eps_p = mu - self.eta    # prior:   belief − preference
        return eps_s, eps_p

    def _free_energy(self, eps_s: np.ndarray, eps_p: np.ndarray) -> float:
        return float(0.5 * np.sum(self.pi_s * eps_s**2)+ 0.5 * np.sum(self.pi_p * eps_p**2))

    def _compute_torque(self, eps_p: np.ndarray) -> float:
        torque = -np.dot(self.kappa_a, eps_p)
        return float(np.clip(torque, -_MAX_TAU, _MAX_TAU))

    def action(self, theta: float, thetadot: float) -> float:

        y = np.array([theta, thetadot])
        # Computing optimal posterior beliefs
        mu_star = self._posterior(y)
        # Also run gradient-descent beliefs
        self._update_beliefs_gd(y)
        # Prediction errors compared to the posterior
        eps_s, eps_p = self._errors(y, mu_star)
        # 4. Cache diagnostics
        self._eps_s = eps_s.copy()
        self._eps_p = eps_p.copy()
        self._eps_m = np.zeros(2)
        self._F = self._free_energy(eps_s, eps_p)
        # 5. Compute torque from prior prediction error
        self._torque = self._compute_torque(eps_p)
        return self._torque

    def get_diagnostics(self) -> dict:

        return { "belief_angle": float(self.mu[0]), "belief_velocity": float(self.mu[1]), "eps_s": self._eps_s.copy(), "eps_p": self._eps_p.copy(),
         "eps_m": self._eps_m.copy(), "free_energy": self._F, "torque": self._torque, "pi_s": self.pi_s.copy(), "pi_p": self.pi_p.copy(), "pi_m": np.zeros(2) }

    def reset(self, theta0: float = 0.0):
        self.mu = np.array([theta0, 0.0])
        self._torque = 0.0
        self._eps_s = np.zeros(2)
        self._eps_p = np.zeros(2)
        self._eps_m = np.zeros(2)
        self._F = 0.0
