"""
Continuous-Time Active Inference Controller for Inverted Pendulum
=================================================================

Replaces PD control with variational free energy minimisation.

The agent maintains *beliefs* (μ) about the pendulum state and updates
them to minimise variational free energy F.  Actions (torques) are chosen
to minimise the same F — making the real world conform to the agent's
prior preferences (upright, still).

Theory
------
Under a Gaussian generative model with:
    - Observation model:  y ~ N(μ, Πₛ⁻¹)
    - Prior:              μ ~ N(η, Πₚ⁻¹)

the optimal posterior belief (minimising F) is:

    μ* = (Πₛ · y + Πₚ · η) / (Πₛ + Πₚ)

This is the precision-weighted Bayesian posterior.

The variational free energy decomposes into:
    F = ½ εₛᵀ Πₛ εₛ  +  ½ εₚᵀ Πₚ εₚ
where:
    εₛ = y - μ     (sensory prediction error)
    εₚ = μ - η     (prior prediction error)

Action minimises F by driving the world toward the prior preference:
    τ = -κ_θ · εₚ[θ]  -  κ_ω · εₚ[ω]

Under linear-Gaussian assumptions, this recovers PD control where:
    - kp ≈ κ_θ · Πₛ_θ / (Πₛ_θ + Πₚ_θ)
    - kd ≈ κ_ω · Πₛ_ω / (Πₛ_ω + Πₚ_ω)
but with explicit state estimation, adaptability, and principled tuning.

In addition to the steady-state posterior, we maintain gradient-descent
belief dynamics for visualisation:
    μ̇ = κ_μ · (Πₛ · εₛ  −  Πₚ · εₚ)

References
----------
- Friston (2010) "The free-energy principle: a unified brain theory?"
- Baltieri & Buckley (2019) "PID Control as Active Inference"
- Lanillos et al. (2021) "Active Inference in Robotics and AI"
"""

import numpy as np

# Physics constants — MUST match the plant in invert.py.
_G       = 9.81
_L       = 1.0
_M       = 1.0
_DAMPING = 0.25
_MAX_TAU = 15.0


class ActiveInferenceController:
    """
    Continuous-time active-inference controller for an inverted pendulum.

    Same interface as ``PDController``:
        torque = controller.action(theta, thetadot)

    Parameters
    ----------
    pi_s_theta, pi_s_omega : float
        Sensory precision on angle / velocity.
        Higher → trust observations more → agent's belief ≈ observation.
    pi_p_theta, pi_p_omega : float
        Prior precision on angle / velocity.
        Higher → stronger pull toward desired state (upright, still).
    kappa_a_theta, kappa_a_omega : float
        Action gains on angle / velocity error channels.
        These directly scale the torque response.
    kappa_mu : float
        Perception learning rate for gradient-descent belief dynamics.
    dt : float
        Simulation time-step.
    n_belief_substeps : int
        Euler substeps per frame for the belief gradient descent.
    """

    def __init__(
        self,
        pi_s_theta:       float = 30.0,
        pi_s_omega:       float = 8.0,
        pi_p_theta:       float = 12.0,
        pi_p_omega:       float = 4.0,
        kappa_a_theta:    float = 35.0,
        kappa_a_omega:    float = 7.0,
        kappa_mu:         float = 30.0,
        dt:               float = 1.0 / 60.0,
        n_belief_substeps: int  = 10,
    ):
        # ── Precisions (diagonal, stored as 2-vectors) ────────────────────
        self.pi_s = np.array([pi_s_theta, pi_s_omega])
        self.pi_p = np.array([pi_p_theta, pi_p_omega])

        # ── Action gains ──────────────────────────────────────────────────
        self.kappa_a = np.array([kappa_a_theta, kappa_a_omega])

        # ── Perception rate ───────────────────────────────────────────────
        self.kappa_mu   = kappa_mu
        self.dt         = dt
        self.n_substeps = n_belief_substeps
        self._sub_dt    = dt / n_belief_substeps

        # ── Prior preferences  η = [target angle, target velocity] ───────
        self.eta = np.array([0.0, 0.0])   # upright & still

        # ── Dynamic beliefs (gradient-descent trajectory for vis) ─────────
        self.mu = np.array([0.0, 0.0])

        # ── Diagnostics ───────────────────────────────────────────────────
        self._torque = 0.0
        self._eps_s  = np.zeros(2)
        self._eps_p  = np.zeros(2)
        self._eps_m  = np.zeros(2)  # kept for HUD compatibility
        self._F      = 0.0

    # ------------------------------------------------------------------
    #  Perception — Bayesian posterior (steady state of gradient descent)
    # ------------------------------------------------------------------
    def _posterior(self, y: np.ndarray) -> np.ndarray:
        """
        Optimal belief under the Gaussian generative model:

            μ* = (Πₛ · y + Πₚ · η) / (Πₛ + Πₚ)

        This is the precision-weighted average of observation and prior.
        """
        return (self.pi_s * y + self.pi_p * self.eta) / (self.pi_s + self.pi_p)

    # ------------------------------------------------------------------
    #  Perception — gradient descent dynamics (for visualisation)
    # ------------------------------------------------------------------
    def _update_beliefs_gd(self, y: np.ndarray):
        """
        Run gradient descent on F to update dynamic beliefs μ.
        These converge to _posterior(y) but the trajectory itself is
        interesting to visualise.

            μ̇ = κ_μ · (Πₛ · (y − μ) − Πₚ · (μ − η))
               = −κ_μ · ∂F/∂μ
        """
        for _ in range(self.n_substeps):
            eps_s = y - self.mu
            eps_p = self.mu - self.eta
            dF_dmu = -self.pi_s * eps_s + self.pi_p * eps_p
            self.mu += (-self.kappa_mu * dF_dmu) * self._sub_dt

        # Wrap angle to [-π, π]
        self.mu[0] = (self.mu[0] + np.pi) % (2 * np.pi) - np.pi

    # ------------------------------------------------------------------
    #  Prediction errors
    # ------------------------------------------------------------------
    def _errors(self, y: np.ndarray, mu: np.ndarray):
        """Compute sensory and prior prediction errors."""
        eps_s = y - mu           # sensory: observation − belief
        eps_p = mu - self.eta    # prior:   belief − preference
        return eps_s, eps_p

    # ------------------------------------------------------------------
    #  Free energy
    # ------------------------------------------------------------------
    def _free_energy(self, eps_s: np.ndarray, eps_p: np.ndarray) -> float:
        """Variational free energy (Laplace / Gaussian)."""
        return float(
            0.5 * np.sum(self.pi_s * eps_s**2)
          + 0.5 * np.sum(self.pi_p * eps_p**2)
        )

    # ------------------------------------------------------------------
    #  Action
    # ------------------------------------------------------------------
    def _compute_torque(self, eps_p: np.ndarray) -> float:
        """
        Action minimises F by changing what the agent will observe next.

        The prior prediction error εₚ = μ* − η represents how far the
        world (filtered through the posterior) is from the desired state.
        The torque is:

            τ = − kappa_a · εₚ

        This is structurally identical to PD control where:
            - kappa_a[0] acts on angle error   (proportional term)
            - kappa_a[1] acts on velocity error (derivative term)

        but the "error" is the *posterior* prediction error, not raw
        sensor error — incorporating the prior belief about the target.
        """
        torque = -np.dot(self.kappa_a, eps_p)
        return float(np.clip(torque, -_MAX_TAU, _MAX_TAU))

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------
    def action(self, theta: float, thetadot: float) -> float:
        """
        Full perception → action cycle.

        Parameters
        ----------
        theta    : float   observed angle  (rad)
        thetadot : float   observed angular velocity  (rad/s)

        Returns
        -------
        torque : float   control torque (Nm), clamped to ±MAX_TORQUE
        """
        y = np.array([theta, thetadot])

        # 1. Compute optimal posterior beliefs
        mu_star = self._posterior(y)

        # 2. Also run gradient-descent beliefs (for visualisation)
        self._update_beliefs_gd(y)

        # 3. Prediction errors w.r.t. the posterior
        eps_s, eps_p = self._errors(y, mu_star)

        # 4. Cache diagnostics
        self._eps_s = eps_s.copy()
        self._eps_p = eps_p.copy()
        self._eps_m = np.zeros(2)
        self._F     = self._free_energy(eps_s, eps_p)

        # 5. Compute torque from prior prediction error
        self._torque = self._compute_torque(eps_p)

        return self._torque

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------
    def get_diagnostics(self) -> dict:
        """Return internal state for HUD visualisation."""
        return {
            "belief_angle":    float(self.mu[0]),
            "belief_velocity": float(self.mu[1]),
            "eps_s":           self._eps_s.copy(),
            "eps_p":           self._eps_p.copy(),
            "eps_m":           self._eps_m.copy(),
            "free_energy":     self._F,
            "torque":          self._torque,
            "pi_s":            self.pi_s.copy(),
            "pi_p":            self.pi_p.copy(),
            "pi_m":            np.zeros(2),
        }

    def reset(self, theta0: float = 0.0):
        """Reset beliefs and action to match a new initial condition."""
        self.mu      = np.array([theta0, 0.0])
        self._torque = 0.0
        self._eps_s  = np.zeros(2)
        self._eps_p  = np.zeros(2)
        self._eps_m  = np.zeros(2)
        self._F      = 0.0
