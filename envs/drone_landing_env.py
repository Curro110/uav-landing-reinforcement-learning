# ============================================================
# envs/drone_landing_env.py
# Entorno Gymnasium para aterrizaje de UAV (Control No Lineal)
# Versiones soportadas: 'ideal', 'real', 'extended'
# ============================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from configs.drone_config import DroneConfig as DC
from utils.quaternion_utils import quat_to_euler, euler_to_quat
from utils.physics import simulate_step

class DroneLandingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    MAX_VEL = 25.0
    MAX_OMEGA = 12.0
    MAX_POS = 30.0

    def __init__(self, render_mode=None, env_version="extended"):
        super().__init__()
        self.render_mode = render_mode
        self.env_version = env_version

        obs_high = np.array([
            25.0, 25.0, 35.0, np.pi, np.pi/2, np.pi,
            25.0, 25.0, 25.0, 12.0, 12.0, 12.0
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(DC.ACT_DIM,), dtype=np.float32)

        self.state = np.zeros(13, dtype=np.float64)
        self.steps = 0
        
        self.motor_alpha = DC.MOTOR_ALPHA
        self.prev_potential = 0.0
        self.prev_dist = 0.0 
        
        # Variables para forzar posicionamiento en validacion
        self.force_x = None
        self.force_y = None
        self.force_z = None

        # Configuracion de version de entorno
        self.total_landings = 0
        self.curriculum_phase = 0
        
        self._configure_environment_version()

        # Fisicas base
        self.gravity = DC.GRAVITY
        self.arm_length = DC.ARM_LENGTH
        self.torque_coeff = DC.TORQUE_COEFF
        self.inertia = DC.INERTIA.copy()
        self.f_min = DC.F_MIN
        self.f_max = DC.F_MAX
        self.dt = DC.DT
        self.substeps = DC.SUBSTEPS
        self.max_steps = DC.MAX_STEPS
        
        self.current_mass = DC.MASS
        self.wind_force = np.zeros(3, dtype=np.float64)

    def _configure_environment_version(self):
        """Configura el curriculum y Domain Randomization segun la version."""
        if self.env_version == "ideal":
            self.phases = [(-3.5, -2.5, 0.5), (-5.0, -3.0, 1.0), (-8.0, -5.0, 1.5)]
            self.phase_thresholds = [20, 50]
            self.dr_enabled = False
        elif self.env_version == "real":
            self.phases = [
                (-3.5, -2.5, 0.5), (-5.0, -3.0, 1.0), 
                (-8.0, -5.0, 1.5), (-12.0, -5.0, 2.0), (-15.0, -8.0, 3.0)
            ]
            self.phase_thresholds = [20, 50, 100, 200]
            self.dr_enabled = True
        elif self.env_version == "extended":
            self.phases = [
                (-3.5, -2.5, 0.5), (-5.0, -3.0, 1.0), (-8.0, -5.0, 1.5),
                (-12.0, -5.0, 2.0), (-15.0, -8.0, 3.0), 
                (-15.0, -10.0, 6.0), (-15.0, -10.0, 10.0)
            ]
            self.phase_thresholds = [20, 50, 100, 200, 300, 450]
            self.dr_enabled = True
        else:
            raise ValueError(f"Version de entorno '{self.env_version}' no reconocida.")

    def _update_curriculum(self):
        for i, threshold in enumerate(self.phase_thresholds):
            if self.total_landings >= threshold:
                new_phase = i + 1
                if new_phase > self.curriculum_phase:
                    self.curriculum_phase = new_phase
                    print(f"[CURRICULUM] Avanzando a la Fase {self.curriculum_phase}")

    def _compute_potential(self, pos, vel, phi, theta):
        dist = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        return (-DC.W_DIST * dist - DC.W_VEL * np.linalg.norm(vel) - DC.W_TILT * (phi**2 + theta**2))

    def _apply_motor_lag(self, commanded):
        self.current_actions = (self.motor_alpha * commanded + (1.0 - self.motor_alpha) * self.current_actions)
        return self.current_actions.copy()

    def _clamp_state(self):
        vel = self.state[7:10]
        vel_norm = np.linalg.norm(vel)
        if vel_norm > self.MAX_VEL: self.state[7:10] = vel * (self.MAX_VEL / vel_norm)
        omega = self.state[10:13]
        omega_norm = np.linalg.norm(omega)
        if omega_norm > self.MAX_OMEGA: self.state[10:13] = omega * (self.MAX_OMEGA / omega_norm)
        self.state[0:3] = np.clip(self.state[0:3], -self.MAX_POS, self.MAX_POS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Domain Randomization (Masas y Viento)
        if self.dr_enabled:
            mass_multiplier = self.np_random.uniform(1.0 - DC.DR_MASS_VAR, 1.0 + DC.DR_MASS_VAR)
            self.current_mass = DC.MASS * mass_multiplier
            wx = self.np_random.uniform(-DC.DR_WIND_MAX, DC.DR_WIND_MAX)
            wy = self.np_random.uniform(-DC.DR_WIND_MAX, DC.DR_WIND_MAX)
            self.wind_force = np.array([wx, wy, 0.0], dtype=np.float64)
        else:
            self.current_mass = DC.MASS
            self.wind_force = np.zeros(3, dtype=np.float64)

        z_min, z_max, xy_range = self.phases[self.curriculum_phase]
        
        # Inyeccion de coordenadas forzadas para validacion
        x = self.force_x if self.force_x is not None else self.np_random.uniform(-xy_range, xy_range)
        y = self.force_y if self.force_y is not None else self.np_random.uniform(-xy_range, xy_range)
        z = self.force_z if self.force_z is not None else self.np_random.uniform(z_min, z_max)
        
        phi = self.np_random.uniform(-0.05, 0.05)
        theta = self.np_random.uniform(-0.05, 0.05)
        psi = self.np_random.uniform(-np.pi, np.pi)
        quat = euler_to_quat(phi, theta, psi)
        
        vel_range = 0.1 + 0.1 * self.curriculum_phase
        vx, vy, vz = self.np_random.uniform(-vel_range, vel_range, size=3)
        
        omega_range = 0.1 + 0.05 * self.curriculum_phase
        p, q, r = self.np_random.uniform(-omega_range, omega_range, size=3)
        
        self.state = np.array([x, y, z, quat[0], quat[1], quat[2], quat[3], vx, vy, vz, p, q, r], dtype=np.float64)
        self.steps = 0
        
        hover_cmd = (self.current_mass * self.gravity / 4.0) / self.f_max * 2.0 - 1.0
        self.current_actions = np.full(4, hover_cmd, dtype=np.float64)
        
        self.prev_potential = self._compute_potential(self.state[0:3], self.state[7:10], phi, theta)
        self.prev_dist = np.linalg.norm(self.state[0:3] - DC.GOAL_POS)
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        commanded = np.asarray(action, dtype=np.float64)
        
        if self.dr_enabled:
            motor_noise = self.np_random.normal(0, DC.DR_MOTOR_NOISE, size=4)
            commanded = np.clip(commanded + motor_noise, -1.0, 1.0)
        
        actual_actions = self._apply_motor_lag(commanded)
        
        self.state = simulate_step(
            self.state, actual_actions, self.current_mass, self.gravity, 
            self.arm_length, self.torque_coeff, self.inertia, 
            self.f_min, self.f_max, self.dt, self.substeps, self.wind_force
        )
        self._clamp_state()
        self.steps += 1
        
        pos, vel, omega = self.state[0:3], self.state[7:10], self.state[10:13]
        phi, theta, psi = quat_to_euler(self.state[3:7])
        
        dist_to_goal = np.linalg.norm(pos - DC.GOAL_POS)
        xy_dist = np.sqrt(pos[0]**2 + pos[1]**2)
        vel_norm = np.linalg.norm(vel)
        
        terminated = landed = crashed = False
        
        # Condiciones de Aterrizaje
        if (pos[2] > DC.LAND_Z_THRESH and vel_norm < DC.LAND_VEL_THRESH and 
            abs(phi) < DC.LAND_TILT_THRESH and abs(theta) < DC.LAND_TILT_THRESH and xy_dist < DC.LAND_XY_THRESH):
            terminated = landed = True
            self.total_landings += 1
            self._update_curriculum()
            
        # Condiciones de Colision / Fuera de limites
        if not landed:
            if pos[2] > DC.CRASH_Z_THRESH and vel_norm > DC.CRASH_VEL: terminated = crashed = True
            if abs(phi) > DC.CRASH_TILT or abs(theta) > DC.CRASH_TILT: terminated = crashed = True
            if xy_dist > DC.OOB_XY or pos[2] < DC.OOB_Z: terminated = crashed = True
            if pos[2] > 0.5: terminated = crashed = True
            
        truncated = (self.steps >= self.max_steps) and not terminated
        
        reward = self._compute_reward(
            pos, vel, omega, phi, theta, dist_to_goal, vel_norm, 
            np.linalg.norm(omega), commanded, landed, crashed, self.steps
        )
        
        self.prev_dist = dist_to_goal
        
        info = self._get_info()
        info.update({
            "landed": landed, 
            "crashed": crashed, 
            "dist_to_goal": dist_to_goal, 
            "velocity_norm": vel_norm,
            "wind": self.wind_force
        })
        return self._get_obs(), reward, terminated, truncated, info

    def _compute_reward(self, pos, vel, omega, phi, theta, dist_to_goal, vel_norm, omega_norm, action, landed, crashed, current_step):
        current_potential = self._compute_potential(pos, vel, phi, theta)
        shaping = 0.99 * current_potential - self.prev_potential
        self.prev_potential = current_potential
        
        progress_bonus = DC.W_PROGRESS * (self.prev_dist - dist_to_goal)
        altitude_penalty = -DC.C_ALT * max(-pos[2], 0.0)
        action_penalty = -DC.C_ACTION * np.sum(action**2)
        
        current_time_sec = current_step * DC.AGENT_DT
        impatience_penalty = -DC.C_TIME if current_time_sec > DC.GRACE_TIME else 0.0
        terminal = DC.R_LAND if landed else (DC.R_CRASH if crashed else 0.0)
        
        return shaping + progress_bonus + altitude_penalty + DC.C_ALIVE + action_penalty + impatience_penalty + terminal

    def _get_obs(self):
        pos = self.state[0:3]
        phi, theta, psi = quat_to_euler(self.state[3:7])
        vel, omega = self.state[7:10], self.state[10:13]
        obs = np.array([pos[0], pos[1], pos[2], phi, theta, psi, vel[0], vel[1], vel[2], omega[0], omega[1], omega[2]], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self):
        return {"step": self.steps, "pos_z": self.state[2], "speed": np.linalg.norm(self.state[7:10])}