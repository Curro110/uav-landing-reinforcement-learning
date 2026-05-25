# ============================================================
# utils/renderer.py
# Motor de renderizado 3D basado en Matplotlib
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
from configs.drone_config import DroneConfig as DC
from utils.quaternion_utils import quat_to_rotation_matrix

class Renderer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.arm_length = DC.ARM_LENGTH

    def render(self, state, action=None, wind=None):
        self.ax.clear()

        # Extraer posición y cuaterniones del estado
        pos = state[0:3]
        quat = state[3:7]
        
        # Limites dinámicos de la cámara
        r = 3.0
        self.ax.set_xlim([pos[0] - r, pos[0] + r])
        self.ax.set_ylim([pos[1] - r, pos[1] + r])
        self.ax.set_zlim([pos[2] - r, pos[2] + r])

        # Etiquetas
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m) - Down')
        
        # Dibujar la plataforma de aterrizaje en el origen (0,0,0)
        theta = np.linspace(0, 2 * np.pi, 100)
        pad_radius = DC.LAND_XY_THRESH
        x_pad = pad_radius * np.cos(theta)
        y_pad = pad_radius * np.sin(theta)
        z_pad = np.zeros_like(x_pad)
        self.ax.plot(x_pad, y_pad, z_pad, color='green', linewidth=2, label='Landing Pad')
        self.ax.scatter([0], [0], [0], color='green', s=100, marker='x')

        # Calcular las posiciones de los 4 rotores (geometría en cruz 'X')
        R = quat_to_rotation_matrix(quat)
        
        # Coordenadas relativas de los brazos
        arm_offset = self.arm_length * np.sqrt(2) / 2.0
        arms_body = np.array([
            [ arm_offset,  arm_offset, 0], # M1 (Front-Right)
            [-arm_offset,  arm_offset, 0], # M2 (Rear-Right)
            [-arm_offset, -arm_offset, 0], # M3 (Rear-Left)
            [ arm_offset, -arm_offset, 0]  # M4 (Front-Left)
        ])

        # Rotar y trasladar los brazos al espacio inercial
        arms_inertial = np.zeros((4, 3))
        for i in range(4):
            arms_inertial[i] = pos + R @ arms_body[i]

        # Dibujar el cuerpo del dron (cruz)
        self.ax.plot([arms_inertial[0, 0], arms_inertial[2, 0]], 
                     [arms_inertial[0, 1], arms_inertial[2, 1]], 
                     [arms_inertial[0, 2], arms_inertial[2, 2]], color='black', linewidth=3)
        
        self.ax.plot([arms_inertial[1, 0], arms_inertial[3, 0]], 
                     [arms_inertial[1, 1], arms_inertial[3, 1]], 
                     [arms_inertial[1, 2], arms_inertial[3, 2]], color='black', linewidth=3)

        # Dibujar los motores/hélices
        self.ax.scatter(arms_inertial[:, 0], arms_inertial[:, 1], arms_inertial[:, 2], color='red', s=50)

        # ═══════════════════════════════════════════════════
        # DIBUJAR VECTOR DE VIENTO (FLECHA CYAN)
        # ═══════════════════════════════════════════════════
        if wind is not None and np.linalg.norm(wind) > 0.01:
            visual_scale = 2.0 # Factor de escala visual para que la flecha sea visible
            self.ax.quiver(
                pos[0], pos[1], pos[2], 
                wind[0]*visual_scale, wind[1]*visual_scale, wind[2]*visual_scale, 
                color='cyan', arrow_length_ratio=0.2, linewidth=2, alpha=0.8, label='Wind Force'
            )

        self.ax.legend(loc='upper left')
        plt.draw()
        plt.pause(0.01)

    def close(self):
        plt.ioff()
        plt.close()