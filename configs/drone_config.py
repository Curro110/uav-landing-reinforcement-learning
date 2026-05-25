# ============================================================
# configs/drone_config.py
# v10 - Sim-to-Real + Impaciencia Fija + Recompensa por Progreso
# ============================================================

import numpy as np

class DroneConfig:

    # FÍSICOS
    MASS = 1.5
    GRAVITY = 9.81
    ARM_LENGTH = 0.25
    TORQUE_COEFF = 0.0245
    
    Ixx = 0.015
    Iyy = 0.015
    Izz = 0.025
    INERTIA = np.array([Ixx, Iyy, Izz], dtype=np.float64)
    
    F_MIN = 0.0
    F_MAX = 8.0

    # DOMAIN RANDOMIZATION (SIM-TO-REAL)
    DR_MOTOR_NOISE = 0.05    # 5% de ruido en los motores
    DR_WIND_MAX = 1.5        # Viento máximo lateral (Newtons)
    DR_MASS_VAR = 0.10       # +/- 10% de variación de masa inicial

    # MOTOR LAG
    MOTOR_ALPHA = 0.15

    # SIMULACIÓN
    DT = 0.01
    SUBSTEPS = 5
    AGENT_DT = DT * SUBSTEPS
    MAX_STEPS = 600

    # CONDICIONES INICIALES
    INIT_POS_X = (-1.0, 1.0)
    INIT_POS_Y = (-1.0, 1.0)
    INIT_POS_Z = (-6.0, -2.0)
    INIT_VEL = (-0.1, 0.1)
    INIT_ANGLE = (-0.05, 0.05)
    INIT_OMEGA = (-0.1, 0.1)

    GOAL_POS = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    # RECOMPENSA Y POTENCIALES
    W_DIST = 10.0
    W_VEL = 1.0
    W_TILT = 1.0

    # ═══════════════════════════════════════════════════
    # NUEVO: PESO DEL PROGRESO (ACERCARSE AL OBJETIVO)
    # ═══════════════════════════════════════════════════
    W_PROGRESS = 20.0  # Fuerte incentivo por reducir la distancia en cada paso

    C_ALT = 0.1        
    C_ACTION = 0.01
    C_ALIVE = 0.05

    # ═══════════════════════════════════════════════════
    # UMBRAL DE TIEMPO (CASTIGO FIJO, NO CRECIENTE)
    # ═══════════════════════════════════════════════════
    GRACE_TIME = 10.0  # 10 segundos para estabilizarse sin castigo
    C_TIME = 1.0       # Castigo de -1.0 constante por cada paso extra

    R_LAND = 500.0
    R_CRASH = -300.0

    # ATERRIZAJE
    LAND_Z_THRESH = -0.3
    LAND_VEL_THRESH = 1.0
    LAND_TILT_THRESH = np.deg2rad(15)
    LAND_XY_THRESH = 1.5

    # CRASH
    CRASH_VEL = 8.0
    CRASH_Z_THRESH = -0.1
    CRASH_TILT = np.deg2rad(80)
    OOB_XY = 25.0
    OOB_Z = -35.0

    OBS_DIM = 12
    ACT_DIM = 4