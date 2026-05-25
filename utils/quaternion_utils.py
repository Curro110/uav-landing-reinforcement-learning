# ============================================================
# utils/quaternion_utils.py
# Álgebra de cuaterniones con aceleración Numba
#
# Convención: q = [w, x, y, z] donde w es la parte ESCALAR
#
# Un cuaternión unitario (||q||=1) representa una rotación 3D.
# q = [1, 0, 0, 0] = identidad (sin rotación)
# ============================================================

import numpy as np
from numba import njit


# ==============================================================
# OPERACIONES BÁSICAS
# ==============================================================

@njit(cache=True)
def quat_multiply(q1, q2):
    """
    Producto Hamilton de dos cuaterniones: q1 ⊗ q2.

    El producto de cuaterniones es como "encadenar" dos rotaciones.
    Si q1 rota y después q2 rota, el resultado es q1 ⊗ q2.

    ¡OJO! El producto NO es conmutativo: q1⊗q2 ≠ q2⊗q1
    (igual que las rotaciones 3D no conmutan).

    Parámetros:
        q1: np.array([w1, x1, y1, z1])
        q2: np.array([w2, x2, y2, z2])

    Retorna:
        np.array([w, x, y, z]) resultado
    """
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,   # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,   # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,   # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2    # z
    ])


@njit(cache=True)
def quat_normalize(q):
    """
    Normaliza un cuaternión para que ||q|| = 1.

    Después de cada paso de integración numérica, el cuaternión
    puede derivar ligeramente de norma 1. Esta función lo corrige.

    Si la norma es casi 0 (cuaternión degenerado), retorna
    la identidad [1,0,0,0] como fallback seguro.
    """
    norm = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)

    if norm < 1e-10:
        # Cuaternión degenerado → retornar identidad
        return np.array([1.0, 0.0, 0.0, 0.0])

    return q / norm


# ==============================================================
# CONVERSIONES
# ==============================================================

@njit(cache=True)
def quat_to_rotation_matrix(q):
    """
    Convierte un cuaternión unitario a matriz de rotación 3×3.

    La matriz R rota vectores del BODY frame al INERCIAL frame.

    Ejemplo: si el dron está inclinado 30° en roll,
    un vector [0, 0, -1] (arriba en body) se convierte
    en un vector que apunta 30° hacia un lado en el mundo.

    Uso: force_inercial = R @ force_body

    Parámetros:
        q: np.array([w, x, y, z]) con ||q|| = 1

    Retorna:
        R: np.array de 3×3
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Precalcular productos (evita repetir multiplicaciones)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = np.array([
        [1.0 - 2.0*(yy + zz),     2.0*(xy - wz),       2.0*(xz + wy)],
        [    2.0*(xy + wz),    1.0 - 2.0*(xx + zz),     2.0*(yz - wx)],
        [    2.0*(xz - wy),        2.0*(yz + wx),    1.0 - 2.0*(xx + yy)]
    ])

    return R


@njit(cache=True)
def quat_to_euler(q):
    """
    Convierte cuaternión a ángulos de Euler (roll, pitch, yaw).

    Secuencia de rotación: ZYX (convención aeronáutica estándar).
    1. Rotar ψ (yaw) alrededor de Z
    2. Rotar θ (pitch) alrededor del nuevo Y
    3. Rotar φ (roll) alrededor del nuevo X

    Retorna:
        phi:   roll  - inclinación lateral      (-π, π)
        theta: pitch - inclinación frontal       (-π/2, π/2)
        psi:   yaw   - orientación de la proa    (-π, π)

    NOTA: pitch está limitado a (-90°, 90°) por la singularidad
    de gimbal lock. Por eso internamente usamos cuaterniones.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # --- Roll (φ) ---
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    phi = np.arctan2(sinr_cosp, cosr_cosp)

    # --- Pitch (θ) ---
    sinp = 2.0 * (w*y - z*x)
    # Clamp para evitar NaN en arcsin si |sinp| > 1 por error numérico
    if sinp > 1.0:
        sinp = 1.0
    elif sinp < -1.0:
        sinp = -1.0
    theta = np.arcsin(sinp)

    # --- Yaw (ψ) ---
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    psi = np.arctan2(siny_cosp, cosy_cosp)

    return phi, theta, psi


@njit(cache=True)
def euler_to_quat(phi, theta, psi):
    """
    Convierte ángulos de Euler a cuaternión.

    Inversa de quat_to_euler(). Usada en reset() para crear
    la orientación inicial a partir de ángulos aleatorios.

    Parámetros:
        phi:   roll  (rad)
        theta: pitch (rad)
        psi:   yaw   (rad)

    Retorna:
        q: np.array([w, x, y, z])
    """
    # Mitades de los ángulos (aparecen en las fórmulas)
    cr = np.cos(phi / 2.0)
    sr = np.sin(phi / 2.0)
    cp = np.cos(theta / 2.0)
    sp = np.sin(theta / 2.0)
    cy = np.cos(psi / 2.0)
    sy = np.sin(psi / 2.0)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy

    return np.array([w, x, y, z])


# ==============================================================
# DERIVADA TEMPORAL (para integración)
# ==============================================================

@njit(cache=True)
def quat_derivative(q, omega):
    """
    Calcula dq/dt dado el cuaternión actual y la velocidad angular.

    La ecuación cinemática del cuaternión es:
        dq/dt = 0.5 * q ⊗ [0, p, q, r]

    Donde [0, p, q, r] es la velocidad angular expresada como
    un "cuaternión puro" (parte escalar = 0).

    Esta derivada se usa dentro del integrador RK4.

    Parámetros:
        q:     np.array([w, x, y, z]) - cuaternión actual
        omega: np.array([p, q, r])    - vel. angular en body frame

    Retorna:
        dqdt: np.array([dw, dx, dy, dz])
    """
    # Construir cuaternión puro de la velocidad angular
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])

    # dq/dt = 0.5 * q ⊗ omega_quat
    return 0.5 * quat_multiply(q, omega_quat)