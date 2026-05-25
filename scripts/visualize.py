# ============================================================
# scripts/visualize.py
# Inferencia visual del modelo neuronal SAC
# ============================================================

import sys
import os
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.drone_landing_env import DroneLandingEnv
from utils.renderer import Renderer

def main():
    parser = argparse.ArgumentParser(description="Visualizacion DRL UAV")
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo (sin extension)")
    parser.add_argument("--version", type=str, default="extended", choices=["ideal", "real", "extended"], help="Version de entorno simulado")
    parser.add_argument("--episodes", type=int, default=5, help="Episodios a ejecutar")
    parser.add_argument("--x", type=float, default=None, help="Posicion X forzada")
    parser.add_argument("--y", type=float, default=None, help="Posicion Y forzada")
    parser.add_argument("--z", type=float, default=None, help="Altitud Z forzada (positiva hacia abajo por convención)")
    args = parser.parse_args()

    model_path = f"{args.model}.zip"
    vec_norm_path = f"{args.model}_vecnormalize.pkl"

    if not os.path.exists(model_path):
        print(f"[ERROR] Modelo no encontrado en: {model_path}")
        return

    print("============================================================")
    print(f" [INFERENCIA] Desplegando agente en entorno: {args.version.upper()}")
    print("============================================================")

    raw_env = DroneLandingEnv(render_mode="human", env_version=args.version)
    
    # Forzar la ultima fase por defecto para validacion
    last_phase_idx = len(raw_env.unwrapped.phases) - 1
    raw_env.unwrapped.curriculum_phase = last_phase_idx 
    
    if args.x is not None:
        raw_env.unwrapped.force_x = args.x
        print(f"[INFO] Inyectando coordenada X: {args.x}")
    if args.y is not None:
        raw_env.unwrapped.force_y = args.y
        print(f"[INFO] Inyectando coordenada Y: {args.y}")
    if args.z is not None:
        z_val = -abs(args.z) 
        raw_env.unwrapped.force_z = z_val
        print(f"[INFO] Inyectando altitud Z: {abs(z_val)}m")
    
    vec_env = DummyVecEnv([lambda: raw_env])

    if os.path.exists(vec_norm_path):
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        vec_env.training = False 
        vec_env.norm_reward = False
    else:
        print("[ERROR] Archivo de normalizacion (.pkl) no encontrado.")
        return

    # Evita el error de mismatch en parametros del optimizador durante inferencia
    custom_objects = {"ent_coef": 0.01}
    model = SAC.load(model_path, env=vec_env, custom_objects=custom_objects)
    renderer = Renderer()

    for ep in range(args.episodes):
        obs = vec_env.reset()
        done = False
        episode_reward = 0.0
        
        print(f"\n--- Ejecucion {ep+1}/{args.episodes} ---")

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            episode_reward += rewards[0]
            done = dones[0]
            info = infos[0]
            
            current_wind = info.get("wind", None)
            renderer.render(raw_env.unwrapped.state, action=action[0], wind=current_wind)

        if info.get("landed", False):
            print(f"[SUCCESS] Aterrizaje completado. Retorno: {episode_reward:.1f}")
        elif info.get("crashed", False):
            print(f"[CRASH] Colision o limites excedidos. Retorno: {episode_reward:.1f}")
        else:
            print(f"[TIMEOUT] Tiempo agotado. Retorno: {episode_reward:.1f}")

    renderer.close()
    raw_env.close()

if __name__ == "__main__":
    main()