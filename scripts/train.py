# ============================================================
# scripts/train.py
# Orquestador de entrenamiento de politicas SAC
# ============================================================

import os
import sys
import argparse
import torch as th

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from envs.drone_landing_env import DroneLandingEnv

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento DRL para UAV")
    parser.add_argument("--timesteps", type=int, default=300000, help="Pasos totales")
    parser.add_argument("--version", type=str, default="extended", choices=["ideal", "real", "extended"], help="Version del entorno")
    parser.add_argument("--resume", action="store_true", help="Reanudar entrenamiento previo")
    args = parser.parse_args()

    log_dir = "logs/tensorboard/"
    chkpt_dir = "logs/checkpoints/"
    eval_dir = "logs/eval/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"[INFO] Inicializando entorno en version: {args.version.upper()}")
    
    raw_env = DroneLandingEnv(env_version=args.version)
    env = Monitor(raw_env)
    vec_env = DummyVecEnv([lambda: env])

    eval_raw_env = DroneLandingEnv(env_version=args.version)
    eval_env = Monitor(eval_raw_env)
    eval_vec_env = DummyVecEnv([lambda: eval_env])

    model_path = os.path.join(chkpt_dir, f"sac_drone_{args.version}.zip")
    vec_norm_path = os.path.join(chkpt_dir, f"sac_drone_{args.version}_vecnormalize.pkl")

    if args.resume and os.path.exists(model_path) and os.path.exists(vec_norm_path):
        print(f"[INFO] Reanudando entrenamiento desde modelo existente.")
        
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True

        eval_vec_env = VecNormalize.load(vec_norm_path, eval_vec_env)
        eval_vec_env.training = False
        eval_vec_env.norm_reward = False

        model = SAC.load(model_path, env=vec_env)
        model.ent_coef = 'auto' 
        model.tensorboard_log = log_dir

    else:
        print("[INFO] Iniciando entrenamiento desde cero.")
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        
        eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_vec_env.training = False
        eval_vec_env.norm_reward = False

        model = SAC(
            "MlpPolicy", vec_env,
            learning_rate=3e-4, buffer_size=200000, learning_starts=10000,
            batch_size=256, tau=0.005, gamma=0.99, ent_coef="auto",
            target_update_interval=1, tensorboard_log=log_dir, verbose=1,
            device="cuda" if th.cuda.is_available() else "cpu"
        )

    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=chkpt_dir,
        log_path=eval_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    print(f"[INFO] Comenzando optimizacion ({args.timesteps} timesteps)...")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback, reset_num_timesteps=False, progress_bar=True)

    model.save(model_path)
    vec_env.save(vec_norm_path)
    print("[SUCCESS] Entrenamiento finalizado y modelo guardado.")

if __name__ == "__main__":
    main()