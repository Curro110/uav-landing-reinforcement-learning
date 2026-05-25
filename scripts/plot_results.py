# ============================================================
# scripts/plot_results.py
# Generador de gráficas académicas a partir de exportaciones JSON
# ============================================================

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def configurar_estilo_academico():
    """Ajusta los parámetros globales de Matplotlib para publicaciones académicas."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        "figure.titlesize": 14,
        "pdf.fonttype": 42,  
        "ps.fonttype": 42
    })

def cargar_y_suavizar(json_path, smoothing_factor=0.9):
    """Lee el JSON de TensorBoard y aplica Media Móvil Exponencial (EMA)."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No se encuentra el archivo: {json_path}")
        
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    
    # TensorBoard exporta los datos históricos como una lista de listas: [wall_time, step, value]
    df = pd.DataFrame(raw_data, columns=['wall_time', 'step', 'value'])
    
    # Filtro EMA para eliminar el ruido de exploración del agente SAC
    df['smoothed'] = df['value'].ewm(alpha=1 - smoothing_factor, adjust=False).mean()
    return df

def generar_grafica(df, title, ylabel, color, output_name):
    """Renderiza la gráfica con el estándar visual académico y la exporta."""
    plt.figure(figsize=(7, 4.5))
    plt.grid(True, linestyle='--')
    
    # Dibujamos los datos reales transparentes de fondo para mostrar la varianza real
    plt.plot(df['step'], df['value'], color=color, alpha=0.18, label='Datos brutos')
    # Dibujamos la línea suavizada principal
    plt.plot(df['step'], df['smoothed'], color=color, linewidth=2.2, label='Tendencia (EMA)')
    
    plt.title(title, fontweight='bold', pad=15)
    plt.xlabel('Pasos de Entrenamiento (Timesteps)')
    plt.ylabel(ylabel)
    plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='none')
    
    # Ajuste estricto de márgenes
    plt.tight_layout()
    
    # Guardamos en formato PNG de alta resolución y PDF vectorial para LaTeX
    plt.savefig(f"{output_name}.png", dpi=300)
    plt.savefig(f"{output_name}.pdf", format='pdf')
    print(f"Archivos generados: {output_name}.png / .pdf")
    plt.close()

def main():
    # Hemos configurado los valores 'default' con los nombres exactos de tus archivos
    parser = argparse.ArgumentParser(description="Procesador Gráfico de Telemetría JSON")
    parser.add_argument("--reward_json", type=str, default="ep_rew_mean.json", help="Ruta al JSON de recompensa")
    parser.add_argument("--length_json", type=str, default="ep_len_mean.json", help="Ruta al JSON de duración")
    parser.add_argument("--critic_json", type=str, default="critic_loss.json", help="Ruta al JSON de pérdida del crítico")
    args = parser.parse_args()

    configurar_estilo_academico()
    
    # Crear la carpeta donde se guardarán las gráficas si no existe
    output_dir = "logs/plots"
    os.makedirs(output_dir, exist_ok=True)

    print("============================================================")
    print(" Iniciando renderizado de gráficas académicas...")
    print("============================================================\n")

    try:
        # 1. Procesar Gráfica de Recompensa Global
        if os.path.exists(args.reward_json):
            print(f"Procesando {args.reward_json}...")
            df_reward = cargar_y_suavizar(args.reward_json, smoothing_factor=0.95)
            generar_grafica(
                df=df_reward,
                title="Evolución de la Recompensa Media (ep_rew_mean)",
                ylabel="Retorno Acumulado",
                color="#1f77b4", # Azul académico
                output_name=f"{output_dir}/grafica_recompensa"
            )
        else:
            print(f"Advertencia: No se encontró {args.reward_json}")

        # 2. Procesar Gráfica de Tiempo de Supervivencia / Duración de Episodio
        if os.path.exists(args.length_json):
            print(f"Procesando {args.length_json}...")
            df_length = cargar_y_suavizar(args.length_json, smoothing_factor=0.92)
            generar_grafica(
                df=df_length,
                title="Duración Media de los Episodios (ep_len_mean)",
                ylabel="Pasos de Control",
                color="#2ca02c", 
                output_name=f"{output_dir}/grafica_duracion"
            )
        else:
            print(f" Advertencia: No se encontró {args.length_json}")

        # 3. Procesar Pérdida del Crítico
        if os.path.exists(args.critic_json):
            print(f"Procesando {args.critic_json}...")
            df_critic = cargar_y_suavizar(args.critic_json, smoothing_factor=0.95)
            generar_grafica(
                df=df_critic,
                title="Convergencia de la Red del Crítico (critic_loss)",
                ylabel="Error de Diferencia Temporal (MSE)",
                color="#d62728", 
                output_name=f"{output_dir}/grafica_critic_loss"
            )
        else:
            print(f" Advertencia: No se encontró {args.critic_json}")

        print("\n¡Proceso finalizado! Las gráficas están en la carpeta 'logs/plots/'")

    except Exception as e:
        print(f"\n Error crítico durante el procesamiento: {e}")

if __name__ == "__main__":
    main()