# utils/experiment_end.py

import os
import datetime
import numpy as np
def record_experiment_results(
    output_dir,
    optimizer_instance,
    start_time,
    s_range, w_range, l_range, height_range
):
    """
    Registra os resultados finais do experimento em um arquivo CSV.
    Cada execução adiciona uma nova linha ao arquivo.

    Args:
        output_dir (str): Diretório onde o arquivo de resultados será salvo.
        optimizer_instance (GeneticOptimizer): A instância do otimizador genético
                                                após a conclusão das gerações.
        start_time (float): O timestamp de início do experimento (time.time()).
        s_range (tuple): Tupla (min, max) do range de s.
        w_range (tuple): Tupla (min, max) do range de w.
        l_range (tuple): Tupla (min, max) do range de l.
        height_range (tuple): Tupla (min, max) do range de height.
    """
    results_file_path = os.path.join(output_dir, "optimization_results.csv")
    
    end_time = datetime.datetime.now()
    duration_seconds = (end_time - start_time).total_seconds()
    duration_hours = int(duration_seconds // 3600)
    duration_minutes = int((duration_seconds % 3600) // 60)

    # Coleta os dados do melhor indivíduo
    best_s = optimizer_instance.best_individual['s'] if optimizer_instance.best_individual else np.nan
    best_w = optimizer_instance.best_individual['w'] if optimizer_instance.best_individual else np.nan
    best_l = optimizer_instance.best_individual['l'] if optimizer_instance.best_individual else np.nan
    best_height = optimizer_instance.best_individual['height'] if optimizer_instance.best_individual else np.nan
    best_delta_amp = optimizer_instance.best_fitness if optimizer_instance.best_fitness != -float('inf') else np.nan

    # Formata os ranges para uma string que o pandas possa ler facilmente
    # Ex: "(0.01e-6, 0.5e-6)"
    s_range_str = f"({s_range[0]:.2e}, {s_range[1]:.2e})"
    w_range_str = f"({w_range[0]:.2e}, {w_range[1]:.2e})"
    l_range_str = f"({l_range[0]:.2e}, {l_range[1]:.2e})"
    height_range_str = f"({height_range[0]:.2e}, {height_range[1]:.2e})"

    # Cabeçalho do CSV (se o arquivo não existir)
    header = "Timestamp,Best_Delta_Amp,Best_s,Best_w,Best_l,Best_height,Num_Generations,Population_Size,Mutation_Rate,s_Range,w_Range,l_Range,height_Range,Duration_Hours,Duration_Minutes\n"

    # Dados da linha atual
    data_row = (
        f"{end_time.strftime('%Y-%m-%d %H:%M:%S')},"
        f"{best_delta_amp:.4e},"
        f"{best_s:.4e},{best_w:.4e},{best_l:.4e},{best_height:.4e},"
        f"{optimizer_instance.generations},"
        f"{optimizer_instance.population_size},"
        f"{optimizer_instance.mutation_rate},"
        f"\"{s_range_str}\"," # Aspas para garantir que a tupla seja lida como uma string única
        f"\"{w_range_str}\","
        f"\"{l_range_str}\","
        f"\"{height_range_str}\","
        f"{duration_hours},{duration_minutes}\n"
    )

    # Escreve no arquivo (modo 'a' para append - adicionar ao final)
    with open(results_file_path, 'a') as f:
        if not os.path.exists(results_file_path) or os.path.getsize(results_file_path) == 0:
            f.write(header) # Escreve o cabeçalho apenas se o arquivo for novo ou vazio
        f.write(data_row)
    
    print(f"\nResultados do experimento salvos em: {results_file_path}")