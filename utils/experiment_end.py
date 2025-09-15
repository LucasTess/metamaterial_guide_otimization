# utils/experiment_end.py (Modificado)

import os
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np

# [MODIFICADO] Adicionado 'total_length_range' à assinatura da função
def record_experiment_results(
    output_directory, 
    optimizer_instance, 
    experiment_start_time,
    s_range, w_range, l_range, height_range, total_length_range,
    generations_processed,
    fit_strategy
):
    timestamp_str = experiment_start_time.strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(output_directory, f"experiment_results_{timestamp_str}.json")
    plot_path = os.path.join(output_directory, f"fitness_history_{timestamp_str}.png")

    current_time = datetime.datetime.now()
    duration = current_time - experiment_start_time

    results_data = {
        "experiment_start_time": experiment_start_time.isoformat(),
        "last_update": current_time.isoformat(),
        "current_duration": str(duration),
        "generations_processed": generations_processed,
        "population_size": optimizer_instance.population_size,
        "mutation_rate": optimizer_instance.mutation_rate,
        "max_generations_set": optimizer_instance.generations,
        "best_individual_so_far": optimizer_instance.best_individual,
        "best_fitness_so_far": optimizer_instance.best_fitness,
        "parameter_ranges": {
            "s": s_range,
            "w": w_range,
            "l": l_range,
            "height": height_range,
            "total_length": total_length_range # [ADICIONADO] Novo range
        },
        "fitness_history": optimizer_instance.fitness_history,
        "fitness_strategy": fit_strategy
    }

    try:
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=4)
    except Exception as e:
        print(f"!!! Erro ao salvar/atualizar resultados do experimento: {e}")

    if optimizer_instance.fitness_history:
        plt.figure(figsize=(10, 6))
        generations = range(1, len(optimizer_instance.fitness_history) + 1)
        plt.plot(generations, optimizer_instance.fitness_history, marker='o', linestyle='-')
        plt.title(f'Histórico de Fitness (Atualizado em: {current_time.strftime("%H:%M:%S")}), {fit_strategy}')
        plt.xlabel('Geração')
        plt.ylabel('Melhor Fitness')
        plt.grid(True)
        try:
            plt.savefig(plot_path)
        except Exception as e:
            print(f"!!! Erro ao salvar/atualizar gráfico de fitness: {e}")
        finally:
            plt.close()
    else:
        print("Nenhum histórico de fitness para plotar.")