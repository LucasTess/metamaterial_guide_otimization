# utils/experiment_end.py

import os
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np

def record_experiment_results(
    output_directory, optimizer_instance, experiment_start_time,
    s_range, w_range, l_range, height_range,
    generations_processed # <--- NOVO ARGUMENTO
):
    """
    Registra os resultados finais do experimento, incluindo o melhor cromossomo,
    o histórico de fitness, e informações de configuração.
    """
    experiment_end_time = datetime.datetime.now()
    duration = experiment_end_time - experiment_start_time

    results_file_name = f"experiment_results_{experiment_end_time.strftime('%Y%m%d_%H%M%S')}.json"
    results_path = os.path.join(output_directory, results_file_name)

    results_data = {
        "experiment_start_time": experiment_start_time.isoformat(),
        "experiment_end_time": experiment_end_time.isoformat(),
        "total_duration": str(duration),
        "total_generations_processed": generations_processed, # <--- INCLUÍDO AQUI
        "population_size": optimizer_instance.population_size,
        "mutation_rate": optimizer_instance.mutation_rate,
        "max_generations_set": optimizer_instance.generations, # Geração máxima configurada
        "best_individual": optimizer_instance.best_individual,
        "best_fitness": optimizer_instance.best_fitness,
        "s_range": s_range,
        "w_range": w_range,
        "l_range": l_range,
        "height_range": height_range,
        "fitness_history": optimizer_instance.fitness_history # Assumindo que o otimizador guarda isso
    }

    try:
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        print(f"\nResultados do experimento salvos em: {results_path}")
    except Exception as e:
        print(f"!!! Erro ao salvar resultados do experimento: {e}")

    # Plotar o histórico de fitness
    if optimizer_instance.fitness_history:
        plt.figure(figsize=(10, 6))
        generations = range(1, len(optimizer_instance.fitness_history) + 1)
        plt.plot(generations, optimizer_instance.fitness_history, marker='o', linestyle='-')
        plt.title('Histórico do Melhor Fitness por Geração')
        plt.xlabel('Geração')
        plt.ylabel('Melhor Delta Amplitude')
        plt.grid(True)
        plot_file_name = f"fitness_history_{experiment_end_time.strftime('%Y%m%d_%H%M%S')}.png"
        plot_path = os.path.join(output_directory, plot_file_name)
        try:
            plt.savefig(plot_path)
            print(f"Gráfico do histórico de fitness salvo em: {plot_path}")
        except Exception as e:
            print(f"!!! Erro ao salvar gráfico do histórico de fitness: {e}")
        plt.close() # Fecha a figura para liberar memória
    else:
        print("Nenhum histórico de fitness para plotar.")