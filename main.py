# main.py

import sys
import os
import datetime
import shutil
import pandas as pd
#import time
#import numpy as np

_lumapi_module_path = "C:\\Program Files\\Lumerical\\v241\\api\\python"

if _lumapi_module_path not in sys.path:
    sys.path.append(_lumapi_module_path)

import lumapi
# Importações dos módulos personalizados
from utils.genetic import GeneticOptimizer
from utils.experiment_end import record_experiment_results
from utils.lumerical_workflow import simulate_generation_lumerical
from utils.post_processing import calculate_delta_amp
from utils.file_handler import clean_simulation_directory
from utils.analysis import generate_correlation_heatmap
# --- Configurações Globais ---
_project_directory = "C:\\Users\\User04\\Documents\\metamaterial_guide_otimization"
_original_fsp_file_name = "guide.fsp"
_geometry_lsf_script_name = "create_guide_fdtd.lsf"
_simulation_lsf_script_name = "run_simu_guide_fdtd.lsf"
_simulation_spectra_directory_name = "simulation_spectra"
_simulation_results_directory_name = "simulation_results"

# --- Novo: Diretório para arquivos temporários geracionais ---
_temp_directory = os.path.join(_project_directory, "temp")
os.makedirs(_temp_directory, exist_ok=True)
print(f"_temp_directory = " + _temp_directory)
# --- Caminho do arquivo FSP base (permanente para o run do script) ---
_temp_fsp_base_path = os.path.join(_project_directory, "guide_temp_base.fsp")

_original_fsp_path = os.path.join(_project_directory, _original_fsp_file_name)
_geometry_lsf_script_path = os.path.join(_project_directory, "resources", _geometry_lsf_script_name)
_simulation_lsf_script_path = os.path.join(_project_directory, "resources", _simulation_lsf_script_name)
_simulation_spectra_directory = os.path.join(_project_directory, _simulation_spectra_directory_name)
_simulation_results_directory = os.path.join(_project_directory, _simulation_results_directory_name)

# Garante que o diretório para os espectros exista
os.makedirs(_simulation_spectra_directory, exist_ok=True)

# --- Configuração do Algoritmo Genético ---
population_size = 3
mutation_rate = 0.2
num_generations = 2

# --- Ranges de Parâmetros ---
s_range = (0.1e-6, 0.25e-6)
w_range = (0.3e-6, 0.7e-6)
l_range = (0.1e-6, 0.25e-6)
height_range = (0.15e-6, 0.3e-6)

# --- Critério de Convergência ---
enable_convergence_check = False
convergence_threshold_percent = 5.0 # 5% de melhoria ou menos entre gerações

print("--------------------------------------------------------------------------")
print("--------------------------------------------------------------------------")
print(" ")
print(f"Iniciando o script principal (main.py) para otimização do guia de onda...")
print(" ")
print("--------------------------------------------------------------------------")
print("--------------------------------------------------------------------------")

# --- Prepara arquivo FSP temporário que será usado como base para os jobs ---
# Copia o arquivo base uma única vez para o diretório principal (não o 'temp')
shutil.copy(_original_fsp_path, _temp_fsp_base_path)
print(f"Copiado {_original_fsp_path} para {_temp_fsp_base_path}")

# Pausa para garantir que o arquivo seja gravado no disco
#time.sleep(1)

if not os.path.exists(_temp_fsp_base_path):
    raise FileNotFoundError(f"Erro: O arquivo base {_temp_fsp_base_path} não foi criado. Verifique as permissões ou o caminho.")

# Garante que o diretório para os espectros exista

if not os.path.exists(_temp_fsp_base_path):
    raise FileNotFoundError(f"Erro: O arquivo base {_temp_fsp_base_path} não foi criado. Verifique as permissões ou o caminho.")

# Instancia o otimizador genético
optimizer = GeneticOptimizer(
    population_size, mutation_rate, num_generations,
    s_range, w_range, l_range, height_range
)
optimizer.initialize_population()
current_population = optimizer.population

experiment_start_time = datetime.datetime.now()

# --- NOVO: Define o caminho do CSV de dados completos UMA VEZ ---
timestamp_str = experiment_start_time.strftime('%Y%m%d_%H%M%S')
full_data_csv_path = os.path.join(_simulation_results_directory, f"full_optimization_data_{timestamp_str}.csv")
realtime_heatmap_path = os.path.join(_simulation_results_directory, f"realtime_correlation_heatmap_{timestamp_str}.png")

# PARAMETROS DE INICIALIZAÇÃO
previous_best_fitness = -float('inf') 
generations_processed = 0
all_individuals_data = []


try:
    # --- Loop Principal de Otimização (Gerações) ---
    with lumapi.FDTD(hide=False) as fdtd: # Inicia uma única sessão Lumerical
        
        for gen_num in range(num_generations):
            generations_processed += 1
            
            print(f"\n--- Processando Geração {gen_num + 1}/{num_generations} ---")
            
            # --- Limpeza dos arquivos temporários da geração ANTERIOR ---
            # O diretório 'temp' agora é usado apenas para os arquivos de simulação temporários
            clean_simulation_directory(_simulation_spectra_directory, file_extension=".h5")
            clean_simulation_directory(_temp_directory, file_extension=".fsp")
            clean_simulation_directory(_temp_directory, file_extension=".log")
            
            # A chamada para a nova função é simples e retorna os resultados
            h5_paths_for_gen = simulate_generation_lumerical(
                fdtd,
                current_population,
                _temp_fsp_base_path,
                _geometry_lsf_script_path,
                _simulation_lsf_script_path,
                _simulation_spectra_directory,
                _temp_directory
            )
            
            # --- Pós-Processamento dos Resultados ---
            print("\n  [Job Manager] Pós-processando os resultados da geração...")
            delta_amp_results_for_gen = []

            for h5_path in h5_paths_for_gen:
                try:
                    delta_amp = calculate_delta_amp(h5_path)
                except Exception as e:
                    print(f"!!! Erro no pós-processamento do arquivo {os.path.basename(h5_path)}: {e}")
                    delta_amp = -float('inf')
                    
                delta_amp_results_for_gen.append(delta_amp)

            # --- Armazena os dados desta geração na lista principal ---
            for i, chromosome in enumerate(current_population):
                individual_data = chromosome.copy()
                individual_data['delta_amp'] = delta_amp_results_for_gen[i]
                individual_data['generation'] = gen_num + 1
                all_individuals_data.append(individual_data)

            # --- Evoluindo a população com base nos resultados ---
            try:
                current_population = optimizer.evolve(delta_amp_results_for_gen)
            except ValueError as e:
                print(f"!!! Erro na evolução da população: {e}")
                break
            # --- ATUALIZA O RELATÓRIO AO FINAL DE CADA GERAÇÃO ---
            print(f"  [Relatório] Atualizando relatório para a Geração {gen_num + 1}...")
            record_experiment_results(
                _simulation_results_directory,
                optimizer,
                experiment_start_time, # Passa o mesmo timestamp de início a cada vez
                s_range, w_range, l_range, height_range,
                generations_processed
            )
            # --- NOVO: ATUALIZA O CSV DE DADOS COMPLETOS A CADA GERAÇÃO ---
            if all_individuals_data:
                df_all_data = pd.DataFrame(all_individuals_data)
                df_all_data.to_csv(full_data_csv_path, index=False)
                print(f"  [Análise] Dados de {len(all_individuals_data)} indivíduos atualizados em: {os.path.basename(full_data_csv_path)}")
                # Gera o novo heatmap de correlação
                generate_correlation_heatmap(full_data_csv_path, realtime_heatmap_path)
                print(f"  [Análise] Heatmap de correlação atualizado e salvo.")
            # ---------------------------------------------------------------------

            # --- LÓGICA DE CONVERGÊNCIA ---
            if enable_convergence_check:
                current_best_fitness = optimizer.best_fitness
                if gen_num > 0:
                    if previous_best_fitness == -float('inf'):
                        print("  [Convergência] Fitness anterior não válido, continuando...")
                    elif current_best_fitness == previous_best_fitness:
                        print(f"  [Convergência] Convergência atingida.")
                        break
                    elif previous_best_fitness != 0:
                        percentage_change = abs((current_best_fitness - previous_best_fitness) / previous_best_fitness) * 100
                        if percentage_change <= convergence_threshold_percent:
                            print(f"  [Convergência] Convergência atingida.")
                            break
                previous_best_fitness = current_best_fitness
            else:
                print("  [Convergência] Checagem de convergência desativada.")
                previous_best_fitness = optimizer.best_fitness

    print("\n--- Otimização Concluída ---")
    if optimizer.best_individual:
        print(f"Melhor cromossomo encontrado: {optimizer.best_individual}")
        print(f"Melhor Delta Amplitude atingido: {optimizer.best_fitness:.4e}")
    else:
        print("Nenhum melhor indivíduo encontrado durante a otimização.")

    # --- Limpeza final após o término da otimização ---
    clean_simulation_directory(_simulation_spectra_directory, file_extension=".h5")
    clean_simulation_directory(_temp_directory, file_extension=".fsp")
    clean_simulation_directory(_temp_directory, file_extension=".log")
    if os.path.exists(_temp_fsp_base_path):
        os.remove(_temp_fsp_base_path)
        print(f"\n[Limpeza Final] Arquivo base removido: {_temp_fsp_base_path}")

except Exception as e:
    print(f"!!! Erro fatal no script principal de otimização: {e}")

print("\nScript principal (main.py) finalizado.")