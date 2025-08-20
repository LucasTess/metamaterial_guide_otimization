# main.py

import sys
import os
import datetime
import time
import shutil

import numpy as np

_lumapi_module_path = "C:\\Program Files\\Lumerical\\v241\\api\\python"

if _lumapi_module_path not in sys.path:
    sys.path.append(_lumapi_module_path)

import lumapi
# Importações dos módulos personalizados
from utils.genetic import GeneticOptimizer
from utils.experiment_end import record_experiment_results
from utils.lumerical_workflow import simulate_generation_lumerical
from utils.post_processing import calculate_delta_amp
from utils.file_handler import clean_simulation_directory, delete_directory_contents
from utils.calculate_spectrum import calculate_generation_spectra

# --- Configurações Globais ---
_project_directory = "C:\\Users\\USUARIO\\OneDrive\\Lumerical\\metamaterial_guide_otimization"
_original_fsp_file_name = "guide.fsp"
_geometry_lsf_script_name = "create_guide_fdtd.lsf"
_simulation_lsf_script_name = "run_simu_guide_fdtd.lsf"
_simulation_spectra_directory_name = "simulation_spectra"

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

# Garante que o diretório para os espectros exista
os.makedirs(_simulation_spectra_directory, exist_ok=True)

# --- Configuração do Algoritmo Genético ---
population_size = 1
mutation_rate = 0.2
num_generations = 1

# --- Critério de Convergência ---
enable_convergence_check = False
convergence_threshold_percent = 5.0 # 5% de melhoria ou menos entre gerações
clean_temp_files_error = False

# --- Ranges de Parâmetros ---
s_range = (0.1e-6, 0.25e-6)
w_range = (0.3e-6, 0.7e-6)
l_range = (0.1e-6, 0.25e-6)
height_range = (0.15e-6, 0.3e-6)


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
previous_best_fitness = -float('inf') 
generations_processed = 0

try:
    # --- Loop Principal de Otimização (Gerações) ---
    with lumapi.FDTD(hide=False) as fdtd: # Inicia uma única sessão Lumerical
        
        for gen_num in range(num_generations):
            generations_processed += 1
            
            print(f"\n--- Processando Geração {gen_num + 1}/{num_generations} ---")
            
            # --- Limpeza dos arquivos temporários da geração ANTERIOR ---
            # O diretório 'temp' agora é usado apenas para os arquivos de simulação temporários
            #clean_simulation_directory(_simulation_spectra_directory, file_extension=".h5")

            delete_directory_contents(_temp_directory)
            # A chamada para a nova função é simples e retorna os resultados
            S_matrixes_for_generation, frequencies = simulate_generation_lumerical(
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
            generation_spectra = calculate_generation_spectra(
                S_matrixes_for_generation,
                current_population,
                frequencies
                )
            delta_amp_results_for_gen = []

            # for h5_path in h5_paths_for_gen:
            #     try:
            #         delta_amp = calculate_delta_amp(h5_path)
            #     except Exception as e:
            #         print(f"!!! Erro no pós-processamento do arquivo {os.path.basename(h5_path)}: {e}")
            #         delta_amp = -float('inf')
                    
            #     delta_amp_results_for_gen.append(delta_amp)

            # --- Evoluindo a população com base nos resultados ---
            try:
                current_population = optimizer.evolve(delta_amp_results_for_gen)
            except ValueError as e:
                print(f"!!! Erro na evolução da população: {e}")
                break

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

    record_experiment_results(
        _simulation_spectra_directory,
        optimizer,
        experiment_start_time,
        s_range, w_range, l_range, height_range,
        generations_processed
    )
    # --- Limpeza final após o término da otimização ---
    if clean_temp_files_error:
            delete_directory_contents(_temp_directory)
    if os.path.exists(_temp_fsp_base_path):
        os.remove(_temp_fsp_base_path)
        print(f"\n[Limpeza Final] Arquivo base removido: {_temp_fsp_base_path}")

except Exception as e:
    if clean_temp_files_error:
            delete_directory_contents(_temp_directory)
    print(f"!!! Erro fatal no script principal de otimização: {e}")

print("\nScript principal (main.py) finalizado.")