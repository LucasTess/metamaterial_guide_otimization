# main.py (com a nova lógica de convergência)

import sys
import os
import datetime
import shutil
import pandas as pd

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
from utils.analysis import run_full_analysis

# --- Configurações Globais ---
_project_directory = "C:\\Users\\User04\\Documents\\metamaterial_guide_otimization"
_original_fsp_file_name = "guide.fsp"
_geometry_lsf_script_name = "create_guide_fdtd.lsf"
_simulation_lsf_script_name = "run_simu_guide_fdtd.lsf"
_simulation_spectra_directory_name = "simulation_spectra"
_simulation_results_directory_name = "simulation_results"

# --- Diretórios ---
_temp_directory = os.path.join(_project_directory, "temp")
os.makedirs(_temp_directory, exist_ok=True)
_temp_fsp_base_path = os.path.join(_project_directory, "guide_temp_base.fsp")
_original_fsp_path = os.path.join(_project_directory, _original_fsp_file_name)
_geometry_lsf_script_path = os.path.join(_project_directory, "resources", _geometry_lsf_script_name)
_simulation_lsf_script_path = os.path.join(_project_directory, "resources", _simulation_lsf_script_name)
_simulation_spectra_directory = os.path.join(_project_directory, _simulation_spectra_directory_name)
_simulation_results_directory = os.path.join(_project_directory, _simulation_results_directory_name)

os.makedirs(_simulation_spectra_directory, exist_ok=True)

# --- Configuração do Algoritmo Genético ---
population_size = 30
mutation_rate = 0.2
num_generations = 100

# --- Ranges de Parâmetros ---
s_range = (0.1e-6, 0.25e-6)
w_range = (0.3e-6, 0.7e-6)
l_range = (0.1e-6, 0.25e-6)
height_range = (0.15e-6, 0.3e-6)

# --- Critério de Convergência ---
# ATIVADO: Agora vamos usar a nova lógica
enable_convergence_check = True
# Otimização irá parar se o melhor fitness não melhorar por 20 gerações
CONVERGENCE_PATIENCE = num_generations*0.2
print("--------------------------------------------------------------------------")
print(f"Iniciando o script principal (main.py) para otimização do guia de onda...")
print("--------------------------------------------------------------------------")

shutil.copy(_original_fsp_path, _temp_fsp_base_path)
print(f"Copiado {_original_fsp_path} para {_temp_fsp_base_path}")

if not os.path.exists(_temp_fsp_base_path):
    raise FileNotFoundError(f"Erro: O arquivo base {_temp_fsp_base_path} não foi criado.")

optimizer = GeneticOptimizer(
    population_size, mutation_rate, num_generations,
    s_range, w_range, l_range, height_range
)
optimizer.initialize_population()
current_population = optimizer.population

experiment_start_time = datetime.datetime.now()
timestamp_str = experiment_start_time.strftime('%Y%m%d_%H%M%S')
full_data_csv_path = os.path.join(_simulation_results_directory, f"full_optimization_data_{timestamp_str}.csv")
realtime_heatmap_path = os.path.join(_simulation_results_directory, f"realtime_correlation_heatmap_{timestamp_str}.png")

generations_processed = 0
all_individuals_data = []

# --- NOVAS VARIÁVEIS PARA A LÓGICA DE CONVERGÊNCIA ---
# Armazena o melhor fitness encontrado até agora
best_fitness_so_far = -float('inf')
# Conta as gerações consecutivas sem melhoria
generations_without_improvement = 0

try:
    with lumapi.FDTD(hide=False) as fdtd:
        for gen_num in range(num_generations):
            generations_processed += 1
            print(f"\n--- Processando Geração {gen_num + 1}/{num_generations} ---")
            
            clean_simulation_directory(_simulation_spectra_directory, file_extension=".h5")
            clean_simulation_directory(_temp_directory, file_extension=".fsp")
            clean_simulation_directory(_temp_directory, file_extension=".log")
            
            h5_paths_for_gen = simulate_generation_lumerical(
                fdtd, current_population, _temp_fsp_base_path,
                _geometry_lsf_script_path, _simulation_lsf_script_path,
                _simulation_spectra_directory, _temp_directory
            )
            
            print("\n  [Job Manager] Pós-processando os resultados da geração...")
            delta_amp_results_for_gen = []
            for h5_path in h5_paths_for_gen:
                try:
                    delta_amp = calculate_delta_amp(h5_path)
                except Exception as e:
                    print(f"!!! Erro no pós-processamento do arquivo {os.path.basename(h5_path)}: {e}")
                    delta_amp = -float('inf')
                delta_amp_results_for_gen.append(delta_amp)

            for i, chromosome in enumerate(current_population):
                individual_data = chromosome.copy()
                individual_data['delta_amp'] = delta_amp_results_for_gen[i]
                individual_data['generation'] = gen_num + 1
                all_individuals_data.append(individual_data)

            # --- MODIFICADO: Salva a população ANTES da evolução para comparar depois ---
            population_before_evolution = [chrom.copy() for chrom in current_population]

            try:
                current_population = optimizer.evolve(delta_amp_results_for_gen)
            except ValueError as e:
                print(f"!!! Erro na evolução da população: {e}")
                break

            print(f"  [Relatório] Atualizando relatório para a Geração {gen_num + 1}...")
            record_experiment_results(
                _simulation_results_directory, optimizer, experiment_start_time,
                s_range, w_range, l_range, height_range, generations_processed
            )
            
            if all_individuals_data:
                df_all_data = pd.DataFrame(all_individuals_data)
                df_all_data.to_csv(full_data_csv_path, index=False)
                print(f"  [Análise] Dados de {len(all_individuals_data)} indivíduos atualizados em CSV.")
                run_full_analysis(full_data_csv_path)
                print(f"  [Análise] Heatmap de correlação atualizado e salvo.")

            # --- LÓGICA DE CONVERGÊNCIA POR ESTAGNAÇÃO DO FITNESS (TOTALMENTE MODIFICADA) ---
            if enable_convergence_check:
                # Pega o melhor fitness encontrado até agora em *toda* a otimização
                current_best_fitness = optimizer.best_fitness

                # Compara com o melhor fitness que tínhamos registrado
                if current_best_fitness > best_fitness_so_far:
                    print(f"  [Convergência] ✅ Novo melhor fitness encontrado: {current_best_fitness:.4e}. Reiniciando contador.")
                    best_fitness_so_far = current_best_fitness
                    generations_without_improvement = 0 # Zera o contador pois houve melhoria
                else:
                    generations_without_improvement += 1 # Incrementa o contador
                    print(f"  [Convergência] ⏳ Nenhuma melhoria no fitness. Gerações sem melhoria: {generations_without_improvement}/{CONVERGENCE_PATIENCE}")

                # Verifica se atingimos o limite de paciência
                if generations_without_improvement >= CONVERGENCE_PATIENCE:
                    print(f"\n  [Convergência] 🛑 O melhor fitness não melhorou por {CONVERGENCE_PATIENCE} gerações consecutivas.")
                    print("  [Convergência] Otimização considerada convergente. Encerrando.")
                    break # Encerra o loop principal de gerações
            # --- FIM DA LÓGICA DE CONVERGÊNCIA MODIFICADA ---

    print("\n--- Otimização Concluída ---")
    if optimizer.best_individual:
        print(f"Melhor cromossomo encontrado: {optimizer.best_individual}")
        print(f"Melhor Delta Amplitude atingido: {optimizer.best_fitness:.4e}")
    else:
        print("Nenhum melhor indivíduo encontrado durante a otimização.")

    # --- Limpeza final ---
    clean_simulation_directory(_simulation_spectra_directory, file_extension=".h5")
    clean_simulation_directory(_temp_directory, file_extension=".fsp")
    clean_simulation_directory(_temp_directory, file_extension=".log")
    if os.path.exists(_temp_fsp_base_path):
        os.remove(_temp_fsp_base_path)
        print(f"\n[Limpeza Final] Arquivo base removido: {_temp_fsp_base_path}")

except Exception as e:
    print(f"!!! Erro fatal no script principal de otimização: {e}")

print("\nScript principal (main.py) finalizado.")