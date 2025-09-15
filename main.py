# main.py (Modificado com o Padrão Strategy)

import sys
import os
import datetime
import shutil
import pandas as pd
import numpy as np

_lumapi_module_path = "C:\\Program Files\\Lumerical\\v241\\api\\python"

if _lumapi_module_path not in sys.path:
    sys.path.append(_lumapi_module_path)

import lumapi
# --- [MODIFICADO] Importações dos módulos personalizados ---
from utils.genetic import GeneticOptimizer
from utils.experiment_end import record_experiment_results
from utils.lumerical_workflow import simulate_generation_lumerical
# Importa as CLASSES de estratégia, não mais uma função específica
from utils.fitness_functions import DeltaAmpStrategy, BandpassStrategy, LowpassStrategy, HighpassStrategy
from utils.file_handler import clean_simulation_directory
from utils.analysis import run_full_analysis

# --- Configurações Globais ---
_project_directory = os.getcwd()
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
population_size = 3
mutation_rate = 0.2
num_generations = 2

# --- Ranges de Parâmetros ---
s_range = (0.1e-6, 0.25e-6)
w_range = (0.3e-6, 0.7e-6)
l_range = (0.1e-6, 0.25e-6)
height_range = (0.15e-6, 0.3e-6)


# --- [NOVO] Seleção da Estratégia de Fitness ---
# Descomente a estratégia que deseja usar para a otimização.
# Apenas UMA estratégia deve estar ativa por vez.
# -------------------------------------------------------------
# Opção 1: Maximizar o contraste do espectro (comportamento original)
fitness_calculator = DeltaAmpStrategy()

# Opção 2: Otimizar para um filtro passa-banda (ainda não implementado)
# c = 299792458  # Velocidade da luz
# center_wavelength = 1.55e-6
# bandwidth_nm = 50e-9
# f_center = c / center_wavelength
# f_upper = c / (center_wavelength - bandwidth_nm / 2)
# f_lower = c / (center_wavelength + bandwidth_nm / 2)
# f_bandwidth = f_upper - f_lower
# fitness_calculator = BandpassStrategy(center_freq=f_center, bandwidth=f_bandwidth)

# Opção 3: Otimizar para um filtro passa-baixas (ainda não implementado)
# cutoff_wavelength = 1.5e-6
# f_cutoff = c / cutoff_wavelength
# fitness_calculator = LowpassStrategy(cutoff_freq=f_cutoff)

# Opção 4: Otimizar para um filtro passa-altas (ainda não implementado)
# cutoff_wavelength = 1.6e-6
# f_cutoff = c / cutoff_wavelength
# fitness_calculator = HighpassStrategy(cutoff_freq=f_cutoff)
# -------------------------------------------------------------

# --- Critério de Convergência ---
enable_convergence_check = True
CONVERGENCE_PATIENCE = 20

# --- Limpeza de Debug ---
debug_clean = True
print("--------------------------------------------------------------------------")
print(f"Iniciando otimização com a estratégia: {fitness_calculator.__class__.__name__}")
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

generations_processed = 0
all_individuals_data = []
best_fitness_so_far = -float('inf')
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
            # --- [MODIFICADO] Lógica de cálculo de fitness genérica ---
            fitness_scores_for_gen = []
            for h5_path in h5_paths_for_gen:
                try:
                    # A chamada agora usa o objeto 'fitness_calculator' selecionado
                    fitness_score = fitness_calculator.calculate(h5_path)
                except Exception as e:
                    print(f"!!! Erro no pós-processamento do arquivo {os.path.basename(h5_path)}: {e}")
                    fitness_score = -float('inf')
                fitness_scores_for_gen.append(fitness_score)

            for i, chromosome in enumerate(current_population):
                individual_data = chromosome.copy()
                # --- [MODIFICADO] Nome da coluna de fitness genérico ---
                individual_data['fitness_score'] = fitness_scores_for_gen[i]
                individual_data['generation'] = gen_num + 1
                all_individuals_data.append(individual_data)

            try:
                # --- [MODIFICADO] Passa a lista de scores genéricos ---
                current_population = optimizer.evolve(fitness_scores_for_gen)
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
                # ATENÇÃO: pode ser necessário ajustar o analysis.py para usar a coluna 'fitness_score'
                run_full_analysis(full_data_csv_path)
                print(f"  [Análise] Gráficos de análise atualizados e salvos.")

            if enable_convergence_check:
                current_best_fitness = optimizer.best_fitness
                if current_best_fitness > best_fitness_so_far:
                    print(f"  [Convergência] ✅ Novo melhor fitness encontrado: {current_best_fitness:.4e}. Reiniciando contador.")
                    best_fitness_so_far = current_best_fitness
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                    print(f"  [Convergência] ⏳ Nenhuma melhoria no fitness. Gerações sem melhoria: {generations_without_improvement}/{CONVERGENCE_PATIENCE}")

                if generations_without_improvement >= CONVERGENCE_PATIENCE:
                    print(f"\n  [Convergência] 🛑 O melhor fitness não melhorou por {CONVERGENCE_PATIENCE} gerações consecutivas.")
                    print("  [Convergência] Otimização considerada convergente. Encerrando.")
                    break

    print("\n--- Otimização Concluída ---")
    if optimizer.best_individual:
        print(f"Melhor cromossomo encontrado: {optimizer.best_individual}")
        # --- [MODIFICADO] Mensagem final genérica ---
        print(f"Melhor Fitness Score atingido: {optimizer.best_fitness:.4e}")
    else:
        print("Nenhum melhor indivíduo encontrado durante a otimização.")
    if debug_clean:
        # --- Limpeza final ---
        clean_simulation_directory(_simulation_spectra_directory, file_extension=".h5")
        clean_simulation_directory(_temp_directory, file_extension=".fsp")
        clean_simulation_directory(_temp_directory, file_extension=".log")
        if os.path.exists(_temp_fsp_base_path):
            os.remove(_temp_fsp_base_path)
            print(f"\n[Limpeza Final] Arquivo base removido: {_temp_fsp_base_path}")

except Exception as e:
    print(f"!!! Erro fatal no script principal de otimização: {e}")
    # Adicionar traceback para mais detalhes em caso de erro
    import traceback
    traceback.print_exc()

print("\nScript principal (main.py) finalizado.")