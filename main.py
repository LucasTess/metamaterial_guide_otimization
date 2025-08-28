
# main.py 
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
from utils.file_handler import delete_directory_contents
from utils.s_matrix_calculations import calculate_mean_S11_for_generation
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

population_size = 30
mutation_rate = 0.2
num_generations = 100


# --- Ranges de Parâmetros ---
Lambda_range = (0.1e-6, 0.6e-6)
DC_range = (0.1, 0.9)
w_range = (0.3e-6, 0.7e-6)
height_range = (0.15e-6, 0.3e-6)

# --- Critério de Convergência ---
enable_convergence_check = True
CONVERGENCE_PATIENCE = 20

print("--------------------------------------------------------------------------")
print(f"Iniciando o script principal (main.py) para otimização do guia de onda...")
print("--------------------------------------------------------------------------")

shutil.copy(_original_fsp_path, _temp_fsp_base_path)
print(f"Copiado {_original_fsp_path} para {_temp_fsp_base_path}")

if not os.path.exists(_temp_fsp_base_path):
    raise FileNotFoundError(f"Erro: O arquivo base {_temp_fsp_base_path} não foi criado.")

optimizer = GeneticOptimizer(
    population_size, mutation_rate, num_generations,
    Lambda_range, DC_range, w_range,  height_range
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
    # --- Loop Principal de Otimização (Gerações) ---
    for gen_num in range(num_generations):
        generations_processed += 1
        
        print(f"\n--- Processando Geração {gen_num + 1}/{num_generations} ---")
        
        # --- PASSO 1: Limpeza dos arquivos temporários ---
        # A limpeza agora é feita ANTES de iniciar a nova sessão do Lumerical.
        delete_directory_contents(_temp_directory)
        
        # --- PASSO 2: Inicia uma NOVA sessão Lumerical para esta geração ---
        with lumapi.FDTD(hide=False) as fdtd:
            
            # A chamada para a simulação agora está dentro do seu próprio bloco 'with'
            S_matrixes_for_generation, frequencies = simulate_generation_lumerical(
                fdtd,
                current_population,
                _temp_fsp_base_path,
                _geometry_lsf_script_path,
                _simulation_lsf_script_path,
                _simulation_spectra_directory,
                _temp_directory
            )
        # --- A sessão 'fdtd' é AUTOMATICAMENTE fechada aqui, liberando os arquivos ---
        
        # --- Pós-Processamento e Evolução (fora do bloco 'with') ---
        print("\n  [Job Manager] Pós-processando os resultados da geração...")
        S11_for_gen = calculate_mean_S11_for_generation(
            S_matrixes_for_generation,
            current_population,
        )
        print("S11 médio da geração:", S11_for_gen)


        for i, chromosome in enumerate(current_population):
            individual_data = chromosome.copy()
            individual_data['S11'] = S11_for_gen[i]
            individual_data['generation'] = gen_num + 1
            all_individuals_data.append(individual_data)



        try:
            current_population = optimizer.evolve(S11_for_gen)
        except ValueError as e:
            print(f"!!! Erro na evolução da população: {e}")
            break

        print(f"  [Relatório] Atualizando relatório para a Geração {gen_num + 1}...")
        record_experiment_results(
            _simulation_results_directory, optimizer, experiment_start_time,
            Lambda_range, DC_range, w_range, height_range, generations_processed
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

    # --- FINALIZAÇÃO E RELATÓRIO FINAL ---
    print("\n--- Otimização Concluída ---")
    if optimizer.best_individual:
        print(f"Melhor cromossomo encontrado: {optimizer.best_individual}")
        print(f"Melhor Fitness (S11 Médio) atingido: {optimizer.best_fitness:.4e}")
    else:
        print("Nenhum melhor indivíduo encontrado durante a otimização.")
except Exception as e:
    print(f"!!! Erro fatal no script principal de otimização: {e}")

finally:
    # --- Limpeza Final ---
    print("\nIniciando limpeza final...")
    delete_directory_contents(_temp_directory)
    if os.path.exists(_temp_fsp_base_path):
        os.remove(_temp_fsp_base_path)
        print(f"[Limpeza Final] Arquivo base removido: {_temp_fsp_base_path}")

print("\nScript principal (main.py) finalizado.")