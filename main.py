# main.py

import sys
import os
import datetime
import threading
import signal
import shutil

# --- Configuração do Python Path para o Lumerical API ---
_lumapi_module_path = "C:\\Program Files\\Lumerical\\v241\\api\\python"

if _lumapi_module_path not in sys.path:
    sys.path.append(_lumapi_module_path)

# --- Importações de Módulos Personalizados ---
from utils.genetic import GeneticOptimizer
from utils.experiment_end import record_experiment_results
from utils.multi_chrom import ParallelChromosomeSimulator

# --- Configurações Globais ---
_project_directory = "C:\\Users\\USUARIO\\OneDrive\\Lumerical\\metamaterial_guide_otimization"
_fsp_file_name = "guide.fsp"
_original_fsp_path = os.path.join(_project_directory, _fsp_file_name)
_create_lsf_script_file_name = os.path.join(_project_directory, "resources", "create_guide_fdtd.lsf")
_run_sim_lsf_script_file_name = os.path.join(_project_directory, "resources", "run_simu_guide_fdtd.lsf")
_simulation_spectra_directory = os.path.join(_project_directory, "simulation_spectra")
_temp_fsp_base_name = "guide_temp_"

os.makedirs(_simulation_spectra_directory, exist_ok=True)

# --- Sinal de Interrupção para Threads ---
stop_simulation_event = threading.Event()

# --- Handler para Ctrl+C ---
def signal_handler(sig, frame):
    print("\nCtrl+C detectado! Sinalizando threads para parar...")
    stop_simulation_event.set()

signal.signal(signal.SIGINT, signal_handler)

print("#########################################################################")
print("Iniciando o script principal (main.py) para otimização do guia de onda...")
print("#########################################################################")

# --- Definição dos Ranges de Parâmetros Construtivos ---
s_range = (0.01e-6, 0.5e-6)
w_range = (0.2e-6, 1.2e-6)
l_range = (0.01e-6, 0.5e-6)
height_range = (0.1e-6, 0.5e-6)

# --- Configuração do Algoritmo Genético ---
population_size = 6
mutation_rate = 0.2
num_generations = 2

# --- Configuração de Multithreading ---
_max_simultaneous_lumerical_sessions = 3

# --- Prepara arquivos FSP temporários ---
temp_fsp_paths = []
for i in range(_max_simultaneous_lumerical_sessions):
    temp_fsp_path = os.path.join(_project_directory, f"{_temp_fsp_base_name}{i}.fsp")
    shutil.copy(_original_fsp_path, temp_fsp_path)
    temp_fsp_paths.append(temp_fsp_path)
    print(f"Copiado {_original_fsp_path} para {temp_fsp_path}")

# --- Instancia o otimizador genético ---
optimizer = GeneticOptimizer(
    population_size, mutation_rate, num_generations,
    s_range, w_range, l_range, height_range
)

# --- Inicializa o ParallelChromosomeSimulator ---
# Passa todas as configurações necessárias para o construtor da classe
parallel_simulator = ParallelChromosomeSimulator(
    max_simultaneous_sessions=_max_simultaneous_lumerical_sessions,
    stop_event=stop_simulation_event,
    project_directory=_project_directory, # Embora não diretamente usado pela classe, é bom manter aqui se for necessário no futuro
    temp_fsp_paths=temp_fsp_paths,
    create_lsf_script_file_name=_create_lsf_script_file_name,
    run_sim_lsf_script_file_name=_run_sim_lsf_script_file_name,
    simulation_spectra_directory=_simulation_spectra_directory
)


# --- Inicializa a primeira população ---
optimizer.initialize_population()
current_chromosomes_for_sim = [{k: chrom[k] for k in ['s', 'w', 'l', 'height']} for chrom in optimizer.population]

experiment_start_time = datetime.datetime.now()

try:
    # --- Loop Principal de Otimização (Gerações) ---
    for gen_num in range(num_generations):
        if stop_simulation_event.is_set():
            print("Sinal de parada detectado. Encerrando otimização.")
            break
        
        print(f"\n--- Processando Geração {gen_num + 1}/{num_generations} ---")
        print(f"  Simulando {len(current_chromosomes_for_sim)} cromossomos em paralelo (até {_max_simultaneous_lumerical_sessions} por vez)...")

        # Chama o método da instância para simular a geração
        delta_amp_results_for_gen = parallel_simulator.simulate_generation(current_chromosomes_for_sim)

        if stop_simulation_event.is_set():
            print("Sinal de parada detectado após simulações. Encerrando otimização.")
            break

        print(f"\n--- Evoluindo população com resultados da Geração {gen_num + 1} ---")
        try:
            current_chromosomes_for_sim = optimizer.evolve(delta_amp_results_for_gen)
        except ValueError as e:
            print(f"!!! Erro na evolução da população: {e}")
            break

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
        s_range, w_range, l_range, height_range
    )

except Exception as e:
    print(f"!!! Erro fatal no script principal de otimização: {e}")
finally:
    # --- Limpeza dos arquivos FSP temporários ---
    print("\nIniciando limpeza dos arquivos FSP temporários...")
    for temp_fsp in temp_fsp_paths:
        if os.path.exists(temp_fsp):
            try:
                os.remove(temp_fsp)
                print(f"Removido: {temp_fsp}")
            except OSError as e:
                print(f"!!! Erro ao remover {temp_fsp}: {e}")
    print("Limpeza concluída.")


print("\nScript principal (main.py) finalizado.")