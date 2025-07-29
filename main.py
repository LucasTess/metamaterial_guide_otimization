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
num_generations = 10 # Limite máximo de gerações

# --- Critério de Convergência ---
enable_convergence_check = True 
convergence_threshold_percent = 5.0 # Porcentagem de mudança mínima para continuar

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
parallel_simulator = ParallelChromosomeSimulator(
    max_simultaneous_sessions=_max_simultaneous_lumerical_sessions,
    stop_event=stop_simulation_event,
    project_directory=_project_directory,
    temp_fsp_paths=temp_fsp_paths,
    create_lsf_script_file_name=_create_lsf_script_file_name,
    run_sim_lsf_script_file_name=_run_sim_lsf_script_file_name,
    simulation_spectra_directory=_simulation_spectra_directory
)

# --- Inicializa a primeira população ---
optimizer.initialize_population()
current_chromosomes_for_sim = [{k: chrom[k] for k in ['s', 'w', 'l', 'height']} for chrom in optimizer.population]

experiment_start_time = datetime.datetime.now()

previous_best_fitness = -float('inf') 
generations_processed = 0 # <--- NOVO: Contador de gerações processadas

try:
    # --- Loop Principal de Otimização (Gerações) ---
    for gen_num in range(num_generations):
        generations_processed += 1 # Incrementa o contador para cada geração tentada
        
        if stop_simulation_event.is_set():
            print("Sinal de parada detectado. Encerrando otimização.")
            break
        
        print(f"\n--- Processando Geração {gen_num + 1}/{num_generations} ---")
        print(f"  Simulando {len(current_chromosomes_for_sim)} cromossomos em paralelo (até {_max_simultaneous_lumerical_sessions} por vez)...")

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

        # --- LÓGICA DE CONVERGÊNCIA (CONDICIONAL) ---
        if enable_convergence_check: # <--- Só executa se a checagem estiver ativada
            current_best_fitness = optimizer.best_fitness

            # A partir da segunda geração (gen_num > 0)
            if gen_num > 0:
                if previous_best_fitness == -float('inf'):
                    print("  [Convergência] Fitness anterior não válido, continuando...")
                elif current_best_fitness == previous_best_fitness:
                    print(f"  [Convergência] Melhor fitness da geração atual ({current_best_fitness:.4e}) é idêntico ao anterior. Convergência atingida.")
                    break
                elif previous_best_fitness != 0:
                    percentage_change = abs((current_best_fitness - previous_best_fitness) / previous_best_fitness) * 100
                    
                    print(f"  [Convergência] Melhor fitness da geração atual: {current_best_fitness:.4e}")
                    print(f"  [Convergência] Melhor fitness da geração anterior: {previous_best_fitness:.4e}")
                    print(f"  [Convergência] Mudança de fitness em relação à geração anterior: {percentage_change:.2f}%")

                    if percentage_change <= convergence_threshold_percent:
                        print(f"  [Convergência] Mudança percentual ({percentage_change:.2f}%) é menor ou igual ao limiar ({convergence_threshold_percent:.2f}%). Convergência atingida.")
                        break
            
            # Atualiza o melhor fitness anterior para a próxima iteração
            previous_best_fitness = current_best_fitness
        else:
            print("  [Convergência] Checagem de convergência desativada. Continuar até o número máximo de gerações.")
            # Se a checagem estiver desativada, ainda precisamos atualizar previous_best_fitness para a próxima iteração
            # caso ela seja reativada, ou para fins de depuração.
            previous_best_fitness = optimizer.best_fitness # Mesmo se não usar, mantenha atualizado

    print("\n--- Otimização Concluída ---")
    if optimizer.best_individual:
        print(f"Melhor cromossomo encontrado: {optimizer.best_individual}")
        print(f"Melhor Delta Amplitude atingido: {optimizer.best_fitness:.4e}")
    else:
        print("Nenhum melhor indivíduo encontrado durante a otimização.")

    # Passa o número de gerações realmente processadas
    record_experiment_results(
        _simulation_spectra_directory,
        optimizer,
        experiment_start_time,
        s_range, w_range, l_range, height_range,
        generations_processed # <--- NOVO ARGUMENTO
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