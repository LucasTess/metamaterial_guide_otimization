# main.py

import sys
import os
import random # Usado para gerar IDs de simulação
import time   # Para pausas e simulação de tempo

# --- Configuração do Python Path para o Lumerical API ---
_lumapi_module_path = "C:\\Program Files\\Lumerical\\v241\\api\\python"

if _lumapi_module_path not in sys.path:
    sys.path.append(_lumapi_module_path)
    import lumapi
# --- Importações de Módulos Personalizados ---
from utils.lumerical_workflow import run_lumerical_workflow 
from utils.post_processing import calculate_delta_amp     
from utils.genetic import GeneticOptimizer                

# --- Configurações Globais ---
_project_directory = "C:\\Users\\USUARIO\\OneDrive\\Lumerical\\metamaterial_guide_otimization"
_fsp_file_name = "guide.fsp"
_create_lsf_script_file_name = os.path.join(_project_directory, "resources", "create_guide_fdtd.lsf")
_run_sim_lsf_script_file_name = os.path.join(_project_directory, "resources", "run_simu_guide_fdtd.lsf")
_simulation_spectra_directory = os.path.join(_project_directory, "simulation_spectra")
_h5_output_file = os.path.join(_simulation_spectra_directory, "current_monitor_data.h5")

print("#########################################################################")
print("Iniciando o script principal (main.py) para otimização do guia de onda...")
print("#########################################################################")
# --- Definição dos Ranges de Parâmetros Construtivos ---
s_range = (0.01e-6, 0.5e-6)   #Artigo: 0.15e-6
w_range = (0.2e-6, 1.2e-6)    #Artigo: 0.5e-6
l_range = (0.01e-6, 0.5e-6)    #Artigo: 0.15e-6
height_range = (0.1e-6, 0.5e-6) #Artigo: 0.22e-6

# --- Configuração do Algoritmo Genético ---
population_size = 4
mutation_rate = 0.1
num_generations = 4 

# Instancia o otimizador genético
optimizer = GeneticOptimizer(
    population_size, mutation_rate, num_generations,
    s_range, w_range, l_range, height_range
)

# --- Inicializa a primeira população ---
optimizer.initialize_population()
current_chromosomes_for_sim = [{k: chrom[k] for k in ['s', 'w', 'l', 'height']} for chrom in optimizer.population]

fdtd_session = None 
try:
    # --- Loop Principal de Otimização (Gerações) ---
    for gen_num in range(num_generations):
        print(f"\n--- Processando Geração {gen_num + 1}/{num_generations} ---")

        delta_amp_results_for_gen = [] # Para armazenar os resultados desta geração

        # --- Loop para Simular CADA Cromossomo da Geração Atual ---
        for i, params in enumerate(current_chromosomes_for_sim):
            # Cria um ID único para o arquivo .h5 deste cromossomo

            print(f"Simulando Cromossomo {i+1} (s={params['s']:.2e}, w={params['w']:.2e}, l={params['l']:.2e}, h={params['height']:.2e})...")

            # 1. Chama run_lumerical_workflow: Realiza a simulação e salva os resultados.
            try:
                with lumapi.FDTD() as fdtd_session:
                    print("\nSessão Lumerical FDTD iniciada.")
                    run_lumerical_workflow(
                        fdtd_session,
                        params['s'],
                        params['w'],
                        params['l'],
                        params['height'],
                        os.path.join(_project_directory, _fsp_file_name),
                        _create_lsf_script_file_name,
                        _run_sim_lsf_script_file_name,
                        _h5_output_file
                    )

            except Exception as e:
                print(f"!!! ERRO na simulação do cromossomo {i+1}: {e}")
                # Penaliza este cromossomo com um delta_amp muito baixo se a simulação falhar
                delta_amp_results_for_gen.append(-float('inf'))
                continue # Pula para o próximo cromossomo

            # 2. Chama calculate_delta_amp: Lê o .h5 e calcula o delta_amp
            try:
                current_delta_amp = calculate_delta_amp(_h5_output_file, monitor_name='in')
                delta_amp_results_for_gen.append(current_delta_amp)
            except Exception as e:
                print(f"!!! ERRO no pós-processamento do cromossomo {i+1}: {e}")
                # Penaliza este cromossomo se o pós-processamento falhar
                delta_amp_results_for_gen.append(-float('inf'))

        # 3. Envia os resultados para o otimizador genético e obtém a próxima geração
        print(f"\n--- Evoluindo população com resultados da Geração {gen_num + 1} ---")
        try:
            # optimizer.evolve retornará a lista de parâmetros para a próxima rodada de simulações
            current_chromosomes_for_sim = optimizer.evolve(delta_amp_results_for_gen)
        except ValueError as e:
            print(f"!!! Erro na evolução da população: {e}")
            break # Sai do loop principal se a evolução falhar

    print("\n--- Otimização Concluída ---")
    if optimizer.best_individual:
        print(f"Melhor cromossomo encontrado: {optimizer.best_individual}")
        print(f"Melhor Delta Amplitude atingido: {optimizer.best_fitness:.4e}")
    else:
        print("Nenhum melhor indivíduo encontrado durante a otimização.")

except Exception as e:
    print(f"!!! Erro fatal no script principal de otimização: {e}")

print("\nScript principal (main.py) finalizado.")