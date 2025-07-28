# utils/multi_chrom.py

import os
import threading
import queue
import lumapi
import numpy as np

# Importa as funções auxiliares que as threads precisarão
from .lumerical_workflow import run_lumerical_workflow
from .post_processing import calculate_delta_amp

class ParallelChromosomeSimulator:
    def __init__(self, max_simultaneous_sessions, stop_event,
                 project_directory, temp_fsp_paths,
                 create_lsf_script_file_name,
                 run_sim_lsf_script_file_name, simulation_spectra_directory):
        
        # Atributos da instância para configuração
        self.max_simultaneous_sessions = max_simultaneous_sessions
        self.stop_event = stop_event
        self.project_directory = project_directory
        self.temp_fsp_paths = temp_fsp_paths
        self.create_lsf_script_file_name = create_lsf_script_file_name
        self.run_sim_lsf_script_file_name = run_sim_lsf_script_file_name
        self.simulation_spectra_directory = simulation_spectra_directory
        
        # Componentes gerenciados pela classe para cada chamada de simulação paralela
        self._results_queue = queue.Queue()
        self._threads = []
        self._thread_limiter = threading.BoundedSemaphore(value=self.max_simultaneous_sessions)

    def _simulate_single_chromosome(self, chrom_id, params):
        """
        Função worker para ser executada por cada thread.
        Realiza a simulação Lumerical e o pós-processamento para um único cromossomo.
        Este método agora acessa os atributos da classe via 'self'.
        """
        # Caminho para o FSP temporário que esta thread usará
        # A escolha do FSP temporário é feita pelo método principal simulate_generation_parallel
        # e o caminho específico é passado para o worker.
        fsp_temp_path_for_this_thread = self.temp_fsp_paths[chrom_id % self.max_simultaneous_sessions]
        
        # Cada thread usará um arquivo .h5 temporário único
        thread_h5_output_file = os.path.join(self.simulation_spectra_directory, f"temp_monitor_data_chrom{chrom_id}.h5")

        current_delta_amp = -float('inf') # Valor padrão em caso de falha

        try:
            if self.stop_event.is_set():
                print(f"  [Thread {chrom_id}] Recebido sinal de parada. Abortando simulação.")
                return # Sai da função imediatamente

            with lumapi.FDTD(hide=True) as fdtd_session:
                print(f"  [Thread {chrom_id}] Sessão Lumerical FDTD iniciada para Cromossomo {chrom_id} usando FSP: {fsp_temp_path_for_this_thread}")
                
                run_lumerical_workflow(
                    fdtd_session,
                    params['s'], params['w'], params['l'], params['height'],
                    fsp_temp_path_for_this_thread, # Usa o FSP temporário da thread
                    self.create_lsf_script_file_name,
                    self.run_sim_lsf_script_file_name,
                    thread_h5_output_file
                )
                print(f"  [Thread {chrom_id}] Simulação do Cromossomo {chrom_id} concluída. Processando resultados...")

                if self.stop_event.is_set():
                    print(f"  [Thread {chrom_id}] Recebido sinal de parada após simulação. Não processando resultados.")
                    return
                
                current_delta_amp = calculate_delta_amp(thread_h5_output_file, monitor_name='in')
                print(f"  [Thread {chrom_id}] Delta Amp para Cromossomo {chrom_id}: {current_delta_amp:.4e}")

        except Exception as e:
            print(f"!!! [Thread {chrom_id}] ERRO na simulação/pós-processamento do Cromossomo {chrom_id}: {e}")
            current_delta_amp = -float('inf') # Penaliza em caso de erro

        finally:
            # Garante que o arquivo temporário seja removido, mesmo em caso de erro ou parada
            if os.path.exists(thread_h5_output_file):
                try:
                    os.remove(thread_h5_output_file)
                except OSError as e:
                    print(f"!!! [Thread {chrom_id}] Erro ao remover arquivo temporário '{thread_h5_output_file}': {e}")
            
            # Só coloca na fila se não foi abortado ou se o resultado é de uma simulação falha
            if not self.stop_event.is_set() or current_delta_amp == -float('inf'):
                self._results_queue.put((chrom_id, current_delta_amp))
            
            # Libera o semáforo para permitir que outras threads iniciem
            self._thread_limiter.release()

    def simulate_generation(self, chromosomes_to_sim):
        """
        Orquestra a simulação de uma geração de cromossomos em paralelo usando threads.
        Este método é o ponto de entrada principal para a simulação paralela de uma geração.
        """
        # Limpa o estado para uma nova geração
        self._results_queue = queue.Queue()
        self._threads = []
        # O _thread_limiter não precisa ser redefinido se seus max_workers são fixos,
        # mas garantir que ele esteja no estado correto para uma nova rodada é bom.
        # Ele já é um BoundedSemaphore, então seu contador deve ser 0 após a rodada anterior.
        # Se você tivesse que redefinir, seria:
        # self._thread_limiter = threading.BoundedSemaphore(value=self.max_simultaneous_sessions)


        # Inicia as threads para simular cada cromossomo
        for i, params in enumerate(chromosomes_to_sim):
            if self.stop_event.is_set(): # Verifica se deve parar antes de iniciar novas threads
                print(f"  [Gerenciador de Threads] Sinal de parada detectado. Não iniciando mais simulações.")
                break # Sai do loop de iniciar threads

            self._thread_limiter.acquire() # Adquire um slot no limite de threads
            
            thread = threading.Thread(
                target=self._simulate_single_chromosome,
                args=(i, params) # Apenas os argumentos específicos da thread
            )
            self._threads.append(thread)
            thread.start()

        # Espera que todas as threads que foram iniciadas terminem
        for thread in self._threads:
            thread.join()

        # Coleta os resultados da fila e organiza na ordem original dos cromossomos
        delta_amp_results_for_gen = [-float('inf')] * len(chromosomes_to_sim) # Inicialize com -inf
        while not self._results_queue.empty():
            chrom_id, delta_amp = self._results_queue.get()
            delta_amp_results_for_gen[chrom_id] = delta_amp
        
        return delta_amp_results_for_gen