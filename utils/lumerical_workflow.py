# lumerical_workflow.py

import lumapi
import os
import h5py
import numpy as np
import time

def prepare_lumerical_job(fdtd, chromosome, fsp_base_path, geometry_lsf_path, simulation_lsf_path,temp_directory,chrom_id):
    """
    Prepara um único arquivo FSP com os parâmetros de um cromossomo e o salva com um nome único.
    
    Args:
        fdtd: A instância da sessão Lumerical FDTD.
        chromosome (dict): Um dicionário contendo os parâmetros do cromossomo.
        fsp_base_path: O caminho base para o arquivo FSP temporário.
        geometry_lsf_path: O caminho para o script LSF que cria a geometria.
        simulation_lsf_path: O caminho para o script LSF que adiciona os elementos de simulação.
        
    Returns:
        O caminho completo para o arquivo FSP salvo.
    """
    # Cria um nome de arquivo FSP único para o cromossomo
    fsp_file_name = f"guide_temp_chrom_id_{chrom_id}.fsp"
    # O arquivo temporário é salvo no mesmo diretório do arquivo base, ou em um diretório temporário.
    print(f"temp_directory = " + temp_directory)
    fsp_path = os.path.join(temp_directory, fsp_file_name)
    print(f"fsp_path = " + fsp_path)
    # Adicionando uma verificação defensiva para garantir que o arquivo base existe
    if not os.path.exists(fsp_base_path):
        raise FileNotFoundError(f"Erro: O arquivo base '{fsp_base_path}' não foi encontrado.")

    # 1. Carrega o arquivo FSP base
    fdtd.load(fsp_base_path)
    fdtd.switchtolayout()

    # 2. Executa o script LSF para criar a geometria
    with open(geometry_lsf_path, 'r') as f:
        create_lsf_content = f.read()
    fdtd.eval(create_lsf_content)

    # 3. Define os parâmetros do cromossomo na geometria
    fdtd.setnamed("Guia Metamaterial", "s", chromosome['s'])
    fdtd.setnamed("Guia Metamaterial", "w", chromosome['w'])
    fdtd.setnamed("Guia Metamaterial", "l", chromosome['l'])
    fdtd.setnamed("Guia Metamaterial", "height", chromosome['height'])

    # 4. Executa o script LSF para adicionar os elementos de simulação
    with open(simulation_lsf_path, 'r') as f:
        simulate_lsf_content = f.read()
    fdtd.eval(simulate_lsf_content)
    
    # 5. Salva o arquivo FSP modificado com um nome único
    fdtd.save(fsp_path)
    fdtd.runsweep()
    return fsp_path

def simulate_generation_lumerical(fdtd, current_population, fsp_base_path, geometry_lsf_path,
                                  simulation_lsf_path, simulation_spectra_directory,temp_directory):
    """
    Prepara e executa as simulações para uma geração inteira de cromossomos usando a fila de jobs.
    Após a execução, lê os resultados de cada arquivo FSP e os salva em arquivos .h5.
    
    Args:
        fdtd: A instância da sessão Lumerical FDTD.
        current_population (list): Uma lista de dicionários, onde cada um representa um cromossomo.
        fsp_base_path: O caminho base para o arquivo FSP temporário.
        geometry_lsf_path: O caminho para o script LSF que cria a geometria.
        simulation_lsf_path: O caminho para o script LSF que adiciona os elementos de simulação.
        simulation_spectra_directory: O diretório onde os arquivos de saída .h5 serão salvos.
        
    Returns:
        Uma lista completa dos caminhos para os arquivos de saída .h5.
    """
    fsp_paths_for_gen = []
    print(f"Preparando e adicionando {len(current_population)} jobs na fila...")
    chrom_id = 0
    for chromosome in current_population:
        chrom_id = chrom_id + 1
        print(chrom_id)
        fsp_path = prepare_lumerical_job(
            fdtd, chromosome, fsp_base_path, geometry_lsf_path, simulation_lsf_path,
            temp_directory,chrom_id
        )
        fsp_paths_for_gen.append(fsp_path)
        
        # Adiciona o arquivo FSP com nome único à fila de jobs
        #fdtd.addjob(fsp_path)
    
    print("\n  [Job Manager] Executando todos os jobs na fila. Isso pode levar um tempo...")
    #fdtd.runjobs()
    #fdtd.runsweep()

    
    print("  [Job Manager] Todos os jobs da geração foram concluídos. Lendo e salvando os resultados...")

    # --- Pós-processamento e salvamento em disco no Python ---
    S_matrixes_for_generation = []

    for fsp_path in fsp_paths_for_gen:
        try:
            # 1. Carrega o arquivo FSP já simulado para extrair os dados
            fdtd.load(fsp_path)
            in_port_spectrum = fdtd.getresult("FDTD::ports::in", "spectrum")
            S_matrix_dataset = fdtd.getsweepresult("s-parameter sweep","S matrix")
            S_matrix = S_matrix_dataset['S']
            S_matrixes_for_generation.append(S_matrix)
            """
                Em resumo, a matriz S para um dispositivo de 2 portas é composta por 4 termos: S11, S12, S21 e S22. No seu array, eles se traduzem em:

                S11: S_matrix[0, 0, :] (Reflexão no porto 1)

                S12: S_matrix[0, 1, :] (Transmissão do porto 2 para o porto 1)

                S21: S_matrix[1, 0, :] (Transmissão do porto 1 para o porto 2)

                S22: S_matrix[1, 1, :] (Reflexão no porto 2)
            """

        except Exception as e:
            print(f"!!! Erro no pós-processamento do arquivo {os.path.basename(fsp_path)}: {e}")
    print(f" Shape todas Matrizes S: {np.shape(S_matrixes_for_generation)}")        
    return S_matrixes_for_generation,in_port_spectrum