# lumerical_workflow.py

import lumapi
import os
import h5py
import numpy as np
import time

def simulate_and_get_s_matrix(fdtd, chromosome, fsp_base_path, geometry_lsf_path, simulation_lsf_path,temp_directory,chrom_id):
    """
    Prepara, simula e extrai a matriz S para um único cromossomo.
    
    Args:
        fdtd: A instância da sessão Lumerical FDTD.
        chromosome (dict): Um dicionário contendo os parâmetros do cromossomo.
        fsp_base_path: O caminho base para o arquivo FSP.
        geometry_lsf_path: O caminho para o script LSF da geometria.
        simulation_lsf_path: O caminho para o script LSF da simulação.
        temp_directory: O diretório para arquivos temporários.
        chrom_id: O ID único do cromossomo.
        
    Returns:
        Um tuplo contendo a matriz S do cromossomo e as frequências da varredura.
        Retorna (None, None) em caso de erro.
    """
    fsp_file_name = f"guide_temp_chrom_id_{chrom_id}.fsp"
    fsp_path = os.path.join(temp_directory, fsp_file_name)
    
    if not os.path.exists(fsp_base_path):
        raise FileNotFoundError(f"Erro: O arquivo base '{fsp_base_path}' não foi encontrado.")

    try:
        # 1. Carrega o arquivo FSP base
        fdtd.load(fsp_base_path)
        fdtd.switchtolayout()

        # 2. Executa o script LSF para criar a geometria
        with open(geometry_lsf_path, 'r') as f:
            create_lsf_content = f.read()
        fdtd.eval(create_lsf_content)

        # 3. Define os parâmetros do cromossomo
        fdtd.setnamed("Guia Metamaterial", "Lambda", chromosome['Lambda'])
        fdtd.setnamed("Guia Metamaterial", "DC", chromosome['DC'])
        fdtd.setnamed("Guia Metamaterial", "w", chromosome['w'])
        fdtd.setnamed("Guia Metamaterial", "height", chromosome['height'])

        # 4. Executa o script LSF para adicionar os elementos de simulação
        with open(simulation_lsf_path, 'r') as f:
            simulate_lsf_content = f.read()
        fdtd.eval(simulate_lsf_content)
        
        # 5. Salva o arquivo FSP modificado
        fdtd.save(fsp_path)

        # 6. Executa a varredura S (sweep) de forma serial
        print(f"  Iniciando varredura S para o cromossomo {chrom_id}...")
        fdtd.runsweep()
        print(f"  Varredura S do cromossomo {chrom_id} concluída.")

        # 7. Coleta o dataset completo da varredura
        S_matrix_dataset = fdtd.getsweepresult("s-parameter sweep", "S matrix")
        S_matrix = S_matrix_dataset['S']
        frequencies = S_matrix_dataset['f']
        
        return S_matrix, frequencies

    except Exception as e:
        print(f"!!! Erro fatal na simulação do arquivo {os.path.basename(fsp_path)}: {e}")
        return None, None

def simulate_generation_lumerical(fdtd, current_population, fsp_base_path, geometry_lsf_path,
                                  simulation_lsf_path, simulation_spectra_directory, temp_directory):
    """
    Executa as simulações para uma geração inteira de cromossomos de forma serial.
    
    Args:
        fdtd: A instância da sessão Lumerical FDTD.
        current_population (list): Uma lista de dicionários, onde cada um representa um cromossomo.
        fsp_base_path: O caminho base para o arquivo FSP.
        geometry_lsf_path: O caminho para o script LSF da geometria.
        simulation_lsf_path: O caminho para o script LSF da simulação.
        simulation_spectra_directory: O diretório para arquivos de saída.
        temp_directory: O diretório para arquivos temporários.
        
    Returns:
        Uma lista de matrizes S e um array NumPy de frequências.
    """
    S_matrixes_for_generation = []
    frequencies = None
    
    print(f"Executando as simulações de forma serial para {len(current_population)} cromossomos...")
    
    chrom_id = 0
    for chromosome in current_population:
        chrom_id += 1
        S_matrix, current_frequencies = simulate_and_get_s_matrix(
            fdtd, chromosome, fsp_base_path, geometry_lsf_path, simulation_lsf_path, temp_directory, chrom_id
        )

        if S_matrix is not None:
            S_matrixes_for_generation.append(S_matrix)
            if frequencies is None:
                frequencies = current_frequencies
    
    if S_matrixes_for_generation:
        print(f" Shape de todas as Matrizes S coletadas: {np.shape(S_matrixes_for_generation)}")
        
    return S_matrixes_for_generation, frequencies