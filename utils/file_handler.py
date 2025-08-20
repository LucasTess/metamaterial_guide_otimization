# file_handler.py
import os
import shutil
def clean_simulation_directory(directory_path, file_extension=None):
    """
    Limpa todos os arquivos em um diretório com uma extensão específica.
    
    Args:
        directory_path (str): O caminho do diretório a ser limpo.
        file_extension (str): A extensão dos arquivos a serem removidos (ex: '.h5', '.fsp').
                               Se for None, todos os arquivos serão removidos.
    """
    if not os.path.exists(directory_path):
        return
        
    print(f"Limpando o diretório de simulação: {directory_path} (arquivos com extensão: {file_extension})...")
    for filename in os.listdir(directory_path):
        if file_extension and not filename.endswith(file_extension):
            continue
            
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Erro ao remover o arquivo {file_path}: {e}")

def delete_directory_contents(directory_path):
    """
    Deleta recursivamente todos os arquivos e pastas dentro de um diretório.
    O diretório pai (directory_path) é preservado.

    Args:
        directory_path (str): O caminho do diretório a ser esvaziado.
    """
    if not os.path.exists(directory_path):
        print(f"Aviso: Diretório não encontrado, não é possível limpar: {directory_path}")
        return

    print(f"Esvaziando o diretório: {directory_path} (todos os arquivos e pastas)...")
    
    # Itera sobre todos os arquivos e diretórios dentro do diretório principal
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        try:
            # Verifica se o item é um arquivo
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Deleta o arquivo
            # Se for um diretório, usa shutil.rmtree para deletar recursivamente
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f'Erro ao deletar {item_path}. Motivo: {e}')

