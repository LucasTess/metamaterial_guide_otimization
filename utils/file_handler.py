import os

# --- Função de Limpeza ---
def clean_simulation_directory(directory_path):
    if not os.path.exists(directory_path):
        return
    print(f"Limpando o diretório de simulação: {directory_path}...")
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Erro ao remover o arquivo {file_path}: {e}")

def remove_file(fsp_path):
        if os.path.exists(fsp_path):
            os.remove(fsp_path)
            print(f"  [Limpeza] Arquivo temporário removido: {os.path.basename(fsp_path)}")