# utils/fitness_functions.py

import h5py
import numpy as np
from abc import ABC, abstractmethod

# ----------------------------------------------------------------------------
# 1. DEFINIÇÃO DA INTERFACE DA ESTRATÉGIA DE FITNESS
# ----------------------------------------------------------------------------

class FitnessStrategy(ABC):
    """
    Classe base abstrata (Interface) para todas as estratégias de cálculo de fitness.
    Garante que toda estratégia concreta implemente um método 'calculate',
    tornando-as intercambiáveis no otimizador.
    """
    @abstractmethod
    def calculate(self, output_h5_path: str) -> float:
        """
        Método principal que calcula o valor de fitness a partir de um arquivo H5.

        Args:
            output_h5_path (str): Caminho para o arquivo H5 com os dados do espectro.

        Returns:
            float: O valor de fitness calculado. Um valor maior é considerado melhor.
        """
        pass

# ----------------------------------------------------------------------------
# 2. IMPLEMENTAÇÃO DAS ESTRATÉGIAS CONCRETAS
# ----------------------------------------------------------------------------

class DeltaAmpStrategy(FitnessStrategy):
    """
    Estratégia de Fitness Concreta: Maximiza o contraste do espectro.

    Esta estratégia corresponde à sua implementação original. O objetivo é maximizar
    a soma das diferenças de amplitude entre picos e vales consecutivos no espectro,
    promovendo dispositivos com alta modulação espectral.
    """
    def calculate(self, output_h5_path: str, monitor_name: str = 'in') -> float:
        """
        Calcula a soma das diferenças de amplitude entre picos e vales.
        Pode ser utilizado tanto em in tanto em through
        Maximiza o fenomeno de interferência visto pela porta
        """
        try:
            with h5py.File(output_h5_path, 'r') as f:
                if 'frequencies_hz' not in f or f'power_{monitor_name}' not in f:
                    print(f"AVISO: Arquivo H5 '{output_h5_path}' não contém os datasets esperados. Retornando -inf.")
                    return -np.inf

                frequencies_hz = f['frequencies_hz'][:].flatten()
                power = f[f'power_{monitor_name}'][:].flatten()

                peaks = []
                valleys = []

                for i in range(1, len(power) - 1):
                    if power[i] > power[i-1] and power[i] > power[i+1]:
                        peaks.append((i, power[i]))
                    elif power[i] < power[i-1] and power[i] < power[i+1]:
                        valleys.append((i, power[i]))
                
                if not peaks or not valleys:
                    return 0.0 # Retorna 0 se não encontrar picos ou vales suficientes

                total_delta_amp = 0.0
                for peak_idx, peak_val in peaks:
                    next_valley_val = None
                    for valley_idx, valley_val in valleys:
                        if valley_idx > peak_idx:
                            next_valley_val = valley_val
                            break
                    
                    if next_valley_val is not None:
                        total_delta_amp += abs(peak_val - next_valley_val)

                return total_delta_amp
        except Exception as e:
            print(f"ERRO ao processar o arquivo {output_h5_path}: {e}")
            return -np.inf # Retorna um fitness muito baixo em caso de erro


class BandpassStrategy(FitnessStrategy):
    """
    [PLACEHOLDER] Estratégia para otimizar um filtro passa-banda.
    O objetivo será maximizar a transmissão dentro de uma banda e minimizá-la fora.
    """
    def __init__(self, center_freq: float, bandwidth: float):
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        # Futuros parâmetros poderiam ser adicionados aqui (ex: fator de roll-off)

    def calculate(self, output_h5_path: str) -> float:
        # TODO: Implementar a lógica de cálculo para o filtro passa-banda.
        # A lógica irá ler o espectro e as frequências, e calcular uma pontuação
        # baseada na transmissão dentro vs. fora da banda definida por center_freq e bandwidth.
        # print(f"AVISO: A lógica para BandpassStrategy(center={self.center_freq}, bw={self.bandwidth}) ainda não foi implementada.")
        return 0.0


class LowpassStrategy(FitnessStrategy):
    """
    [PLACEHOLDER] Estratégia para otimizar um filtro passa-baixas.
    O objetivo será maximizar a transmissão abaixo de uma frequência de corte.
    """
    def __init__(self, cutoff_freq: float):
        self.cutoff_freq = cutoff_freq

    def calculate(self, output_h5_path: str) -> float:
        # TODO: Implementar a lógica de cálculo para o filtro passa-baixas.
        # A lógica irá recompensar a alta transmissão abaixo da cutoff_freq
        # e penalizar a transmissão acima dela.
        # print(f"AVISO: A lógica para LowpassStrategy(cutoff={self.cutoff_freq}) ainda não foi implementada.")
        return 0.0


class HighpassStrategy(FitnessStrategy):
    """
    [PLACEHOLDER] Estratégia para otimizar um filtro passa-altas.
    O objetivo será maximizar a transmissão acima de uma frequência de corte.
    """
    def __init__(self, cutoff_freq: float):
        self.cutoff_freq = cutoff_freq

    def calculate(self, output_h5_path: str) -> float:
        # TODO: Implementar a lógica de cálculo para o filtro passa-altas.
        # A lógica irá recompensar a alta transmissão acima da cutoff_freq
        # e penalizar a transmissão abaixo dela.
        # print(f"AVISO: A lógica para HighpassStrategy(cutoff={self.cutoff_freq}) ainda não foi implementada.")
        return 0.0