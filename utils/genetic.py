import random
import numpy as np

class GeneticOptimizer:
    def __init__(self, population_size, mutation_rate, generations,
                 s_range, w_range, l_range, height_range):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.param_ranges = {
            's': s_range,
            'w': w_range,
            'l': l_range,
            'height': height_range
        }
        self.population = []
        self.best_individual = None
        self.best_fitness = -float('inf')

        # Definir os valores de referência do artigo AQUI
        self.reference_params = {
            's': 0.15e-6,
            'w': 0.5e-6,
            'l': 0.15e-6,
            'height': 0.22e-6
        }
        # Definir a amplitude da variação inicial/mutação em torno dos valores de referência
        # Por exemplo, 10% do valor de referência para mutação/inicialização localizada
        self.initial_mutation_amplitude = {
            's': 0.5 * self.reference_params['s'], # Ex: 50%
            'w': 0.5 * self.reference_params['w'],
            'l': 0.5 * self.reference_params['l'],
            'height': 0.5 * self.reference_params['height']
        }


    def _constrain_param(self, param_name, value):
        """Garanta que o valor esteja dentro dos ranges definidos."""
        min_val, max_val = self.param_ranges[param_name]
        return max(min_val, min(max_val, value))


    def create_chromosome(self):
        chromosome = {
            's': random.uniform(*self.param_ranges['s']),
            'w': random.uniform(*self.param_ranges['w']),
            'l': random.uniform(*self.param_ranges['l']),
            'height': random.uniform(*self.param_ranges['height'])
        }
        return chromosome

    def initialize_population(self):
        """
        Inicializa a primeira população.
        Alguns cromossomos serão aleatórios, outros baseados nos valores do artigo.
        """
        self.population = []
        num_ref_chromosomes = max(1, self.population_size // 5) # Ex: 20% da população inicia perto da referência

        # Adiciona cromossomos baseados nos valores de referência
        for _ in range(num_ref_chromosomes):
            new_chrom = {}
            for param, ref_val in self.reference_params.items():
                # Pequena variação em torno do valor de referência
                variation_amplitude = self.initial_mutation_amplitude[param]
                val = random.uniform(ref_val - variation_amplitude, ref_val + variation_amplitude)
                new_chrom[param] = self._constrain_param(param, val) # Garante que está dentro do range geral
            self.population.append(new_chrom)

        # Preenche o restante da população com cromossomos completamente aleatórios
        while len(self.population) < self.population_size:
            self.population.append(self.create_chromosome())

        random.shuffle(self.population) # Mistura a população inicial

    def calculate_fitness(self, delta_amp):
        if np.isinf(delta_amp) or np.isnan(delta_amp):
            return -float('inf')
        return delta_amp

    def select_parents(self):
        pool = random.sample(self.population, min(5, len(self.population)))
        parent1 = max(pool, key=lambda x: x.get('fitness', -float('inf')))

        pool = random.sample(self.population, min(5, len(self.population)))
        parent2 = max(pool, key=lambda x: x.get('fitness', -float('inf')))

        return parent1, parent2

    def crossover(self, parent1, parent2):
        child1 = {}
        child2 = {}
        keys = list(self.param_ranges.keys())
        crossover_point = random.randint(1, len(keys) - 1)

        for i, key in enumerate(keys):
            if i < crossover_point:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]
        
        return child1, child2

    def mutate(self, chromosome):
        """
        Aplica mutação a um cromossomo.
        A mutação agora pode ser mais direcionada (pequena variação) ou aleatória (grande salto).
        """
        if random.random() < self.mutation_rate:
            param_to_mutate = random.choice(list(self.param_ranges.keys()))
            param_range = self.param_ranges[param_to_mutate]
            current_val = chromosome[param_to_mutate]

            # Decidir se a mutação é um pequeno ajuste ou um salto maior
            if random.random() < 0.8: # 80% de chance de pequena mutação (ajuste o percentual)
                # Mutação pequena: variação em torno do valor atual
                # A amplitude da variação diminui com o tempo/gerações, se desejar
                mutation_step = (param_range[1] - param_range[0]) * 0.05 # Ex: 5% do range total
                new_val = current_val + random.uniform(-mutation_step, mutation_step)
            else:
                # Mutação maior: valor aleatório dentro do range total
                new_val = random.uniform(*param_range)
            
            chromosome[param_to_mutate] = self._constrain_param(param_to_mutate, new_val)
        return chromosome

    def evolve(self, current_generation_delta_amps):
        if len(current_generation_delta_amps) != len(self.population):
            raise ValueError("O número de resultados de delta_amp não corresponde ao tamanho da população.")

        for i, individual in enumerate(self.population):
            individual['fitness'] = self.calculate_fitness(current_generation_delta_amps[i])
            if individual['fitness'] > self.best_fitness:
                self.best_fitness = individual['fitness']
                self.best_individual = {k: individual[k] for k in self.param_ranges.keys()}
                self.best_individual['fitness'] = self.best_fitness

        new_population = []
        if self.best_individual and self.best_fitness > -float('inf'):
            elite_chromosome = {k: self.best_individual[k] for k in self.param_ranges.keys()}
            new_population.append(elite_chromosome)

        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            new_population.extend([child1, child2])
        
        self.population = new_population[:self.population_size]

        return [{k: chrom[k] for k in self.param_ranges.keys()} for chrom in self.population]