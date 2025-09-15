# utils/genetic.py (Modificado)

import random
import numpy as np

class GeneticOptimizer:
    # [MODIFICADO] Adicionado 'total_length_range' à assinatura do método
    def __init__(self, population_size, mutation_rate, generations,
                 s_range, w_range, l_range, height_range, total_length_range):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.param_ranges = {
            's': s_range,
            'w': w_range,
            'l': l_range,
            'height': height_range,
            'total_length': total_length_range  # [ADICIONADO] Novo parâmetro
        }
        self.population = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.fitness_history = []

        # Os valores de referência e mutação não precisam ser alterados
        # a menos que você queira uma inicialização focada para total_length também.
        self.reference_params = {
            's': 0.15e-6,
            'w': 0.5e-6,
            'l': 0.15e-6,
            'height': 0.22e-6,
            'total_length': 40e-6 # Exemplo de valor de referência
        }
        self.initial_mutation_amplitude = {
            's': 0.5 * self.reference_params['s'],
            'w': 0.5 * self.reference_params['w'],
            'l': 0.5 * self.reference_params['l'],
            'height': 0.5 * self.reference_params['height'],
            'total_length': 0.2 * self.reference_params['total_length'] # Ex: variação de 20%
        }
        self.local_mutation_step = {
            param: (self.param_ranges[param][1] - self.param_ranges[param][0]) * 0.05
            for param in self.param_ranges
        }


    def _constrain_param(self, param_name, value):
        min_val, max_val = self.param_ranges[param_name]
        return max(min_val, min(max_val, value))


    def create_chromosome(self, reference_based=False):
        chromosome = {}
        if reference_based and self.population_size > 1: # Evita erro se a população for 1
            for param, ref_val in self.reference_params.items():
                if param in self.initial_mutation_amplitude:
                    variation_amplitude = self.initial_mutation_amplitude[param]
                    val = random.uniform(ref_val - variation_amplitude, ref_val + variation_amplitude)
                    chromosome[param] = self._constrain_param(param, val)
        else:
            for param in self.param_ranges:
                chromosome[param] = random.uniform(*self.param_ranges[param])
        return chromosome


    def initialize_population(self):
        self.population = []
        num_ref_based = self.population_size // 2
        num_random = self.population_size - num_ref_based

        for _ in range(num_ref_based):
            self.population.append(self.create_chromosome(reference_based=True))

        for _ in range(num_random):
            self.population.append(self.create_chromosome(reference_based=False))

        random.shuffle(self.population)


    def calculate_fitness(self, fitness_score):
        if np.isinf(fitness_score) or np.isnan(fitness_score):
            return -float('inf')
        return fitness_score


    def select_parents(self):
        # Seleção por torneio
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


    def mutate(self, chromosome, mutation_type='local'):
        if random.random() < self.mutation_rate:
            param_to_mutate = random.choice(list(self.param_ranges.keys()))
            param_range = self.param_ranges[param_to_mutate]
            current_val = chromosome[param_to_mutate]

            if mutation_type == 'local':
                mutation_step = self.local_mutation_step[param_to_mutate]
                new_val = current_val + random.uniform(-mutation_step, mutation_step)
            elif mutation_type == 'global':
                new_val = random.uniform(*param_range)
            else: # Fallback
                mutation_step = self.local_mutation_step[param_to_mutate]
                new_val = current_val + random.uniform(-mutation_step, mutation_step)
            
            chromosome[param_to_mutate] = self._constrain_param(param_to_mutate, new_val)
        return chromosome


    def evolve(self, current_generation_fitness_scores):
        if len(current_generation_fitness_scores) != len(self.population):
            raise ValueError("O número de scores de fitness não corresponde ao tamanho da população.")

        for i, individual in enumerate(self.population):
            individual_fitness = self.calculate_fitness(current_generation_fitness_scores[i])
            individual['fitness'] = individual_fitness

            if individual_fitness > self.best_fitness:
                self.best_fitness = individual_fitness
                self.best_individual = {k: individual[k] for k in self.param_ranges.keys()}
                self.best_individual['fitness'] = self.best_fitness

        self.fitness_history.append(self.best_fitness)

        new_population = []
        if self.best_individual:
            elite_chromosome = {k: self.best_individual[k] for k in self.param_ranges.keys()}
            new_population.append(elite_chromosome)

        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents()
            child = random.choice(self.crossover(parent1, parent2))
            
            if random.random() < 0.5:
                child = self.mutate(child, mutation_type='local')
            else:
                child = self.mutate(child, mutation_type='global')

            new_population.append(child)

        self.population = new_population
        return [{k: chrom[k] for k in self.param_ranges.keys()} for chrom in self.population]