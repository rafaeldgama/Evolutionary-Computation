import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from benchmark import PymooWrapper
from optimizers.pso import ParticleSwarmOptimizer
from optimizers.nsga2 import NSGAII
from components.crossover import SimulatedBinaryCrossover
from components.mutations import PolynomialMutation
from components.selections import CrowdedTournamentSelection

# problem = Ackley()
# optimizer = ParticleSwarmOptimizer(problem=problem,
#                                    direction='min',
#                                    population_size=100,
#                                    dimensions=2,
#                                    bounds=[-5,5])

# fitness = []
# for generation in tqdm(range(1,150+1)):
#     best_pos, best_fit = optimizer.evolve()
#     fitness.append(best_fit)
#     print(f"Generation: {generation} | Best fitness: {best_fit}")

if __name__ == "__main__":

    n_var = 30
    problem = PymooWrapper("zdt1", n_var=n_var)
    true_pf = problem.get_true_pareto_front()

    optimizer = NSGAII(problem=problem,
                       population_size=100,
                       dimensions=n_var,
                       bounds=problem.bounds,
                       crossover_operator=SimulatedBinaryCrossover(probability=0.9, eta_c=20),
                       selection_operator=CrowdedTournamentSelection(),
                       mutation_operator=PolynomialMutation(mutation_rate=1.0/n_var, bounds=problem.bounds, eta_m=20))

    generations = 500

    plt.ion()
    fig, ax = plt.subplots(figsize=(8,6))

    for gen in range(generations):

        optimizer.evolve()

        if (gen+1)%5 == 0 or gen == 0:
            ax.clear()
            if true_pf is not None:
                ax.plot(true_pf[:, 0], true_pf[:, 1], color='red', label='True Pareto Front', zorder=1)

            current_fitness = optimizer.fitness
            ax.scatter(current_fitness[:, 0], current_fitness[:, 1], 
                       facecolors='none', edgecolors='blue', label='Population', zorder=2)

            # Formatação do Gráfico
            ax.set_title(f"NSGA-II: ZDT1 - Generation {gen + 1}")
            ax.set_xlabel("Objective 1 (Min)")
            ax.set_ylabel("Objective 2 (Min)")
            # ax.set_xlim(-0.1, 1.1)
            # ax.set_ylim(-0.1, 1.1)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Desenha o frame e pausa brevemente
            plt.draw()
            plt.pause(0.01)
            
    print("Evolution process concluded!")
    
    # Desliga o modo interativo e mantém o gráfico final aberto
    plt.ioff() 
    plt.show()