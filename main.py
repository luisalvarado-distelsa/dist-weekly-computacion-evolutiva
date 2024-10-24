import random
import numpy as np
from typing import Literal, Tuple

# Función de fitness
def fitness(x):
    return -(x - 3)**2 + 10

# Métodos de cruce
def cruce_aritmetico(padre1, padre2):
    alfa = random.random()
    hijo = alfa * padre1 + (1 - alfa) * padre2
    return hijo

def cruce_un_punto(padre1, padre2):
    punto_corte = random.randint(0, 1)
    if punto_corte == 0:
        return padre1
    else:
        return padre2

def cruce_dos_puntos(padre1, padre2):
    punto1 = random.uniform(-10, 10)
    punto2 = random.uniform(-10, 10)
    hijo = (padre1 + padre2) / 2 + (punto1 - punto2) / 2
    return hijo

def cruce_uniforme(padre1, padre2):
    if random.random() < 0.5:
        return padre1
    else:
        return padre2

# Métodos de mutación
def mutacion_gaussiana(individuo, sigma=0.1):
    return individuo + random.gauss(0, sigma)

def mutacion_uniforme(individuo):
    return random.uniform(-10, 10)

def mutacion_polinomial(individuo, eta=20):
    u = random.random()
    if u < 0.5:
        delta = (2 * u)**(1 / (eta + 1)) - 1
    else:
        delta = 1 - (2 * (1 - u))**(1 / (eta + 1))
    return individuo + delta

# Algoritmo Evolutivo
def algoritmo_evolutivo(
    tam_poblacion: int=10,
    num_generaciones: int=20,
    rango: Tuple=(-10, 10),
    metodo_cruce: Literal['aritmetico', 'un_punto', 'dos_puntos', 'uniforme']='aritmetico',
    metodo_mutacion: Literal['gaussiana', 'uniforme', 'polinomial']='gaussiana',
    epsilon: float=0.001,
    solucion: float=0
):
    # Generar población inicial
    poblacion = [random.uniform(rango[0], rango[1]) for _ in range(tam_poblacion)]

    for generacion in range(num_generaciones):
        # Evaluar fitness
        fitness_poblacion = [fitness(ind) for ind in poblacion]

        # Selección de los dos mejores individuos
        padres = [x for _, x in sorted(zip(fitness_poblacion, poblacion), reverse=True)][:2]

        # Cruce
        if metodo_cruce == 'aritmetico':
            hijo = cruce_aritmetico(padres[0], padres[1])
        elif metodo_cruce == 'un_punto':
            hijo = cruce_un_punto(padres[0], padres[1])
        elif metodo_cruce == 'dos_puntos':
            hijo = cruce_dos_puntos(padres[0], padres[1])
        elif metodo_cruce == 'uniforme':
            hijo = cruce_uniforme(padres[0], padres[1])
        else:
            raise ValueError("Método de cruce no válido.")

        # Mutación
        if metodo_mutacion == 'gaussiana':
            hijo_mutado = mutacion_gaussiana(hijo)
        elif metodo_mutacion == 'uniforme':
            hijo_mutado = mutacion_uniforme(hijo)
        elif metodo_mutacion == 'polinomial':
            hijo_mutado = mutacion_polinomial(hijo)
        else:
            raise ValueError("Método de mutación no válido.")

        # Reemplazo del peor individuo
        indice_peor = fitness_poblacion.index(min(fitness_poblacion))
        poblacion[indice_peor] = hijo_mutado

        # Mejor individuo y error
        mejor_individuo = padres[0]
        mejor_fitness = fitness(mejor_individuo)
        error = abs(mejor_individuo - solucion)
        # Mostrar resultados de la generación
        print(f"Generación {generacion+1}: Mejor Individuo = {mejor_individuo:.7f}, Fitness = {mejor_fitness:.7f}, Error = {error:.7f}")
        # Condición de parada
        if error < epsilon:
            break

    # Resultado final
    mejor_individuo = max(poblacion, key=fitness)
    mejor_fitness = fitness(mejor_individuo)
    return mejor_individuo, mejor_fitness, generacion

# Ejecución del algoritmo
if __name__ == "__main__":
    mejor_ind, mejor_fit, generacion = algoritmo_evolutivo(
        tam_poblacion=10,
        num_generaciones=1000000,
        rango=(-10, 10),
        metodo_cruce='dos_puntos',
        metodo_mutacion='gaussiana',
        epsilon=0.001,
        solucion=3
    )
    print(f"\nMejor individuo encontrado: x = {mejor_ind:.7f}, Fitness = {mejor_fit:.7f}, Generación = {generacion+1}")
