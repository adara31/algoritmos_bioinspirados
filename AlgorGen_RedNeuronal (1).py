import numpy
import tensorflow.keras
import gc
import random

# Cargar los datos
data_inputs = numpy.load('../phyton/dataset_inputs.npy')
data_outputs = numpy.load('../phyton/dataset_outputs.npy')
data_outputs = tensorflow.keras.utils.to_categorical(data_outputs)

# Crear el modelo de la red neuronal convolucional
input_layer = tensorflow.keras.layers.Input(shape=(100, 100, 3))
conv_layer1 = tensorflow.keras.layers.Conv2D(filters=5,
                                             kernel_size=7,
                                             activation="relu")(input_layer)
max_pool1 = tensorflow.keras.layers.MaxPooling2D(pool_size=(5, 5),
                                                 strides=5)(conv_layer1)
conv_layer2 = tensorflow.keras.layers.Conv2D(filters=3,
                                             kernel_size=3,
                                             activation="relu")(max_pool1)
flatten_layer = tensorflow.keras.layers.Flatten()(conv_layer2)
dense_layer = tensorflow.keras.layers.Dense(15, activation="relu")(flatten_layer)
output_layer = tensorflow.keras.layers.Dense(4, activation="softmax")(dense_layer)

model = tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)

# Obtener la cantidad total de pesos del modelo
initial_weights = model.get_weights()
num_weights = numpy.sum([numpy.prod(w.shape) for w in initial_weights])

# Convertir la lista de pesos del modelo a un solo vector
def weights_to_vector(weights):
    return numpy.concatenate([w.flatten() for w in weights])

# Convertir un vector de pesos en la lista de pesos con las formas originales
def vector_to_weights(vector, model):
    weights = []
    offset = 0
    for layer_weights in model.get_weights():
        shape = layer_weights.shape
        size = numpy.prod(shape)
        weights.append(vector[offset:offset+size].reshape(shape))
        offset += size
    return weights

# Función de aptitud (fitness) para evaluar una solución
def evaluate(individual):
    global model, data_inputs, data_outputs
    weights = vector_to_weights(individual, model)
    model.set_weights(weights)
    
    # Realizar predicciones y calcular la pérdida categórica cruzada
    predictions = model(data_inputs)
    cce = tensorflow.keras.losses.CategoricalCrossentropy()
    fitness = 1.0 / (cce(data_outputs, predictions).numpy() + 0.00000001)  # Fitness inverso de la pérdida
    
    return fitness

# Función de cruce de dos padres (promedio de los pesos)
def combine(parentA, parentB, cRate):
    if random.random() <= cRate:
        return (parentA + parentB) / 2  # Cruce por promedio
    else:
        return parentA

# Función de mutación (ajustar ligeramente los pesos con ruido gaussiano)
def mutate(individual, mRate):
    for i in range(len(individual)):
        if random.random() <= mRate:
            individual[i] += numpy.random.normal(0, 0.1)  # Mutar con ruido gaussiano
    return individual

# Selección por torneo
def select(population, evaluation, tournamentSize):
    winner = numpy.random.randint(0, len(population))
    for _ in range(tournamentSize - 1):
        rival = numpy.random.randint(0, len(population))
        if evaluation[rival] > evaluation[winner]:
            winner = rival
    return population[winner]

# Algoritmo genético para optimizar los pesos de la red neuronal
def geneticAlgorithm(n, populationSize, cRate, mRate, generations):
    # Inicializar la población de individuos (pesos iniciales)
    population = [weights_to_vector(model.get_weights()) + numpy.random.normal(0, 0.1, n) for _ in range(populationSize)]
    evaluation = [evaluate(individual) for individual in population]

    # Mejor individuo inicial
    bestIndividual = max(population, key=lambda ind: evaluate(ind))
    bestEvaluation = evaluate(bestIndividual)

    # Proceso evolutivo
    for generation in range(generations):
        newPopulation = []
        for _ in range(populationSize):
            parentA = select(population, evaluation, 3)
            parentB = select(population, evaluation, 3)
            offspring = combine(parentA, parentB, cRate)
            offspring = mutate(offspring, mRate)
            newPopulation.append(offspring)

        # Evaluar la nueva población
        population = newPopulation
        evaluation = [evaluate(ind) for ind in population]

        # Actualizar el mejor individuo
        currentBest = max(population, key=lambda ind: evaluate(ind))
        currentBestEval = evaluate(currentBest)
        if currentBestEval > bestEvaluation:
            bestIndividual = currentBest
            bestEvaluation = currentBestEval

        # Imprimir información de la generación
        print(f"Generación {generation + 1}: Mejor Fitness = {bestEvaluation}")

        # Liberar memoria innecesaria
        gc.collect()

    return bestIndividual, bestEvaluation

# Ejecutar el algoritmo genético
solution, evaluation = geneticAlgorithm(num_weights, 20, 0.9, 0.1, 100)

# Imprimir la mejor solución encontrada
print(f"Mejor Fitness: {evaluation}")

# Convertir la mejor solución en pesos y asignarla al modelo
best_weights = vector_to_weights(solution, model)
model.set_weights(best_weights)
