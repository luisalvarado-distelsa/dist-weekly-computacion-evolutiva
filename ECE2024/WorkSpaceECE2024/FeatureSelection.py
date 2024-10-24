import csv
import numpy as np
from sklearn import svm
from sklearn import metrics
import afssa
import matplotlib.pyplot as plt

datadir = 'D:/xampp/htdocs/mgilr/ece2024/contents/WorkSpaceECE2024/DatasetECE2024/'
#datadir = '/opt/lampp/htdocs/public/mgilr/ece2024/contents/WorkSpaceECE2024/DatasetECE2024/'

training_data      = np.genfromtxt(datadir + '01_Training.csv', delimiter=',', skip_header=1)
training_classes   = np.genfromtxt(datadir + '01_TrainingClasses.txt')

validation_data    = np.genfromtxt(datadir + '02_Validation.csv', delimiter=',', skip_header=1)
validation_classes = np.genfromtxt(datadir + '02_ValidationClasses.txt')

training_data   = training_data   / training_data.max(axis=0)
validation_data = validation_data / validation_data.max(axis=0)

# Se cargan los nombres de las caracteristicas
# para saber al final, cuales se seleccionan:
lista_caracteristicas = []
with open(datadir + '01_Training.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    for row in csv_reader:
        lista_caracteristicas.append(row)
        break

# Se definen las iteraciones maximas del SA:
max_iters = 2000

# Se genera una solucion inicial de forma
sol_ini = np.random.randint(low=0, high=2, size=(training_data.shape[1]), dtype=int) >= 0

# Se realiza la Seleccion Automatica de caracteristicas:
best_sol,   best_effclass, best_effeat, \
best_effmo, best_model,    perf_data = afssa.search(training_data,
                                                    training_classes,
                                                    validation_data,
                                                    validation_classes,
                                                    max_iters,
                                                    sol_ini)
print("Best Solution Found:")
afssa.__print_performance__(best_effclass, best_effeat, best_effmo, best_sol, lista_caracteristicas, True)

# Entrenamiento y clasificacion utilizando todas las caracteristicas:
model = svm.SVC()
model.fit(training_data, training_classes)
preds = model.predict(validation_data)
accuracy = metrics.accuracy_score(validation_classes, preds)
print("Validation Accuracy With All Features: %f" %(accuracy))

# Entrenamiento y clasificacion utilizando 
# solo las caracteristicas seleccionadas
# por el Recocido Simulado:
model = svm.SVC()
model.fit(training_data[:, best_sol], training_classes)
preds = model.predict(validation_data[:, best_sol])
accuracy = metrics.accuracy_score(validation_classes, preds)
print("Validation Accuracy With Selected Features: %f" %(accuracy))

# Generacion de Graficas para visualizar el desempenio del Algoritmo:
x_data = np.linspace(1, max_iters, max_iters)
fig, axs = plt.subplots()
axs.plot(x_data, perf_data[:, 0], label='Fitness - Global Best', linewidth=3)
axs.plot(x_data, perf_data[:, 1], label='Fitness - Current Best', linewidth=1, linestyle='-')
axs.set_xlabel('Iteracion')
axs.set_ylabel('Fitness')
axs.grid(True)
axs.legend()

fig, axs = plt.subplots()
axs.plot(x_data, perf_data[:, 2], label='Classification Accuracy - Global Best', linewidth=3)
axs.plot(x_data, perf_data[:, 3], label='Classification Accuracy - Current Best', linewidth=1, linestyle='-')
axs.set_xlabel('Iteracion')
axs.set_ylabel('Classification Accuracy')
axs.grid(True)
axs.legend()


fig, axs = plt.subplots()
axs.plot(x_data, perf_data[:, 4], label='Feature Decreasing Rate - Global Best', linewidth=3)
axs.plot(x_data, perf_data[:, 5], label='Feature Decreasing Rate - Current Best', linewidth=1, linestyle='-')
axs.set_xlabel('Iteracion')
axs.set_ylabel('Feature Decreasing Rate')
axs.grid(True)
axs.legend()

plt.show()