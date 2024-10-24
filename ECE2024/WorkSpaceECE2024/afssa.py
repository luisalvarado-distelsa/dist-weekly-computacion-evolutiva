import math
from random import random as rand
from random import randint as randi
import numpy as np
from sklearn import svm
from sklearn import metrics

WEIGHT_CLASSIFICATION    = 0.95
WEIGHT_FEATURE_SELECTION = 0.05

def search(training_data,   training_classes, 
           validation_data, validation_classes,    
           max_iters:int=10, 
           sol_ini=None):
    best_effmo = -1
    sol_size = training_data.shape[1]
    new_sol = sol_ini
    t = np.linspace(1, 0, max_iters)
        
    perf_data = np.zeros((t.shape[0], 6))
    
    for iter in range(0, max_iters):
        print('Running Iteration %d of %d' %((iter+1), max_iters), end='  ')    
        model = svm.SVC()
        model.fit(training_data[:, new_sol], training_classes)
                        
        preds = model.predict(validation_data[:, new_sol])
        new_effclass = metrics.accuracy_score(validation_classes, preds)
        
        new_effeat = 1 - (np.sum(new_sol) / sol_size)
        new_effmo = (new_effclass * WEIGHT_CLASSIFICATION) + (new_effeat * WEIGHT_FEATURE_SELECTION)
        
        __print_performance__(new_effclass, new_effeat, new_effmo, new_sol)
        
        if iter == 0 or new_effmo > best_effmo:
            best_sol      = new_sol
            best_effclass = new_effclass
            best_effeat   = new_effeat
            best_effmo    = new_effmo
            best_model    = model
            
        perf_data[iter, :] = [best_effmo, new_effmo, best_effclass, new_effclass, best_effeat, new_effeat]
            
        deltae = (best_effmo - new_effmo) * t[iter]
        metropolis = math.exp(-deltae / t[iter]) if t[iter] > 0 else 0
        r = rand()
        if deltae < 0 or r < metropolis:
            new_sol = __generate_solution__(metropolis, deltae, sol_size)
    return best_sol, best_effclass, best_effeat, best_effmo, best_model, perf_data

def __generate_solution__(metropolis, deltae, sol_size):
    new_sol = []
    for i in range(0, sol_size):
        t = abs((rand() * metropolis) - deltae)
        f = t > 0.5
        new_sol.append(f)
    new_sol = np.array(new_sol)
    if np.sum(new_sol) < 1:
        new_sol = []
        for i in range(0, sol_size):
            new_sol.append(rand() > 0.5)
        new_sol = np.array(new_sol)
    return new_sol

def __print_performance__(effclass, effeat, effmo, sol, feat_names=None, print_feats:bool=False):
    # Se imprimen los valores de eficiencia de clasificacion, discriminacion de caracteristicas
    # y fitness multi-objetivo:
    print('EffClass: %f, Effeat: %f [%d Features], EffMO: %f' %(effclass, effeat, np.sum(sol), effmo))
    
    # Se imprimen las caracteristicas seleccionadas:
    if feat_names is not None and len(feat_names[0]) == sol.shape[0] and print_feats:
        print("Caracteristicas Seleccionadas:")
        for i in range(0, sol.shape[0]):
            if sol[i] == 1:
                print('\t' + feat_names[0][i])