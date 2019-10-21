#############################################################################################################################
#####################################--Imports--#############################################################################
#############################################################################################################################
import numpy as np
import random as rand
import time
import multiprocessing as mp

from itertools import repeat
from itertools import count
from itertools import starmap
from functools import partial
#############################################################################################################################
#####################################--Class Declarations--##################################################################
#############################################################################################################################
class DE_Handler(object):
    """ Handler class holding the necessary predefined parameters for Differential Evolution
    and implementing the DE algorithm

    properties:
    - F:                    mutation scale factor, positive real number typically less than 1
    - Cr:                   crossover constant, positive real number between 0 and 1
    - G:                    maximum number of generations/iterations for DE algorithm
    - Np:                   population size
    - population:           initial population the algorithm is executed on
    - ObjectiveFunction:    the function whose parameters need to be optimized by the algorithm
    - minimizeFlag:         specifies whether it is an Minimization or an Maximization problem
    - minBounds:            minimum values for parameters to find
    - maxBounds:            maximum values for parameters to find
    """

    def __init__(self, F, Cr, G, Np, population, ObjectiveFunction, minimizeFlag, minBounds, maxBounds):
        self.F = F
        self.Cr = Cr
        self.G = G
        self.Np = Np
        self.population = population
        self.ObjectiveFunction = ObjectiveFunction
        self.minimizeFlag = minimizeFlag
        self.minBounds = minBounds
        self.maxBounds = maxBounds
#############################################################################################################################
    def EvaluatePopulation(self):
        """ Finds the member of the current population that yields the best value for the Objective Function
        (doesn't need to be called externally)
        """
        denormalization = lambda p : self.minBounds + p * (self.maxBounds - self.minBounds)
        denormPopulation = np.array(list(map(denormalization, self.population)))
        values = np.array(list(map(self.ObjectiveFunction, denormPopulation)))
        #values = np.array([self.ObjectiveFunction(self.minBounds + self.population[p] * (self.maxBounds - self.minBounds)) for p in range(self.Np)])
        
        if self.minimizeFlag == True:
            best = (self.population[np.argmin(values)], np.amin(values))
        else:
            best = (self.population[np.argmax(values)], np.amax(values))

        return best, values
#############################################################################################################################
    def GetMutantVector(self, memberIndex, bestParams):
        """ Calculates the Mutant Vectors for the current population
        (doesn't need to be called externally)

        - bestParams: param vector of the best member in current population
        - memberIndex: index of the currently evaluated member of the present population
        """

        randChoice = np.random.randint(0, self.Np-1, 6)
        randNumbers = np.random.choice(randChoice[randChoice != memberIndex], 2)

        v = (bestParams + (self.population[randNumbers[0]] - self.population[randNumbers[1]]) * self.F)
        v = np.select([v < 0, v > 1], [rand.random(), rand.random()], v)

        return v
#############################################################################################################################
    def SelectVector(self, v, p, value):
        """ Crosses the population with the Mutant Vectors and creates the Trial Vectors
        (doesn't need to be called externally)

        - v: the mutant vector of the currently evaluated member
        - p: the currently evaluated member
        - value: return value of the objective function for the current member
        """

        r = rand.randint(0, v.size-1)
        L = 1
        while rand.random() <= self.Cr and L < v.size:
            L = L + 1

        u = np.array([v[(r+i)%v.size] if i < L else p[(r+i)%v.size] for i in range(v.size)])

        valueU = self.ObjectiveFunction(self.minBounds + u * (self.maxBounds - self.minBounds))

        if (valueU <= value and self.minimizeFlag == True) or (valueU >= value and self.minimizeFlag == False):
            return u
        else:
            return p
#############################################################################################################################
    def DE_Optimization(self):
        """ executes the whole optimization process of the DE-Algorithm
        (doesn't need to be called externally)
        """

        best, currentValues = self.EvaluatePopulation()
        mutantVectors = np.array(list(map(partial(self.GetMutantVector, bestParams=best[0]), range(self.Np))))
        #self.population = [self.SelectVector(self.GetMutantVector(j, best[0]), self.population[j], currentValues[j]) for j in range(self.Np)]
        self.population = np.array(list(starmap(self.SelectVector, zip(mutantVectors, self.population, currentValues))))
        return best[1]
#############################################################################################################################
    def DE_GetBestParameters(self):
        """ starts the optimization process and then returns the last found best parameters
        """
        t0 = time.time()
        bestValueHistory = np.array([self.DE_Optimization() for _ in range(self.G)])
        #bestValueHistory = np.array(list(map(self.DE_Optimization, repeat(1, self.Np))))
        #bestValueHistory = np.array(list(starmap(self.DE_Optimization, repeat((), self.G))))
        print(time.time()-t0)
        best, currentValues = self.EvaluatePopulation()
        return best, bestValueHistory


