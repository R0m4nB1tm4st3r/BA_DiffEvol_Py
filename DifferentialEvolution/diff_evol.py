#############################################################################################################################
#####################################--Imports--#############################################################################
#############################################################################################################################
import numpy as np
import random as rand
import time
import plot

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

    def __init__(self, F, Cr, G, Np, population, ObjectiveFunction, minimizeFlag, minBounds, maxBounds, objArgs):
        self.F = F
        self.Cr = Cr
        self.G = G
        self.Np = Np
        self.ObjectiveFunction = ObjectiveFunction
        self.minimizeFlag = minimizeFlag
        self.minBounds = minBounds
        self.maxBounds = maxBounds
        self.population = np.array([self.minBounds + population[p] * (self.maxBounds - self.minBounds) for p in range(self.Np)])
        self.objArgs = objArgs

        self.x = np.arange(self.G + 1, dtype=np.float_)
        self.y = np.arange(self.G + 1, dtype=np.float_)
        self.z = np.arange(self.G + 1, dtype=np.float_)
        self.genIndex = 0

        rand.seed(789)
        np.random.seed(456)
#############################################################################################################################
    def EvaluatePopulation(self):
        """ Finds the member of the current population that yields the best value for the Objective Function
        (doesn't need to be called externally)
        """

        values = np.array(list((self.ObjectiveFunction(self.population[p], *self.objArgs) for p in range(self.Np))))

        if self.minimizeFlag == True:
            best = (self.population[np.argmin(values)], np.amin(values))
        else:
            best = (self.population[np.argmax(values)], np.amax(values))

        if best[0].size == 2:
            self.x[self.genIndex] = best[0][0]
            self.y[self.genIndex] = best[0][1]
            self.z[self.genIndex] = best[1]
            self.genIndex = self.genIndex + 1

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
        #v = np.select([v < 0, v > 1], [rand.random(), rand.random()], v)
        v = np.array([rand.uniform(self.minBounds[i], self.maxBounds[i]) \
            if v[i] < self.minBounds[i] or v[i] > self.maxBounds[i] \
            else v[i] \
            for i in range(v.size)])

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

        #valueU = self.ObjectiveFunction(self.minBounds + u * (self.maxBounds - self.minBounds), *self.objArgs)
        valueU = self.ObjectiveFunction(u, *self.objArgs)

        if (valueU < value and self.minimizeFlag == True) or (valueU > value and self.minimizeFlag == False):
            return u
        else:
            return p
#############################################################################################################################
    def DE_Optimization(self):
        """ executes the whole optimization process of the DE-Algorithm
        (doesn't need to be called externally)
        """

        best, currentValues = self.EvaluatePopulation()
        self.population = np.array([self.SelectVector(self.GetMutantVector(j, best[0]), self.population[j], currentValues[j]) for j in range(self.Np)])

        return best[1]
#############################################################################################################################
    def DE_GetBestParameters(self):
        """ starts the optimization process and then returns the last found best parameters
        """
        t0 = time.time()

        bestValueHistory = np.array(list((self.DE_Optimization() for _ in range(self.G))))
        print(time.time()-t0)
        best, currentValues = self.EvaluatePopulation()
        if best[0].size == 2:
            plot.SurfacePlot3D(self.x, self.y, self.z)
        return best, bestValueHistory


