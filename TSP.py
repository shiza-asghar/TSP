import numpy as np
import random
#create a population of some size using random shuffle by taking one scheme to another
def create_population(size,no_of_cities):
    population = []
    for i in range(size):
        population.append(random.sample(range(no_of_cities),no_of_cities))
    return population
#calcuate distance function using the 2d distance list given as input
def CalDistance(route,distancesList):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distancesList[route[i]][route[i + 1]]
    total_distance += distancesList[route[-1]][route[0]]  # Return to the starting city
    return total_distance
#for every element in the population, calculate the fitness of the element and store it in a list(using the calculate distance function)
def DistanceAndFitness(population,popsize,distancesList):
    FitnessList = []
    for i in range(popsize):
        d = CalDistance(population[i],distancesList)
        #using the fitness = 1/(d+1)
        FitnessList.append(1/(d+1))
        #if calclated distance is less than record distance then record distance will be set to calculated distance and the best route will be set to the current route 
        global Record_Distance
        global BestRoute
        if d <  Record_Distance:
            Record_Distance = d
            BestRoute = population[i]
    return FitnessList
#normalize fitness values  by dividing by the sum of all fitness values
def Normalize(FitnessList):
    s = sum(FitnessList)
    newFit = [each/s for each in FitnessList]
    return newFit
#pick one with highest probability
def SelectOne(population,probabilities):
    index = 0
    r = random.random()
    while r > 0:
        r = r - probabilities[index]
        index += 1
    return population[index-1]
#for crossover you have to take 2 rotes using the probability thing function and then use em to make new order
def CrossOver(ChoiceA,ChoiceB):
    crossover_point = np.random.randint(1, len(ChoiceA) - 1)
    child = np.concatenate((ChoiceA[:crossover_point], np.setdiff1d(ChoiceB,ChoiceA[:crossover_point])))
    return list(child)
#mutate by shuffling 2 neighbours in the route 
def Mutate(individual):
    mutation_point1, mutation_point2 = np.random.choice(len(individual), 2, replace=False)
    individual[mutation_point1], individual[mutation_point2] = individual[mutation_point2], individual[mutation_point1]
    return individual
#make next generation ( take 2 routes,crossover em and  mutate it and add it to new generation run it no of population times)
def MakeNextGeneration(population,popsize,probabilities):
    newGen = []
    for i in range(popsize):
        ChoiceA = SelectOne(population,probabilities)
        ChoiceB = SelectOne(population,probabilities)
        while ChoiceA == ChoiceB:
            ChoiceB = SelectOne(population,probabilities)
        newOne = CrossOver(ChoiceA,ChoiceB)
        newGen.append(Mutate(newOne))
    return newGen
#the genetic algorithm using the above functions running till generations time
def GeneticAlgo(no_of_cities,distancesList,no_ofGen,popsize):
    #record distance will be set to infinity and BestRoute to []
    global Record_Distance
    Record_Distance = float('inf')
    global BestRoute
    BestRoute = []
    CurrGEN =  create_population(popsize,no_of_cities)
    #run it till no og generations and return the best route and record distance
    for _ in range(no_ofGen):
        Fitness = Normalize(DistanceAndFitness(CurrGEN,popsize,distancesList))
        NextGen = MakeNextGeneration(CurrGEN,popsize,Fitness)
        #mutated genration will be the new population
        CurrGEN = NextGen
    return Record_Distance,BestRoute
#the main function to input everything and run the genetic algorithm
def main():
    distancesList = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])
    no_of_cities = 4
    no_ofGen =  100
    popsize = 50
    dis,route = GeneticAlgo(no_of_cities,distancesList,no_ofGen,popsize)
    print(f'the optimized solution is {route} with the distance of {dis}')
main()



