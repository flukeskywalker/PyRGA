import GA
from GA import *
import numpy as np

def square(x):
    term1 = (x[0]*x[0]+x[1]-11.0)*(x[0]*x[0]+x[1]-11.0);
    term2 = (x[0]+x[1]*x[1]- 7.0)*(x[0]+x[1]*x[1]- 7.0);
    term3 = term1+term2;
    return -1*term3
    # return -np.sum(np.power(x,2))
ga=GA(square, dim=2, popsize=40, ngen=50, pc=0.9, pm=0.1, etac=2, etam=100)
ga.setbounds(np.zeros(10), 10*np.ones(10))
#ga.pop_init()
print ga.run()

