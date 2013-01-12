import numpy as np
import random
import math
class GA: # popsize must be multiple of 4
    def __init__(self, obj, dim, popsize, ngen, pc, pm, etac, etam):
        self.EPSILON = 10e-6
        self.INFINITY = 10e6
        self.pop = []
        self.fits = []
        self.obj = obj
        self.dim = dim
        self.popsize = popsize
        self.ngen = ngen
        self.pc = pc
        self.pm = pm
        self.etac = etac
        self.etam = etam
        self.RIGID = 0
        self.lowb = -self.INFINITY*np.ones(self.dim)
        self.highb = self.INFINITY*np.ones(self.dim)
        self.tourneylist = range(0, self.popsize)
        self.tourneysize = 2 # works for 2 for now
        self.bestmemyet = np.zeros(self.dim)
        self.bestfityet = -np.inf
        self.pop_init()
    def pop_init(self):
        self.pop = [np.random.rand(self.dim) for _ in range(self.popsize)]
        for member in self.pop:
            for i in range(self.dim):
                member[i] = self.lowb[i] + member[i]*(self.highb[i]-self.lowb[i])
        self.fits = [self.obj(member) for member in self.pop]
        #self.pop_print()
        return
    def setbounds(self, lows, highs):
        for i in range(self.dim):
            self.lowb[i] = lows[i]
            self.highb[i] = highs[i]
        self.pop_init()
        return
    def run(self):
        for gen in range(self.ngen):
            print "Generation ", gen,
            self.pop = self.getnewpop()
            self.eval_pop()
            #self.pop_print()
        return [self.bestmemyet, self.bestfityet]
    def getnewpop(self):
        newpop = []
        #self.tourneylist = xrange(0, self.popsize)
        random.shuffle(self.tourneylist)
        self.tourneypos = 0
        for i in xrange(0, self.popsize, 2):
            [p1, p2] = self.getparents() #return parents, not just indices
            [c1, c2] = self.xover(p1, p2) #return children, not just indices
            c1 = self.mutate(c1)
            c2 = self.mutate(c2)
            newpop.append(c1)
            newpop.append(c2)
        return newpop
    def getparents(self):
        if (self.popsize - self.tourneypos) < self.tourneysize:
            random.shuffle(self.tourneylist)
            self.tourneypos = 0
        if (self.fits[self.tourneylist[self.tourneypos]]>self.fits[self.tourneylist[self.tourneypos+1]]):
            p1 = self.pop[self.tourneylist[self.tourneypos]]
        else:
            p1 = self.pop[self.tourneylist[self.tourneypos+1]]
        self.tourneypos += self.tourneysize
        if (self.fits[self.tourneylist[self.tourneypos]]>self.fits[self.tourneylist[self.tourneypos+1]]):
            p2 = self.pop[self.tourneylist[self.tourneypos]]
        else:
            p2 = self.pop[self.tourneylist[self.tourneypos+1]]
        self.tourneypos += self.tourneysize
        return [p1, p2]
    def xover(self, p1, p2): # Here p1 and p2 are pop members
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p2)
        if random.random()<=self.pc: # do crossover
            for i in xrange(p1.size):
                if random.random()<0.5: # 50% variables crossover
                    [c1[i], c2[i]] = self.crossvars(p1[i], p2[i], self.lowb[i], self.highb[i])
                else:
                    [c1[i], c2[i]] = [p1[i], p2[i]]
        else:
            c1 = p1
            c2 = p2
        return [c1, c2]
    def crossvars(self, p1, p2, low, high): # Here p1 and p2 are variables
        if p1>p2:
            p1, p2 = p2, p1 # p1 must be smaller
        mean = (p1+p2)*0.5
        diff = (p2-p1)
        dist = max(min(p1-low, high-p2), 0)
        if (self.RIGID and diff > self.EPSILON):
            alpha = 1.0 + (2.0*dist/diff)
            umax = 1.0 - (0.5/pow(alpha, (self.etac+1.0)))
            seed = umax*random.random()
        else:
            seed = random.random()
        beta = self.getbeta(seed)
        if (abs(diff*beta) > self.INFINITY):
            beta = self.INFINITY/diff
        c2 = mean + beta*0.5*diff
        c1 = mean - beta*0.5*diff
        c1 = max(low, min(c1, high))
        c2 = max(low, min(c2, high))
        return [c1, c2]
    def getbeta(self, seed):
        if (1 - seed) < self.EPSILON:
            seed = 1 - self.EPSILON
        seed = max(0.0, seed)
        if seed < 0.5:
            beta = pow(2.0*seed, (1.0/(self.etac+1.0)))
        else:
            beta = pow((0.5/(1.0-seed)), (1.0/(self.etac+1.0)))
        return beta
    def getdelta(self, seed, delta_low, delta_high):
        if seed >= 1.0 - (self.EPSILON/1e3):
            return delta_high
        if seed <= (self.EPSILON/1e3):
            return delta_low
        if seed <= 0.5:
            dummy = 2.0*seed + (1.0 - 2.0*seed)*pow((1+delta_low), (self.etam+1.0))
            delta = pow(dummy, (1.0/(self.etam+1.0))) - 1.0
        else:
            dummy = 2.0*(1.0 - seed) + 2.0*(seed - 0.5)*pow((1-delta_high), (self.etam+1.0))
            delta = 1.0 - pow(dummy, (1.0/(self.etam+1.0)))
        return delta
    def mutate(self, member):
        mut_member = np.zeros_like(member)
        for i in xrange(member.size):
            low = self.lowb[i]
            high = self.highb[i]
            if random.random() <= self.pm: # pm is simply the prob of a variable to mutate
                if self.RIGID:
                    value = member[i]
                    delta_low = max((low-value)/(high-low), -1.0)
                    delta_high = min((high-value)/(high-low), 1.0)
                    if abs(delta_low)<abs(delta_high):
                        delta_high = -delta_low
                    else:
                        delta_low = -delta_high
                else:
                    delta_low = -1.0
                    delta_high = 1.0
                seed = random.random()
                delta = self.getdelta(seed, delta_low, delta_high)*(high-low)
                mut_member[i] = max(low, min(member[i] + delta, high))
            else:
                mut_member[i] = member[i]
        return mut_member
    def eval_pop(self):
        self.fits = [self.obj(member) for member in self.pop]
        bestindex = np.argmax(self.fits)
        bestmember = self.pop[bestindex]
        bestfitness = self.fits[bestindex]
        if bestfitness > self.bestfityet:
            self.bestfityet = bestfitness
            self.bestmemyet = bestmember
        print "Current best: ", bestfitness, "Best yet: ", self.bestfityet
    def pop_print(self):
        for i in range(self.popsize):
            print self.pop[i], self.fits[i]
        return
