from torch._C import dtype
from primitives import standard_env
import torch
from graph_based_sampling import topological, sub_in_vals
from daphne import daphne
import numpy as np
from evaluation_based_sampling import eval
import matplotlib.pyplot as plt
from statistics import variance, mean
import seaborn as sns
import scipy.stats as stats
import math
from plots import plots 



def bb_eval(exp, env):
    "Evaluation function for the deterministic target language of the graph based representation."
    if isinstance(exp, str):    # variable reference
        return env.find(exp)[exp]
    elif not isinstance(exp, list): # constant 
        return torch.tensor(exp)   
    op, *args = exp      
    if op == 'if':             # conditional
        (test, conseq, alt) = args
        exp = (conseq if bb_eval(test, env) else alt)
        return bb_eval(exp, env)
    else:
        proc = bb_eval(op, env)
        vals = [x[0] for x in (eval(arg, None, env) for arg in args)]
        r = proc(*vals)
        return r

def grad_log_prob(q, c):
    lp = q.log_prob(c)
    lp.backward()   
    ret = [p.grad.clone().detach() for p in q.Parameters()]     
    return ret
    
def sample_from_q(graph, sigma):
    "This function does ancestral sampling"
    V, A, P = graph[1]['V'], graph[1]['A'], graph[1]['P']
    
    V_sorted = topological(V, A)

    values = {}
    env = standard_env()

    for v in V_sorted:
        op, exp = P[v][0], P[v][1]
        if op == "sample*":

            p = bb_eval(sub_in_vals(exp, values), env)   # prior dist

            if v not in sigma['q']: # sigma and sigma['q'] are both dictionnaries (nested)
                pg = p.make_copy_with_grads()
                sigma['q'][v] = pg   # prior
                sigma['O'].param_groups[0]['params'] += pg.Parameters()                
            
            c = sigma['q'][v].sample()

            sigma['G'][v] = grad_log_prob(sigma['q'][v], c)
            sigma['O'].zero_grad()

            log_W_v = p.log_prob(c) - sigma['q'][v].log_prob(c)
            sigma['logW'] = sigma['logW'] + log_W_v

            env[v] = c
            values[v] = c


        elif op == "observe*":

            exp2 = P[v][2]
            p = bb_eval(exp, env)
            c = bb_eval(exp2, env)

            sigma['logW'] = sigma['logW'] + p.log_prob(c)
            
            env[v] = c
            values[v] = c

    # The last entry in the graph is the return value of the program.
    return bb_eval(sub_in_vals(graph[-1], values), env), sigma


def optimizer_step(sigma, g_hat):

    for v in g_hat:
        d = sigma['q'][v]

        for i, param in enumerate(d.Parameters()):
            try:
                param.grad = torch.tensor(-g_hat[v][i]).type(param.grad.dtype)
            except:
                param.grad = torch.tensor(-g_hat[v]).type(param.grad.dtype)

    sigma['O'].step()
    sigma['O'].zero_grad()
        
    return sigma['q']


def elbo_gradients(G_vals, logW_vals):
    g_hat = {}
    dom = set([key for dic in G_vals for key in dic])
    L = len(logW_vals)

    for v in dom:
        try:
            n_parameters = len(G_vals[0][v][0].numpy())     # Categorical 
        except:
            n_parameters = len(G_vals[0][v])

        F = np.zeros((L,n_parameters))
        G = np.zeros((L,n_parameters))
        for l in range(L):
            try:
                G[l,:] = G_vals[l][v]
            except:
                G[l,:] = G_vals[l][v][0].numpy()            # Categorical 

            F[l,:] = logW_vals[l] * G[l,:]

        var = []
        co = []

        for i in range(n_parameters):
            co_matrix = np.cov(F[:,i], G[:,i])
            co.append(co_matrix[0,1])
            var.append(co_matrix[1,1])
        
        if sum(var) != 0:
            b_hat = sum(co)/sum(var)
        else:
            b_hat = 1.0 

        g_hat[v] = np.sum(F - b_hat*G, axis=0)/L 

    return g_hat


def BBVI_sampler(graph, L,T, lr):
    return_values = []
    prob_sequence = []
    prob_means= []

    sigma = {'logW':0, 'q':{}, 'G':{}, 'O':torch.optim.Adam([torch.tensor(0)], lr=lr)}

    for t in range(T):  # number of samples in q-distribution space
        G_sequence = []
        prob_sequence_l = []
        return_sequence_l = []
        for l in range(L): # number of samples to estimate the gradient
            sigma['logW'] = 0
            sigma['G'] = {}
            r_tl, sigma_tl = sample_from_q(graph, sigma)
            return_sequence_l.append(r_tl)
            return_values.append(r_tl)
            prob_sequence_l.append(sigma_tl['logW'].item())
            prob_sequence.append(sigma_tl['logW'].item())
            G_sequence.append(sigma_tl['G'].copy())

        g_hat = elbo_gradients(G_sequence, prob_sequence_l)
        prob_means.append(mean(prob_sequence_l))


        sigma['q'] = optimizer_step(sigma_tl, g_hat)

    return return_values, prob_sequence, prob_means, sigma['q']

def weighted_avg(X, weights):
    return (weights.dot(X)) / weights.sum()

if __name__ == '__main__':

    # for i in range(1,6):
    for i in [2]:
        graph = daphne(['graph','-i','C:/Users/jlovr/CS532-HW4/BBVI/programs/{}.daphne'.format(i)])

        sigma = {'logW':0, 'q':{}, 'G':{}}

        L= 20
        T =3000
        lr= 0.03
        
        print('\n\n\nProgram {}:'.format(i))
        return_values, prob_sequence, prob_means, sigma['q'] = BBVI_sampler(graph, L, T, lr)
        print(sigma['q']) 
        plots(i, return_values, prob_sequence, prob_means, sigma['q'])

