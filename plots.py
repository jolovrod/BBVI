import matplotlib.pyplot as plt
import numpy as np
from statistics import variance as var, mean
from numpy import asarray
from numpy import arange
from numpy import meshgrid
import torch
import seaborn as sns
import scipy.stats as stats
import math


def weighted_avg(X, weights):
    return (weights.dot(X)) / weights.sum()

def plots(i, return_values, prob_sequence, prob_means, q):

    num_samples = len(return_values)

    # ELBO trace plot
    plt.figure(figsize=(5,4))
    plt.xlabel("Iterations")
    plt.ylabel("ELBO")
    plt.title("ELBO trace plot program " + str(i) + " with BBVI")
    plt.plot(prob_means)
    figstr = "elbo_plot/program_"+str(i)
    plt.savefig(figstr)
    # plt.show()
    print("Last ELBO", prob_sequence[-1])

    if i == 1: 

        plt.figure(figsize=(5,4))
        plt.xlabel("mu")
        plt.ylabel("density")
        plt.title("Weighted posterior density of mu (program " + str(i) + ") with BBVI")
        sns.distplot(return_values, hist_kws={'weights': np.exp(prob_sequence)}, kde=False)
        figstr = "posterior_plot/program_"+str(i)
        plt.savefig(figstr)
        # plt.show()

        plt.figure(figsize=(5,4))
        plt.xlabel("mu")
        plt.ylabel("density")
        plt.title("Plot of final q for program " + str(i) + " with BBVI")
        figstr = "q_plot/program_"+str(i)
        mu = float(q['sample2'].Parameters()[0])
        variance = float(q['sample2'].Parameters()[1])
        s = math.sqrt(variance)
        x = np.linspace(mu - 3*s, mu + 3*s, 100)
        plt.plot(x, stats.norm.pdf(x, mu, s))
        plt.savefig(figstr)
        # plt.show()

        W = np.exp(prob_sequence)
        means = weighted_avg(return_values, W)
        vars = weighted_avg((return_values - means)**2, W)

        print("mean", means)
        print("variance", vars)


    elif i == 2: 

        for n in range(num_samples):
            return_values[n] = [float(x) for x in return_values[n]]
        
        variables = np.array(return_values,dtype=object).T.tolist()

        for d in range(len(variables)):

            plt.figure(figsize=(5,4))
            if d == 0: 
                xname = "slope"
                v = "sample1"
            else: 
                xname = "bias"
                v = "sample2"
            plt.xlabel(xname)
            plt.ylabel("density")
            plt.title("Weighted posterior density of " + xname + " (program " + str(i) + ") with BBVI")
            sns.distplot(variables[d], hist_kws={'weights': np.exp(prob_sequence)}, kde=False)
            figstr = "posterior_plot/program_"+str(i)+ "_" + xname
            plt.savefig(figstr)
            # plt.show()

            plt.figure(figsize=(5,4))
            plt.xlabel("mu")
            plt.ylabel("density")
            plt.title("Plot of final q for program " + str(i) + " with BBVI")
            figstr = "q_plot/program_"+str(i)+ "_" + xname
            mu = float(q[v].Parameters()[0])
            variance = float(q[v].Parameters()[1])
            s = math.sqrt(abs(variance))
            x = np.linspace(min(variables[d]), max(variables[d]), 100)
            plt.plot(x, stats.norm.pdf(x, mu, s))
            plt.savefig(figstr)
            # plt.show()

            W = np.exp(prob_sequence)
            means = weighted_avg(variables[d], W)
            vars = weighted_avg((variables[d] - means)**2, W)

            print("mean", means)
            print("variance", vars)

    elif i == 3: 

        W = np.exp(prob_sequence)
        means = weighted_avg(return_values, W)
        vars = weighted_avg((return_values - means)**2, W)

        print("mean", means)
        print("variance", vars)

    elif i==4:

        W0, b0, W1, b1 = [],[],[],[]
        for n in range(num_samples):
            W0_n, b0_n, W1_n, b1_n = return_values[n]
            W0.append(W0_n.numpy().flatten())
            b0.append(b0_n.numpy().flatten())
            W1.append(W1_n.numpy().flatten())
            b1.append(b1_n.numpy().flatten())
        
        objects = [W0, b0, W1, b1]
        strs = ["W0", "b0", "W1", "b1"]

        for j in range(4):
            variables = np.array(objects[j]).T.tolist()

            means = [mean(variables[d]) for d in range(len(variables))]
            plt.figure(figsize=(12,10), dpi= 80)
            if j == 2:
                sns.heatmap(np.asarray(means).reshape(10,10), cmap='RdYlGn', center=0, annot=True)
            else:
                sns.heatmap(np.asarray(means).reshape(10,1), cmap='RdYlGn', center=0, annot=True)
            plt.title('Heatmap for the posterior mean of ' + strs[j])
            figstr = "posterior_plot/program_"+str(i)+"_"+strs[j] + "_mean"
            plt.savefig(figstr)


            vars = [var(variables[d]) for d in range(len(variables))]
            plt.figure(figsize=(12,10), dpi= 80)
            if j == 2:
                sns.heatmap(np.asarray(vars).reshape(10,10), cmap='RdYlGn', center=0, annot=True)
            else:
                sns.heatmap(np.asarray(vars).reshape(10,1), cmap='RdYlGn', center=0, annot=True)
            plt.title('Heatmap for the posterior variance of ' + strs[j])
            figstr = "posterior_plot/program_"+str(i)+"_"+strs[j] + "_variance"
            plt.savefig(figstr)

    if i == 5: 

        plt.figure(figsize=(5,4))
        plt.xlabel("s")
        plt.ylabel("density")
        plt.title("Weighted posterior density of s (program " + str(i) + ") with BBVI")
        sns.distplot(return_values, hist_kws={'weights': np.exp(prob_sequence)}, kde=False)
        figstr = "posterior_plot/program_"+str(i)
        plt.savefig(figstr)
        # plt.show()

        plt.figure(figsize=(5,4))
        plt.xlabel("s")
        plt.ylabel("density")
        plt.title("Plot of final q for program " + str(i) + " with BBVI")
        figstr = "q_plot/program_"+str(i)
        a = float(q['sample2'].Parameters()[0])
        variance = float(q['sample2'].Parameters()[1])
        b = math.sqrt(variance)
        x = np.linspace(0, 20, 100)
        plt.plot(x, stats.gamma.pdf(x, a=a, scale=b))
        plt.savefig(figstr)
        # plt.show()

        W = np.exp(prob_sequence)
        means = weighted_avg(return_values, W)
        vars = weighted_avg((return_values - means)**2, W)

        print("mean", means)
        print("variance", vars)