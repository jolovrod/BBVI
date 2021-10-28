import matplotlib.pyplot as plt
import numpy as np
from statistics import variance, mean
from numpy import asarray
from numpy import arange
from numpy import meshgrid
import torch


def weighted_avg(X, weights):
    return (weights.dot(X)) / weights.sum()

def plots(i, samples, prob_sequence, alg, P=None):
    num_samples = len(samples)

    if i in [3,4]:
        
        if alg == "_IS":
            W = np.exp(prob_sequence)
            means = weighted_avg(samples, W)
            vars = weighted_avg((samples - means)**2, W)

        else:
            means = mean(np.array(samples, dtype=float))
            vars = variance(np.array(samples, dtype=float))

        print("mean", means)
        print("variance", vars)


    if i == 1:
        for _ in range(num_samples):
            samples = [float(x) for x in samples]
        
        plt.figure(figsize=(5,4))
        plt.xlabel("mu")
        plt.ylabel("frequency")
        plt.title("Histogram program 1" + alg)
        plt.hist(samples)
        figstr = "histograms/program_"+str(i) + alg
        plt.savefig(figstr)

        if alg == "_IS":
            print("HERE")
            W = np.exp(prob_sequence)
            plt.figure(figsize=(5,4))
            plt.xlabel("mu")
            plt.ylabel("value")
            plt.title("Posterior program 1" + alg)
            plt.bar(samples, W)
            figstr = "histograms/newprogram_"+str(i) + alg
            plt.savefig(figstr)
            

        plt.figure(figsize=(5,4))
        plt.xlabel("Iterations")
        plt.ylabel("mu")
        plt.title("Trace plot program 1" + alg)
        plt.plot(samples)
        figstr = "trace_plots/program_"+str(i) + alg
        plt.savefig(figstr)


        plt.figure(figsize=(5,4))
        plt.xlabel("Iterations")
        plt.ylabel("log prob")
        plt.title("log prob plot program 1 " + alg)
        plt.plot(prob_sequence)
        figstr = "prob_plots/program_"+str(i) + alg
        plt.savefig(figstr)

        if alg == "_IS":
            W = np.exp(prob_sequence)
            means = weighted_avg(samples, W)
            vars = weighted_avg((samples - means)**2, W)
        else:
            means = mean(np.array(samples, dtype=float))
            vars = variance(np.array(samples, dtype=float))

        print("mean", means)
        print("variance", vars)


    elif i in [2,5]:
        for n in range(num_samples):
            samples[n] = [float(x) for x in samples[n]]
        
        variables = np.array(samples,dtype=object).T.tolist()
        for d in range(len(variables)):
            plt.figure(figsize=(5,4))
            plt.hist(variables[d])
            if d==0 and i==2:
                xvarname = "slope"
            elif d==0 and i==5:
                xvarname = "x"
            elif d==1 and i==2:
                xvarname = "bias"
            else:
                xvarname = "y"

            plt.xlabel(xvarname)
            plt.ylabel("frequency")
            figstr = "histograms/program_"+str(i)+"_var_"+str(d)+alg
            plt.savefig(figstr)

            plt.figure(figsize=(5,4))
            plt.xlabel("Iterations")
            plt.ylabel(xvarname)
            plt.title("Trace plot program " + str(i) + " " + xvarname + " " + alg)
            plt.plot(variables[d])
            figstr = "trace_plots/program_"+str(i)+"_var_"+str(d)+alg
            plt.savefig(figstr)


        plt.figure(figsize=(5,4))
        plt.xlabel("Iterations")
        plt.ylabel("log prob")
        plt.title("log prob plot program " + str(i) + alg)
        plt.plot(prob_sequence)
        figstr = "prob_plots/program_"+str(i)+alg
        plt.savefig(figstr)

        if alg == "_IS":
            W = np.exp(prob_sequence)
            means = weighted_avg(samples, W)
            vars = weighted_avg((samples - means)**2, W)
        else:
            means = ["{:.5f}".format(mean(variables[d])) for d in range(len(variables))]
            vars = ["{:.5f}".format(variance(variables[d])) for d in range(len(variables))]

        print("mean", means)
        print("variance", vars)

        # if i==5:
        #     from hmc import U as objective

        #     bounds = asarray([[-25.0, 25.0], [-25.0, 25.0]])
        #     xaxis = arange(bounds[0,0], bounds[0,1], .5)
        #     yaxis = arange(bounds[1,0], bounds[1,1], .5)
        #     objective_results = []
        #     ydict = {'observe3':torch.tensor(7.0)}
        #     for x in xaxis:
        #         results = []
        #         for y in yaxis:
        #             xdict = {'sample1':torch.tensor(x), 'sample2':torch.tensor(y)}
        #             results.append(objective(xdict, ydict, P))
        #         objective_results.append(results)

        #     x, y = meshgrid(torch.from_numpy(xaxis), torch.from_numpy(yaxis))

        #     plt.figure(figsize=(5,4))
        #     plt.xlabel("x")
        #     plt.ylabel("y")
        #     plt.title("100 samples of x, y with log-probability contour " + alg)
        #     plt.contourf(x, y, objective_results, levels=100, cmap='Spectral')
        #     X = variables[0][:100]
        #     Y = variables[1][:100]
        #     plt.plot(X, Y, 'bo')  # plot x and y using blue circle marker            
        #     figstr = "contour/"+alg
        #     plt.savefig(figstr)

