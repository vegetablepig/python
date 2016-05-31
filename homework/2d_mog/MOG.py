import math, random, copy
import numpy as np
import sys


import Gnuplot

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

mg_plt = Gnuplot.Gnuplot()
mg_res_plt = Gnuplot.Gnuplot()
 
gauss_pt_2d = []
Em_data = []
mean1 = [3,3]
cov1 = [[1,0.0],[0.0,1]]
gauss_pt_2d = np.random.multivariate_normal(mean1,cov1,500)

pltdata1=[]
for item in gauss_pt_2d:
    pltdata1.append([item[0],item[1]])
    Em_data.append([item[0],item[1]])
Gauss_data1 = Gnuplot.PlotItems.Data(pltdata1, with_="points pointtype 7 pointsize 0", title="2d gauss distribution points with mean(3,3)")


mean2 = [-3,-3]
cov2 = [[1,0],[0,1]]

gauss_pt_2d = np.random.multivariate_normal(mean2,cov2,800)

pltdata2=[]
for item in gauss_pt_2d:
    pltdata2.append([item[0],item[1]])
    Em_data.append([item[0],item[1]])
Gauss_data2 = Gnuplot.PlotItems.Data(pltdata2, with_="points pointtype 7 pointsize 0", title="2d gauss distribution points with mean(-3,-3)")

mg_plt.plot(Gauss_data1, Gauss_data2)


def expectation_maximization(t, nbclusters=2, nbiter=3, normalize=False,\
        epsilon=0.001, monotony=False, datasetinit=True):
    def pnorm(x, m, s):
        """ 
        Compute the multivariate normal distribution with values vector x,
        mean vector m, sigma (variances/covariances) matrix s
        """

        xmt = np.matrix(x-m).transpose()
        for i in xrange(len(s)):
            if s[i,i] <= sys.float_info[3]: # min float
                s[i,i] = sys.float_info[3]
        sinv = np.linalg.inv(s)   #inv of s
        xm = np.matrix(x-m)
        return (2.0*math.pi)**(-len(x)/2.0)*(1.0/math.sqrt(np.linalg.det(s)))\
                *math.exp(-0.5*(xm*sinv*xmt))

    def draw_params(c):
        tmpmu = np.array([random.uniform(min_max[f][0], min_max[f][1])\
                for f in xrange(nbfeatures)], np.float64)

        return {'mu': tmpmu,\
                'sigma': np.matrix(np.diag(\
                [(min_max[f][1]-min_max[f][0])/2.0\
                for f in xrange(nbfeatures)])),\
                'proba': 1.0/nbclusters}

    nbobs = t.shape[0]
    nbfeatures = t.shape[1]

    min_max = []

    # find xranges for each features
    for f in xrange(nbfeatures):
        min_max.append((t[:,f].min(), t[:,f].max()))
    
    ### Normalization
    if normalize:
        for f in xrange(nbfeatures):
            t[:,f] -= min_max[f][0]
            t[:,f] /= (min_max[f][1]-min_max[f][0])
    min_max = []
    for f in xrange(nbfeatures):
        min_max.append((t[:,f].min(), t[:,f].max()))


    result = {}
    quality = 0.0
    random.seed()
    Pclust = np.ndarray([nbobs,nbclusters], np.float64) # P(c|x)
    Px = np.ndarray([nbobs,nbclusters], np.float64) # P(x|c)


    # Step 1: draw nbclusters sets of parameters
    params = [draw_params(c) for c in xrange(nbclusters)]     ##computer the parameter of each cluster which including mu,sigma  index like params[0]['mu'] etc.
    print "params",params
    old_log_estimate = sys.maxint         # init, not true/real
    log_estimate = sys.maxint/2 + epsilon # init, not true/real
    estimation_round = 0


    # Iterate until convergence (EM is monotone) <=> < epsilon variation
    while (abs(log_estimate - old_log_estimate) > epsilon\
             and (not monotony or log_estimate < old_log_estimate)):
        restart = False
        old_log_estimate = log_estimate


        # Step 2: compute P(Cluster|obs) for each observations

        for o in xrange(nbobs):
            for c in xrange(nbclusters):
                Px[o,c] = pnorm(t[o,:],\
                        params[c]['mu'], params[c]['sigma'])  #prob(x|c)


        for o in xrange(nbobs):
            for c in xrange(nbclusters):
                Pclust[o,c] = Px[o,c]*params[c]['proba']


        for o in xrange(nbobs):
            tmpSum = 0.0
            for c in xrange(nbclusters):
                tmpSum += params[c]['proba']*Px[o,c]
            Pclust[o,:] /= tmpSum


        # Step 3: update the parameters (sets {mu, sigma, proba})

        print  " estimation#:", estimation_round, " params:", params

        for c in xrange(nbclusters):
            tmpSum = math.fsum(Pclust[:,c])

                
            params[c]['proba'] = tmpSum/nbobs

            if params[c]['proba'] <= 1.0/nbobs:
                restart = True
                print "Restarting, p:",params[c]['proba']
                break

            m = np.zeros((1,nbfeatures), np.float64)

            for o in xrange(nbobs):
                m += t[o,:]*Pclust[o,c]
            params[c]['mu'] = m/tmpSum
            s = np.matrix(np.diag(np.zeros(nbfeatures, np.float64)))
            for o in xrange(nbobs):
                s += Pclust[o,c]*(np.matrix(t[o,:]-params[c]['mu']).transpose()*\
                        np.matrix(t[o,:]-params[c]['mu']))
            params[c]['sigma'] = s/tmpSum
            print "------------------sigma",c
            print params[c]['sigma']

        # Boundary conditions
        if not restart:
            restart = True
            for c in xrange(1,nbclusters):
                if not np.allclose(params[c]['mu'], params[c-1]['mu'])\
                or not np.allclose(params[c]['sigma'], params[c-1]['sigma']):
                    restart = False
                    break
        if restart:                # restart if all converges to only
            old_log_estimate = sys.maxint          # init, not true/real
            log_estimate = sys.maxint/2 + epsilon # init, not true/real
            params = [draw_params() for c in xrange(nbclusters)]
            continue

        # Step 4: compute the log estimate
        log_estimate = math.fsum([math.log(math.fsum(\
                [Px[o,c]*params[c]['proba'] for c in xrange(nbclusters)]))\
                for o in xrange(nbobs)])
        print "(EM) old and new log estimate: ",\
                old_log_estimate, log_estimate
        estimation_round += 1


    result['quality'] = quality
    result['params'] = copy.deepcopy(params)
    result['clusters'] = [[o for o in xrange(nbobs) if Px[o,c] == max(Px[o,:])]\
            for c in xrange(nbclusters)]
    print "final result"
    return result



def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)



def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip



MOG_res={}
MOG_res=expectation_maximization(np.matrix(Em_data))
mu = MOG_res['params'][0]['mu']
sigma = MOG_res['params'][0]['sigma']

print MOG_res['params'][0]

pltdata3=[]
pltdata3.append([Em_data[c] for c in MOG_res['clusters'][0]])
Gauss_data3 = Gnuplot.PlotItems.Data(pltdata3, with_="points pointtype 7 pointsize 0")
pltdata3 = []
pltdata3.append([Em_data[c] for c in MOG_res['clusters'][1]])
Gauss_data4 = Gnuplot.PlotItems.Data(pltdata3, with_="points pointtype 7 pointsize 0")
mg_res_plt.plot(Gauss_data3,Gauss_data4,title="the result of MOG")
raw_input()

