import numpy as np

def objectFun(var): ##F(x,y)=(x-3)^2+(y-8)^2

    x_ = var[0]
    y_ = var[1]
    return (x_-10)**2 + y_**2 - y_*10 + 26
    

def diff(fun, axis, dims, var, delta=1e-5):
    if axis >=dims:
        return
    tmp = np.zeros(dims)
    tmp[axis]=delta

    diffm = (fun(var+tmp)-fun(var))/delta
    return diffm


def ComputerHesse(fun, x, dims, delta=1e-5):
    hesse = np.zeros((dims,dims))
    
    for i in xrange(dims):        
        for j in xrange(dims):
            param = np.zeros(dims)
            param[j] = delta
            diffm1 = diff(fun,i,dims,x+param)
            diffm2 = diff(fun,i,dims,x)
            hesse[i][j] = (diffm1 - diffm2)/delta
    return hesse



def levenberg_marquardt(fun, dims, begins, iterals=50, eposio=1e-10):
    mu = eposio*10
    solution = begins
    iteral = 0
    
    while True:
        g = np.zeros(dims)
        for i in xrange(dims):
            g[i] = diff(fun, i, dims, solution )
        if np.sqrt(sum(g.T*g))<eposio or iteral >iterals:
            break
        G = ComputerHesse(fun, solution, dims)
        while True:
            Ev = np.linalg.eigvals(G+mu*np.eye(G.shape[0])) # caculate the eigenvalues
            isPositive = True
            for j in xrange(np.size(Ev)):
                if Ev[j]<-eposio:
                    isPositive = False
            if isPositive:
                break
            else:
                mu = 4 * mu

        print "Iteration",solution
        correction = np.linalg.solve(G+mu*np.eye(G.shape[0]), -g)
        fk = fun(solution)

        solution_new = solution + correction
        fk_new = fun(solution_new)

        Qk = (g*correction).sum() + 0.5*np.mat(correction)*np.mat(G)*np.mat(correction).T


        Rk = (fk_new - fk)/Qk

        if Rk>0:
            solution = solution_new

        if Rk<0.25:
            mu = 4*mu
        elif Rk>0.75:
            mu = mu/2       
        else:
            pass

        iteral += 1

    return solution,fk_new


if __name__ == '__main__':
    
    sol, funval = levenberg_marquardt(objectFun, 2, np.array([400,500]))
    print "The minimal value ",funval, " reaches at ",sol

            
        
        
        
    
    
    
        


