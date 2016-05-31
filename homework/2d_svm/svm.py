import sys, math
import numpy as np
import Gnuplot


def tolerance(v1, v2, norm):
    # tolerance level
    sum_ = 0.0
    for k in v1.keys()+v2.keys():
        t1 = 0
        t2 = 0
        if k in v1:
            t1 = v1[k]
        if k in v2:
            t2 = v2[k]
        sum_ += (t1-t2) * (t1-t2)
        
    return math.sqrt(sum_/norm)


def init_alpha(num_examples):
    alpha_s = []
    for i in range(num_examples):
        alpha_s.append((0.0,i))
    return alpha_s


def dot(v1, v2):
    sum = 0.0
    if len(v1) < len(v2):
        for id in v1.keys():
            if id in v2:
                sum += v1[id] * v2[id]
    else:
        for id in v2.keys():
            if id in v1:
                sum += v1[id] * v2[id]
    return sum


def kernel(v1, v2):
    # linear kernel
    return dot(v1,v2)


def compute_Q(es,height):
    q = []
    for i in range(height):
        q.append([])
        for j in range(height):
            if i>=j:
                q[i].append(kernel(es[i],es[j]))
                
    return q

# retrieve kernel matrix element on i,j 
def ret_Q(q,i,j):
    if i>=j:
        return q[i][j]
    else:
        return q[j][i]


def read_examples(path):
    
    ret_class = []
    ret_val = []
    filepath = open(path)
    for line in filepath:
        temp = {}
        strt = line.split(" ")
        col_count = len(strt)-1
        
        ret_class.append(int(strt[0]))
        for i in range(col_count) :
            temp[str(i)] = float(strt[i+1])
        
        ret_val.append(temp)
    
    return len(ret_class), col_count, ret_class, ret_val



def process(C = 1.0, max_iter = 100, tol = 0.001):
    
    alpha_sparse = {} # current sparse alpha list
    
    print "loading examples..."
    
    print "example matrix: " , height_e, ",", width_e
    print "kernelizing..."
    q = compute_Q(es, height_e)
    
    alpha_s = init_alpha(height_e)
    bias = 0

    # solve the problem
    for i in range(max_iter):
        print "nnew iteration:", i+1
        gamma = 1

        alpha_s.sort(None, None, True)
        print >> sys.stderr, alpha_s[0:30]
        print >> sys.stderr, 'sparsity: ', len(alpha_sparse),':',height_e
        
        alpha_s_prim = alpha_sparse.copy()
        
        z_max = float("-infinity"); z_min= float("infinity")
        
        for t_id in range(len(alpha_s)):
            # update from the largest alpha
            alpha = alpha_s[t_id][0]
            j = alpha_s[t_id][1]
            t = 0.0
            
            for k in alpha_sparse.keys():
                t += cs[k]* alpha_sparse[k] * ret_Q(q,j,k)
            # check z_max and z_min for bias computation
            if cs[j]>0:
                if t < z_min:
                    z_min = t
            else:
                if t > z_max:
                    z_max = t
                    
            learning_rate = gamma * (1/ret_Q(q,j,j))
            delta = learning_rate * ( 1 - t * cs[j] )

            alpha += delta
            if alpha < 0 :
                alpha = 0.0 
            if alpha > C:
                alpha = C

            alpha_s[t_id] = alpha,j

            if math.fabs(alpha - 0.0) >= 1e-10:
                alpha_sparse[j]=alpha
            else:
                if j in alpha_sparse:
                    del alpha_sparse[j]

        bias = (z_max+z_min)/2.0 # the bias
        tol1 = tolerance(alpha_sparse, alpha_s_prim, float(height_e))
            
        print "tolerance:", tol1
        if tol1 < tol:
            print "nfinished in", i+1, "iterations"
            break
    
    svm_res ={'sv_s':[], 'id_s':[], 'alpha_s':[]}

    for t_id, alpha in alpha_sparse.items():
        svm_res['sv_s'].append(es[t_id])
        svm_res['id_s'].append(t_id)
        svm_res['alpha_s'].append(cs[t_id]*alpha)
    svm_res['bias'] = bias

    print "cs: ", cs
    print "es: ", es
    print "svm_res: ", svm_res
        
    return svm_res

# main process
height_e, width_e, cs,es = read_examples("data/trainInstanceLabelPairs.txt")
svm = process()
print 'support vectors:', svm['sv_s']
print 'example IDs:', svm['id_s']
print 'lagrange multipliers:',svm['alpha_s']
print 'bias:', svm['bias']

A = 0
B = 0
C = 0
for i in range(len(svm['sv_s'])):
    A += svm['alpha_s'][i] * svm['id_s'][i] * svm['sv_s'][i]['0']
    B += svm['alpha_s'][i] * svm['id_s'][i] * svm['sv_s'][i]['1']

for i in range(len(svm['sv_s'])):
    if svm['alpha_s'][i] != 0:
        C = svm['id_s'][i] - A*svm['sv_s'][i]['0'] - B*svm['sv_s'][i]['1']
    break

print "A = ", A
print "B = ", B
print "C = ", C

class0 = []
class1 = []

for i in range(len(cs)): 
    if cs[i] == 1:
        class0.append([es[i]['0'], es[i]['1']])
    elif cs[i] == -1:
        class1.append([es[i]['0'], es[i]['1']])
    else:
        print "error"
        
class0 = np.array(class0)
class1 = np.array(class1)

func = "(-1)*" \
       + "("+ str(A) +")" \
       + "/" \
       + "("+str(B)+")" \
       + "*x+" \
       + "(-1)*" \
       + "("+str(C)+")" \
       + "/" + "("+str(B)+")"

print func

#if __name__ == "__main__":
    
g = Gnuplot.Gnuplot(debug=1)
g.title('SVM') # (optional)
g('set data style linespoints')

d0 = Gnuplot.Data(class0, title = "Class 0")
d1 = Gnuplot.Data(class1, title = "Class 1")
d2 = Gnuplot.Func(func, title = "Separating Plane")

g.plot(d0, d1, d2)

raw_input()
    