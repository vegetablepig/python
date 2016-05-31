from numpy import *
import random as rand
import Gnuplot
from scipy.optimize import leastsq

num = 10 # sample points
p_num = 2000

ax = Gnuplot.Gnuplot()
ax('set data style lines')

x_val = linspace(0,1.0,num) # x coordinates of sample points

x_real_val = linspace(0,1.0,p_num) # x coor


def sinFun(x):
    """
    sin(x) by x in radius
    """
    m_sin = sin(2*pi*x)
    return m_sin

y_true = sinFun(x_real_val)

data1 = []
for i in range(p_num):
    data1.append([x_real_val[i],y_true[i]])
    
# plot sin(x)
plot1 = Gnuplot.PlotItems.Data(data1, with_="lines", title="sin function") # sin(x)


# generate random points
y_val=[]
ya = sinFun(x_val);
for yy in ya:
    yy = rand.gauss(yy,0.2) # gaussian noise
    y_val.append(yy)

data2 = []
for i in range(num):
    data2.append([x_val[i],y_val[i]])
plot2 = Gnuplot.PlotItems.Data(data2, with_="point", title="gauss noise")

# begin fitting
def Poly(x, p, degree ):
    val = 0
    for i in range(0,degree+1):
        dx = 1.0
        for j in range(0,i):
            dx = dx*x
        val+=p[i]*dx
    return val


def Ew_Poly( p, deg, y, x):
    err = y-Poly( x, p, deg)
    return err


def Ew_Poly1( p, deg, y, x,local_lambda):
    err = y-Poly( x, p, deg)
    err = append(err,sqrt(local_lambda)*p)
    return err


def EwSquare_Poly( p, deg, y, x, lambda_,w):
    err = 0
    i=0
    for xx in x:
        err+= 0.5*(y[i]-Poly( xx, p, deg))**2
        i+=1
    max=w[0]
    for ww in w:
        if ww>max:
            max = ww
    return err + lambda_ * max**2



# fitting with 3-degree polynomials
p3=[0, 0, 0, 0]

polysq3 = leastsq(Ew_Poly, p3, args=(3, y_val, x_val))

y_fit3 = Poly(x_real_val,polysq3[0],3)
data3 = []
for i in range(p_num):
    data3.append([x_real_val[i],y_fit3[i]])
    
plot3 = Gnuplot.PlotItems.Data(data3, with_="line", title="M = 3")

print polysq3[0]

# fitting with 9-degree polymonials
p9=[0,0,0,0,0,0,0,0,0,0] # factors

polysq9 = leastsq(Ew_Poly, p9, args=(9, y_val, x_val))

y_fit9 = Poly(x_real_val,polysq9[0],9)
data4 = []
for i in range(p_num):
    data4.append([x_real_val[i],y_fit9[i]])
    
plot4 = Gnuplot.PlotItems.Data(data4, with_="line",
                               title="M = 9 Square minimal sample=100")

# fitting with 9-degree polymonials (regularized)
p9m=[0,0,0,0,0,0,0,0,0,0] # factors

ploysq9m = leastsq(Ew_Poly1, p9m, args=(9, y_val, x_val, e**(-18)))  #
y_fit9m = Poly(x_real_val,ploysq9m[0],9)
data5 = []
for i in range(p_num):
    data5.append([x_real_val[i],y_fit9m[i]])
    
plot5 = Gnuplot.PlotItems.Data(data5, with_="line",
                               title="M = 9 regularization ln(lambda)=-18")

ax.plot(plot1,plot2,plot5)
raw_input()
