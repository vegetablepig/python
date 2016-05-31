import os
import numpy
import Gnuplot
import numpy.linalg as lin

def write_char_list(charlist,a,b):
    fp = open('optdigits-orig_3.tra', 'a')
    for item in charlist:
        fp.write(item)
        a = numpy.array(item)
        a.transpose()

    fp.write('\n\n')
    fp.close()


def select_three():
    row_count = 33
    k = 0
    charList = []
    a = []
    b = []
    fileHandle = open('digit.tra', 'r')
    fileList = fileHandle.readlines()
    # if os.path.exists('optdigits-orig_3.tra') is True:
    #    os.remove('optdigits-orig_3.tra')
        
    for fileLine in fileList:
        row_count -= 1
        if row_count != 0:
            charList.append(fileLine)
        else:
            row_count = 33     
            if fileLine == ' 3\n':
                k += 1
                write_char_list(charList,a,b)
            charList = []
    
    fileHandle.close()
    
def getMatrixs():
    feature, featureMatrix = [], []
    
    fileHandle = open('optdigits-orig_3.tra', 'r')
    fileList = fileHandle.readlines()
    for fileLine in fileList:
        line = fileLine.rstrip()
        if line != "":
            feature += [int(x) for x in line]
        else:
            if(len(feature)>0):                
                featureMatrix.append(feature)
            feature = []
    

    dataMatrix = numpy.array(featureMatrix)
    print dataMatrix.T  
    print 'have already transformed the samples into matrixs.'
    print numpy.shape(dataMatrix.T)
    fp = open('result', 'a')
    fp.write(str(featureMatrix))
    fp.close()
    return dataMatrix.T


select_three()
mat = getMatrixs()
res = mat.T - mat.mean(1)
res = res.T
u,s,v = lin.svd(res,full_matrices=True)

print numpy.shape(u)
print numpy.shape(s)
print numpy.shape(v)

u_d2 = u[:,0:2].T      
data = numpy.dot(u_d2,res)
print numpy.shape(data)
data1 = data.T



####################
# begin ploting
ax = Gnuplot.Gnuplot()
x = []
y = []
pltdata=[]
for item in data1:
    x.append(item[0]) 
    y.append(item[1])
    pltdata.append([item[0],item[1]])
plot = Gnuplot.PlotItems.Data(pltdata, with_="points pointtype 7 pointsize "
                                             "0", title="PCA coordinates of 199 samples")
ax.xlabel('first principal component')
ax.ylabel('second principal component')
ax.title('PCA analysis with 2 components')

firstPline = Gnuplot.PlotItems.Data(((min(x),0),(max(x),0)), with_="lines linetype 2 linewidth 2")
secondPline = Gnuplot.PlotItems.Data(((0,min(y)),(0,max(y))), with_="lines linetype 3 linewidth 2")

ax.plot(plot, firstPline, secondPline)

