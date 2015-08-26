import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('TkAgg') # used to avoid attribute error from moviepy
import matplotlib.pyplot as plt
import math
from tsne import bh_sne
# used in varima
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd


''' This file tests the PCA for working with Isometries and rotational data 

'''

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

def rotateArray(array, angle):
    rotation = np.array([[np.cos(angle), -1*np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = np.dot(rotation, array.T)
    return rotated.T

def castPCA(array):
    pca = PCA(n_components = array.shape[1])
    transform = pca.fit_transform(array)
    return transform

def castPCA2(array):
    pca = PCA(n_components = 2)
    transform = pca.fit_transform(array)
    return transform

def castTSNE(array):
    tsne = TSNE(n_components = array.shape[1])
    transform = tsne.fit_transform(array)
    return transform


def getRandomTriangle(nSamples):
    tri = np.random.rand(nSamples,2)
    tri = tri[tri[:,0] + tri[:,1] <= 1, :]
    return tri

def getUniformSquare(start,end,step,nSamples):
    coordinates = []
    i = 0
    for x in my_range(start,end, step):
      for y in my_range(start, end, step):
        if i < nSamples:
            coordinates.append([x, y])
            i += 1

    sqr = np.asarray(coordinates)
    return sqr

def getUniformTriangle(start,end,step,nSamples):
    coordinates = []
    yEnd = start
    i = 0
    for x in my_range(start,end,step):
        for y in my_range(start, yEnd, step):
            if i < nSamples:
                coordinates.append([x,y])
                i += 1
        yEnd += step
    tri = np.asarray(coordinates)
    return tri

def getRandomTriangle3D(nSamples):
    tri = np.random.rand(nSamples,3)
    tri = tri[tri[:,0] + tri[:,1] +tri[:,2] <= 1, :]
    return tri
    
def reflectX(array):
    reflect = np.zeros(array.shape)
    reflect[:,0] = array[:,0]
    reflect[:,1] = -1 * array[:,1]
    return reflect

def reflectY(array):
    reflect = np.zeros(array.shape)
    reflect[:,1] = array[:,1]
    reflect[:,0] = -1 * array[:,0]
    return reflect
 
def reflectXY(array):
    reflect = np.zeros(array.shape)
    reflect[:,1] = -1*array[:,1]
    reflect[:,0] = -1*array[:,0]
    return reflect

# rotates to orthogonal basis - therefore can just check for reflection and perfect rotation
def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in xrange(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R)

def twoD_histogram(array, labels, shape, filename):
    x = array[:,0]
    y = array[:,1]
    z = labels[:,0]
    # Bin the data onto a 10x10 grid
    # Have to reverse x & y due to row-first indexing
    zi, yi, xi = np.histogram2d(y, x, bins=(shape,shape), weights=z, normed=False)
    counts, _, _ = np.histogram2d(y, x, bins=(shape,shape))

    zi = zi / counts
    zi = np.ma.masked_invalid(zi)

    # ZI is the newly produced image, where the buckets have been filled!
    print "zi shape", zi.shape
    # print "xi shape", xi.shape
    # print "yi shape", yi.shape

    # print "ZI", zi
    # print "XI", xi
    # print "YI", yi

    fig, ax = plt.subplots()
    ax.pcolormesh(xi, yi, zi, edgecolors='black')
    scat = ax.scatter(x, y, c=z, s=15)
    fig.colorbar(scat)
    ax.margins(0.05)
    plt.savefig(filename)
    plt.show()

    return zi

def main_original():
    # Generate Random Triangle
    tri = getRandomTriangle(100)
    
    # Generate Random Traingle, rotate and offset
    tri2 = getRandomTriangle(100) 
    tri2 = rotateArray(tri2, np.pi/4)
    tri2[:,0] = tri2[:,0] + 10
    tri2[:,1] = tri2[:,1] + 5

    # Generate Random Triange, Rotate and offset (diff amounts)
    tri3 = getRandomTriangle(100) 
    tri3 = rotateArray(tri3, 5*np.pi/3)
    tri3[:,0] = tri3[:,0] + 10
    tri3[:,1] = tri3[:,1]

    # add both to the np array
    tri = np.append(tri, tri2, axis=0)
    tri = np.append(tri, tri3, axis=0)

    # Generate two rotations of the original data.
    triR1 = rotateArray(tri, 11*np.pi/6)
    triR2 = rotateArray(tri, np.pi/4)

    # plot original 3
    plt.scatter(tri[:,0], tri[:,1], c='b')
    # plt.savefig('rotatedDataOriginal.png')
    # plt.show()

    # Plot the three sets of triangles
    plt.scatter(tri[:,0], tri[:,1], color = 'b')
    # plt.scatter(triR1[:,0], triR1[:,1], color = 'r')
    # plt.scatter(triR2[:,0], triR2[:,1], color = 'g')
    # plt.savefig('rotatedData2.png')
    plt.show()
    plt.close()

    # Get the projected data in the PC spaces
    triCast = castPCA(tri)
    triCastR1 = castPCA(triR1)
    triCastR2 = castPCA(triR2)


    print "TRI: ", tri.shape, "\n", tri
    print "TRICAST: ", triCast.shape, "\n", triCast

    plt.scatter(triCast[:,0], triCast[:,1], color = [0.2,0.9,0.1])
    # plt.scatter(triCastR1[:,0], triCastR1[:,1], color = [0.2,0.9,0.1])
    # plt.scatter(triCastR2[:,0], triCastR2[:,1], color = 'g')  
    
    # To see which of the rotations overlap in 
    # print triCast - triCastR1
    # print triCast - triCastR2
    # print triCastR1 - triCastR2
    # plt.savefig('pcaData2.png')
    plt.show() 
    plt.close()
    
    # Create the possible reflections for each rotation (X, XY, Y)
    triCastR1RefX = reflectX(triCastR1)
    triCastR1RefY = reflectY(triCastR1)
    triCastR1RefXY = reflectXY(triCastR1)
    triCastR2RefX = reflectX(triCastR2)
    triCastR2RefY = reflectY(triCastR2)
    triCastR2RefXY = reflectXY(triCastR2)
    
    # Visualise the possible reflections for the first rotation.
    plt.scatter(triCast[:,0], triCast[:,1], color = 'b')
    plt.scatter(triCastR1RefY[:,0], triCastR1RefY[:,1], color = 'r')
    plt.scatter(triCastR1RefX[:,0], triCastR1RefX[:,1], color = 'g')
    plt.scatter(triCastR1RefXY[:,0], triCastR1RefXY[:,1], color = 'y')
    # plt.savefig('possible_reflections.png')
    # plt.show()
    plt.close()

    # See which reflection overlaps the original.
    # print triCast - triCastR1RefX
    # print triCast - triCastR1RefY
    # print triCast - triCastR1RefXY
    # print triCast - triCastR1


    # print "FOR THE SECOND ONE"
    # print triCast - triCastR2RefX
    # print triCast - triCastR2RefY
    # print triCast - triCastR2RefXY
    # print triCast - triCastR2

    
    plt.close()

def new_main():
    # Generate Random Triangle
    # tri = getRandomTriangle(100)
    tri = getUniformTriangle(1,3,0.1,100)

    # offset and rotate the triangle
    tri2 = getUniformTriangle(1,3,0.1,100)
    tri2 = rotateArray(tri2, np.pi/4)    
    tri2[:,0] = tri2[:,0] + 10
    tri2[:,1] = tri2[:,1] + 5

    # Generate Random Square
    sqr = getUniformSquare(0,1,0.1,100)

    sqr2 = getUniformSquare(0,1,0.1,100)
    sqr2 = rotateArray(sqr2, 5*np.pi/3)
    sqr2[:,0] = sqr2[:,0] + 10
    sqr2[:,1] = sqr2[:,1]

    sqr3 = getUniformSquare(0,1,0.1,100)
    sqr3 = rotateArray(sqr3, 9*np.pi/2)
    sqr3[:,0] = sqr3[:,0] + 10
    sqr3[:,1] = sqr3[:,1]

    # print "shapes before (t,s)", tri.shape, sqr2.shape
    tri_labels = np.full((100, 1), 10, dtype=np.int)
    tri2_labels = np.full((100, 1), 8, dtype=np.int)
    sqr_labels = np.full((100,1), 1, dtype=np.int)
    sqr2_labels = np.full((100,1), 3, dtype=np.int)
    sqr3_labels = np.full((100,1), 5, dtype=np.int)

    appended = np.append(tri, sqr, axis=0)
    appended = np.append(appended, tri2, axis=0)
    appended = np.append(appended, sqr2, axis=0)
    appended = np.append(appended, sqr3, axis=0)

    appended_labels = np.append(tri_labels, sqr_labels, axis=0)
    appended_labels = np.append(appended_labels, tri2_labels, axis=0)
    appended_labels = np.append(appended_labels, sqr2_labels, axis=0)
    appended_labels = np.append(appended_labels, sqr3_labels, axis=0)

    print "appended-shape", appended.shape
    print "appended labels", appended_labels.shape

    histo1 = twoD_histogram(appended, appended_labels, 40, "histo1.png")

    rotate_appended = rotateArray(appended, np.pi/4)    

    histo2 = twoD_histogram(rotate_appended, appended_labels, 40, "histo2.png")

    histo1_rs = np.reshape(histo1, (1,-1))
    histo2_rs = np.reshape(histo2, (1,-1))

    histo_app = np.append(histo1_rs, histo2_rs, axis=0)
    histo_lab = [1,2]

    histo_bh = bh_sne(histo_app, perplexity=0.3, theta=0.5)
    plt.scatter(histo_bh[:,0], histo_bh[:,1], c=histo_lab)
    plt.savefig('BHafter2d.png')
    plt.show()

    # plot initial
    plt.scatter(sqr[:,0], sqr[:,1], c='b')
    plt.scatter(sqr2[:,0], sqr2[:,1], c='g')
    plt.scatter(sqr3[:,0], sqr3[:,1], c='k')
    plt.scatter(tri[:,0], tri[:,1], c='y')
    plt.scatter(tri2[:,0], tri2[:,1], c='r')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((-1,13,-5,10))
    plt.savefig('rotatedDataOriginal.png')
    plt.show()

    # apply pca
    triCast = castPCA(tri)
    tri2Cast = castPCA(tri2)
    sqrCast = castPCA(sqr)
    sqr2Cast = castPCA(sqr2)
    sqr3Cast = castPCA(sqr3)

    # reshape so can append
    triCastL = np.reshape(triCast, (1,-1))
    tri2CastL = np.reshape(tri2Cast, (1,-1))
    sqrCastL = np.reshape(sqrCast, (1,-1))
    sqr2CastL = np.reshape(sqr2Cast, (1,-1))
    sqr3CastL = np.reshape(sqr3Cast, (1,-1))


    all_ = [triCastL, tri2CastL, sqrCastL, sqr2CastL, sqr3CastL]

    # remove NaNs and infintes
    new = []
    for i in all_:
        # i = i[~np.isnan(i)]
        i = i[np.logical_not(np.isnan(i))]
        print i.shape
        new.append(i)

    new_ = np.asarray(new)
    new_ = np.around(new_, decimals=3)
    new_[new_ > 1e8] = 0

    print new_

    sne_col = bh_sne(new_, perplexity=1.0, theta=0.5)

    labels = np.asarray([1,2,3,4,5])

    plt.scatter(sne_col[:,0], sne_col[:,1], c=labels)
    plt.savefig('BHafterPCA.png')
    plt.show()

    # plot PCA cast
    plt.scatter(sqrCast[:,0], sqrCast[:,1], c='b')
    plt.scatter(sqr2Cast[:,0], sqr2Cast[:,1], c='g')
    plt.scatter(sqr3Cast[:,0], sqr3Cast[:,1], c='k')
    plt.scatter(triCast[:,0], triCast[:,1], c='y')
    plt.scatter(tri2Cast[:,0], tri2Cast[:,1], c='r')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((-2,2,-2,2))
    plt.savefig('PCArotations.png')
    plt.show()

    # apply pca varimax to make them orthogonally similar
    triVari = varimax(triCast)
    tri2Vari = varimax(tri2Cast)
    sqrVari = varimax(sqrCast)
    sqr2Vari = varimax(sqr2Cast)
    sqr3Vari = varimax(sqr3Cast)

    # plot initial
    plt.scatter(sqrVari[:,0], sqrVari[:,1], c='b')
    plt.scatter(sqr2Vari[:,0], sqr2Vari[:,1], c='g')
    plt.scatter(triVari[:,0], triVari[:,1], c='y')
    plt.scatter(tri2Vari[:,0], tri2Vari[:,1], c='r')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((-2,2,-2,2))
    plt.savefig('VarimaxRotations.png')
    plt.show()

    triList = np.reshape(triVari, (1,-1))
    tri2List = np.reshape(tri2Vari, (1,-1))
    sqrList = np.reshape(sqrVari, (1,-1))
    sqr2List = np.reshape(sqr2Vari, (1,-1))
    sqr3List = np.reshape(sqr3Vari, (1,-1))

    all_ = [triList, tri2List, sqrList, sqr2List, sqr3List]

    new = []
    for i in all_:
        # i = i[~np.isnan(i)]
        i = i[np.logical_not(np.isnan(i))]
        print i.shape
        new.append(i)

    new_ = np.asarray(new)
    new_[new_ > 1e8] = 0

    print new_

    sne_co = bh_sne(new_, perplexity=1.0, theta=0.5)

    print sne_co
    

    labels = np.asarray([1,2,3,4,5])

    plt.scatter(sne_co[:,0], sne_co[:,1], c=labels)
    plt.savefig('BHrotations.png')
    plt.show()


if __name__ == '__main__':
    
    # main_original()
    new_main()

    # pca doesn't use the labels, but LDA eoes
    # from sklearn.lda import LDA