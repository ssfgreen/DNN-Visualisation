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

    fig, ax = plt.subplots()
    ax.pcolormesh(xi, yi, zi, edgecolors='black')
    scat = ax.scatter(x, y, c=z, s=15)
    fig.colorbar(scat)
    ax.margins(0.05)
    plt.savefig(filename)
    plt.show()

    return zi

def new_main():
    # Generate Triangle
    tri = getUniformTriangle(1,3,0.1,100)

    # Offset and rotate the triangle
    tri2 = getUniformTriangle(1,3,0.1,100)
    tri2 = rotateArray(tri2, np.pi/4)    
    tri2[:,0] = tri2[:,0] + 10
    tri2[:,1] = tri2[:,1] + 5

    # Generate Square
    sqr = getUniformSquare(0,1,0.1,100)

    # Rotate Square
    sqr2 = getUniformSquare(0,1,0.1,100)
    sqr2 = rotateArray(sqr2, 5*np.pi/3)
    sqr2[:,0] = sqr2[:,0] + 10
    sqr2[:,1] = sqr2[:,1]

    # Rotate Square
    sqr3 = getUniformSquare(0,1,0.1,100)
    sqr3 = rotateArray(sqr3, 9*np.pi/2)
    sqr3[:,0] = sqr3[:,0] + 10
    sqr3[:,1] = sqr3[:,1]

    # create artifical labels 
    tri_labels = np.full((100, 1), 10, dtype=np.int)
    tri2_labels = np.full((100, 1), 8, dtype=np.int)
    sqr_labels = np.full((100,1), 1, dtype=np.int)
    sqr2_labels = np.full((100,1), 3, dtype=np.int)
    sqr3_labels = np.full((100,1), 5, dtype=np.int)

    # append artificial data for histogram testing
    appended = np.append(tri, sqr, axis=0)
    appended = np.append(appended, tri2, axis=0)
    appended = np.append(appended, sqr2, axis=0)
    appended = np.append(appended, sqr3, axis=0)

    # append artificial labels for histogram testing
    appended_labels = np.append(tri_labels, sqr_labels, axis=0)
    appended_labels = np.append(appended_labels, tri2_labels, axis=0)
    appended_labels = np.append(appended_labels, sqr2_labels, axis=0)
    appended_labels = np.append(appended_labels, sqr3_labels, axis=0)

    # check shape
    print "appended-shape", appended.shape
    print "appended labels", appended_labels.shape

    # create duplicated dataset, that is simply rotated
    rotate_appended = rotateArray(appended, np.pi/4)    

    # convert to histogram images of artificial and rotated datasets
    histo1 = twoD_histogram(appended, appended_labels, 40, "histo1.png")
    histo2 = twoD_histogram(rotate_appended, appended_labels, 40, "histo2.png")

    # reshape to make ready for Barnes Hut SNE
    histo1_rs = np.reshape(histo1, (1,-1))
    histo2_rs = np.reshape(histo2, (1,-1))

    histo_app = np.append(histo1_rs, histo2_rs, axis=0)
    histo_lab = [1,2]

    # Run Barnes Hut SNE
    histo_bh = bh_sne(histo_app, perplexity=0.3, theta=0.5)

    # Plot BH SNE version of histogram images
    plt.scatter(histo_bh[:,0], histo_bh[:,1], c=histo_lab)
    plt.savefig('BHafter2d.png')
    plt.show()

    # Plot origininal dataset of rotated trianges and squares
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

    # put into list to make more efficient transformations (i.e - not like above)
    all_ = [triCastL, tri2CastL, sqrCastL, sqr2CastL, sqr3CastL]

    # remove NaNs and infintes!!
    new = []
    for i in all_:
        # i = i[~np.isnan(i)]
        i = i[np.logical_not(np.isnan(i))]
        print i.shape
        new.append(i)

    # rounding caused errors?
    new_ = np.asarray(new)
    new_ = np.around(new_, decimals=3)
    new_[new_ > 1e8] = 0

    # run BN SNE on the PCA transformed dataset
    sne_col = bh_sne(new_, perplexity=1.0, theta=0.5)

    # 5 labels = 5 shapes (2 tri, 3 sqr)
    labels = np.asarray([1,2,3,4,5])

    # PLOT
    plt.scatter(sne_col[:,0], sne_col[:,1], c=labels)
    plt.savefig('BHafterPCA.png')
    plt.show()

    # plot PCA cast without the application on BN_SNE
    plt.scatter(sqrCast[:,0], sqrCast[:,1], c='b')
    plt.scatter(sqr2Cast[:,0], sqr2Cast[:,1], c='g')
    # plt.scatter(sqr3Cast[:,0], sqr3Cast[:,1], c='k')
    plt.scatter(triCast[:,0], triCast[:,1], c='y')
    plt.scatter(tri2Cast[:,0], tri2Cast[:,1], c='r')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((-2,2,-2,2))
    plt.savefig('PCArotations.png')
    plt.show()

    # apply pca varimax to make them orthogonally similar (better classificaiton)
    triVari = varimax(triCast)
    tri2Vari = varimax(tri2Cast)
    sqrVari = varimax(sqrCast)
    sqr2Vari = varimax(sqr2Cast)
    sqr3Vari = varimax(sqr3Cast)

    # plot after having applied varimax and pca
    plt.scatter(sqrVari[:,0], sqrVari[:,1], c='b')
    plt.scatter(sqr2Vari[:,0], sqr2Vari[:,1], c='g')
    plt.scatter(triVari[:,0], triVari[:,1], c='y')
    plt.scatter(tri2Vari[:,0], tri2Vari[:,1], c='r')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((-2,2,-2,2))
    plt.savefig('VarimaxRotations.png')
    plt.show()

    # put into list for better manipulation
    triList = np.reshape(triVari, (1,-1))
    tri2List = np.reshape(tri2Vari, (1,-1))
    sqrList = np.reshape(sqrVari, (1,-1))
    sqr2List = np.reshape(sqr2Vari, (1,-1))
    sqr3List = np.reshape(sqr3Vari, (1,-1))

    all_ = [triList, tri2List, sqrList, sqr2List, sqr3List]

    # remove infinites and NaNs
    new = []
    for i in all_:
        # i = i[~np.isnan(i)]
        i = i[np.logical_not(np.isnan(i))]
        print i.shape
        new.append(i)

    new_ = np.asarray(new)
    new_[new_ > 1e8] = 0

    # lot the Varimax / PCA attempt
    sne_co = bh_sne(new_, perplexity=1.0, theta=0.5)

    labels = np.asarray([1,2,3,4,5])

    plt.scatter(sne_co[:,0], sne_co[:,1], c=labels)
    plt.savefig('BHrotations.png')
    plt.show()


if __name__ == '__main__':
    
    new_main()

    # pca doesn't use the labels, but LDA eoes
    # from sklearn.lda import LDA