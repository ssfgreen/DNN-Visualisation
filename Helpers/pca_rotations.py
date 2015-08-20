import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math

''' This file tests the PCA for working with Isometries and rotational data 

'''

def rotateArray(array, angle):
    rotation = np.array([[np.cos(angle), -1*np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = np.dot(rotation, array.T)
    return rotated.T

def castPCA(array):
    pca = PCA(n_components = array.shape[1])
    transform = pca.fit_transform(array)
    return transform

def getRandomTriangle(nSamples):
    tri = np.random.rand(nSamples,2)
    tri = tri[tri[:,0] + tri[:,1] <= 1, :]
    return tri

def getUniformSquare(nSamples):
    coordinates = []
    size = int(math.sqrt(nSamples))
    for x in range(size):
      for y in range(size):
        coordinates.append([x, y])
    sqr = np.asarray(coordinates)
    return sqr

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
    plt.savefig('rotatedDataOriginal.png')
    # plt.show()

    # Plot the three sets of triangles
    plt.scatter(tri[:,0], tri[:,1], color = 'b')
    plt.scatter(triR1[:,0], triR1[:,1], color = 'r')
    plt.scatter(triR2[:,0], triR2[:,1], color = 'g')
    plt.savefig('rotatedData2.png')
    # plt.show()
    plt.close()

    # Get the projected data in the PC spaces
    triCast = castPCA(tri)
    triCastR1 = castPCA(triR1)
    triCastR2 = castPCA(triR2)

    print "TRI: ", tri.shape, "\n", tri
    print "TRICAST: ", triCast.shape, "\n", triCast

    plt.scatter(triCast[:,0], triCast[:,1], color = [0.2,0.9,0.1])
    plt.scatter(triCastR1[:,0], triCastR1[:,1], color = [0.2,0.9,0.1])
    plt.scatter(triCastR2[:,0], triCastR2[:,1], color = 'g')  
    
    # To see which of the rotations overlap in 
    # print triCast - triCastR1
    # print triCast - triCastR2
    # print triCastR1 - triCastR2
    plt.savefig('pcaData2.png')
    # plt.show() 
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
    plt.savefig('possible_reflections.png')
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
    tri = getRandomTriangle(100)

    # Generate Random Square
    sqr = getUniformSquare(100)

    plt.scatter(sqr[:,0], sqr[:,1], c='b')
    plt.scatter(tri[:,0], tri[:,1], c='y')
    # plt.savefig('rotatedDataOriginal.png')
    plt.show()

if __name__ == '__main__':
    
    main_original()