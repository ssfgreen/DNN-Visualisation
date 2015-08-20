# from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt
import base64

''' This file opens a csv file,
    re-saves another csv to place into contiguous memory space
    required to process into base64, then saves a base64 string to file
 '''


# standard importing
my_data = np.genfromtxt('mnist_coords.csv', delimiter=',')
my_labels = np.genfromtxt('mnist_ys.csv', delimiter=',')

# getting X, y and labels - also trims the NaNs
# TODO: should only do this if the no. dims is > 2!
X = my_data[:1000,0]
y = my_data[:1000,1]
label = my_labels[:1000,0]

lab64 = my_labels[:,0]

# saving the csv, so upon next import the data is c-congruient (in consecutive memory locations)
np.savetxt("test_csv.csv", lab64, delimiter=",")

# new data imported with float32, so it's possible to base64 encode like Colah
new_data = np.genfromtxt('test_csv.csv', dtype=np.float32, delimiter=',')
# print new_data
print type(new_data[0])
print new_data.shape
print new_data

# prints the type of the data at slot 0
# print type(new_data[0])

# encoding to base 64 or the Float32 data
new_data = base64.encodestring(new_data)
print new_data

with open("save_b64.txt", "w") as text_file:
    text_file.write(new_data)

# testing for c-congruency
# np.ascontiguousarray(lab64, dtype=np.float64)
# print lab64.flags['C_CONTIGUOUS']

# Plotting functions

# N = 500
# # x = np.random.rand(N)
# # y = np.random.rand(N)
# # colors = np.random.rand(N)
# area = 20 #np.pi * (15 * np.random.rand(N))**2 # 0 to 15 point radiuses

# # for some reason the input from output has a NaN appended: which is the '\n' character.
# plt.scatter(X, y, s=area, c=label, alpha=0.5)
# plt.show()