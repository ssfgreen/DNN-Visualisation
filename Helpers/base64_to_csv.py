import numpy as np
import matplotlib.pyplot as plt
import base64

# my_data = np.genfromtxt('mnist_ys.txt')
with open ("mnist_ys.txt", "r") as myfile:
    data=myfile.read()
print data

r = base64.decodestring(data)
print "r", r
q = np.frombuffer(r, dtype=np.float32)
print "q", q

# it works and decodes it!!!
np.savetxt("save_b64.csv", q, delimiter=",")

