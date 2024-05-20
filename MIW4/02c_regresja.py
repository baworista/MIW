import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt('dane7.txt')

x = a[:,[0]]
y = a[:,[1]]

c = np.hstack([x*x*x, x*x, x, np.ones(x.shape)])
v = np.linalg.inv(c.T@c)@c.T @ y

print(c)

c1 = np.hstack([1/x, np.ones(x.shape)])
v1 = np.linalg.pinv(c1) @ y

c2 = np.hstack([x, np.ones(x.shape)])
v2 = np.linalg.pinv(c2) @ y



plt.plot(x, y, 'ro')
plt.plot(x,v[0]*x*x*x + v[1]*x*x + v[2]*x + v[3],)
plt.plot(x,v1[0]/x + v1[1])
plt.plot(x,v2[0]*x + v2[1])
plt.show()

