import numpy as np
print('ciao')
import matplotlib.pyplot as plt


plt.figure('test', figsize=(30,30))
x = np.asarray((1,2,3,4,5))
y = np.asarray((1,2,3,4,5))

plt.plot(x, y, 'o-', ms=10, lw=5, color='lightblue')
plt.plot(x, y**2, 'o-', ms=10, lw=5, color='red')
plt.show()