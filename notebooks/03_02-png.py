import numpy as np
import matplotlib.pyplot as plt

n = 256

x = np.linspace(0, 2*np.pi, n)

X, Y = np.meshgrid(x, x)



u = (
    np.sin(4 * X)
    + 0.5 * np.sin(20 * X)
)


plt.imshow(u, cmap="gray")
plt.axis("off")

plt.show()

