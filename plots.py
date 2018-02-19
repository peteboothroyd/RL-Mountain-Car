import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def gaussian_reward():
  X, X_DOT = np.meshgrid(np.linspace(-1,1,num=250), np.linspace(-2,2,num=250))
  states = np.array([X, X_DOT]).T
  r = multivariate_normal.pdf(states, [0.6, 0.0], 0.05**2) / 66.84
  r = np.rot90(r)
  contour = plt.contourf(X, X_DOT, r)
  plt.colorbar(contour, shrink=0.5)
  plt.xlabel("x")
  plt.ylabel("dx")
  plt.xlim([-1,1])
  plt.ylim([-2,2])
  plt.title("Reward Function")
  plt.savefig("gaussian_reward.png", dpi=300)

def hill():
  x1 = np.linspace(-1,0, num=150)
  x2 = np.linspace(0,1,num=150)
  y1 = x1 * x1 + x1
  y2 = x2 / np.sqrt(1 + 5 * x2**2)
  x = np.concatenate([x1, x2])
  y = np.concatenate([y1, y2])

  end_height = 0.6/math.sqrt(1+5*0.6**2)

  start, = plt.plot([-0.5], [-0.25], marker='*', markersize=10, color="red", label="Start")
  end, = plt.plot([0.6], [end_height], marker='o', markersize=10, color="green", label="Finish")

  plt.plot(x,y)
  plt.legend(handles=[start, end])
  plt.xlabel("x")
  plt.ylabel("height")
  plt.xlim([-1,1])
  plt.ylim([-0.3,0.5])
  plt.show()


if __name__ == "__main__":
  gaussian_reward()