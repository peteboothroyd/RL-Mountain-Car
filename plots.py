import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import multivariate_normal
import pandas as pd

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

def entropy_plot():
  entropy_csv = pd.read_csv('/Users/peterboothroyd/Downloads/entropy.csv')
  pg_loss_csv = pd.read_csv('/Users/peterboothroyd/Downloads/pg_loss.csv')

  # data = pg_loss_csv.values
  data = entropy_csv.values

  vals = data[:,2]
  t_steps = data[:,1]*80

  N=10
  averaged_vals = np.convolve(vals, np.ones((N,))/N, mode='same')

  fig, ax = plt.subplots()
  ax.plot(t_steps, averaged_vals)
  ax.set_xlabel("Training Step")
  ax.set_ylabel("Policy Gradient Loss")
  ax.set_ylim(0, 1.5)
  # ax.set_title('Destabilised Policy Gradient Loss during Entropy Collapse')
  ax.set_title('Collapsing Entropy during A2C Learning ')
  ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%0.00e'))
  plt.show()

if __name__ == "__main__":
  entropy_plot()