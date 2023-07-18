import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

def plot_heat_data(Z,dom, title, BCs=True, ax=None):
    """Visualize temperature data in space and time."""
    x_all = dom
    if ax is None:
        _, ax = plt.subplots(1, 1)

    # Plot a few snapshots over the spatial domain.
    sample_columns = [0, 2, 5, 10, 20, 40, 80, 160, 320, 400]
    color = iter(plt.cm.viridis_r(np.linspace(.05, 1, len(sample_columns))))

    leftBC, rightBC = [0], [0]
    for j in sample_columns:
        if BCs==True:
          q_all = np.concatenate([leftBC, Z[:,j], rightBC])
        else:
          q_all = Z[:, j]
        ax.plot(x_all, q_all, color=next(color), label=fr"$q(x,t_{{{j}}})$")

    ax.set_xlim(x_all[0], x_all[-1])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$q(x,t)$")
    ax.legend(loc=(1.05, .05))
    ax.set_title(title)

def optimal_svd_vecs(threshold_energy, singular_values):
  for j in range(len(singular_values)):
    if sum(singular_values[:j]**2) / sum(singular_values**2) > threshold_energy:
      return j+1
  
  return len(singular_values)

class OpInfA_torch(nn.Module):
    def __init__(self, n_rows, n_cols):
        super(OpInfA_torch, self).__init__()
        self.weights = nn.Parameter(torch.randn(n_rows, n_cols))

    def forward(self, x):
        output = torch.matmul(self.weights, x)
        return output


def train_loop(optimizer, loss_fn, data, A_inf, n_epochs = 100000):
    for epoch in range(n_epochs):
        
        Q_dot = torch.Tensor(data[0])
        Q_hat = torch.Tensor(data[1])
        rhs = A_inf(Q_hat)
        lhs = Q_dot
        
        loss = loss_fn(rhs,lhs)
        # loss = torch.mean(torch.square(rhs-lhs))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1)%2000==0:
            print("Loss :", loss.item())
            
def predict_from_torch(A_inf, q_0_hat, Vr, t):
  A_inf = A_inf.weights.cpu().detach().numpy()
  Q_hat_0 = q_0_hat
  Q_inf = np.zeros((Vr.shape[0], len(t)))
  Q_inf[:, 0] = np.linalg.multi_dot([Vr, Q_hat_0])
  current_q = Q_hat_0
  dt = t[1]-t[0]
  for i in range(len(t)-1):
    current_q = np.linalg.multi_dot([A_inf, current_q])*dt + current_q
    Q_inf[:, i+1] = np.linalg.multi_dot([Vr, current_q])
  
  return Q_inf

def predict_from_scratch(A_inf, q_0_hat, Vr, t):
  Q_hat_0 = q_0_hat
  Q_inf = np.zeros((Vr.shape[0], len(t)))
  Q_inf[:, 0] = np.linalg.multi_dot([Vr, Q_hat_0])
  current_q = Q_hat_0
  dt = t[1]-t[0]
  for i in range(len(t)-1):
    current_q = np.linalg.multi_dot([A_inf, current_q])*dt + current_q
    Q_inf[:, i+1] = np.linalg.multi_dot([Vr, current_q])
  
  return Q_inf