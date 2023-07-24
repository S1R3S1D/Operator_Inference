import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp

def plot_heat_data(Z,dom, title, BCs=True, ax=None, legend=True):
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
    if legend==True:
      ax.legend(loc=(1.15, .05))
    ax.set_title(title)

def optimal_svd_vecs(threshold_energy, singular_values):
  for j in range(len(singular_values)):
    if sum(singular_values[:j]**2) / sum(singular_values**2) > threshold_energy:
      return j
  
  return len(singular_values)

def x_2(x):
    s = int((x.shape[0]*(x.shape[0]+1))/2)
    x2 = np.zeros((s, x.shape[1]))
    k = 0
    rows, cols = np.triu_indices(x.shape[0])
    for i, j in zip(rows, cols):
        x2[k] = x[i]*x[j]
        k += 1
    return x2
  
class OpInfA_torch(nn.Module):
    def __init__(self, n_rows, n_cols):
        super(OpInfA_torch, self).__init__()
        self.weights = nn.Parameter(torch.randn(n_rows, n_cols))

    def forward(self, x):
        output = torch.matmul(self.weights, x)
        return output


class OpInfs_torch(nn.Module):
  def __init__(self, Q_dim, model_type="Li"):
    super(OpInfs_torch, self).__init__()
    N = Q_dim[0]
    self.model_type = model_type
    S = int((N*(N+1))/2)
    if "Li" in model_type:
      self.A = nn.Parameter(torch.zeros(N, N))
    if "B" in model_type:
      self.B = nn.Parameter(torch.zeros(N, S))
          
  def forward(self, x):
    output = torch.zeros_like(x)
    if "Li" in self.model_type:
      output += torch.matmul(self.A, x)
    if "B" in self.model_type:
      output += torch.matmul(self.B, torch.tensor(x_2(x), dtype=torch.float32))
    return output

  def predict(self, q_0_hat, Vr, t):
    with torch.no_grad():
      if "Li" in self.model_type:
        A_inf = self.A.cpu().detach().numpy()
      if "B" in self.model_type:
        B_inf = self.B.cpu().detach().numpy()
      
      Q_hat_0 = q_0_hat
      
      Q_inf = np.zeros((Vr.shape[0], len(t)))
      Q_inf[:, 0] = np.linalg.multi_dot([Vr, Q_hat_0])

      
      current_q = Q_hat_0.reshape(-1, 1)
      
      dt = t[1]-t[0]
      for i in range(len(t)-1):
        c_q = current_q.copy()
        if "Li" in self.model_type:
          current_q += np.linalg.multi_dot([A_inf, c_q])*dt
        
        if "B" in self.model_type:
          current_q += np.linalg.multi_dot([B_inf, x_2(c_q)])*dt
        
        Q_inf[:, i+1] = np.linalg.multi_dot([Vr, current_q])[:, 0]
    
    return Q_inf

def train_loop(optimizer, loss_fn, data, oper_inf, n_epochs = 100000):
    Q_dot = torch.Tensor(data[0].copy())
    Q_hat = torch.Tensor(data[1].copy())
    
    lhs = Q_dot
    
    for epoch in range(n_epochs):
      
        rhs = oper_inf(Q_hat)
        
        loss = loss_fn(rhs,lhs)
        
        
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

def predict_from_scratch(q_0_hat, Vr, t, A_inf = None, B_inf = None):
  Q_hat_0 = q_0_hat
  Q_inf = np.zeros((Vr.shape[0], len(t)))
  Q_inf[:, 0] = np.linalg.multi_dot([Vr, Q_hat_0])
  current_q = Q_hat_0.reshape(-1, 1)
  dt = t[1]-t[0]
  for i in range(len(t)-1):
    if A_inf is not None and B_inf is not None:
      current_q = np.linalg.multi_dot([A_inf, current_q])*dt + np.linalg.multi_dot([B_inf, x_2(current_q)])*dt + current_q
    elif A_inf is not None:
      current_q = np.linalg.multi_dot([A_inf, current_q])*dt + current_q
    elif B_inf is not None:
      current_q = np.linalg.multi_dot([B_inf, x_2(current_q)])*dt + current_q
  
    Q_inf[:, i+1] = np.linalg.multi_dot([Vr, current_q])[:, 0]
  
  return Q_inf

def frobenius_norm_error(A, B):
  return np.sqrt(np.sum((A-B)**2))

def compare(q_0, Vr, Q_data, rom, torch_rom, A_inf, B_inf, x_all, t):
  q_0_hat = Vr.T@q_0
  dt = t[1]-t[0]
  t0 = t[0]
  tf = t[-1]
  Q_ROM_scratch_uno = predict_from_scratch(A_inf=A_inf, B_inf=B_inf, q_0_hat=q_0_hat, Vr=Vr, t=t)
  Q_ROM_Opinf_uno = rom.predict(q_0_hat, t, method="BDF", max_step=dt)
  Q_ROM_Torch_uno = torch_rom.predict(q_0_hat, Vr, t)
  

  fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2)
  plt.subplots_adjust(hspace=0.5, wspace=0.3)
  
  plot_heat_data(Q_data,x_all, f"Actual Data",BCs=False, ax=ax1, legend=False)
  plot_heat_data(Q_ROM_Torch_uno, x_all, f"Pytorch ROM, ε {frobenius_norm_error(Q_data, Q_ROM_Torch_uno):.2e}", BCs=False, ax=ax2, legend=False)
  plot_heat_data(Q_ROM_Opinf_uno, x_all, f"OpInf ROM, ε {frobenius_norm_error(Q_data, Q_ROM_Opinf_uno):.2e}", BCs=False, ax=ax3, legend=False)
  plot_heat_data(Q_ROM_scratch_uno, x_all, f"Scratch ROM, ε {frobenius_norm_error(Q_data, Q_ROM_scratch_uno):.2e}", BCs=False, ax=ax4, legend=True)
  
  plt.show()

def get_reduced_matrices(A, Vr):
  return np.linalg.multi_dot([np.transpose(Vr), A]) 

def get_time_derivatives(Q, dt):
  return (Q[:, 1:] - Q[:, :-1])/dt