import torch.nn as nn

class QNetwork(nn.Module):
  def __init__(self, state_size, action_size, hidden_size, hidden_size_final, norm=False):
    super().__init__()
    self.action_size = action_size
    self.state_size = state_size
    if not norm:
      self.net = nn.Sequential(nn.Linear(state_size, hidden_size),
                              nn.ReLU(),
                              nn.Linear(hidden_size, hidden_size),
                              nn.ReLU(),
                              nn.Linear(hidden_size, hidden_size_final),
                              nn.ReLU(),
                              nn.Linear(hidden_size_final, action_size))
    else:
      self.net = nn.Sequential(nn.Linear(state_size, hidden_size),
                              nn.ReLU(),
                              nn.LayerNorm(hidden_size),
                              nn.Linear(hidden_size, hidden_size),
                              nn.ReLU(),
                              nn.LayerNorm(hidden_size),
                              nn.Linear(hidden_size, hidden_size_final),
                              nn.ReLU(),
                              nn.LayerNorm(hidden_size_final),
                              nn.Linear(hidden_size_final, action_size))

  def forward(self, x):
    """Estimate q-values given state

      Args:
          state (tensor): current state, size (batch x state_size)

      Returns:
          q-values (tensor): estimated q-values, size (batch x action_size)
    """
    return self.net(x)