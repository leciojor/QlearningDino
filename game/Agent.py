import torch
from model.Qnet import QNetwork
from model.DuelQnet import DuelingQNetwork

import pygame

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class Agent:

    def __init__(self, model, duel=False):
        state_size = 8
        action_size = 3
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not duel:
            self.model = QNetwork(state_size, action_size, 128, 64, True).to(device)
        else:
             self.model = DuelingQNetwork(state_size, action_size, 128).to(device)
        self.model.load_state_dict(torch.load(model, map_location=device))
        self.model.eval()

    def get_model_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.argmax(self.model(state.unsqueeze(0)))
        if action == 1:
            return {pygame.K_UP: True, pygame.K_DOWN: False}
        elif action == 2:
            return {pygame.K_UP: False, pygame.K_DOWN: True}
        else: 
            return {pygame.K_UP: False, pygame.K_DOWN: False}

