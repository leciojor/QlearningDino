import random
import pygame

SCREEN_WIDTH = 1100

class Obstacle:
    WIDTH = 40
    HEIGHT = 40

    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, self.WIDTH, self.HEIGHT)

    def update(self, speed):
        self.rect.x -= speed


class SmallCactus(Obstacle):
    def __init__(self):
        y = 325
        x = SCREEN_WIDTH 
        super().__init__(x, y)


class LargeCactus(Obstacle):
    def __init__(self):
        y = 300
        x = SCREEN_WIDTH
        super().__init__(x, y)


class Bird(Obstacle):
    def __init__(self):
        y = 250
        x = SCREEN_WIDTH
        super().__init__(x, y)


class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5
    WIDTH = 40
    HEIGHT = 40

    def __init__(self):
        self.rect = pygame.Rect(self.X_POS, self.Y_POS, self.WIDTH, self.HEIGHT)
        self.jump_vel = self.JUMP_VEL
        self.is_jumping = False
        self.is_ducking = False

    def update(self, action):
        if action == "jump" and not self.is_jumping:
            self.is_jumping = True
            self.is_ducking = False
        elif action == "duck" and not self.is_jumping:
            self.is_ducking = True
        elif action == "run":
            self.is_ducking = False

        if self.is_jumping:
            self.rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
            if self.jump_vel < -self.JUMP_VEL:
                self.is_jumping = False
                self.jump_vel = self.JUMP_VEL
                self.rect.y = self.Y_POS
        elif self.is_ducking:
            self.rect.y = self.Y_POS_DUCK
        else:
            self.rect.y = self.Y_POS


class Env:
    def __init__(self):
        self.dino = Dinosaur()
        self.obstacle = None
        self.game_speed = 20
        self.done = False

    def reset(self):
        self.dino = Dinosaur()
        self.obstacle = self._spawn_obstacle()
        self.game_speed = 20
        self.done = False
        return self.get_state()

    def _spawn_obstacle(self):
        choice = random.choice(["small", "large", "bird"])
        if choice == "small":
            return SmallCactus()
        elif choice == "large":
            return LargeCactus()
        else:
            return Bird()

    def step(self, action):
        if self.done:
            return self.get_state(), 0, self.done, {}
        action_map = {0: "run", 1: "jump", 2: "duck"}
        action = action_map[action]


        self.dino.update(action)
        self.obstacle.update(self.game_speed)

        reward = 0.1 
        if self.dino.rect.colliderect(self.obstacle.rect):
            reward = -100
            self.done = True
            return self.get_state(), reward, self.done, {}

        if self.obstacle.rect.right < self.dino.rect.left:
            if isinstance(self.obstacle, Bird) and action == "duck":
                reward += 10
            elif isinstance(self.obstacle, (SmallCactus, LargeCactus)) and action == "jump":
                reward += 10
            else:
                reward += 5 
            self.obstacle = self._spawn_obstacle()
        else:
            if action == "jump" and not isinstance(self.obstacle, (SmallCactus, LargeCactus)):
                reward -= 1
            elif action == "duck" and not isinstance(self.obstacle, Bird):
                reward -= 0.5

        reward += self.game_speed * 0.01

        return self.get_state(), reward, self.done, {}

    def get_state(self):
        dist_to_obstacle = self.obstacle.rect.x - self.dino.rect.x
        dino_y = self.dino.rect.y
        obs_y = self.obstacle.rect.y
        obs_width = self.obstacle.rect.width

        is_small = isinstance(self.obstacle, SmallCactus)
        is_large = isinstance(self.obstacle, LargeCactus)
        is_bird = isinstance(self.obstacle, Bird)

        one_hot = [int(is_small), int(is_large), int(is_bird)]

        return [
            dist_to_obstacle,
            dino_y,
            obs_y,
            obs_width,
            self.game_speed,
            *one_hot
        ]
