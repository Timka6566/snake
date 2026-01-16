import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20
SPEED = 200


class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w, self.h = w, h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 25, bold=True)
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, Point(
            self.head.x-BLOCK_SIZE, self.head.y), Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        old_dist = np.sqrt((self.head.x - self.food.x) **
                           2 + (self.head.y - self.food.y)**2)
        self._move(action)
        self.snake.insert(0, self.head)
        new_dist = np.sqrt((self.head.x - self.food.x) **
                           2 + (self.head.y - self.food.y)**2)

        reward = 0.15 if new_dist < old_dist else -0.25
        game_over = False

        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 20
            self._place_food()
            self.frame_iteration = 0
        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def _update_ui(self):
        self.display.fill((15, 15, 15))  

        for i, pt in enumerate(self.snake):
            rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)

            if i == 0:  
                pygame.draw.rect(self.display, (0, 200, 255),
                                 rect) 
                eye_size = 4
                if self.direction == Direction.RIGHT:
                    pygame.draw.circle(
                        self.display, (255, 255, 255), (pt.x + 15, pt.y + 5), eye_size)
                    pygame.draw.circle(
                        self.display, (255, 255, 255), (pt.x + 15, pt.y + 15), eye_size)
                elif self.direction == Direction.LEFT:
                    pygame.draw.circle(
                        self.display, (255, 255, 255), (pt.x + 5, pt.y + 5), eye_size)
                    pygame.draw.circle(
                        self.display, (255, 255, 255), (pt.x + 5, pt.y + 15), eye_size)
                elif self.direction == Direction.UP:
                    pygame.draw.circle(
                        self.display, (255, 255, 255), (pt.x + 5, pt.y + 5), eye_size)
                    pygame.draw.circle(
                        self.display, (255, 255, 255), (pt.x + 15, pt.y + 5), eye_size)
                elif self.direction == Direction.DOWN:
                    pygame.draw.circle(
                        self.display, (255, 255, 255), (pt.x + 5, pt.y + 15), eye_size)
                    pygame.draw.circle(
                        self.display, (255, 255, 255), (pt.x + 15, pt.y + 15), eye_size)

            else:  
                color_val = max(50, 255 - (i * 3))
                color = (0, color_val // 2, color_val)
                pygame.draw.rect(self.display, color, rect)
                pygame.draw.rect(self.display, (15, 15, 15), rect, 1)

        pygame.draw.circle(self.display, (255, 0, 0),
                           (self.food.x + 10, self.food.y + 10), 8)
        pygame.draw.circle(self.display, (255, 100, 100),
                           (self.food.x + 10, self.food.y + 10), 4)

        text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(text, [10, 10])  
        pygame.display.flip()

    def _move(self, action):
        cw = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = cw.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = cw[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = cw[(idx + 1) % 4]
        else:
            new_dir = cw[(idx - 1) % 4]
        self.direction = new_dir
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)
