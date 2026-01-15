import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(12, 256, 3)
        self.model.load()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        p_l = Point(head.x - BLOCK_SIZE, head.y)
        p_r = Point(head.x + BLOCK_SIZE, head.y)
        p_u = Point(head.x, head.y - BLOCK_SIZE)
        p_d = Point(head.x, head.y + BLOCK_SIZE)

        p_l2 = Point(head.x - 2*BLOCK_SIZE, head.y)
        p_r2 = Point(head.x + 2*BLOCK_SIZE, head.y)
        p_u2 = Point(head.x, head.y - 2*BLOCK_SIZE)
        p_d2 = Point(head.x, head.y + 2*BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(p_r)) or (dir_l and game.is_collision(p_l)) or (
                dir_u and game.is_collision(p_u)) or (dir_d and game.is_collision(p_d)),
            (dir_u and game.is_collision(p_r)) or (dir_d and game.is_collision(p_l)) or (
                dir_l and game.is_collision(p_u)) or (dir_r and game.is_collision(p_d)),
            (dir_d and game.is_collision(p_r)) or (dir_u and game.is_collision(p_l)) or (
                dir_r and game.is_collision(p_u)) or (dir_l and game.is_collision(p_d)),
            (dir_r and game.is_collision(p_r2)) or (dir_l and game.is_collision(p_l2)) or (
                dir_u and game.is_collision(p_u2)) or (dir_d and game.is_collision(p_d2)),
            dir_l, dir_r, dir_u, dir_d,
            game.food.x < head.x, game.food.x > head.x,
            game.food.y < head.y, game.food.y > head.y
        ]
        return np.array(state, dtype=int)

    def count_reachable_space(self, start_pt, game):
        """Алгоритм Flood Fill для подсчета доступного места"""
        if game.is_collision(start_pt):
            return 0

        visited = set()
        queue = deque([start_pt])
        visited.add(start_pt)
        count = 0

        max_scan = len(game.snake) * 2 + 10

        while queue and count < max_scan:
            curr = queue.popleft()
            count += 1

            for dx, dy in [(BLOCK_SIZE, 0), (-BLOCK_SIZE, 0), (0, BLOCK_SIZE), (0, -BLOCK_SIZE)]:
                next_pt = Point(curr.x + dx, curr.y + dy)
                if (next_pt not in visited and
                    0 <= next_pt.x < game.w and 0 <= next_pt.y < game.h and
                        next_pt not in game.snake):
                    visited.add(next_pt)
                    queue.append(next_pt)
        return count

    def get_action(self, state, game):
        self.epsilon = max(0, 80 - self.n_games)

        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)

        actions_by_priority = torch.argsort(
            prediction, descending=True).tolist()

        if random.randint(0, 200) < self.epsilon:
            random_idx = random.randint(0, 2)
            actions_by_priority = [random_idx] + \
                [i for i in range(3) if i != random_idx]

        best_move = [1, 0, 0]  
        max_space = -1

        possible_moves_info = []
        for move_idx in actions_by_priority:
            move = [0, 0, 0]
            move[move_idx] = 1
            next_pt = self._get_next_head(move, game)

            if not game.is_collision(next_pt):
                space = self.count_reachable_space(next_pt, game)
                possible_moves_info.append((move, space, move_idx))


        snake_len = len(game.snake)

        safe_moves = [m for m in possible_moves_info if m[1] >= snake_len]

        if safe_moves:
            for pref_idx in actions_by_priority:
                for sm in safe_moves:
                    if sm[2] == pref_idx:
                        return sm[0]
        elif possible_moves_info:
            possible_moves_info.sort(key=lambda x: x[1], reverse=True)
            return possible_moves_info[0][0]

        return [1, 0, 0]

    def _get_next_head(self, action, game):
        cw = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = cw.index(game.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = cw[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = cw[(idx + 1) % 4]
        else:
            new_dir = cw[(idx - 1) % 4]

        x, y = game.head.x, game.head.y
        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE
        return Point(x, y)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        sample = random.sample(self.memory, BATCH_SIZE) if len(
            self.memory) > BATCH_SIZE else self.memory
        s, a, r, ns, d = zip(*sample)
        self.trainer.train_step(s, a, r, ns, d)

    def train_short_memory(self, s, a, r, ns, d):
        self.trainer.train_step(s, a, r, ns, d)


def train():
    agent, game, record = Agent(), SnakeGameAI(), 0
    while True:
        s_old = agent.get_state(game)
        move = agent.get_action(s_old, game)
        reward, done, score = game.play_step(move)
        s_new = agent.get_state(game)
        agent.train_short_memory(s_old, move, reward, s_new, done)
        agent.remember(s_old, move, reward, s_new, done)
        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
            print(f'Game {agent.n_games} | Score: {score} | Best: {record}')


if __name__ == '__main__':
    train()
