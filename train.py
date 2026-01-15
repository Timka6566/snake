from agent import Agent
from game import SnakeGameAI


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    print("Начинаем обучение...")

    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(
            state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1

            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Игра: {agent.n_games}, Счет: {score}, Рекорд: {record}')


if __name__ == '__main__':
    train()
