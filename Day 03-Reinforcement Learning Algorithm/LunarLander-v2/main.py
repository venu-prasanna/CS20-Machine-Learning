import gym
import numpy as np
from model import DQN
import matplotlib.pyplot as plt
import tensorflow


def train_dqn(episode):
    loss = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0])
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0
        max_steps = 3000
        for i in range(max_steps):
            action = agent.act(state)
            env.render()
            next_state, reward, done, _ = env.step(int(action))
            score += reward
            next_state = np.reshape(next_state, (1, 8))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)

        # Average score of last 100 episode
        is_solved = np.mean(loss[-100:])
        if is_solved > 200:
            name = "save-" + str(e) + "ep"
            agent.save(name)
            print('\n Task Completed! \n')
            break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))
    return loss


env = gym.make('LunarLander-v2')
env.seed(0)
np.random.seed(0)

if __name__ == '__main__':
    with tensorflow.device('/GPU:0'):
        print(env.observation_space)
        print(env.action_space)
        episodes = 400
        loss = train_dqn(episodes)
        plt.plot([i + 1 for i in range(0, len(loss), 2)], loss[::2])
        plt.show()
