import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10000
SHOW_EVERY = 200
STATS_EVERY = 50

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
DISCRETE_OS_WIN_SIZE = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

epsilon = 1
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES//2
EPSILON_DECAY_VALUE = epsilon/(END_EPSILON_DECAY - START_EPSILON_DECAY)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

def get_discrete_state(state):
    dis_state = (state - env.observation_space.low) / DISCRETE_OS_WIN_SIZE
    return tuple(dis_state.astype(np.int))

for episode in range(EPISODES):
    episode_reward = 0

    discrete_state = get_discrete_state(env.reset())
    done = False

    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        if episode % SHOW_EVERY == 0:
            env.render()
        
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            curren_q = q_table[discrete_state + (action,)]
            new_q = (1- LEARNING_RATE) * curren_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            print(f"We made it by episode {episode}")
            q_table[discrete_state + (action,)] = 0
        
        discrete_state = new_discrete_state
    if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
        epsilon -= EPSILON_DECAY_VALUE

    ep_rewards.append(episode_reward)

    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards[-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()