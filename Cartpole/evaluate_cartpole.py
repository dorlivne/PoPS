import gym
from model import CartPoleDQN
from configs import CartpoleConfig as dense_config
import numpy as np
from utils.plot_utils import plot_weights

def main():
    agent = CartPoleDQN(input_size=dense_config.input_size,
                        output_size=dense_config.output_size, model_path=dense_config.model_path_overtrained) # TODO delete overtrained option in evaluate, prune and in config
    agent.print_num_of_params()
    agent.test_mode()
    score = evaluate_cartepole(agent=agent, n_epoch=5)
    if score > dense_config.OBJECTIVE_SCORE:
        print("Objective score achieved, saving ready model to " + dense_config.ready_path)
        agent.save_model(path=dense_config.ready_path_overtrained)
        plot_weights(agent, "extra_train", 1, 1)


def evaluate_cartepole(agent: CartPoleDQN, n_epoch=100, render=False):
    env = gym.make('CartPole-v0')
    mean_reward = np.zeros(n_epoch)
    agent_epsilon = agent.epsilon
    assert agent_epsilon == 0
    for e in range(n_epoch):
        state = env.reset()
        done = False
        epoch_reward = 0
        while not done:  # while not in terminal
            if render:
                env.render()
            q_values = agent.get_q(state=np.expand_dims(state, axis=0))
            action = agent.select_action(qValues=q_values)
            next_state, reward, done, _ = env.step(action)
            epoch_reward += reward
            state = next_state
        mean_reward[e] = epoch_reward
    mean_reward = np.mean(mean_reward)
    print("---evaluation of agent during {} episodes is {}".format(n_epoch, mean_reward))
    return mean_reward


if __name__ == '__main__':
    main()
