import gym
from model import CartPoleDQN, CartPoleDQNTarget
from configs import CartpoleConfig as dense_config
from utils.Memory import PrioritizedExperienceReplay
import numpy as np
from utils.logger_utils import get_logger
from PONG.train_gym import train_on_batch


def main():
    agent = CartPoleDQN(input_size=dense_config.input_size,
                        output_size=dense_config.output_size, model_path=dense_config.model_path)
    target_agent = CartPoleDQNTarget(input_size=dense_config.input_size, output_size=dense_config.output_size)
    agent.print_num_of_params()
    target_agent.print_num_of_params()
    logger = get_logger("train_Cartpole_agent")
    fit(logger, agent, target_agent, dense_config.n_epoch)


def fit(logger, agent: CartPoleDQN, target_agent: CartPoleDQNTarget, n_epochs=1500, synch=True):
    logger.info("------Building env------")
    env = gym.make('CartPole-v0')
    last_mean_100_reward = [0] * 100
    exp_replay = PrioritizedExperienceReplay(size=dense_config.memory_size, alpha=dense_config.ALPHA_PER)
    logger.info("------Commence training------")
    degradation = 1 / dense_config.EXPLORE
    agent.set_degradation(degradation)
    total_steps = 0
    i = 0  # for convergence purposes
    for e in range(n_epochs):
        state = env.reset()
        print("agent epsilon : {}".format(agent.epsilon))
        done = False
        epoch_reward = 0
        while not done:  # while not in terminal
            total_steps += 1
            q_values = agent.get_q(state=np.expand_dims(state, axis=0))
            action = agent.select_action(qValues=q_values)
            next_state, reward, done, _ = env.step(action)
            epoch_reward += reward
            exp_replay.add_memory(state=state, action=action, reward=reward, next_state=next_state, is_done=done)
            state = next_state
            agent.lower_epsilon()
            if total_steps < dense_config.OBSERVE:
                continue
            train_on_batch(agent=agent, target_dqn=target_agent, exp_replay=exp_replay, e=e, config=dense_config)
            if synch and total_steps % dense_config.UPDTATE_FREQ == 0:
                agent.save_model()
                target_agent.sync(agent_path=agent.model_path)
                print("Update target DQN")
        last_mean_100_reward[e % 100] = epoch_reward
        if e < 100:
            print("Episode ", e, " / {} finished with reward {}".format(n_epochs, epoch_reward))
        else:
            mean_100_reward = sum(last_mean_100_reward) / 100
            print("Episode ", e,
                  " / {} finished with reward of {} and the last 100 average reward is {} ".format(n_epochs,
                                                                                                   epoch_reward,
                                                                                                   mean_100_reward))
            logger.info(
                "Episode {} / {} finished with reward of {} and the last 100 average reward is {} ".format(e, n_epochs,
                                                                                                           epoch_reward,
                                                                                                           mean_100_reward))
            if mean_100_reward > dense_config.OBJECTIVE_SCORE:
                agent.save_model()
                logger.info("Goal achieved!, at episode {} to {}, with average reward of {}".format(e - 100, e,
                                                                                                    mean_100_reward))
                i += 1
                if i % 50 == 0:
                    break
            else:
                i = 0
    try:
        del env
    except ImportError:
        pass


if __name__ == '__main__':
    main()