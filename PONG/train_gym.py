import gym
from model import DQNPong, PongTargetNet
from utils.wrappers import wrap_deepmind
from configs import DensePongAgentConfig as dense_config
from utils.Memory import ExperienceReplayMultistep, MultiStepPrioritizedExperienceReplay
import numpy as np
from utils.logger_utils import get_logger
from argparse import ArgumentParser

USE_PER = 1
FLAGS = 0


def train():
    logger = get_logger("train_pong_student")
    agent = DQNPong(input_size=dense_config.input_size, output_size=dense_config.output_size,
                    model_path=FLAGS.model_path, scope=dense_config.scope,
                    epsilon_stop=dense_config.final_epsilon, epsilon=dense_config.initial_epsilon)
    target_agent = PongTargetNet(input_size=dense_config.input_size, output_size=dense_config.output_size)
    agent.print_num_of_params()
    target_agent.print_num_of_params()
    fit(logger, agent, target_agent, FLAGS.n_epoch)


def main():
    train()


def train_on_batch(agent, target_dqn, exp_replay, e, config=dense_config):
    batch_size = config.batch_size if not FLAGS else FLAGS.batch_size
    batch_size = batch_size * config.steps_per_train
    # Multi step variation,
    # the batch size should match the delta of steps between training
    if not USE_PER:
        mini_batch = exp_replay.getMiniBatch(batch_size=batch_size)
        weights, indexes = np.ones(np.shape(mini_batch)[0], config.output_size), None
    else:
        n_epoch = config.n_epoch if not FLAGS else FLAGS.n_epoch
        mini_batch, weights, indexes = exp_replay.getMiniBatch(batch_size=batch_size,
                                             beta=config.beta_schedule(beta0=config.BETA0_PER,
                                                                             e=e, n_epoch=n_epoch))
        weights = np.expand_dims(weights, axis=-1)
        weights = weights * np.ones(1, config.output_size)
    state_batch, action_batch, reward_batch, dones_batch, next_state_batch = [], [], [], [], []
    for exp in mini_batch:
        state_batch.append(exp.state)
        action_batch.append(exp.action)
        reward_batch.append(exp.reward)
        dones_batch.append(exp.done)
        if dones_batch[-1]:
            next_state_batch.append(exp.state)   # this is just to prevent nope, the terminal states are masked anyway
        else:
            next_state_batch.append(exp.next_state)
    target_batch = target_dqn.get_q(state_batch)  # target.get_q(state_batch) --> (batch_size,action_dim)
    next_qvalues_batch = target_dqn.get_q(state=next_state_batch)  # target.get_q(next_state) --> (batch_size,action_dim)
    for i, reward in enumerate(reward_batch):   # iteration over batch_size
            target = reward + (agent.gamma * np.max(next_qvalues_batch[i])) * (1 - dones_batch[i]) # create target_value --> scalar
            target_batch[i][action_batch[i]] = target  # target_batch (target.get_q(state_batch))[i][action[i]) = target
            # calculating the TD_error for each experience
    _, td_errors = agent.learn(target_batch=target_batch, learning_rate=config.learning_rate_schedule(e),
                            input=state_batch, weights=weights)
    if USE_PER:
        new_priority = np.abs(td_errors) + config.EPS_PER  # we add epsilon so that every transaction has a chance
        new_priority = [priority[action_batch[i]] for i, priority in enumerate(new_priority)]
        exp_replay.update_priorities(indexes=indexes, priorities=new_priority)


def fit(logger, agent, target_agent, n_epoch, update=True):
    logger.info("Start :  training agent ")
    env = gym.make("PongNoFrameskip-v4")
    env = wrap_deepmind(env, frame_stack=True)
    if USE_PER:
        exp_replay = MultiStepPrioritizedExperienceReplay(size=dense_config.memory_size, gamma=agent.gamma,
                                                          alpha=dense_config.ALPHA_PER)
    else:
        exp_replay = ExperienceReplayMultistep(size=dense_config.memory_size, gamma=agent.gamma)
    degradation = dense_config.steps_per_train / dense_config.EXPLORE
    agent.set_degradation(degradation)
    last_100_epochs_reward = np.zeros(100)
    total_steps = 0
    best_reward = -21.0
    i = 0
    for e in range(n_epoch):
        state = env.reset()
        state = np.asarray(state)
        print("agent epsilon : {}".format(agent.epsilon))
        done = False
        epoch_reward = 0.0
        while not done:
            total_steps += 1
            q_values = agent.get_q(state=np.reshape(state, (1, state.shape[0], state.shape[1], state.shape[2])))
            action = agent.select_action(qValues=q_values)
            next_state, reward, done, _ = env.step(action + 1) # 1 for up 2 for stay 3 for down, action is from 0 to 2 so we need an offset
            next_state = np.asarray(next_state)
            exp_replay.add_memory(state, action, reward, next_state, done,
                                  total_steps % dense_config.steps_per_train == 0)  # transaction are inserted after steps per train
            state = next_state
            epoch_reward += reward
            if total_steps % dense_config.steps_per_train == 0:
                agent.lower_epsilon()
            if total_steps < dense_config.OBSERVE:
                continue
            if total_steps % dense_config.steps_per_train == 0:
                train_on_batch(agent, target_agent, exp_replay, e)
            if update and total_steps % dense_config.UPDTATE_FREQ == 0:
                agent.save_model()
                target_agent.sync(agent_path=agent.model_path)
                print("Update target DQN")
        last_100_epochs_reward[e % 100] = epoch_reward
        if e < 100:
            if best_reward < epoch_reward:
                logger.info("Best Reward : episode {} / {}, reward {}".format(e, n_epoch, epoch_reward))
                best_reward = epoch_reward
            print("Episode ", e, " / {} finished with reward {}".format(n_epoch, epoch_reward))
        else:
            mean_100_reward = sum(last_100_epochs_reward) / 100
            if best_reward < mean_100_reward:
                print("Best Reward : episode {} to {}, with average reward of {}".format(e - 100, e, mean_100_reward))
                best_reward = mean_100_reward
            print("Episode ", e, " / {} finished with reward of {} and the last 100 average reward is {} ".format(n_epoch, epoch_reward, mean_100_reward))
            logger.info("Episode {} / {} finished with reward of {} and the last 100 average reward is {} ".format(e,n_epoch, epoch_reward, mean_100_reward))

            if mean_100_reward > 20.0:
                agent.save_model()
                logger.info("Goal achieved!, at episode {} to {}, with average reward of {}".format(e - 100, e, mean_100_reward))
                i += 1
                if i % 5 == 0:
                    break
                else:
                    i = 0
    try:
        del env
    except ImportError:
        pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=dense_config.model_path,
        help=' Directory where to save model checkpoint.')
    parser.add_argument(
        '--n_epoch',
        type=int,
        default=dense_config.n_epoch,
        help='number of epoches')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=dense_config.n_epoch,
        help='number of epoches')
    FLAGS, unparsed = parser.parse_known_args()
    main()
