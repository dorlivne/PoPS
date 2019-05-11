from configs import LunarLanderConfig as dense_config
from model import ActorLunarlander, CriticLunarLander, Actor, DQNAgent, CriticLunarLanderTarget
from utils.logger_utils import get_logger
from argparse import ArgumentParser
from utils.Memory import ExperienceReplay
import gym
import numpy as np
FLAGS = 0




def main():
    logger = get_logger('train_lunarlander')
    actor = ActorLunarlander(input_size=dense_config.input_size, output_size=dense_config.output_size,
                             model_path=FLAGS.actor_path)
    critic = CriticLunarLander(input_size=dense_config.input_size, output_size=dense_config.critic_output,
                               model_path=FLAGS.critic_path)

    train(logger, actor, critic, epochs=FLAGS.n_epoch)


def learn_on_mini_batch(e, actor: Actor, critic: DQNAgent, critic_target: DQNAgent, exp_replay: ExperienceReplay, config=dense_config):
    batch_size = FLAGS.batch_size
    mini_batch = exp_replay.getMiniBatch(batch_size)
    state_batch, action_batch, reward_batch, dones_batch, next_state_batch = [], [], [], [], []
    for exp in mini_batch:
        state_batch.append(exp.state)
        action_batch.append(exp.action)
        reward_batch.append(exp.reward)
        dones_batch.append(exp.done)
        if dones_batch[-1]:
            next_state_batch.append(exp.state)  # this is just to prevent nope, the terminal states are masked anyway
        else:
            next_state_batch.append(exp.next_state)
    Actor_Y_Batch = np.zeros((mini_batch.__len__(), actor.output_size[-1]))
    Critic_Y_Batch = np.zeros((mini_batch.__len__(), 1))
    critic_batch_output_for_state = critic_target.get_q(state_batch)
    critic_batch_output_for_next_state = critic_target.get_q(next_state_batch)
    for i, reward in enumerate(reward_batch):  # iteration over batch_size
        target = reward + (critic.gamma * np.max(critic_batch_output_for_next_state[i])) * (1 - dones_batch[i])  # create target_value --> scalar
        Critic_Y_Batch[i] = target  # target_batch (target.get_q(state_batch))[i][action[i]) = target
        Actor_Y_Batch[i][action_batch[i]] = target - critic_batch_output_for_state[i]
        # Q(s,a) - V(s) = Advantage for stability
    critic.learn(target_batch=Critic_Y_Batch, learning_rate=config.learning_rate_schedule_critic(e),
                 input=state_batch)
    actor.learn(target_batch=Actor_Y_Batch, learning_rate=config.learning_rate_schedule_actor(e), input=state_batch)


def preprocess_state(state):
    return np.reshape(state, newshape=(1, np.shape(state)[0]))


def train(logger, actor: ActorLunarlander, critic: CriticLunarLander, epochs=int(1e4)):
    env = gym.make('LunarLander-v2')
    logger.info("Start :  training agent ")
    print("Start :  training agent ")
    exp_replay = ExperienceReplay(size=dense_config.memory_size)
    last100Scores = [0] * 100
    step = 0
    i = 0
    critic_target = CriticLunarLanderTarget(input_size=dense_config.input_size, output_size=dense_config.critic_output)
    for e in range(epochs):
        done = False
        state = env.reset()
        epoch_reward = 0
        step += 1
        while not done and not (step % 1000 == 0):
            step += 1
            action_dist = actor.get_q(preprocess_state(state))
            action = actor.select_action(action_dist)
            next_state, reward, done, info = env.step(action)
            exp_replay.add_memory(state=state, action=action, reward=reward, next_state=next_state, is_done=done)
            epoch_reward += reward
            if step > FLAGS.OBSERVE:
                learn_on_mini_batch(e=e, actor=actor, critic=critic, critic_target=critic_target, exp_replay=exp_replay)
                if step % dense_config.UPDATE_FREQ == 0:
                    critic.save_model()
                    print("Critic target net updated")
                    critic_target.sync(agent_path=critic.model_path)
            state = next_state
        last100Scores[e % 100] = epoch_reward

        if e > 100 and step > FLAGS.OBSERVE:
            mean_score = np.mean(last100Scores)
            print("episode {}/{} , reward: {}, average reward last 100 is {}".format(e, epochs, epoch_reward,
                                                                                     mean_score))
            logger.info("episode {}/{} , reward: {} average reward last 100 is {}".format(e, epochs, epoch_reward,
                                                                                          mean_score))
            if mean_score >= dense_config.OBJECTIVE_SCORE:
                i += 1
                if i % 10 == 0:
                    logger.info("Finish: agent achieved the objective for 10 consecutive episodes ")
                    actor.save_model()
                    critic.save_model()
                    break
            else:
                i = 0
    actor.save_model()
    critic.save_model()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--actor_path',
        type=str,
        default=dense_config.actor_path,
        help=' path where to load initial model.')
    parser.add_argument(
        '--critic_path',
        type=str,
        default=dense_config.critic_path,
        help=' path where to load initial model.')
    parser.add_argument(
        '--n_epoch',
        type=int,
        default=dense_config.n_epoch,
        help='number of epoches to do policy distillation')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=dense_config.batch_size,
        help='number of epoches')
    parser.add_argument(
        '--eval_epochs',
        type=int,
        default=5,
        help='number of epoches to evaluate the models during the process')
    parser.add_argument(
        '--OBSERVE',
        type=int,
        default=dense_config.OBSERVE,
        help='number of steps to observe the environment before training')
    FLAGS, unparsed = parser.parse_known_args()
    main()