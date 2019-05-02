from configs import LunarLanderConfig as dense_config
from model import ActorLunarlander, DQNAgent
from argparse import ArgumentParser
import gym
import numpy as np
from Lunarlander.train_lunarlander import preprocess_state


def main():
    actor = ActorLunarlander(input_size=dense_config.input_size, output_size=dense_config.output_size,
                             model_path=FLAGS.actor_path)
    actor.load_model()
    score = evaluate(agent=actor, n_epoch=FLAGS.eval_epochs, render=FLAGS.render)
    if score > 200:
        print("Objective score achieved, saving ready model to " + FLAGS.ready_path)
        actor.save_model(path=FLAGS.ready_path)


def evaluate(agent: DQNAgent, n_epoch=10, render=False):
    """
    evaluate the agent
    :param agent: agent to be evaluated
    :param n_epoch: number of epoch to evaluate, the bigger the more accurate the evaluation is
    :param render: if you want to visualize the evaluation
    :return: score of the evaluation
    """
    env = gym.make('LunarLander-v2')
    score = []
    for e in range(n_epoch):
        done = False
        state = env.reset()
        epoch_reward = 0
        step = 1
        while not done and not (step % 1000 == 0):
            step += 1
            if render:
                env.render()
            action_dist = agent.get_q(preprocess_state(state))
            action = agent.select_action(action_dist)
            next_state, reward, done, info = env.step(action)
            epoch_reward += reward
            state = next_state
        print("episode {}/{} , reward: {}".format(e, n_epoch, epoch_reward))
        score.append(epoch_reward)
    score = np.mean(score)
    return score


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--actor_path',
        type=str,
        default=dense_config.actor_path,
        help=' path where to load initial model.')
    parser.add_argument(
        '--ready_path',
        type=str,
        default=dense_config.actor_ready_path,
        help=' path where to output the evaluated model.')
    parser.add_argument(
        '--eval_epochs',
        type=int,
        default=100,
        help='number of epoches to evaluate the models during the process')
    parser.add_argument(
        '--render',
        type=bool,
        default=False,
        help='render')
    FLAGS, unparsed = parser.parse_known_args()
    main()