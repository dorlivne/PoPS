import gym
from model import DQNPong, StudentPong, DQNAgent
from utils.wrappers import wrap_deepmind
from configs import DensePongAgentConfig as dense_config
from configs import PrunePongAgentConfig as prune_config
from configs import StudentPongConfig as student_config
import numpy as np
from argparse import ArgumentParser

OBJECTIVE_SCORE = 18.0


def main():
    agent = DQNPong(input_size=dense_config.input_size, output_size=dense_config.output_size,
                        model_path=dense_config.model_path, scope=dense_config.scope)
    agent.load_model()
    score = evaluate(agent, n_epoch=10, render=FLAGS.render)
    print("FINAL_SCORE: average over {} epoch is {}".format(FLAGS.n_epoch, score))
    agent.print_num_of_params()
    if score > OBJECTIVE_SCORE:
        print("Objective score achieved, saving ready model to " + FLAGS.ready_path)
        agent.save_model(path=dense_config.ready_path)


def evaluate(agent: DQNAgent, n_epoch=10, render=False):
    """
    evaluate the agent
    :param agent: agent to be evaluated
    :param n_epoch: number of epoch to evaluate, the bigger the more accurate the evaluation is
    :param render: if you want to visualize the evaluation
    :return: score of the evaluation
    """
    env = gym.make("PongNoFrameskip-v4")
    env = wrap_deepmind(env, frame_stack=True)
    final_score = []
    for e in range(n_epoch):
        state = env.reset()
        state = np.asarray(state)
        done = False
        epoch_reward = 0.0
        while not done:
            if render:
                env.render()
            q_values = agent.get_q(state=np.reshape(state, (1, state.shape[0], state.shape[1], state.shape[2])))
            action = agent.select_action(qValues=q_values, explore=False)
            next_state, reward, done, _ = env.step(action + 1)
            # 1 for up 2 for stay 3 for down, action is from 0 to 2 so we need an offset
            next_state = np.asarray(next_state)
            state = next_state
            epoch_reward += reward
        print("Episode ", e, " / {} finished with reward {}".format(n_epoch, epoch_reward))
        final_score.append(epoch_reward)
    final_score = np.mean(final_score)
    try:
        del env
    except ImportError:
        pass
    return final_score


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
        default=10,
        help='number of epoches')
    parser.add_argument(
        '--render',
        type=int,
        default=True,
        help='visualize the evaluation')
    parser.add_argument(
        '--ready_path',
        type=str,
        default=dense_config.ready_path,
        help='path to save the ready_to_prune model')
    FLAGS, unparsed = parser.parse_known_args()
    main()
