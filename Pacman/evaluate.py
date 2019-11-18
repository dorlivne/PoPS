import gym
from model import DQNPacman, DQNAgent,StudentPacman, PacmanTargetNet
from configs import DensePacmanAgentConfig as dense_config
from configs import StudentPacmanConfig as student_config
from configs import PrunePacmanAgentConfig as prune_config
import numpy as np
from argparse import ArgumentParser
from PIL import Image
from copy import deepcopy
from Pacman.processor import process_observation, process_state_batch, init_state, append_frame

OBJECTIVE_SCORE = 100
INPUT_SHAPE = (84, 84)


def main():
    # agent = DQNPacman(input_size=dense_config.input_size, output_size=dense_config.output_size,
    #                     model_path=dense_config.model_path, scope=dense_config.scope)
    # agent.load_model(prune_config.best_path)
    # if score > OBJECTIVE_SCORE:
    #     print("Objective score achieved, saving ready model to " + FLAGS.ready_path)
    #     agent.save_model(path=dense_config.ready_path)
    agent = StudentPacman(input_size=student_config.input_size,
                                       output_size=student_config.output_size,
                                       model_path=r"/media/dorliv/50D03BD3D03BBE52/Study/Masters/PoPS/Pacman/PoPS_Iterative/policy_step_1 (copy)",
                                       tau=student_config.tau,
                                       redundancy=[0.989990234375, 0.989990234375, 0.989990234375, 0.9900001992984694, 0.9900390625],
                                       pruning_freq=prune_config.pruning_freq,
                                       sparsity_end=prune_config.sparsity_end,
                                       target_sparsity=prune_config.target_sparsity,
                                       prune_till_death=True)
    agent.load_model()
    # teacher = DQNPacman(input_size=dense_config.input_size, output_size=dense_config.output_size,
    #                     model_path=dense_config.model_path, scope=dense_config.scope)
    # teacher.load_model(path=dense_config.ready_path)  # load teacher
    # nnz_params_at_each_layer = teacher.get_number_of_nnz_params_per_layer()
    # nnz_params_at_each_layer_student = agent.get_number_of_nnz_params_per_layer()
    # teacher.print_num_of_params()
    a = agent.print_num_of_params()
    score = evaluate(agent, n_epoch=10, render=FLAGS.render, verbose=True)
    print("FINAL_SCORE: average over {} epoch is {}".format(FLAGS.n_epoch, score))



def evaluate(agent: DQNAgent, n_epoch=10, render=False, verbose=False, record=False, video_path=None):
    """
    evaluate the agent
    :param agent: agent to be evaluated
    :param n_epoch: number of epoch to evaluate, the bigger the more accurate the evaluation is
    :param render: if you want to visualize the evaluation
    :return: score of the evaluation
    """
    env = gym.make('MsPacmanDeterministic-v4')
    if record:
        video_save_location = "./vid" if not video_path else video_path
        env = gym.wrappers.Monitor(env, video_save_location, video_callable=lambda episode_id: True, force=True)
    final_score = []
    for e in range(n_epoch):
        state = init_state()
        observation = env.reset()
        observation = process_observation(observation)
        done = False
        epoch_reward = 0.0
        while not done:
            state = append_frame(state, observation)
            if render:
                env.render()
            q_values = agent.get_q(state=np.expand_dims(state, axis=0))
            action = agent.select_action(qValues=q_values, explore=False)
            next_observation, reward, done, _ = env.step(action)
            next_observation = process_observation(next_observation)
            observation = next_observation
            epoch_reward += reward
        if verbose:
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