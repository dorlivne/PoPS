import gym
from model import DQNPacman, PacmanTargetNet, StudentPacman
from configs import DensePacmanAgentConfig as dense_config
from configs import PrunePacmanAgentConfig as prune_config
from configs import StudentPacmanConfig as student_config
from Pacman.evaluate import evaluate
from utils.Memory import ExperienceReplayMultistep, MultiStepPrioritizedExperienceReplay, Supervised_ExperienceReplay,\
    Supervised_Prioritzed_ExperienceReplay
import numpy as np
from Pacman.train import train_on_batch, train_on_batch_with_benchmark
from utils.plot_utils import plot_graph, plot_nnz_vs_accuracy
from utils.logger_utils import get_logger
from multiprocessing import Process, Queue
from Pacman.processor import process_observation, process_state_batch, init_state, append_frame
from Pacman.accumulate_experience_Pacman import accumulate_experience
USE_PER = 0


def prune_DQN():
    logger = get_logger("prune_pacman_agent_using_DQN")
    prune_model = DQNPacman(input_size=prune_config.input_size, output_size=prune_config.output_size,
                          model_path=prune_config.model_path, scope=prune_config.scope,
                          epsilon_stop=prune_config.final_epsilon, epsilon=prune_config.initial_epsilon,
                          pruning_end=prune_config.pruning_end,
                          pruning_freq=prune_config.pruning_freq,
                          sparsity_end=prune_config.sparsity_end,
                          target_sparsity=prune_config.target_sparsity,
                          prune_till_death=True)
    target_model = PacmanTargetNet(input_size=dense_config.input_size, output_size=dense_config.output_size)
    logger.info("loading models")
    target_model.load_model(dense_config.ready_path)
    prune_model.load_model(dense_config.ready_path)
    prune_model.reset_global_step()
    logger.info("Commencing iterative pruning")
    sparsity_vs_accuracy = iterative_pruning(logger, prune_model, target_model, prune_config.n_epoch)
    print("dqn finished")
    plot_graph(sparsity_vs_accuracy, "sparsity_vs_accuracy", figure_num=1)
    prune_model.sess.close()


def iterative_pruning(logger, agent, target_agent, n_epoch, benchmarking=False):
    env = gym.make('MsPacmanDeterministic-v4')
    if USE_PER:
        exp_replay = MultiStepPrioritizedExperienceReplay(size=dense_config.memory_size, gamma=agent.gamma,
                                                         alpha=dense_config.ALPHA_PER)
    else:
        exp_replay = ExperienceReplayMultistep(size=dense_config.memory_size, gamma=agent.gamma)
    total_steps = 0
    sparsity_vs_accuracy = [[], []]
    counter_for_consecutive_failed_trials = 0
    counter_for_last_evaluation = 0
    stop_prune = False
    finished = False
    last_sparsity_measure = -1
    sparsity_converged = 0
    for e in range(n_epoch):
        state = init_state()
        observation = env.reset()
        observation = process_observation(observation)
        done = False
        state = append_frame(state, observation)
        while not done:
            total_steps += 1
            q_values = agent.get_q(state=np.expand_dims(state, axis=0))
            action = agent.select_action(qValues=q_values, explore=False)
            next_observation, reward, done, _ = env.step(action)
            next_observation = process_observation(next_observation)
            next_state = append_frame(state, next_observation)
            exp_replay.add_memory(state, action, reward, next_state, done,
                                  total_steps % dense_config.steps_per_train == 0)  # transaction are inserted after steps per train
            state = next_state
            if total_steps < prune_config.OBSERVE:  # filling up the experience replay
                continue
            if total_steps % dense_config.steps_per_train == 0:
                if not benchmarking:
                    train_on_batch(agent, target_agent, exp_replay, e, config=prune_config)
                else:
                    train_on_batch_with_benchmark(agent, target_agent, exp_replay, e, config=prune_config)
                if not stop_prune:  # this signal is down when the agent needs to recover from pruning
                     agent.prune()


        if e % 10 == 0:
            score = evaluate(agent=agent, n_epoch=student_config.eval_prune)
            sparsity = agent.get_model_sparsity()
            if last_sparsity_measure < sparsity:  # expect the sparsity to get bigger
                NNZ = agent.get_number_of_nnz_params()  # for paper
                sparsity_vs_accuracy[1].append(score)
                sparsity_vs_accuracy[0].append(NNZ)  # 0 is sparsity and 1 is score
                last_sparsity_measure = sparsity
                sparsity_converged = 0
            elif score > sparsity_vs_accuracy[1][-1]:  # better performance for current sparsity
                sparsity_vs_accuracy[1][-1] = score
            if last_sparsity_measure >= sparsity:  # sparsity remained un changed
                sparsity_converged += 1
            print("Episode {} / {} : accuracy is {} with sparsity {} , reward {}"
                  .format(e, n_epoch, sparsity_vs_accuracy[1][-1], sparsity, score))
            logger.info("Episode {} / {} : accuracy is  {} with sparsity {} "
                        .format(e, n_epoch, sparsity_vs_accuracy[1][-1], sparsity))

            if total_steps >= prune_config.OBSERVE:
                if score > dense_config.OBJECTIVE_SCORE:
                    if stop_prune:
                        stop_prune = False
                        freeze_global_step = agent.unfreeze_global_step()
                        logger.info("agent got back on the horse and managed to score 18 plus,"
                                    " continue pruning with global step {}".format(freeze_global_step))
                    logger.info("Saved best model with average score of {} and NNZ params {}".format(sparsity_vs_accuracy[1][-1], sparsity_vs_accuracy[0][-1]))
                    agent.save_model(prune_config.best_path)
                if score < dense_config.LOWER_BOUND or finished:
                    if not stop_prune and not finished:
                        stop_prune = True
                        freeze_global_step = agent.freeze_global_step()  # algorithm works with global step
                        logger.info("stopped pruning due to low results, global step is {}".format(freeze_global_step))
                    counter_for_consecutive_failed_trials += 1
                    logger.info("consecutive trials failed {}".format(counter_for_consecutive_failed_trials))
                    if counter_for_consecutive_failed_trials >= 10 or sparsity_converged >= 10 or finished:
                        if finished:
                            counter_for_last_evaluation += 1
                            stop_prune = False
                            if counter_for_last_evaluation > 10:
                                logger.info("Episode {} / {} : Done".format(e, n_epoch))
                                break
                        else:
                            logger.info("Episode {} / {} : Finished due to low accuracy for 10 consecutive trials".format(e, n_epoch))
                        finished = True
                else:
                    counter_for_consecutive_failed_trials = 0
    try:
        del env
    except ImportError:
        pass
    return sparsity_vs_accuracy

#
# def prune_policy_dist(queue=None):
#     logger = get_logger("prune_pong_agent_using_Policy_dist")
#     prune_model = StudentPong(input_size=prune_config.input_size, output_size=prune_config.output_size,
#                           model_path=student_config.model_path_policy_dist_pruned, scope=prune_config.scope,
#                           epsilon=prune_config.initial_epsilon,
#                           pruning_end=prune_config.pruning_end,
#                           pruning_freq=prune_config.pruning_freq,
#                           sparsity_end=prune_config.sparsity_end,
#                           target_sparsity=prune_config.target_sparsity,
#                           prune_till_death=True)
#     target_model = PongTargetNet(input_size=dense_config.input_size, output_size=dense_config.output_size)
#     prune_model.print_num_of_params()
#     logger.info("loading models")
#     print("loading models")
#     target_model.load_model(dense_config.ready_path)  # load ready model
#     prune_model.load_model(student_config.model_path_policy_dist_ready)  # load output of train via policy_dist
#     prune_model.reset_global_step()
#     logger.info("Commencing iterative pruning")
#     sparsity_vs_accuracy = iterative_pruning_policy_distilliation(agent=prune_model, target_agent=target_model,
#                                                                   iterations=student_config.n_epochs, logger=logger)
#     print("dist finished")
#     plot_graph(sparsity_vs_accuracy, "sparsity_vs_accuracy", figure_num=1, file_name="sparsity_vs_accuracy_with_dist")
#     if queue is not None:
#         queue.put(sparsity_vs_accuracy)
#     prune_model.sess.close()

#
# def main():
#
#     DQN_Queue = Queue()
#     policy_Queue = Queue()
#     p_DQN = Process(target=prune_DQN, args=(DQN_Queue,))
#     p_policy_dist = Process(target=prune_policy_dist, args=(policy_Queue,))
#
#     p_DQN.start()
#     p_policy_dist.start()
#     sparsity_vs_accuracy_dqn = DQN_Queue.get()
#     sparsity_vs_accuracy_policy = policy_Queue.get()
#     plot_nnz_vs_accuracy(data_policy=sparsity_vs_accuracy_policy, data_pruned=sparsity_vs_accuracy_dqn,
#                          legend=('IPP', 'MBG Pruning'), xlabel='NNZ params', ylabel='Accuracy',
#                          title='', filename='results_prune.png')
#     p_DQN.join()
#     p_policy_dist.join()
#     """
#     plot_nnz_vs_accuracy(data_policy=[[1,1600000],[-21,21]], data_pruned=[[1,1600000],[-21,21]],
#                          legend=('IPP', 'MBGP   p'), xlabel='Non-Zero parameters', ylabel='Score',
#                          title='', filename='results_prune.png')
#      """


# def iterative_pruning_policy_distilliation(logger, agent, target_agent, iterations=100, use_per=False,
#                                            config=student_config, best_path=student_config.model_path_policy_dist_best,
#                                            arch_type=0, lower_bound=0.0, accumulate_experience_fn=accumulate_experience,
#                                            evaluate_fn=evaluate, objective_score=18.0):
#     initial_score = evaluate_fn(agent=agent)
#     sparsity_vs_accuracy = [[], []]
#     sparsity_vs_accuracy[1].append(initial_score)
#     sparsity_vs_accuracy[0].append(agent.get_number_of_nnz_params()) # change back to sparsiy
#     if use_per:
#         exp_replay = Supervised_Prioritzed_ExperienceReplay(size=config.memory_size,
#                                                             alpha=config.ALPHA_PER)
#     else:
#         exp_replay = Supervised_ExperienceReplay(size=config.memory_size)
#     stop_prune_arg = False
#     m = 0
#     cnt = 0
#     sparsity_measure = 10e6  # put 0 when done
#     for i in range(iterations):
#         logger.info("-- ITERATION number " + str(i) + "/" + str(iterations) + ": accumulating experience from teacher --")
#         print("-- ITERATION number " + str(i) + "/" + str(iterations) + ": accumulating experience from teacher --")
#         accumulate_experience_fn(teacher=target_agent, exp_replay=exp_replay, config=config)
#         logger.info("-- ITERATION number " + str(i) + "/" + str(iterations) +
#                     ": finished accumulating experience from teacher starting to prune and fine-tune the student --")
#         print("-- ITERATION number " + str(i) + "/" + str(iterations) +
#               ": finished accumulating experience from teacher starting to prune and fine-tune the student -- ")
#         score_list, sparsity_list, stop_prune_arg = train_student(logger=logger, student=agent,
#                                                                   exp_replay=exp_replay,
#                                                                   prune=True,
#                                                                   lr=config.learning_rate_schedule_prune(i, arch_type),
#                                                                   stop_prune_arg=stop_prune_arg,
#                                                                   epoch=i, use_per=use_per, best_path=best_path,
#                                                                   config=config, evaluate_fn=evaluate_fn, objective_score=objective_score,
#                                                                   lower_bound=lower_bound)
#
#         for j, score in enumerate(score_list):
#             if sparsity_list[j] < sparsity_vs_accuracy[0][-1]:  # revered for paper if get smaller
#                 sparsity_vs_accuracy[1].append(score)
#                 sparsity_vs_accuracy[0].append(sparsity_list[j])  # nnz params get smaller
#             if sparsity_list[j] == sparsity_vs_accuracy[0][-1] and score > sparsity_vs_accuracy[1][-1]:
#                 sparsity_vs_accuracy[1][-1] = score
#
#         mean_score = np.mean(score_list)
#         logger.info("-- iteration number " + str(i) + ": student evaluation after pruning procedeure is: "
#                     + str(mean_score) + " --")
#         print("-- iteration number " + str(i) + ": student evaluation after pruning procedeure is: "
#               + str(mean_score) + " --")
#
#         if mean_score < lower_bound:
#             m += 1
#             logger.info("-- iteration {} / {} : {} consecutive trials with low score --"
#                         .format(i, iterations, m))
#             if m % 5 == 0:
#                 logger.info("-- iteration {} / {} : Finished due to low accuracy for 5 consecutive trials --"
#                             .format(i, iterations))
#                 break
#         else:
#             m = 0
#
#         if abs(sparsity_measure - sparsity_vs_accuracy[0][-1]) < 1e-3:
#                 cnt += 1
#                 logger.info("-- iteration {} / {} : {} consecutive trials with same sparsity --"
#                             .format(i, iterations, cnt))
#                 if cnt == 5:
#                     logger.info("sparsity converged, ending pruning procedure")
#                     break
#         else:
#                 cnt = 0
#         if sparsity_vs_accuracy[0][-1] < sparsity_measure:  # reversed for paper, symbolize NNZ params instead of sparsity
#             sparsity_measure = sparsity_vs_accuracy[0][-1]
#     return sparsity_vs_accuracy


def prune_benchmark():
    logger = get_logger("prune_pong_agent_benchmark")
    prune_model = DQNPacman(input_size=prune_config.input_size, output_size=prune_config.output_size,
                          model_path=prune_config.model_path, scope=prune_config.scope,
                          epsilon_stop=prune_config.final_epsilon, epsilon=prune_config.initial_epsilon,
                          pruning_end=prune_config.pruning_end,
                          pruning_freq=prune_config.pruning_freq,
                          sparsity_end=prune_config.sparsity_end,
                          target_sparsity=prune_config.target_sparsity,
                          prune_till_death=True)
    target_model = PacmanTargetNet(input_size=dense_config.input_size, output_size=dense_config.output_size)
    logger.info("loading models")
    print("loading models")
    target_model.load_model(dense_config.ready_path)
    prune_model.load_model(dense_config.ready_path)
    prune_model.change_loss_to_benchmark_loss()
    prune_model.reset_global_step()
    logger.info("Commencing iterative pruning")
    sparsity_vs_accuracy = iterative_pruning(logger, prune_model, target_model, prune_config.n_epoch, benchmarking=True)
    print("benchmark finished")
    plot_graph(sparsity_vs_accuracy, "sparsity_vs_accuracy_benchmark", figure_num=1)


if __name__ == '__main__':
    prune_benchmark()
