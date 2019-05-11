from configs import StudentPongConfig as student_config
from PONG.evaluate import evaluate
from utils.Memory import Supervised_ExperienceReplay, Supervised_Prioritzed_ExperienceReplay
import numpy as np
from policy_distilliation_train import accumulate_experience, train_student

"""
import gym
from model import DQNPong, PongTargetNet, StudentPong
from utils.wrappers import wrap_deepmind
from configs import DensePongAgentConfig as dense_config
from configs import PrunePongAgentConfig as prune_config
from PONG.train_gym import train_on_batch, USE_PER
from utils.plot_utils import plot_graph, plot_nnz_vs_accuracy
from utils.logger_utils import get_logger
from multiprocessing import Process, Queue
from utils.Memory import ExperienceReplayMultistep, MultiStepPrioritizedExperienceReplay


def prune_DQN(queue=None):
    logger = get_logger("prune_pong_agent_using_DQN")
    prune_model = DQNPong(input_size=prune_config.input_size, output_size=prune_config.output_size,
                          model_path=prune_config.model_path, scope=prune_config.scope,
                          epsilon_stop=prune_config.final_epsilon, epsilon=prune_config.initial_epsilon,
                          pruning_end=prune_config.pruning_end,
                          pruning_freq=prune_config.pruning_freq,
                          sparsity_end=prune_config.sparsity_end,
                          target_sparsity=prune_config.target_sparsity,
                          prune_till_death=True)
    target_model = PongTargetNet(input_size=dense_config.input_size, output_size=dense_config.output_size)
    logger.info("loading models")
    target_model.load_model(dense_config.ready_path)
    prune_model.load_model(dense_config.ready_path)
    prune_model.reset_global_step()
    logger.info("Commencing iterative pruning")
    sparsity_vs_accuracy = iterative_pruning(logger, prune_model, target_model, prune_config.n_epoch)
    print("dqn finished")
    plot_graph(sparsity_vs_accuracy, "NNZ_vs_accuracy", figure_num=1)
    if queue is not None:
        queue.put(sparsity_vs_accuracy)
    prune_model.sess.close()


def iterative_pruning(logger, agent, target_agent, n_epoch):
    env = gym.make("PongNoFrameskip-v4")
    env = wrap_deepmind(env, frame_stack=True)
    if USE_PER:
        exp_replay = MultiStepPrioritizedExperienceReplay(size=dense_config.memory_size, gamma=agent.gamma,
                                                          alpha=dense_config.ALPHA_PER)
    else:
        exp_replay = ExperienceReplayMultistep(size=dense_config.memory_size, gamma=agent.gamma)
    total_steps = 0
    sparsity_vs_accuracy = [[], []]
    i = 0
    m = 0
    stop_prune = False
    freeze_global_step = 0
    last_sparsity_measure = -1
    for e in range(n_epoch):
        state = env.reset()
        state = np.asarray(state)
        done = False
        while not done:
            total_steps += 1
            q_values = agent.get_q(state=np.reshape(state, (1, state.shape[0], state.shape[1], state.shape[2])))
            action = agent.select_action(qValues=q_values)
            next_state, reward, done, _ = env.step(action + 1) # 1 for up 2 for stay 3 for down, action is from 0 to 2 so we need an offset
            next_state = np.asarray(next_state)
            exp_replay.add_memory(state, action, reward, next_state, done,
                                  total_steps % dense_config.steps_per_train == 0)  # transaction are inserted after steps per train
            state = next_state
            if total_steps < prune_config.OBSERVE:  # filling up the experience replay
                continue
            if total_steps % dense_config.steps_per_train == 0:  # fine tuning and prune block
                train_on_batch(agent, target_agent, exp_replay, e, config=prune_config)
                if not stop_prune:  # this signal is down when the agent needs to recover from pruning
                     agent.prune()

        if e % 10 == 0:
            score = evaluate(agent=agent, n_epoch=5)
            sparsity = agent.get_model_sparsity()
            if last_sparsity_measure < sparsity:  # got pruned
                NNZ = agent.get_number_of_nnz_params()
                sparsity_vs_accuracy[1].append(score)
                sparsity_vs_accuracy[0].append(NNZ)
                last_sparsity_measure = sparsity
                m = 0
            elif score > sparsity_vs_accuracy[1][-1]:  # better performance for current sparsity
                sparsity_vs_accuracy[1][-1] = score
                m = 0
            else:
                m += 1
            print("Episode {} / {} : accuracy is {} with sparsity {} , reward {}"
                  .format(e, n_epoch, sparsity_vs_accuracy[1][-1], sparsity_vs_accuracy[0][-1], score))
            logger.info("Episode {} / {} : accuracy is  {} with sparsity {}"
                        .format(e, n_epoch, sparsity_vs_accuracy[1][-1], sparsity_vs_accuracy[0][-1]))
            if total_steps >= prune_config.OBSERVE:
                if score > dense_config.OBJECTIVE_SCORE:
                    if stop_prune:
                        stop_prune = False
                        freeze_global_step = agent.set_global_step(freeze_global_step)
                        logger.info("agent got back on the horse and managed to score 18 plus,"
                                    " continue pruning with global step {}".format(freeze_global_step))
                    logger.info("Saved best model with average score of {} and sparsity {}".format(sparsity_vs_accuracy[1][-1], sparsity_vs_accuracy[0][-1]))
                    agent.save_model(prune_config.best_path)
                if score < dense_config.LOWER_BOUND:
                    if not stop_prune:
                        stop_prune = True
                        freeze_global_step = agent.print_global_step()  # algorithm works with global step
                        logger.info("stopped pruning due to low results, global step is {}".format(freeze_global_step))
                    i += 1
                    if i % 10 == 0 or m % 10 == 0:
                        logger.info("Episode {} / {} : Finished due to low accuracy for 10 consecutive trials")
                        break
                else:
                    i = 0

    try:
        del env
    except ImportError:
        pass
    return sparsity_vs_accuracy


def prune_policy_dist(queue=None):
    logger = get_logger("prune_pong_agent_using_Policy_dist")
    prune_model = StudentPong(input_size=prune_config.input_size, output_size=prune_config.output_size,
                          model_path=student_config.model_path_policy_dist_pruned, scope=prune_config.scope,
                          epsilon=prune_config.initial_epsilon,
                          pruning_end=prune_config.pruning_end,
                          pruning_freq=prune_config.pruning_freq,
                          sparsity_end=prune_config.sparsity_end,
                          target_sparsity=prune_config.target_sparsity,
                          prune_till_death=True)
    target_model = PongTargetNet(input_size=dense_config.input_size, output_size=dense_config.output_size)
    prune_model.print_num_of_params()
    logger.info("loading models")
    print("loading models")
    target_model.load_model(dense_config.ready_path)  # load ready model
    prune_model.load_model(student_config.model_path_policy_dist_ready)  # load output of train via policy_dist
    prune_model.reset_global_step()
    logger.info("Commencing iterative pruning")
    sparsity_vs_accuracy = iterative_pruning_policy_distilliation(agent=prune_model, target_agent=target_model,
                                                                  iterations=student_config.n_epochs, logger=logger)
    print("dist finished")
    plot_graph(sparsity_vs_accuracy, "NNZ_vs_accuracy", figure_num=1, file_name="NNZ_vs_accuracy_with_dist")
    if queue is not None:
        queue.put(sparsity_vs_accuracy)
    prune_model.sess.close()


def main():
    DQN_Queue = Queue()
    policy_Queue = Queue()
    p_DQN = Process(target=prune_DQN, args=(DQN_Queue,))
    p_policy_dist = Process(target=prune_policy_dist, args=(policy_Queue,))

    p_DQN.start()
    p_policy_dist.start()
    sparsity_vs_accuracy_dqn = DQN_Queue.get()
    sparsity_vs_accuracy_policy = policy_Queue.get()
    plot_nnz_vs_accuracy(data_policy=sparsity_vs_accuracy_policy, data_pruned=sparsity_vs_accuracy_dqn,
                         legend=('pruning_with_policy_dist', 'pruning'), xlabel='NNZ', ylabel='accuracy',
                         title='NNZ_vs_accuracy')
    p_DQN.join()
    p_policy_dist.join()
    
if __name__ == '__main__':
    main()

"""

def iterative_pruning_policy_distilliation(logger, agent, target_agent, iterations=100, use_per=False,
                                           config=student_config, best_path=student_config.prune_best,
                                           arch_type=0, lower_bound=0.0, accumulate_experience_fn=accumulate_experience,
                                           evaluate_fn=evaluate, objective_score=18.0):
    """
    orchestrating the pruning phase, the teacher feeds experience via the "accumulate_experience_fn" function,
    then we prune the student and fine-tune it using Policy Distillation via the "train_student" function.
    on default each iteration is worth 20k iterations of batch_training in the "train_student" function
    the procedure stops once the agent doesnt recuperate or if the sparsity converges for 5 consecutive trials
    :param logger: a proxy to a log file
    :param agent: the student model, which we prune
    :param target_agent: the teacher model
    :param iterations: num of iterations
    :param use_per: a Flag to signal if we want to use Prioritized Experience Replay
    :param config: some customer-defined variables such as memory size
    :param best_path: a path to output the pruned model which solves the environment
    :param arch_type: helps to determine the learning rate
    :param lower_bound: if the model scores a score which is under the lower bound, the pruning stops until the agent recuperates
    :param accumulate_experience_fn: a function in which the teacher interacts with the environment and feeds the ER
    :param evaluate_fn: a function to evaluate the agent
    :param objective_score: the objective score of the environment , if the agent scores higher then the objective
                            then the agent is considered to be able to solve the environment.
    :return: sparsity_vs_accuracy : which holds information on the performance of the agent during the procedure
             sparse_model : the sparse_model is saved in best_path, the sparse_model should be able to solve the environment
    """
    initial_score = evaluate_fn(agent=agent)
    NNZ_vs_accuracy = [[], []]
    NNZ_params_measure = agent.get_number_of_nnz_params()  # initial number of parameters
    NNZ_vs_accuracy[1].append(initial_score)
    NNZ_vs_accuracy[0].append(NNZ_params_measure)
    if use_per:
        exp_replay = Supervised_Prioritzed_ExperienceReplay(size=config.memory_size,
                                                            alpha=config.ALPHA_PER)
    else:
        exp_replay = Supervised_ExperienceReplay(size=config.memory_size)

    stop_prune_arg = False
    m = 0  # counter for number of consecutive iteration the model was evaluated below the lower bound
    r = 0  # counter for number of consecutive iteration the model didn't prune due to low results
    cnt = 0  # counter for number of consecutive iteration the model's sparsity remained the same
    learning_rate_multiplier = 1.0  # dynamic learning rate when the model cannot recuperate for more then 5 iterations
    plus = True
    multiplier = 10
    for i in range(iterations):
        logger.info("-- ITERATION number " + str(i) + "/" + str(iterations) + ": accumulating experience from teacher --")
        print("-- ITERATION number " + str(i) + "/" + str(iterations) + ": accumulating experience from teacher --")
        accumulate_experience_fn(teacher=target_agent, exp_replay=exp_replay, config=config)
        logger.info("-- ITERATION number " + str(i) + "/" + str(iterations) +
                    ": finished accumulating experience from teacher starting to prune and fine-tune the student --")
        print("-- ITERATION number " + str(i) + "/" + str(iterations) +
              ": finished accumulating experience from teacher starting to prune and fine-tune the student -- ")
        score_list, NNZ_params_list, stop_prune_arg = train_student(logger=logger, student=agent,
                                                                  exp_replay=exp_replay,
                                                                  prune=True,
                                                                  lr=config.learning_rate_schedule_prune(i, arch_type) * learning_rate_multiplier,
                                                                  stop_prune_arg=stop_prune_arg,
                                                                  epoch=i, use_per=use_per, best_path=best_path,
                                                                  config=config, evaluate_fn=evaluate_fn, objective_score=objective_score,
                                                                  lower_bound=lower_bound)

        for j, score in enumerate(score_list):
            if NNZ_params_list[j] < NNZ_vs_accuracy[0][-1]:  # if the number of Non zero params is lower that means we pruned
                NNZ_vs_accuracy[1].append(score)
                NNZ_vs_accuracy[0].append(NNZ_params_list[j])

        mean_score = np.mean(score_list)
        logger.info("-- iteration number " + str(i) + ": student evaluation after pruning procedeure is: "
                    + str(mean_score) + " --")
        print("-- iteration number " + str(i) + ": student evaluation after pruning procedeure is: "
              + str(mean_score) + " --")

        if mean_score < lower_bound:
            m += 1
            if m % 5 == 0:
                logger.info("-- iteration {} / {} : Finished due to low accuracy for 5 consecutive trials --"
                            .format(i, iterations))
                break
        else:
            m = 0

        if stop_prune_arg:
            r += 1
            if r >= 5:
                # if the model cant recuperate we drunk-walk the learning rate to help it get un-stuck,
                # if the learning rate scheduler is done carefully, the algorithm shouldn't get here
                # in the paper we didn't really got to use it but it is a nice-to-have dynamic feature to help
                # models to get unstuck
                logger.info("Changing learning rate")
                if plus:
                    learning_rate_multiplier *= multiplier
                    plus = False
                    multiplier *= 10
                else:
                    learning_rate_multiplier /= multiplier
                    plus = True
                    multiplier *= 10
            if NNZ_params_measure == NNZ_vs_accuracy[0][-1]:
                cnt += 1
                if cnt == 5:
                    logger.info("sparsity converged, ending pruning procedure")
                    break
            else:
                cnt = 0

        else:
            multiplier = 10.0
            r = 0
        # if we pruned some weights we expect that the number of Non zero parameters will drop
        if NNZ_vs_accuracy[0][-1] < NNZ_params_measure:  # update the measure of NNZ params to check if the sparsity converged
            NNZ_params_measure = NNZ_vs_accuracy[0][-1]
    return NNZ_vs_accuracy

