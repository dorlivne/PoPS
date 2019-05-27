from configs import StudentPongConfig as student_config
from PONG.evaluate import evaluate
from utils.Memory import Supervised_ExperienceReplay, Supervised_Prioritzed_ExperienceReplay
import numpy as np
from train import accumulate_experience, train_student


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

