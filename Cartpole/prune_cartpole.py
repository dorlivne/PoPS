import gym
from model import CartPoleDQNTarget, CartPoleDQN
from configs import CartpoleConfig as dense_config
from configs import PruneCartpoleConfig as prune_config
from Cartpole.evaluate_cartpole import evaluate_cartepole
from utils.Memory import Supervised_ExperienceReplay
from utils.logger_utils import get_logger
from utils.plot_utils import plot_graph
import numpy as np
from policy_distilliation_train import policy_distilliation_batch_train
from prune import iterative_pruning_policy_distilliation


def accumulate_experience_cartpole(teacher: CartPoleDQNTarget,
                                   exp_replay: Supervised_ExperienceReplay, config=dense_config):
    env = gym.make('CartPole-v0')
    steps = 0
    while 1:
        state = env.reset()
        state = np.asarray(state)
        done = False
        while not done:
            steps += 1
            teacher_q_value = teacher.get_q(state=np.expand_dims(state, axis=0))
            action = teacher.select_action(teacher_q_value)
            next_state, reward, done, _ = env.step(action)
            exp_replay.add_memory(state, teacher_q_value, action)  # feeding the experience replay
            state = next_state
        if steps > config.OBSERVE:  # we have OBSERVE  number of exp in exp_replay
            break


def main():
    teacher = CartPoleDQNTarget(input_size=dense_config.input_size, output_size=dense_config.output_size)
    teacher.load_model(path=dense_config.ready_path)
    to_be_pruned_agent = CartPoleDQN(input_size=dense_config.input_size,
                                     output_size=dense_config.output_size,
                                     model_path=prune_config.model_path,
                                     scope="pruned_cartpole",
                                     pruning_end=prune_config.pruning_end,
                                     pruning_freq=prune_config.pruning_freq,
                                     sparsity_start=prune_config.sparsity_start,
                                     sparsity_end=prune_config.sparsity_end,
                                     target_sparsity=prune_config.target_sparsity)
    to_be_pruned_agent.load_model(dense_config.ready_path)
    logger = get_logger("pruning_cartpole")
    logger.info("--------- Commencing pruning procedure ---------")
    to_be_pruned_agent.epsilon = 0.0
    to_be_pruned_agent.reset_global_step()
    sparsity_vs_accuracy = iterative_pruning_policy_distilliation(logger=logger, agent=to_be_pruned_agent,
                                                                  target_agent=teacher,
                                                                  config=prune_config, best_path=prune_config.best_model,
                                                                  lower_bound=prune_config.LOWER_BOUND,
                                                                  accumulate_experience_fn=accumulate_experience_cartpole,
                                                                  evaluate_fn=evaluate_cartepole)
    plot_graph(data=sparsity_vs_accuracy, name="accuracy_vs_sparsity")


def prune_student_and_train(logger, lr, student: CartPoleDQN,
                                    exp_replay: Supervised_ExperienceReplay, iterations=50000,
                                    prune=False, stop_prune=True, obj_prune=None):
    """
    :param lr: learning rate
    :param student: student net to be trained
    :param exp_replay: Supervised ER
    :param iterations: number of training iterations
    :param prune: flag to enable pruning
    :param stop_prune: flag to stop pruning for fine_tuning during pruning
    :param obj_prune: if none the idea is to prune until accuracy drops under a certain threshold. relevant only when pruning
    """
    objective_achieved = False
    j = 0
    print("learning rate is {}".format(lr))
    objective_achieved_score = 0
    # --------- Fitting the agent with policy_dist ---------
    for iter in range(iterations):
        if prune and not stop_prune:
            student.prune()
        _ = policy_distilliation_batch_train(student=student, exp_replay=exp_replay, learning_rate=lr,
                                             config=dense_config)
        if iter % 1000 == 0:
            nnz_params = student.get_number_of_nnz_params()
            sparsity = student.get_model_sparsity()
            score_before_final_tune = evaluate_cartepole(agent=student, n_epoch=10)
            logger.info("iter number {}/{} , student has {} NNZ params with sparsity {} and scored {}"
                        .format(iter, iterations, nnz_params, sparsity, score_before_final_tune))
            print("iter number {}/{} , student has {} NNZ params with sparsity {}  and scored {}"
                  .format(iter, iterations, nnz_params, sparsity, score_before_final_tune))

            #  --------- Pruning sector ---------
            if prune:
                    print("global step is " + str(student.print_global_step()))


                    # --------- sparsity objective sector ---------
                    if obj_prune is not None:
                        if nnz_params < obj_prune and not stop_prune:  # objective achieved
                            logger.info("sparsity objective of {} NNZ_params has been achieved with {} NNZ_params "
                                        .format(obj_prune, nnz_params))
                            objective_achieved = True
                            student.save_model()
                            objective_achieved_score = score_before_final_tune
                            prune = False  # this will cause the algorithm to only train the model with no pruning
                            student.freeze_global_step()  # added

                    # --------- accuracy objective sector ---------
                    elif score_before_final_tune > prune_config.OBJECTIVE_SCORE:  # prune until death, saving best models
                        logger.info("saving best model with sparsity {} and score {}"
                                    .format(sparsity, score_before_final_tune))
                        print("saving best model with sparsity {} and score {}".format(sparsity, score_before_final_tune))
                        student.save_model(path=prune_config.best_model)

                    # --------- pruning w.r.t the score ---------
                    if score_before_final_tune < 140 and not stop_prune:
                        stop_prune = True
                        freeze_global_step = student.freeze_global_step()
                        logger.info("stopped pruning due to low results, global step is {}".format(freeze_global_step))
                    if score_before_final_tune > dense_config.OBJECTIVE_SCORE and stop_prune:
                        stop_prune = False
                        freeze_global_step = student.unfreeze_global_step()
                        logger.info("agent got back on the horse and managed to score 195 plus,"
                                    " continue pruning with global step {}".format(freeze_global_step))


                    # --------- stopping rule sector ---------
                    if stop_prune:
                        j += 1
                        if j % 10 == 0 and obj_prune is not None: # this is for sparsity objective sector
                            logger.info("model preforming bad, but pruning must continue to achieve the sparsity objective")
                            stop_prune = False  # to continue pruning untill objective achieved
                            student.unfreeze_global_step()  # so it could continue pruning from the last ckpt

                        if j % 50 == 0 and obj_prune is None:  # this is for accuracy objective sector pruning
                            logger.info("Couldn't prune model any further, loading best model")
                            print("Couldn't prune model any further, loading best model")
                            student.load_model(path=prune_config.best_model)
                            objective_achieved = True
                            break
                    else:
                        j = 0


            # --------- policy distillation sector ---------
            else:
                if score_before_final_tune > prune_config.OBJECTIVE_SCORE or score_before_final_tune > objective_achieved_score :
                    objective_achieved_score = score_before_final_tune
                    j += 1
                    if j == 10:
                        student.save_model()
                        print("finished policy_distilling student")
                        logger.info("finished policy_distilling student")
                        objective_achieved = True
                        break
                else:
                        j = 0

    if obj_prune is not None and objective_achieved:
        # when pruning, we want to fine_tune after the OBJ has
        # been achieved, that is why we freeze the global step, so that future pruning will not be biased by the
        # fine tuning without pruning
        student.unfreeze_global_step()

    return objective_achieved, stop_prune



def prune_cartpole(logger, teacher: CartPoleDQNTarget, agent: CartPoleDQN, obj_prune=None, arch_type=0):
    """

    :param teacher: teacher net
    :param agent: to be pruned agent
    :param obj_prune: if not None then we have a target NNZ_Params to reach
    :return:  a pruned agent
    """
    stop_prune = False
    iteration = 0
    exp_replay = Supervised_ExperienceReplay(size=dense_config.memory_size)
    objective_achieved = False
    while not objective_achieved:
        iteration += 1
        logger.info("iteration number " + str(iteration) + ": accumulating experience from teacher")
        print("iteration number " + str(iteration) + ": accumulating experience from teacher")
        accumulate_experience_cartpole(teacher=teacher, exp_replay=exp_replay)
        logger.info("iteration number " + str(iteration) + ": finished accumulating experience from teacher")
        print("iteration number " + str(iteration) + ": finished accumulating experience from teacher")
        objective_achieved, stop_prune = prune_student_and_train(logger=logger,
                                                                 lr=prune_config.learning_rate_schedule(iteration, arch_type),
                                                                 exp_replay=exp_replay, student=agent,
                                                                 stop_prune=stop_prune, prune=True,
                                                                 obj_prune=obj_prune)
    score = evaluate_cartepole(agent=agent, n_epoch=100)
    nnz_params = agent.get_number_of_nnz_params()
    sparsity = agent.get_model_sparsity()
    logger.info("pruned model is with sparsity {} and {} nnz_params with average score of {}"
                .format(sparsity, nnz_params, score))
    print("pruned model is with sparsity {} and {} nnz_params with average score of {}"
          .format(sparsity, nnz_params, score))
    return score


def policy_dist(logger, teacher: CartPoleDQNTarget, agent: CartPoleDQN, n_epochs=dense_config.n_epoch, arch_type=0):
    """
    :param teacher: teacher net
    :param agent: to be trained agent
    :return: a policy_distilled agent
    """
    exp_replay = Supervised_ExperienceReplay(size=dense_config.memory_size)
    objective_achieved = False
    for e in range(n_epochs):
        logger.info("epoch number " + str(e) + ": accumulating experience from teacher")
        print("epoch number " + str(e) + ": accumulating experience from teacher")
        accumulate_experience_cartpole(teacher=teacher, exp_replay=exp_replay)
        logger.info("iteration number " + str(e) + ": finished accumulating experience from teacher")
        print("iteration number " + str(e) + ": finished accumulating experience from teacher")
        objective_achieved, _ = prune_student_and_train(logger=logger,
                                                        lr=prune_config.learning_rate_schedule(e, arch_type),
                                                        exp_replay=exp_replay, student=agent)
        if objective_achieved:
            break
    if not objective_achieved:
        logger.warning("WARNING: objective not achieved, saving agent anyway")
        print("WARNING: objective not achieved, saving agent anyway")
        agent.save_model()


if __name__ == '__main__':
    main()