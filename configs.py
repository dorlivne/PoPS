

IMAGE_DIM = (210, 160, 3)
INPUT_DIM = (None, 84, 84, 4)  # NONE for batch size
OUTPUT_DIM = (None, 3)  # up down and nothing


class StudentPongConfig:
    input_size = INPUT_DIM
    output_size = OUTPUT_DIM
    iterative_DPPD = r'DPPD_Iterative'
    prune_best = r'saved_models/best_prune'
    batch_size = 256
    memory_size = 100000  # 100k
    OBSERVE = 25000  # Used to be 50k
    scope = 'PongStudent'
    tau = 0.01
    n_epochs = 100
    ALPHA_PER = 0.6
    EPS_PER = 1e-6
    BETA0_PER = 0.4
    eval_prune = 10

    @staticmethod
    def learning_rate_for_10_and_up(epoch: int):
        if epoch <= 5:
            return 1e-5  # usually  1e-5 for 6% and up and 3 1e-4 for lower
        if 5 < epoch <= 10:
            return 5e-6
        if 10 < epoch <= 20:
            return 1e-6
        else:
            return 5e-7


    @staticmethod
    def learning_rate_for_10_and_down(epoch: int):
        if epoch <= 5:
            return 8e-4
        if 5 < epoch <= 10:
            return 7e-4
        if 10 < epoch <= 15:
            return 5e-4
        if 15 < epoch <= 20:
            return 1e-4
        if 20 < epoch <= 30:
            return 5e-5
        if 30 < epoch <= 40:
            return 2e-6
        if 40 < epoch <= 45:
            return 1e-6
        if 45 < epoch <= 50:
            return 5e-7
        else:
            return 1e-7


    @staticmethod
    def beta_schedule(beta0, e: int, n_epoch: int):
        return min(beta0 + ((1 - beta0) / n_epoch) * e, 1.0)



    @staticmethod
    def learning_rate_schedule(epoch: int, arch_type=0):
        if arch_type == 0:
            return StudentPongConfig.learning_rate_for_10_and_up(epoch)
        else:
            return StudentPongConfig.learning_rate_for_10_and_down(epoch)

    @staticmethod
    def learning_rate_for_10_and_up_prune(epoch: int):
        if epoch <= 20:
            return 5.0e-6  # usually  1e-5 for 6% and up and 3 1e-4 for lower
        if 20 < epoch <= 40:
            return 1e-6
        if 40 < epoch <= 60:
            return 5e-7
        else:
            return 1e-7

    @staticmethod
    def learning_rate_for_10_and_down_prune(epoch: int):
        if epoch <= 20:
            return 1.0e-5  # usually  1e-5 for 6% and up and 3 1e-4 for lower
        if 20 < epoch <= 40:
            return 5e-6
        if 40 < epoch <= 60:
            return 1e-6
        else:
            return 5e-7

    @staticmethod
    def learning_rate_schedule_prune(epoch: int, arch_type=0):
        if arch_type == 0:
            return StudentPongConfig.learning_rate_for_10_and_up_prune(epoch)
        else:
            return StudentPongConfig.learning_rate_for_10_and_down_prune(epoch)


class DensePongAgentConfig:

    input_size = INPUT_DIM
    output_size = OUTPUT_DIM
    n_epoch = 3500
    gamma = 0.99
    initial_epsilon = 1.0
    final_epsilon = 0.02
    memory_size = 100000  # 100k
    batch_size = 32
    model_path = r'saved_models/network_dense_Pong'
    ready_path = r'saved_models/network_dense_Pong_ready'
    EXPLORE = 100000  # 100k
    OBSERVE = 10000  # 10k
    UPDTATE_FREQ = 1000
    steps_per_train = 4
    scope = 'PongDQN'
    ALPHA_PER = 0.6
    EPS_PER = 1e-6
    BETA0_PER = 0.4

    @staticmethod
    def learning_rate_schedule(epoch : int):
        if epoch <= 300:
            return 1e-4
        if 300 < epoch <= 500:
            return 5e-5
        if 500 < epoch <= 700:
            return 1e-5
        if 700 < epoch <= 1200:
            return 5e-6
        else:
            return 1e-6

    @staticmethod
    def beta_schedule(beta0, e: int, n_epoch: int):
        return min(beta0 + ((1 - beta0) / n_epoch) * e, 1.0)


class PrunePongAgentConfig:

    input_size = INPUT_DIM
    output_size = OUTPUT_DIM
    n_epoch = 3000
    gamma = 0.99
    initial_epsilon = 0.05
    final_epsilon = 0.05
    memory_size = 100000  # 100k
    batch_size = 32
    model_path = r'saved_models/network_prune_Pong'
    best_path = r'saved_models/network_prune_Pong_best'
    steps_per_train = 4
    OBSERVE = 25000  # 10k
    pruning_end = -1
    target_sparsity = 0.99  # used to be 0.9
    pruning_freq = 50
    initial_sparsity = 0
    sparsity_start = 0
    sparsity_end = int(7.5e5)   # random big number
    ALPHA_PER = 0.6
    EPS_PER = 1e-6
    BETA0_PER = 0.4
    scope = 'Pruned_PongDQN'

    @staticmethod
    def beta_schedule(beta0, e: int, n_epoch: int):
        return min(beta0 + ((1 - beta0) / n_epoch) * e, 1.0)

    @staticmethod
    def learning_rate_for_10_and_up(epoch: int):
        if epoch <= 5:
            return 5e-6  # usually  1e-5 for 6% and up and 3 1e-4 for lower
        if 5 < epoch <= 10:
            return 1e-6
        if 10 < epoch <= 20:
            return 1e-6
        else:
            return 1e-6

    @staticmethod
    def learning_rate_for_10_and_down(epoch: int):
        if epoch <= 5:
            return 1e-5
        if 5 < epoch <= 10:
            return 5e-5
        if 10 < epoch <= 20:
            return 5e-6
        else:
            return 1e-6

    @staticmethod
    def learning_rate_schedule(epoch: int, arch_type=0):
        if arch_type == 0:
            return StudentPongConfig.learning_rate_for_10_and_up(epoch)
        else:
            return StudentPongConfig.learning_rate_for_10_and_down(epoch)


class CartpoleConfig:
    input_size = (None, 4)
    output_size = (None, 2)
    model_path = 'saved_models/Cart_pole/network_dense'
    ready_path = 'saved_models/Cart_pole/‏‏network_dense_ready'
    n_epoch = 5000
    batch_size = 128
    memory_size = 100000
    EXPLORE = 100000  # 100k
    OBSERVE = 25000  # 10k
    UPDTATE_FREQ = 1000
    ALPHA_PER = 0.6
    EPS_PER = 1e-6
    BETA0_PER = 0.4
    OBJECTIVE_SCORE = 195
    steps_per_train = 1

    @staticmethod
    def learning_rate_schedule(epoch : int):
        if epoch <= 2500:
            return 1e-3
        if 2500 < epoch <= 3500:
            return 5e-4
        if 3500 < epoch <= 4500:
            return 2e-4
        else:
            return 1e-4


    @staticmethod
    def beta_schedule(beta0, e: int, n_epoch: int):
        return min(beta0 + ((1 - beta0) / n_epoch) * e, 1.0)


class PruneCartpoleConfig:
    input_size = (None, 4)
    output_size = (None, 2)
    model_path = 'saved_models/Cart_pole/network_prune'
    best_model = 'saved_models/Cart_pole/‏network_prune_best'
    iterative_DPPD = r'DPPD_iterative'
    policy_dist = r'saved_models/‏network_policy_dist'
    n_epoch = 125
    batch_size = 128
    memory_size = 100000
    EXPLORE = 100000  # 100k
    OBSERVE = 10000  # 10k
    UPDTATE_FREQ = 1000
    ALPHA_PER = 0.6
    EPS_PER = 1e-6
    BETA0_PER = 0.4
    OBJECTIVE_SCORE = 190
    LOWER_BOUND = 80
    steps_per_train = 1
    pruning_end = -1
    target_sparsity = 0.99
    pruning_freq = 10
    initial_sparsity = 0
    sparsity_start = 0
    sparsity_end = int(5e5)  # random big number
    tau = 0.01
    epsilon = 0.0
    eval_prune = 25


    @staticmethod
    def learning_rate_for_40_and_up(epoch : int):
        if epoch <= 20:
            return 1e-5
        if 20 < epoch <= 50:
            return 5e-6
        if 50 < epoch <= 80:
            return 2e-6
        else:
            return 1e-6


    @staticmethod
    def learning_rate_for_40_and_down(epoch: int):
        if epoch <= 20:
            return 1e-4
        if 20 < epoch <= 50:
            return 5e-5
        if 50 < epoch <= 80:
            return 2e-5
        else:
            return 1e-5

    @staticmethod
    def learning_rate_for_10_and_down(epoch: int):
        if epoch <= 20:
            return 1e-3
        if 20 < epoch <= 50:
            return 5e-4
        if 50 < epoch <= 80:
            return 2e-4
        else:
            return 1e-4

    @staticmethod
    def learning_rate_for_10_and_down_pruning(epoch: int):
        if epoch <= 20:
            return 1e-7
        if 20 < epoch <= 50:
            return 8e-8
        if 50 < epoch <= 80:
            return 5e-8
        else:
            return 1e-9

    @staticmethod
    def learning_rate_schedule(epoch: int, arch_type=0):
        if arch_type == 0:
            return PruneCartpoleConfig.learning_rate_for_40_and_up(epoch)
        if arch_type == 1:
            return PruneCartpoleConfig.learning_rate_for_40_and_down(epoch)
        if arch_type == 2:
            return PruneCartpoleConfig.learning_rate_for_10_and_down(epoch)
        if arch_type == 3:
            return PruneCartpoleConfig.learning_rate_for_10_and_down_pruning(epoch)

    @staticmethod
    def learning_rate_schedule_prune(epoch:int, arch_type=0):
            return PruneCartpoleConfig.learning_rate_schedule(epoch, arch_type)
            # just for compatibility with existing structure


class LunarLanderConfig:
    input_size = (None, 8)
    output_size = (None, 4)
    critic_output = (None, 1)
    actor_path = 'saved_models/lunarlander/actor_dense'
    critic_path = 'saved_models/lunarlander/critic_dense'
    actor_ready_path = 'saved_models/lunarlander/‏‏actor_dense_ready'
    critic_ready_path = 'saved_models/lunarlander/critic_dense_ready'
    n_epoch = 10000
    batch_size = 512
    memory_size = 100000
    OBSERVE = 10000  # 10k
    UPDATE_FREQ = 1000 # used to be 500
    OBJECTIVE_SCORE = 200

    @staticmethod
    def learning_rate_schedule_actor(epoch: int):
        if epoch <= 1500:
            return 1e-5
        if 1500 < epoch <= 2500:
            return 5e-6
        if 2500 < epoch <= 3500:
            return 1e-6
        else:
            return 5e-7

    @staticmethod
    def learning_rate_schedule_critic(epoch: int):
        if epoch <= 1500:
            return 1e-4
        if 1500 < epoch <= 2500:
            return 5e-5
        if 2500 < epoch <= 3500:
            return 1e-5
        else:
            return 5e-6

class StudentLunarLanderConfig:
    input_size = (None, 8)
    output_size = (None, 4)
    iterative_DPPD = r'DPPD_results'
    n_epochs = 100
    tau = 0.01
    memory_size = 100000
    batch_size = 512
    OBSERVE = 10000  # 10k
    OBJECTIVE_SCORE = 200
    LOWER_BOUND = 100
    pruning_end = -1
    target_sparsity = 0.99
    pruning_freq = 10
    initial_sparsity = 0
    sparsity_start = 0
    sparsity_end = int(5e5)  # random big number
    eval_prune = 25
    @staticmethod
    def learning_rate_for_40_and_up(epoch: int):
        if epoch <= 20:
            return 1e-5
        if 20 < epoch <= 40:
            return 5e-6
        if 40 < epoch <= 60:
            return 1e-6
        else:
            return 5e-7

    @staticmethod
    def learning_rate_for_40_and_down(epoch: int):
        if epoch <= 20:
            return 1e-5
        if 20 < epoch <= 40:
            return 5e-6
        if 40 < epoch <= 60:
            return 1e-6
        else:
            return 5e-7

    @staticmethod
    def learning_rate_for_10_and_down(epoch: int):
        if epoch <= 20:
            return 1e-3
        if 20 < epoch <= 40:
            return 5e-4
        if 40 < epoch <= 60:
            return 1e-4
        else:
            return 5e-5

    @staticmethod
    def learning_rate_for_10_and_down_pruning(epoch: int):
        if epoch <= 20:
            return 1e-6
        if 20 < epoch <= 40:
            return 5e-7
        if 40 < epoch <= 60:
            return 1e-7
        else:
            return 5e-8

    @staticmethod
    def learning_rate_schedule(epoch: int, arch_type=0):
        if arch_type == 0:
            return StudentLunarLanderConfig.learning_rate_for_40_and_up(epoch)
        if arch_type == 1:
            return StudentLunarLanderConfig.learning_rate_for_40_and_down(epoch)
        if arch_type == 2:
            return StudentLunarLanderConfig.learning_rate_for_10_and_down(epoch)
        if arch_type == 3:
            return StudentLunarLanderConfig.learning_rate_for_10_and_down_pruning(epoch)
    @staticmethod
    def learning_rate_schedule_prune(epoch:int, arch_type=0):
            return PruneCartpoleConfig.learning_rate_schedule(epoch, arch_type)
            # just for compatibility with existing structure
