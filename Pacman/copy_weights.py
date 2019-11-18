from model import DQNPacman, StudentPacman
from configs import DensePacmanAgentConfig as dense_config
from configs import StudentPacmanConfig as student_config


def copy_weights(output_path, teacher_path):
    teacher = DQNPacman(input_size=dense_config.input_size, output_size=dense_config.output_size,
                      model_path=dense_config.model_path, scope=dense_config.scope,
                      epsilon_stop=dense_config.final_epsilon, epsilon=0.0)
    teacher.load_model(path=teacher_path)  # load teacher
    teacher_weights = teacher.get_weights()
    student = StudentPacman(input_size=student_config.input_size,
                          output_size=student_config.output_size,
                          model_path=output_path,
                          tau=student_config.tau,
                          epsilon=0.0)
    student.print_num_of_params()
    teacher.print_num_of_params()
    student.copy_weights(weights=teacher_weights)
    student.save_model()
    print("created initial model at {}".format(output_path))



