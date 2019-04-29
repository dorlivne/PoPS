from model import  CartPoleDQN, StudentCartpole
from configs import CartpoleConfig as dense_config
from configs import PruneCartpoleConfig as student_config


def copy_weights(output_path: str, teacher_path: str):
    teacher = CartPoleDQN(input_size=dense_config.input_size, output_size=dense_config.output_size,
                          model_path=dense_config.model_path, epsilon=0.0)
    teacher.load_model(path=teacher_path)  # load teacher
    weights = teacher.get_weights()
    student = StudentCartpole(input_size=student_config.input_size,
                          output_size=student_config.output_size,
                          model_path=output_path,
                          tau=student_config.tau,
                          epsilon=0.0)
    student.copy_weights(weights=weights)
    student.save_model()


