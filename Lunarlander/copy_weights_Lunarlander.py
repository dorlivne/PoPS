

from model import ActorLunarlander,StudentActorLunarlander
from configs import LunarLanderConfig as dense_config
from configs import StudentLunarLanderConfig as student_config


def copy_weights(output_path, teacher_path):
    teacher = ActorLunarlander(input_size=dense_config.input_size, output_size=dense_config.output_size,
                          model_path=dense_config.actor_path)
    teacher.load_model(path=teacher_path)  # load teacher
    weights = teacher.get_weights()
    student = StudentActorLunarlander(input_size=student_config.input_size,
                          output_size=student_config.output_size,
                          model_path=output_path,
                          tau=student_config.tau)
    student.copy_weights(weights=weights)
    student.save_model()
