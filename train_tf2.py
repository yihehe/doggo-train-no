from absl import flags
import tensorflow.compat.v2 as tf
from train_and_eval_tf2 import runTrainLoop

FLAGS = flags.FLAGS


def main(unused_argv):
    flags.mark_flag_as_required('model_dir')
    flags.mark_flag_as_required('pipeline_config_path')
    tf.config.set_soft_device_placement(True)

    runTrainLoop(
        FLAGS.pipeline_config_path,
        FLAGS.model_dir,
        FLAGS.checkpoint_every_n)


if __name__ == '__main__':
    tf.compat.v1.app.run()
