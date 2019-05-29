import pickle
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

GENERATE_N_SAMPLES = 100000
OUTPUT_DIR = "samples/generate.txt"
MAX_N_EXAMPLES = 10000000
BATCH_SIZE = 100
SEQ_LEN = 10
DIM = 128
checkpoints_path = "pretrained/checkpoints/"
model_path = "pretrained/protobuf/model.pb"
name = "Discriminator.Output_1/BiasAdd"

with open("pretrained/char2int.pickle", "rb") as f:
    charmap = pickle.load(f)
    f.close()

with open("pretrained/chars.pickle", "rb") as f:
    chars = pickle.load(f)
    f.close()


def ckpt2pb(checkpoints_path, saver_model_name):
    ckpt = tf.train.get_checkpoint_state(checkpoints_path)
    if not (ckpt and ckpt.model_checkpoint_path):
        raise ("ERROR: Not found checkpoints.")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
        saver.restore(sess, ckpt.model_checkpoint_path)
        output_graph_def = convert_variables_to_constants(sess, sess.graph_def, [name])
        with tf.gfile.FastGFile(saver_model_name, "wb") as f:
            f.write(output_graph_def.SerializeToString())
    return


def pb_test(path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:

            def generate_samples():
                samples = sess.run(output_tensor)
                samples = np.argmax(samples, axis=2)
                decoded_samples = []
                for i in range(len(samples)):
                    decoded = []
                    for j in range(len(samples[i])):
                        decoded.append(chars[samples[i][j]])
                    decoded_samples.append(tuple(decoded))
                return decoded_samples

            def save(samples):
                with open(OUTPUT_DIR, 'a') as f:
                    for s in samples:
                        s = "".join(s).replace('`', '')
                        f.write(s + "\n")

            output_node_name = "Reshape_2:0"
            output_tensor = sess.graph.get_tensor_by_name(output_node_name)

            samples = []
            count = int(np.ceil(GENERATE_N_SAMPLES / BATCH_SIZE))
            start = time.time()
            for i in range(count):
                samples.extend(generate_samples())
            save(samples)
            print(
                "wrote {} samples to {} in {:.2f} seconds.".format(count * BATCH_SIZE, OUTPUT_DIR, time.time() - start))
            print("finished.")


if __name__ == "__main__":
    ckpt2pb(checkpoints_path, model_path)
    pb_test(model_path)
