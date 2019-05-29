import pickle
import time

import numpy as np
import tensorflow as tf

from model import Generator

MODEL_NAME = "./pretrained/checkpoints/checkpoint_13399.ckpt"
GENERATE_N_SAMPLES = 100000
OUTPUT_DIR = "./samples/generate.txt"
BATCH_SIZE = 64
SEQ_LEN = 10
DIM = 128

with open("pretrained/char2int.pickle", "rb") as f:
    charmap = pickle.load(f)
    f.close()

with open("pretrained/chars.pickle", "rb") as f:
    chars = pickle.load(f)
    f.close()

fake_inputs = Generator(BATCH_SIZE, SEQ_LEN, DIM, len(charmap))

with tf.Session() as session:
    def generate_samples():
        samples = session.run(fake_inputs)
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


    saver = tf.train.Saver()
    saver.restore(session, MODEL_NAME)

    samples = []
    then = time.time()
    start = time.time()
    for i in range(int(GENERATE_N_SAMPLES / BATCH_SIZE)):

        samples.extend(generate_samples())

        # append to OUTPUT_DIR file every 1000 batches
        if i % 1000 == 0 and i > 0:
            save(samples)
            samples = []  # flush

            print('wrote {} samples to {} in {:.2f} seconds. {} total.'.format(1000 * BATCH_SIZE, OUTPUT_DIR,
                                                                               time.time() - then, i * BATCH_SIZE))
            then = time.time()

    save(samples)
    print('finished in {:.2f} seconds'.format(time.time() - start))
