import os
import pickle
import re
import sys
import time

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.plot
import untils
from model import Generator, Discriminator

sys.path.append(os.getcwd())

# fill in the path to the extracted files here!
DATA_DIR = './data/train.txt'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in train.py!')

BATCH_SIZE = 64  # Batch size
ITERS = 200000  # How many iterations to train for
SEQ_LEN = 10  # Sequence length in characters
DIM = 128  # Model dimensionality. This is fairly slow and overfits, even on
# Billion Word. Consider decreasing for smaller datasets.
CRITIC_ITERS = 10  # How many critic iterations per generator iteration. We
# use 10 for the results in the paper, but 5 should work fine
# as well.
LAMBDA = 10  # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 10000000  # Max number of data examples to load. If data loading
# is too slow or takes too much RAM, you can decrease
# this (at the expense of having less training data).

lib.print_model_settings(locals().copy())

lines, charmap, chars = untils.load_dataset(max_length=SEQ_LEN, max_n_examples=MAX_N_EXAMPLES)
print("char2int: ", charmap)
print("chars   : ", chars)

with open("pretrained/char2int.pickle", "wb") as f:
    pickle.dump(charmap, f)

with open("pretrained/chars.pickle", "wb") as f:
    pickle.dump(chars, f)

real_inputs_discrete = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
fake_inputs = Generator(BATCH_SIZE, SEQ_LEN, DIM, len(charmap))
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims - 1)

disc_real = Discriminator(real_inputs, SEQ_LEN, DIM, len(charmap))
disc_fake = Discriminator(fake_inputs, SEQ_LEN, DIM, len(charmap))

disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
gen_cost = -tf.reduce_mean(disc_fake)

# WGAN lipschitz-penalty
alpha = tf.random_uniform(
    shape=[BATCH_SIZE, 1, 1],
    minval=0.,
    maxval=1.
)

differences = fake_inputs - real_inputs
interpolates = real_inputs + (alpha * differences)
gradients = tf.gradients(Discriminator(interpolates, SEQ_LEN, DIM, len(charmap)), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
disc_cost += LAMBDA * gradient_penalty

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                                                                          var_list=disc_params)


# Dataset iterator
def inf_train_gen():
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines) - BATCH_SIZE + 1, BATCH_SIZE):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i + BATCH_SIZE]],
                dtype='int32'
            )


# During training we monitor JS divergence between the true & generated ngram
# distributions for n=1,2,3,4. To get an idea of the optimal values, we
# evaluate these statistics on a held-out set first.
true_char_ngram_lms = [untils.NgramLanguageModel(i + 1, lines[10 * BATCH_SIZE:], tokenize=False) for i in range(4)]
validation_char_ngram_lms = [untils.NgramLanguageModel(i + 1, lines[:10 * BATCH_SIZE], tokenize=False) for i in
                             range(4)]

for i in range(4):
    print("validation set JSD for n={}: {}".format(i + 1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
true_char_ngram_lms = [untils.NgramLanguageModel(i + 1, lines, tokenize=False) for i in range(4)]

model_saver = tf.train.Saver(max_to_keep=3)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state('./pretrained/checkpoints/')
    if ckpt and ckpt.model_checkpoint_path:
        model_saver.restore(session, ckpt.model_checkpoint_path)


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


    gen = inf_train_gen()

    print("------ Training...")
    for iteration in range(ITERS):
        start_time = time.time()

        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)

        # Train critic
        for i in range(CRITIC_ITERS):
            _data = gen.__next__()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_inputs_discrete: _data}
            )

        lib.plot.plot('time', time.time() - start_time)
        lib.plot.plot('train disc cost', _disc_cost)

        if iteration % 100 == 99:

            if ckpt:
                fname = "./samples/samples_{}.txt".format(
                    int(re.sub("\D", "", ckpt.model_checkpoint_path)) + 1 + iteration)
                model_name = "./pretrained/checkpoints/checkpoint_{}.ckpt".format(
                    int(re.sub("\D", "", ckpt.model_checkpoint_path)) + 1 + iteration)
                model_saver.save(session, model_name)
            else:
                fname = "./samples/samples_{}.txt".format(iteration)
                model_name = "./pretrained/checkpoints/checkpoint_{}.ckpt".format(iteration)

            samples = []
            for i in range(10):
                samples.extend(generate_samples())

            for i in range(4):
                lm = untils.NgramLanguageModel(i + 1, samples, tokenize=False)
                lib.plot.plot('js{}'.format(i + 1), lm.js_with(true_char_ngram_lms[i]))

            with open(fname, 'w') as f:
                for s in samples:
                    s = "".join(s).replace("`", "")
                    f.write(s + "\n")

            lib.plot.flush()
            model_saver.save(session, model_name)

        lib.plot.tick()
