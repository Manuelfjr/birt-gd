from tqdm import tqdm
import tensorflow as tf
b = tf.Variable(0.5, trainable=True, dtype=tf.float32)
a = tf.Variable(2, trainable=True, dtype=tf.float32)
variables = [a, b]

for _ in tqdm(range(100)):
    with tf.GradientTape() as g:
        g.watch(variables)
        pred = tf.math.tanh(b) * tf.math.softplus(a)
        current_loss = (1 - pred)**2
    _a, _b = g.gradient(current_loss, variables)
    a.assign_sub(tf.math.scalar_mul(0.1, _a))
    b.assign_sub(tf.math.scalar_mul(0.1, _b))