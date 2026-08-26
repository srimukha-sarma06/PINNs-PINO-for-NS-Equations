import tensorflow as tf
import deepxde as dde

#Enforicing the stream function psi here to satisfy the continuity automatically(not being used rn)
n_sensors = 2

net = dde.nn.DeepONet(
    [n_sensors]+[96]*2, # n_sensors + pin, l and d(dimensions of the pipe)
    [3]+[96]*2,
    "tanh",
    "Glorot normal",
    num_outputs=3,
    multi_output_strategy="split_both"
)


dummy_branch = tf.zeros((1, 2))
dummy_trunk = tf.zeros((1, 3))
_ = net((dummy_branch, dummy_trunk))


net.load_weights("PINO_TUTORIAL/pino_models_no_stream/pino_model_no_stream-1596.weights.h5")
print("Weights loaded successfully!")

@tf.function(input_signature=[
    tf.TensorSpec(shape=[1, 2], dtype=tf.float32, name="branch_inputs"),
    tf.TensorSpec(shape=[1, 3], dtype=tf.float32, name="trunk_inputs")
])
def serve_tf_lite(branch_inputs, trunk_inputs):
    outputs = net((branch_inputs, trunk_inputs))
    x = trunk_inputs[:, 0:1]
    y = trunk_inputs[:, 1:2]
    t = trunk_inputs[:, 2:3]

    Pin = branch_inputs[:, 0:1]
    Pout = branch_inputs[:, 1:2]

    u = outputs[:, 0:1]
    v = outputs[:, 1:2]
    p_hat = outputs[:, 2:3]

    p = (1.0 - x) * Pin + x*Pout + x*(1.0 - x)*p_hat

    return tf.concat([u, v, p], axis=1)

#Cant apply the output transforms here, has to be done on the board itself(post processing).

concrete_func = serve_tf_lite.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("pino_model_no_stream.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite model saved successfully!")