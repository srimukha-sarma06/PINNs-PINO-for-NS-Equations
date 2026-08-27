import tensorflow as tf
import deepxde as dde
#custom keras model for the deeponet

#custom keras model for the deeponet

class keras_deeponet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.bf = tf.keras.layers.Dense(96, activation="tanh")
        self.bu_1 = tf.keras.layers.Dense(32, activation="tanh")
        self.bv_1 = tf.keras.layers.Dense(32, activation="tanh")
        self.bp_1 = tf.keras.layers.Dense(32, activation="tanh")

        self.bu_2 = tf.keras.layers.Dense(32, activation="tanh")
        self.bv_2 = tf.keras.layers.Dense(32, activation="tanh")
        self.bp_2 = tf.keras.layers.Dense(32, activation="tanh")

        self.t0 = tf.keras.layers.Dense(96, activation="tanh")
        self.tu_1 = tf.keras.layers.Dense(32, activation="tanh")
        self.tv_1 = tf.keras.layers.Dense(32, activation="tanh")
        self.tp_1 = tf.keras.layers.Dense(32, activation="tanh")

        self.tu_2 = tf.keras.layers.Dense(32, activation="tanh")
        self.tv_2 = tf.keras.layers.Dense(32, activation="tanh")
        self.tp_2 = tf.keras.layers.Dense(32, activation="tanh")

        self.bias_u = tf.Variable(0.0)
        self.bias_v = tf.Variable(0.0)
        self.bias_p = tf.Variable(0.0)

        # Extracts from (x, y, t)
        self.ext_x = tf.constant([[1.0], [0.0], [0.0]], dtype=tf.float32)
        self.ext_y = tf.constant([[0.0], [1.0], [0.0]], dtype=tf.float32)
        # Extracts from (Pin, Pout)
        self.ext_pin = tf.constant([[1.0], [0.0]], dtype=tf.float32)
        self.ext_pout = tf.constant([[0.0], [1.0]], dtype=tf.float32)

    def call(self, inputs):
        trunk_inputs = inputs[1]
        branch_inputs = inputs[0]

        branch_x = self.bf(branch_inputs)
        branch_u1 = self.bu_1(branch_x)
        branch_v1 = self.bv_1(branch_x)
        branch_p1 = self.bp_1(branch_x)

        branch_u2 = self.bu_2(branch_u1)
        branch_v2 = self.bv_2(branch_v1)
        branch_p2 = self.bp_2(branch_p1)

        trunk_x = self.t0(trunk_inputs)
        trunk_u1 = self.tu_1(trunk_x)
        trunk_v1 = self.tv_1(trunk_x)
        trunk_p1 = self.tp_1(trunk_x)

        trunk_u2 = self.tu_2(trunk_u1)
        trunk_v2 = self.tv_2(trunk_v1)
        trunk_p2 = self.tp_2(trunk_p1)

        u_hat = tf.reduce_sum(branch_u2 * trunk_u2, axis=1, keepdims=True) + self.bias_u
        v_hat = tf.reduce_sum(branch_v2 * trunk_v2, axis=1, keepdims=True) + self.bias_v
        p_hat = tf.reduce_sum(branch_p2 * trunk_p2, axis=1, keepdims=True) + self.bias_p

        x = tf.matmul(trunk_inputs, self.ext_x)
        y = tf.matmul(trunk_inputs, self.ext_y)

        pin = tf.matmul(branch_inputs, self.ext_pin)
        pout = tf.matmul(branch_inputs, self.ext_pout)

        p = pin * (1.0 - x) + pout * x + x*(1.0 - x)*p_hat
        u = y * (1.0 - y) * u_hat
        v = y * (1.0 - y) * v_hat
        

        return tf.concat([u, v, p], axis=1)   

    @property
    def regularizer(self):
        return None         


#Enforicing the stream function psi here to satisfy the continuity automatically(not being used rn)
n_sensors = 2

net = keras_deeponet()


dummy_branch = tf.zeros((1, 2))
dummy_trunk = tf.zeros((1, 3))
_ = net((dummy_branch, dummy_trunk))


net.load_weights("PINO_TUTORIAL/pino_models_no_stream/custom_keras_model_deeper-5188.weights.h5")
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

converter = tf.lite.TFLiteConverter.from_keras_model(net)
tflite_model = converter.convert()

with open("custom_keras_model_deeper.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite model saved successfully!")