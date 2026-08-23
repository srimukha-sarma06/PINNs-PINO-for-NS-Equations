import tensorflow as tf
n_sensors = 1

layer_size_branch = [n_sensors, 64, 64]
layer_size_trunk = [3, 64, 64]

class Deeponet(tf.keras.Model):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk):
        super(Deeponet, self).__init__()
        self.layer_sizes_branch = layer_size_branch
        self.layer_sizes_trunk = layer_size_trunk

        self.branch_network = tf.keras.Sequential([tf.keras.layers.Dense(layer, activation="tanh")] for layer in layer_sizes_branch)

        self.trunk_network = tf.keras.Sequential([tf.keras.layers.Dense(layer, activation="tanh")] for layer in layer_sizes_trunk)


    def create_network(self, inputs):
        branch_inputs, trunk_inputs = inputs

        x1 = self.branch_network(branch_inputs)
        x2 = self.trunk_network(trunk_inputs)

        element_wise = tf.multiply(x1, x2)

        outputs = tf.reduce_sum(element_wise, axis=-1, keepdims=True)

        return outputs


deeponet = Deeponet(layer_size_branch, layer_size_trunk)
deeponet.load_weights("/PINO_TUTORIAL/pino_models_tf/pino_01-15005.weights.h5")

print("Weights loaded successfully")
converter = tf.lite.TFLiteConverter.from_keras_model(deeponet)
tflite_model = converter.convert()

# 4. Save the standalone file to disk
with open("pin0_model.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite model saved successfully!")