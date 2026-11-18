import tensorflow as tf

layer_sizes = [3, 100, 100, 100, 100, 3]
activation_func = "tanh"

keras_model = tf.keras.Sequential()

keras_model.add(tf.keras.layers.Input(shape=(layer_sizes[0],)))

# Hidden Layers
for hidden_units in layer_sizes[1:-1]:
    keras_model.add(tf.keras.layers.Dense(units=hidden_units, activation=activation_func))

# Output Layer (PINNs typically use linear/no activation for predictions)
keras_model.add(tf.keras.layers.Dense(units=layer_sizes[-1], activation=None))

# 2. Populate the weights directly from your H5 file
# (This bypasses any need for compilation, loss, or optimizer matching)
keras_model.load_weights("TRON/models_2/model_7-exponential_decay.weights.h5") 
print("Weights mapped successfully to the network architecture!")

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# 4. Save the standalone file to disk
with open("pinn_model.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite model saved successfully!")