import tensorflow as tf

# Check for available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
    for gpu in gpus:
        print("Found GPU:", gpu)
else:
    print("No GPUs found. TensorFlow is using the CPU.")