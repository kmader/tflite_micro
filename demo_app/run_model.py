import tensorflow as tf
import numpy as np, time
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path='local_min_cnn.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
print('# Input')
for i, c_input in enumerate(input_details):
    for k, v in c_input.items():
        print('\t{}:{}={}'.format(i, k, v))
print('# Output')
for i, c_input in enumerate(output_details):
    for k, v in c_input.items():
        print('\t{}:{}={}'.format(i, k, v))

# run the model
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
start = time.time()
interpreter.invoke()
output_data = {c_det['name']: interpreter.get_tensor(c_det['index'])
 for c_det in output_details}
end = time.time()
print('{:2.2f}ms'.format(1e3*(end-start)))
