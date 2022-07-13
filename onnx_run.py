import onnxruntime as ort
import os
import numpy as np
import sys
import pickle
import time
 
model_dir = 'yolov5s-v12_sim.onnx'
device = 'cpu'
num_threads = 1
 
if device == 'cpu':
    providers = ['CPUExecutionProvider']
else:
    providers = ['CUDAExecutionProvider']

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = num_threads
sess = ort.InferenceSession(
    model_dir,
    sess_options=sess_options,
    providers=providers)
inputs_dict = {}
input_list = list()
np.random.seed(6)

import pickle
input_dict = dict()

input_dict["images"] = np.random.randn(1, 3, 640, 640).astype("float32")
# input_dict["right"] = np.random.randn(1, 3, 480, 640).astype("float32")
# input_dict["flow_init"] = np.random.randn(1, 2, 240, 320).astype("float32")
with open('onnx_inputs.pkl', 'wb') as inp:
    pickle.dump(input_dict, inp)

print(sess.get_inputs()[0].name)
inputs_dict[sess.get_inputs()[0].name] = input_dict["images"]
# print(sess.get_inputs()[1].name)
# inputs_dict[sess.get_inputs()[1].name] = input_dict["right"]
# print(sess.get_inputs()[2].name)
# inputs_dict[sess.get_inputs()[2].name] = input_dict["flow_init"]


warmup_count = 0
count = 1
for i in range(warmup_count):
    result = sess.run(None, input_feed=inputs_dict)

start = time.time()
for i in range(count):
    result = sess.run(None, input_feed=inputs_dict)
    # with open('onnx_outputs.pkl', 'wb') as out:
    #     pickle.dump(result, out)
    print(result[0].shape)
elapsed_ms = (time.time() - start) * 1000
print("Latency: {:.2f} ms".format(elapsed_ms / count))