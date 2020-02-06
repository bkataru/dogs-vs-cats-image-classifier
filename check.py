import tensorflow as tf

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from tensorflow.python.client import device_lib
print("*" * 50)
print("checking\n", device_lib.list_local_devices())
print(tf.test.gpu_device_name())