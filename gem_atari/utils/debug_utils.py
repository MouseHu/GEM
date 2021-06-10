import math
import numpy as np

def check_nan(np_arr):
	len = np.prod(np.array(np_arr.shape))
	np_arr = np_arr.reshape(len)
	for i in range(len):
		if math.isnan(np_arr[i]):
			print('nan found')

def check_inf(np_arr):
	len = np.prod(np.array(np_arr.shape))
	np_arr = np_arr.reshape(len)
	for i in range(len):
		if math.isinf(np_arr[i]):
			print('inf found')