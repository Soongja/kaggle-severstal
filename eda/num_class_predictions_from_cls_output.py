import numpy as np

cls_output = np.load('cls_output_191024e.npy')
print(cls_output)
print(cls_output.dtype)
print(cls_output.shape)

cls_output = (cls_output >= 0.4).astype(np.uint8)
print(cls_output)

cls_1_sum = np.sum(cls_output[:,0])
cls_2_sum = np.sum(cls_output[:,1])
cls_3_sum = np.sum(cls_output[:,2])
cls_4_sum = np.sum(cls_output[:,3])

print(cls_1_sum)
print(cls_2_sum)
print(cls_3_sum)
print(cls_4_sum)