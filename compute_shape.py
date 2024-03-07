import numpy as np
batch = 1
in_channel = 3
out_channel = 32
padding = 1
dilation = 1
kernel_size = 3
stride = 1
in_height = 150
in_width = 150

conv2d = 1

if conv2d:
    out_height = np.floor( 1/stride *( in_height + 2 * padding - dilation * (kernel_size - 1) - 1) + 1)
    out_width = np.floor( 1/stride *( in_height + 2 * padding - dilation * (kernel_size - 1) - 1) + 1)
else:
    out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    out_width = (in_height - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1

print("height: ", out_height, "width: ", out_width)






