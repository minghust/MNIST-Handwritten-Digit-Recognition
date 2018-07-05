#
## Created by Ming Zhang on 2018-07-04
#

import struct
from array import array

def LoadData(image_path, lable_path):
    with open(lable_path, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        labels = array("B", file.read())

    with open(image_path, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        image_data = array("B", file.read())

    # append all images to an array
    images = []
    for i in range(size):
        images.append([0] * rows * cols)

    for i in range(size):
        images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

    return images, labels