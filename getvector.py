import Image
def img2vector(imgFile):
    img = Image.open(imgFile).convert('L')
    img_arr = np.array(img, 'i') # 20px * 20px 灰度图像
    img_normlization = np.round(img_arr/255) # 对灰度值进行归一化
    img_arr2 = np.reshape(img_normlization, (1,-1)) # 1 * 400 矩阵
    return img_arr2

if __name__ == '__main__':
    img2vector('test/0.png')