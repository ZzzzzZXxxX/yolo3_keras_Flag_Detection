import os
from yolo import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

path_test = "./sample/"
path_test_1 = ""
path_result = "./result/result.csv"


def detect_img(yolo,path_result):
    # while True:
    #     img = input('Input image filename:')
    #     try:
    #         image = Image.open(img)
    #     except:
    #         print('Open Error! Try again!')
    #         continue
    #     else:
    #         r_image, temp = yolo.detect_image(image)
    #         r_image.show()
    #         plt.imshow(r_image)
    #         plt.show()
    # yolo.close_session()
    filenames = os.listdir(path_test)
    info = []
    for filename in tqdm(filenames):
        img_path = path_test + filename
        try:
            image = Image.open(img_path)
            r_image, temp = yolo.detect_image(image)
            # r_image.show()
            plt.imshow(r_image)
            plt.show()
            temp.insert(0, filename)
            # print(temp)
            info.append(temp)
        except Exception as e:
            print("错误文件："+img_path)

    with open(path_result, 'w')as file:
        for cutWords in info:
            file.write('\t'.join(cutWords) + '\n')


if __name__ == '__main__':


    detect_img(YOLO(), path_result)
