import numpy as np
import os
from xml.etree import ElementTree


class XML_preprocessor(object):

    def __init__(self, data_path):
        self.path_prefix = data_path
        self.data = []
        self._preprocess_XML()

    def _preprocess_XML(self):
        filenames = os.listdir(self.path_prefix)
        # filenames = filenames[:500]
        print(filenames)

        for filename in filenames:
            temp = []
            print(filename)
            if filename == '.DS_Store':
                continue
            tree = ElementTree.parse(self.path_prefix + filename)
            root = tree.getroot()
            image_name = root.find('filename').text
            temp.append('model_data/train/' + image_name)
            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text)
                    ymin = float(bounding_box.find('ymin').text)
                    xmax = float(bounding_box.find('xmax').text)
                    ymax = float(bounding_box.find('ymax').text)
                class_name = object_tree.find('name').text
                if class_name == '40' or class_name == '39':
                    if class_name == '40':
                        temp.append(str(int(xmin)) + "," + str(int(ymin)) + "," + str(int(xmax)) + "," + str(
                            int(ymax)) + "," + str(2))
                    if class_name == '39':
                        temp.append(str(int(xmin)) + "," + str(int(ymin)) + "," + str(int(xmax)) + "," + str(
                            int(ymax)) + "," + str(1))
                else:
                    temp.append(
                        str(int(xmin)) + "," + str(int(ymin)) + "," + str(int(xmax)) + "," + str(int(ymax)) + "," + str(
                            0))
                # temp.append(
                #     str(int(xmin)) + "," + str(int(ymin)) + "," + str(int(xmax)) + "," + str(int(ymax)) + "," + str(
                #         int(class_name) - 1))
                # temp.append(
                #     str(int(xmin)) + "," + str(int(ymin)) + "," + str(int(xmax)) + "," + str(int(ymax)) + "," + str(
                #         0))

            self.data.append(temp)


# ## example on how to use it
# import pickle
#
#
data = XML_preprocessor("./model_data/label_train/").data
# pickle.dump(data, open('data/train.pkl', 'wb'))


with open('./model_data/kitti_simple_label.txt', 'w')as file:
    for cutWords in data:
        file.write(' '.join(cutWords) + '\n')
