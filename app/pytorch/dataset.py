import os, cv2
import numpy as np
from torch.utils.data.dataset import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, label_path, train_data_path):
        self._train_data_path = train_data_path
        self._parse_label_path(label_path)

    def _parse_label_path(self, label_path):
        truth = {}
        with open(label_path, "r") as read_file:
            for line in read_file.readlines():
                data = line.split(" ")
                truth[data[0]] = []
                for data_element in data[1:]:
                    truth[data[0]].append([int(j) for j in data_element.split(",")])
        self._truth = truth

    def __len__(self):
        return len(self._truth.keys())

    def __getitem__(self, index):
        image_path = list(self._truth.keys())[index]
        bounding_boxes = np.array(self._truth.get(image_path), dtype=np.float)

        image_path = os.path.join(self._train_data_path, image_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (400, 400))

        out_image = np.asarray(image, dtype=np.float)

        out_bounding_boxes = np.zeros([1, 4])
        out_bounding_boxes = bounding_boxes[:, 0:4]

        out_label = np.zeros(26)
        out_label[int(bounding_boxes[:, -1])] = 1

        return (out_image, out_bounding_boxes, out_label)
