import numpy as np
import pandas as pd
import cv2
import re
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import streamlit as st
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt


def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r


class WheatTestDataset(Dataset):

    def __init__(self, image, transforms=None):
        super().__init__()
        self.transforms = transforms
        self.image = [image]

    def __getitem__(self, index):
        image = cv2.cvtColor(np.asarray(self.image[index]), cv2.COLOR_BGR2RGB).astype(np.float32)
        # st.write('image', image)
        # image = np.asarray(self.image[index]).astype(np.float32)
        image /= 255.0

        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']

        return np.asarray(image)

    def __len__(self) -> int:
        return len(self.image)


# Albumentations
def get_test_transform():
    return A.Compose([
        # A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    st.header("""
    WELCOME TO GLOBAL WHEAT HEAD CHALLENGE!
    """)

    WEIGHTS_FILE = 'fasterrcnn_resnet50_fpn_best.pth'
    # load a model; pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2  # 1 class (wheat) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Load the trained weights
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
    model.eval()
    x = model.to(device)
    detection_threshold = 0.6
    results = []
    outputs = None
    images = None

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Detecting...")
        test_dataset = WheatTestDataset(image, get_test_transform())
        test_data_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            drop_last=False,
            collate_fn=collate_fn
        )

        for images in test_data_loader:
            images = torch.Tensor([images[0][0], images[1][0], images[2][0]])
            images = torch.reshape(images, (3, 1024, 1024))
            images = (images,)
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for i, image in enumerate(images):
                boxes = outputs[i]['boxes'].data.cpu().numpy()
                scores = outputs[i]['scores'].data.cpu().numpy()

                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                scores = scores[scores >= detection_threshold]

                boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

                for j in zip(boxes, scores):
                    result = {
                        'Detected Boxes': "{} {} {} {}".format(j[0][0], j[0][1], j[0][2], j[0][3]),
                        'Confidence%': j[1]
                    }

                    results.append(result)

    if len(results) != 0:
        # print out results
        sample = images[0].permute(1, 2, 0).cpu().numpy()
        boxes = outputs[0]['boxes'].data.cpu().numpy()
        scores = outputs[0]['scores'].data.cpu().numpy()
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        fig, ax = plt.subplots(1, 1, figsize=(32, 16))
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img=sample,
                          pt1=(x1, y1),
                          pt2=(x2, y2),
                          color=(0, 0, 255), thickness=3)

        ax.set_axis_off()
        im = ax.imshow(sample)
        st.pyplot()
        st.write("# Results")
        st.dataframe(pd.DataFrame(results))
    else:
        st.write("")
        st.write("""
        No wheat heads detected in the image!
        """)