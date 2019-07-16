from pagecontent.detection.settings import PageContentSettings
import numpy as np
from typing import List
from functools import partial
from pagecontent.detection.datatypes import ImageData
import multiprocessing
import tqdm
from pagesegmentation.lib.predictor import PredictSettings
from pagecontent.pixelclassifier.pixelclassifier import PCPredictor
from matplotlib import pyplot as plt
from PIL import Image
from pagecontent.postprocessing import box_detection
import cv2
from pagecontent.detection.util import compute_char_height
import os


class PageContentDetection:

    def __init__(self, settings: PageContentSettings):
        self.settings = settings
        self.predictor = None
        if settings.model:
            pcsettings = PredictSettings(
                mode='meta',
                network=os.path.abspath(settings.model),
                output=None,
                high_res_output=False
            )
            self.predictor = PCPredictor(pcsettings, settings.target_line_space_height)

    def detect_paths(self, image_paths: List[str]):
        def read_img(path):
            return np.array(Image.open(path))

        return self.detect(list(map(read_img, image_paths)))

    def detect(self, images : List[np.array]):
        create_data_partial = partial(create_data, avg_letter_height=self.settings.line_space_height)
        if len(images) <= 1:
            data = list(map(create_data_partial, images))
        else:
            with multiprocessing.Pool(processes=self.settings.processes) as p:
                data = [v for v in tqdm.tqdm(p.imap(create_data_partial, images), total=len(images))]
        for i, prob in enumerate(self.predictor.predict(data)):
            norm = prob / np.max(prob) if self.settings.model_foreground_normalize else prob
            pred = (norm > self.settings.model_foreground_threshold)
            data[i].pixel_classifier_prediction = prob

            if self.settings.debug_model:
                f, ax = plt.subplots(1, 2, sharex='all', sharey='all')
                ax[0].imshow(prob)
                ax[1].imshow(pred)
                plt.show()
            yield self.detect_border(pred * 255, data[i].image)

    def detect_border(self, data, image):
        pred_page_coords = box_detection.find_boxes(data.astype(np.uint8), mode='min_rectangle')
        if len(pred_page_coords) > 1:
            mask = np.zeros(image.shape)
            for box in pred_page_coords:
                pts = np.array(box, np.int32)
                pts = pts.reshape((-1, 1, 2))
                mask = cv2.fillPoly(mask, [pts], (255, 255, 255, 255))
            plt.imshow(mask)
            plt.show()
            #preprocess_boxes()
        if self.settings.debug:

            original_img = np.stack((image,) * 3, axis=-1)
            if pred_page_coords is not None:
                cv2.polylines(original_img, [pred_page_coords[:, None, :]], True, (0, 0, 255), thickness=5)
            else:
                print('No box found in {}'.format('on this page'))
            plt.figure(figsize=(10, 10))
            plt.imshow(original_img)
            plt.show()
        return pred_page_coords


def draw_polygon(box):
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    img = cv2.polylines(img, [pts], True, (0, 255, 255))


def preprocess_boxes(contours):
    LENGTH = len(contours)
    status = np.zeros((LENGTH, 1))

    for i, cnt1 in enumerate(contours):
        x = i
        if i != LENGTH - 1:
            for j, cnt2 in enumerate(contours[i + 1:]):
                x = x + 1
                val = min(status[i], status[x])
                status[x] = status[i] = val


    unified = []
    maximum = int(status.max()) + 1
    for i in range(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)

    cv2.drawContours(img, unified, -1, (0, 255, 0), 2)
    cv2.drawContours(thresh, unified, -1, 255, -1)

def create_data(image: np.ndarray, avg_letter_height: int) -> ImageData:
    binary_image = image.astype(np.uint8) / 255
    if avg_letter_height == 0:
        avg_letter_height = compute_char_height(image)

    image_data = ImageData(image=binary_image, average_letter_height=avg_letter_height,
                           binary_image=binary_image)
    return image_data


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model = os.path.join(project_dir, 'demo/model/model')
    setting_predictor = PageContentSettings(debug=True, model=model, debug_model=True, model_foreground_normalize=True,
                                            line_space_height=0)

    line_detector = PageContentDetection(setting_predictor)

    page_path = os.path.join(project_dir, 'demo/images/0077.bin.png')
    for _pred in line_detector.detect_paths([page_path]):
        pass
