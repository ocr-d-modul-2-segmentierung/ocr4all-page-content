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
from shapely.geometry import Polygon
import math
from itertools import chain
from pagecontent.detection.util import alpha_shape


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
        pred_page_coords = box_detection.find_boxes(data.astype(np.uint8), mode='min_rectangle',
                                                    min_area=self.settings.min_contour_area,
                                                    min_area_ratio=self.settings.min_area_ratio)
        if len(pred_page_coords) >= 1:
            points = np.array(list(chain.from_iterable(pred_page_coords)))
            edges = alpha_shape(points, image.size)
            polys = polygons(edges)
            polys = np.flip(points[polys], axis=1)
            poly = Polygon(polys)
            if self.settings.debug:
                x, y = poly.exterior.xy
                mask = np.zeros(image.shape)
                for box in pred_page_coords:
                    pts = np.array(box, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    mask = cv2.fillPoly(mask, [pts], (255, 255, 255, 255))
                f, ax = plt.subplots(1, 2, True, True)
                ax[0].imshow(mask)
                ax[1].imshow(image)
                ax[1].plot(y, x)
                plt.show()
            return poly
        return None


def create_data(image: np.ndarray, avg_letter_height: int) -> ImageData:
    binary_image = image.astype(np.uint8) / 255
    if avg_letter_height == 0:
        avg_letter_height = compute_char_height(image)

    image_data = ImageData(image=binary_image, average_letter_height=avg_letter_height,
                           binary_image=binary_image)
    return image_data


def polygons(edges: List[int]):
    # Generates polygons from Delaunay edges
    if len(edges) == 0:
        return []

    edges = list(edges.copy())

    shapes = []

    initial = edges[0][0]
    current = edges[0][1]
    points = [initial]
    del edges[0]
    while len(edges) > 0:
        found = False
        for idx, (i, j) in enumerate(edges):
            if i == current:
                points.append(i)
                current = j
                del edges[idx]
                found = True
                break
            if j == current:
                points.append(j)
                current = i
                del edges[idx]
                found = True
                break

        if not found:
            shapes.append(points)
            initial = edges[0][0]
            current = edges[0][1]
            points = [initial]
            del edges[0]

    if len(points) > 1:
        shapes.append(points)

    return shapes


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model = os.path.join(project_dir, 'demo/model/model')
    setting_predictor = PageContentSettings(debug=True, model=model, debug_model=True, model_foreground_normalize=True,
                                            line_space_height=0)

    line_detector = PageContentDetection(setting_predictor)

    page_path = os.path.join(project_dir, 'demo/images/0191.png')

    for _pred in line_detector.detect_paths([page_path]):
        pass
