import numpy as np
import os, cv2, glob, json
import matplotlib.pyplot as plt

MAPPING = {
    # clothes
    "giallo": 2,
    "verde": 3,
    "grigio": 1,
    # clothes2
    "gallo": 2,
    "quadri": 1,
    # ropes
    "black": 2,
    "red": 3,
    "red_dot": 1,
}


PATH = "/home/lar/Downloads/test_clothes2_seg"
OUT_FOLER_1 = "gt_imgs"
OUT_FOLER_2 = "gt_masks"
os.makedirs(os.path.join(PATH, OUT_FOLER_1), exist_ok=True)
os.makedirs(os.path.join(PATH, OUT_FOLER_2), exist_ok=True)

files_json = glob.glob(os.path.join(PATH, "*.json"))

for file_json in files_json:

    with open(file_json, "r") as f:
        data = json.load(f)["shapes"]

    img_name = os.path.basename(file_json).replace(".json", ".png")
    img = cv2.imread(os.path.join(PATH, img_name), cv2.IMREAD_COLOR)

    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for shape in data:
        points = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [points], MAPPING[shape["label"]])

    if False:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[1].imshow(mask)
        plt.show()

    cv2.imwrite(os.path.join(PATH, OUT_FOLER_1, img_name), img)
    cv2.imwrite(os.path.join(PATH, OUT_FOLER_2, img_name), mask)
