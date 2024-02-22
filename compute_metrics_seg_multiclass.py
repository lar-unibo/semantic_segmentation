import numpy as np
import os, cv2, glob
import matplotlib.pyplot as plt


def get_metrics(y_pred, y_true):
    if y_pred.ndim == 3:
        y_pred = cv2.cvtColor(y_pred, cv2.COLOR_RGB2GRAY) / 255
    else:
        y_pred = y_pred / 255

    if y_true.ndim == 3:
        y_true = cv2.cvtColor(y_true, cv2.COLOR_RGB2GRAY) / 255
    else:
        y_true = y_true / 255

    y_pred = (y_pred).astype(float)
    y_true = (y_true).astype(float)

    true_positives = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    true_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    false_positives = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    false_negatives = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    assert not np.isnan(true_negatives)
    assert not np.isnan(false_positives)
    assert not np.isnan(false_negatives)
    assert not np.isnan(true_positives)

    if true_positives > 0:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        iou = true_positives / (true_positives + false_positives + false_negatives)
    else:
        precision = recall = f1_score = iou = 0

    return {
        "precision": precision,
        "recall": recall,
        "dice": f1_score,
        "iou": iou,
    }


def compute_metrics_pairwise(y_true_list, y_pred_list):
    out = {}
    for k, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
        metrics = get_metrics(y_pred, y_true)
        out[k] = metrics
    return out


def get_binary_masks(mask):
    masks = []
    for c in np.unique(mask):
        mask_tmp = np.zeros(mask.shape, dtype=np.uint8)
        mask_tmp[mask == c] = 255
        masks.append(mask_tmp)
    return masks


def compute(y_true, y_pred):
    masks_true = get_binary_masks(y_true)
    masks_pred = get_binary_masks(y_pred)
    return compute_metrics_pairwise(masks_true, masks_pred)


if __name__ == "__main__":
    MAIN_FOLDER = "/home/lar/dev/labeling_dlo_sam/data/TEST_SEGMENTATION/clothes2"

    MASKS_TO_TEST = [
        "CLOTHES2_virtuous-rabbit-34",
        "CLOTHES2_beaming-bao-38",
        "CLOTHES2_lambent-festival-36",
        "CLOTHES2_radiant-rooster-37",
        # "CLOTHES_abundant-dragon-32",
        # "CLOTHES_brilliant-rocket-33",
        # "CLOTHES_lambent-ox-31",
        # "CLOTHES_twinkling-festival-30",
    ]

    gt_imgs_dir = os.path.join(MAIN_FOLDER, "gt_imgs")
    gt_labels_dir = os.path.join(MAIN_FOLDER, "gt_masks")

    for mask_to_test in MASKS_TO_TEST:

        iou_scores, dice_scores = [], []

        path = os.path.join(MAIN_FOLDER, mask_to_test)
        img_files = glob.glob(os.path.join(path, "*"))

        for img_path in img_files:
            IMG_NAME = img_path.split("/")[-1].split(".")[0]
            true_path = os.path.join(gt_labels_dir, IMG_NAME + ".png")

            y_true = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)
            y_pred = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            y_pred = cv2.resize(y_pred, (y_true.shape[1], y_true.shape[0]))

            score = compute(y_true, y_pred)

            ious = [v["iou"] for k, v in score.items()]
            dices = [v["dice"] for k, v in score.items()]

            iou_scores.append(np.mean(ious))
            dice_scores.append(np.mean(dices))

        print(f"Model: {mask_to_test}")
        print(f"Mean IoU: {np.mean(iou_scores):.3f} - Mean Dice: {np.mean(dice_scores):.3f}")
        print()
