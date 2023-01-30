import numpy as np 
import os
import cv2
import glob

def get_metrics(y_pred, y_true, loss=float('nan')):
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

 

def compute_metrics_from_list_masks(y_true, y_pred_list):
    max_score_iou = 0
    max_score_dice = 0
    for pred in y_pred_list:
        score = get_metrics(y_true, pred)
        if score["iou"] > max_score_iou: 
            max_score_iou = score["iou"]
        if score["dice"] > max_score_dice: 
            max_score_dice = score["dice"]

    return max_score_iou, max_score_dice,  

def get_binary_masks(color_mask_pil):
    '''
    get list of masks from a single colored mask in PIL format, the background is assumed to be 0
    '''
    color_mask_np = np.array(color_mask_pil)  
    colors = np.array([c[1] for c in color_mask_pil.getcolors() if c[1] != 0])

    masks = []
    for c in colors:
        mask = np.zeros(color_mask_np.shape, dtype=np.uint8)
        mask[color_mask_np == c] = 255
        masks.append(mask)
    return masks


def compute(true_path, pred_path, threshold=127):

    # GT       
    y_true = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)
    mask = np.zeros(np.array(y_true).shape, dtype=np.uint8)
    mask[np.array(y_true) != 0] = 255

    # PRED
    y_pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    y_pred = cv2.resize(y_pred, (y_true.shape[1], y_true.shape[0]))
    mask_pred = np.zeros(y_pred.shape, dtype=np.uint8)
    mask_pred[np.array(y_pred) > threshold] = 255
   
    return get_metrics(mask, mask_pred)


if __name__ == "__main__":

    MAIN_FOLDER = ""
    MASK_NAME = "CP_Best_unique-firefly-20"  
    THRESHOLD = 77

    print(MASK_NAME, THRESHOLD)

    
    gt_imgs_dir = os.path.join(MAIN_FOLDER, "gt_imgs")
    gt_labels_dir = os.path.join(MAIN_FOLDER, "gt_labels")

    iou_scores, dice_scores = [], []

    path = os.path.join(MAIN_FOLDER, MASK_NAME)
    imgs = glob.glob(os.path.join(path, "*"))

    for img_path in imgs:

        IMG_NAME = img_path.split("/")[-1].split(".")[0]
        true_path = os.path.join(gt_labels_dir, IMG_NAME + ".png")
        score = compute(true_path=true_path, pred_path=img_path, threshold=THRESHOLD)

        #print(IMG_NAME, " IOU: ", np.round(score["iou"],3), " DICE: ", np.round(score["dice"], 3))
        
        iou_scores.append(score["iou"])
        dice_scores.append(score["dice"])

    print("{0}   \t| iou: {1:.3}, dice: {2:.3f}".format(type, np.mean(iou_scores)*100, np.mean(dice_scores)*100))

    print(" ")