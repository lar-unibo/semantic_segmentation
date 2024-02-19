import glob, cv2, os, torch, arrow
import numpy as np
import matplotlib.pyplot as plt

from dataset import BasicDataset
import model as network
from utils import MODEL_MAP

CKPT_TO_LOAD = "checkpoints/CP_Best_riveting-quiver-11.pth"
PRED_DIR = "/home/alessio/dev/LABELING_DLO_SAM/data/test4_clothes"
SAVE = False
PLOT = True


def predict_img_binary(net, img, device):
    img = torch.from_numpy(BasicDataset.pre_process(np.array(img))).unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0).cpu()
        full_mask = probs.squeeze().cpu().numpy()
    return full_mask


def predict_img_multiclass(net, img, device):
    img = torch.from_numpy(BasicDataset.pre_process(np.array(img))).unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img).squeeze(0)
        full_mask = torch.softmax(output, dim=0).cpu().numpy()

    if False:
        if full_mask.shape[0] == 4:
            fig, axs = plt.subplots(1, 4, figsize=(15, 15))
            for i in range(4):
                axs[i].imshow(full_mask[i])
            plt.tight_layout()
            plt.show()

    full_mask = np.argmax(full_mask, axis=0)
    return full_mask


if __name__ == "__main__":
    checkpoint = torch.load(CKPT_TO_LOAD, map_location=torch.device("cpu"))

    img_h, img_w = checkpoint["img_h"], checkpoint["img_w"]
    num_classes = checkpoint["num_classes"]

    print("model selected: ", MODEL_MAP[checkpoint["backbone"]])
    model = MODEL_MAP[checkpoint["backbone"]](num_classes=num_classes, output_stride=16)
    network.convert_to_separable_conv(model.classifier)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("Model loaded !")

    test_imgs = glob.glob(os.path.join(PRED_DIR, "*.png"))
    test_imgs = [t for t in test_imgs if "mask" not in os.path.basename(t)]
    print("Found {} images to predict".format(len(test_imgs)))
    if SAVE:
        checkpoint_name = os.path.basename(CKPT_TO_LOAD).split(".")[0]
        save_subdir = os.path.join(PRED_DIR, "pred_" + checkpoint_name)
        os.makedirs(save_subdir, exist_ok=True)

    for i, image_file in enumerate(test_imgs):
        print("\nPredicting image {} ...".format(image_file))

        img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_w, img_h))

        # PREDICT
        if num_classes == 1:
            mask = predict_img_binary(net=model, img=img, device=device)
        else:
            mask = predict_img_multiclass(net=model, img=img, device=device)

        if SAVE:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(save_subdir, os.path.basename(image_file)), mask)

        if PLOT:
            fix, axs = plt.subplots(1, 2, figsize=(15, 15))
            axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[1].imshow(mask)
            plt.tight_layout()
            plt.show()
