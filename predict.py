import glob, cv2, os, torch, arrow
import numpy as np
from dataset import BasicDataset
import model as network
from utils import MODEL_MAP

CKPT_TO_LOAD = "checkpoints/CP_Best_unique-firefly-20.pth"
PRED_DIR = "/home/lar/dev/FCD/test_set/gt_imgs"
SAVE = True


def predict_img(net, img, device):
    net.eval() 

    img = torch.from_numpy(BasicDataset.pre_process(np.array(img)))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0).cpu()
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask  
 
if __name__ == "__main__":

    checkpoint = torch.load(CKPT_TO_LOAD, map_location=torch.device('cpu'))

    img_h, img_w = checkpoint["img_h"], checkpoint["img_w"]

    print("model selected: ", MODEL_MAP[checkpoint["backbone"]])
    model = MODEL_MAP[checkpoint["backbone"]](num_classes=1, output_stride=16)
    network.convert_to_separable_conv(model.classifier)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("Model loaded !")

    test_dir = PRED_DIR
    test_imgs = glob.glob(test_dir+"/*")
    test_imgs = [t for t in test_imgs if not os.path.isdir(t)] 

    if SAVE:
        checkpoint_name = os.path.basename(CKPT_TO_LOAD).split(".")[0]
        save_subdir = os.path.join("results", checkpoint_name)
        os.makedirs(save_subdir, exist_ok=True)

    for i, image_file in enumerate(test_imgs):
        print("\nPredicting image {} ...".format(image_file))
	
        img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_w, img_h))

        # PREDICT 
        start_time = arrow.utcnow()        
        mask = predict_img(net=model, img=img, device=device) 
        end_time = arrow.utcnow()	
        print("Total time: {} milliseconds".format((end_time - start_time).total_seconds() * 1000))       

        mask = mask / np.max(mask)
        result = (mask * 255).astype(np.uint8)

        if SAVE:
            pred_save_path = os.path.join(save_subdir, os.path.basename(os.path.normpath(image_file)))
            cv2.imwrite(pred_save_path, result)
        else:
            cv2.imshow("output", result)
            cv2.waitKey(0)
        