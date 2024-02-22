import os, sys, wandb, torch, glob, cv2
import numpy as np
from tqdm import tqdm
import albumentations as aug
import model as network
from dataset import BasicDataset
from utils import dice_coeff, set_seeds, PolyLR, MODEL_MAP
from torchvision.utils import make_grid

from compute_metrics_seg_multiclass import compute

CLOTHES2 = {
    "num_classes": 3,
    "dataset_name": "/home/lar/dev/labeling_dlo_sam/data/clothes2_filtered",
    "test_set": "/home/lar/dev/labeling_dlo_sam/data/TEST_SEGMENTATION/clothes2",
}

CLOTHES = {
    "num_classes": 4,
    "dataset_name": "/home/lar/dev/labeling_dlo_sam/data/clothes_filtered",
    "test_set": "/home/lar/dev/labeling_dlo_sam/data/TEST_SEGMENTATION/clothes",
}

ROPES = {
    "num_classes": 4,
    "dataset_name": "/home/lar/dev/labeling_dlo_sam/data/ropes_filtered",
    "test_set": "/home/lar/dev/labeling_dlo_sam/data/TEST_SEGMENTATION/ropes",
}

DATA_CONFIG = CLOTHES2


hyperparameter_defaults = dict(
    num_classes=DATA_CONFIG["num_classes"],
    backbone="swinT",
    epochs=200,
    lr=1e-4,
    batchsize=4,
    pretrained_backbone=True,
    early_stopping_patience=5,
    early_stopping_min_epochs=20,
    freq_validation_per_epoch=1,
    ckpt_pre_train="",
    ckpt_resume="",
    dataset_name=DATA_CONFIG["dataset_name"],
    dataset_path="",
    transforms="transforms_base",
    test_set=DATA_CONFIG["test_set"],
    test_every_n_epocochs=1,
    scheduler="poly",  # step, cosine, poly
    warmup_steps=1000,
    warmup_lr=1e-8,
    img_width=1024,
    img_height=576,
    seed=0,
)


DATASET_TYPE = hyperparameter_defaults["dataset_name"].split("/")[-1].split("_")[0]
print(f"DATASET_TYPE: {DATASET_TYPE}")

wandb.init(
    config=hyperparameter_defaults, project="deformable_objects_segmentation", entity="acaporali", mode="online"
)
config = wandb.config

# set random seed
set_seeds(config.seed)

transforms_base = aug.Compose(
    [
        aug.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=0, p=0.8),
        aug.Flip(p=0.5),
        aug.Perspective(scale=(0, 0.1), p=0.5),
        aug.RandomBrightnessContrast(contrast_limit=[0.1, 0.1], brightness_limit=[-0.1, 0.1]),
    ],
    p=1,
)


def benchmark_multiclass(net, device, global_step):
    net.eval()

    def predict_img_multiclass(net, img, device):
        img = torch.from_numpy(BasicDataset.pre_process(np.array(img))).unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            output = net(img).squeeze(0)
            full_mask = torch.softmax(output, dim=0).cpu().numpy()
        full_mask = np.argmax(full_mask, axis=0)
        return full_mask

    path_imgs = os.path.join(config["test_set"], "gt_imgs")
    path_labels = os.path.join(config["test_set"], "gt_masks")
    test_imgs = glob.glob(os.path.join(path_imgs, "*"))
    test_imgs = [t for t in test_imgs if not os.path.isdir(t)]

    score_iou, score_dice = [], []
    for i, image_file in enumerate(test_imgs):
        img_name = image_file.split("/")[-1]

        img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (config["img_width"], config["img_height"]))

        # PREDICT
        mask = predict_img_multiclass(net=net, img=img, device=device)

        label = cv2.imread(os.path.join(path_labels, img_name), cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (config["img_width"], config["img_height"]), interpolation=cv2.INTER_NEAREST)

        score = compute(label, mask)

        ious = [v["iou"] for k, v in score.items()]
        dices = [v["dice"] for k, v in score.items()]

        score_iou.append(np.mean(ious))
        score_dice.append(np.mean(dices))

    test_score_iou = np.mean(score_iou)
    test_score_dice = np.mean(score_dice)
    print(f"Test Score:  IoU {test_score_iou:.5f}, Dice {test_score_dice:.5f}")

    wandb.log({"test_score_iou": test_score_iou, "test_score_dice": test_score_dice}, step=global_step)
    net.train()


def eval_net(net, loader, device, global_step, log_img=True):
    net.eval()
    tot, loss = 0, 0
    n_val = len(loader)
    with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
        for img, mask in loader:
            img = img.to(device=device)
            mask = mask.to(device=device)

            with torch.no_grad():
                pred = net(img)
                loss += criterion(pred, mask)

            if config["num_classes"] == 1:
                pred = torch.sigmoid(pred)

                # dice coeff
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, mask).item()

            if log_img:
                wandb.log({"val_images": wandb.Image(make_grid(pred).to(torch.float))}, step=global_step)
                log_img = not log_img

            pbar.update()

    net.train()
    return tot / n_val, loss / n_val


def train_net(
    net, device, config, optimizer, scheduler, global_step, current_epoch, lr=0.001, save_last_cp=False, save_cp=False
):
    # Dataset ----------------------------------------------------------------------------------
    train_dataset = BasicDataset(
        train_dir,
        config["img_height"],
        config["img_width"],
        num_classes=config["num_classes"],
        transform=eval(config["transforms"]),
    )
    val_dataset = BasicDataset(
        val_dir,
        config["img_height"],
        config["img_width"],
        num_classes=config["num_classes"],
        transform=eval(config["transforms"]),
    )
    n_val = len(val_dataset)
    n_train = len(train_dataset)

    source_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batchsize"], shuffle=True, num_workers=8, pin_memory=True, drop_last=True
    )
    source_val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batchsize"], shuffle=False, num_workers=8, pin_memory=True, drop_last=True
    )

    ### TRAIN
    epochs_no_improve = 0
    min_loss, min_epoch_loss = 1000, 1000
    for epoch in range(current_epoch, config["epochs"]):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{config["epochs"]}', unit="img") as pbar:
            for img, mask in source_train_loader:
                # learning rate warmup
                if config["warmup_steps"] > 0 and global_step <= config["warmup_steps"]:
                    lr = config["warmup_lr"] + (config["lr"] - config["warmup_lr"]) * float(
                        global_step / config["warmup_steps"]
                    )
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr

                optimizer.zero_grad()

                img = img.to(device)
                mask = mask.to(device)
                outputs = net(img)
                loss = criterion(outputs, mask)

                ##########
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

                wandb.log({"train_loss": loss.item()}, step=global_step)
                pbar.set_postfix(**{"loss (batch)": loss.item()})
                pbar.update(img.shape[0])
                global_step += 1

                if global_step % (n_train // (config["freq_validation_per_epoch"] * config["batchsize"])) == 0:
                    val_score, val_loss = eval_net(
                        net=net, loader=source_val_loader, device=device, global_step=global_step
                    )
                    wandb.log(
                        {"val_loss": val_loss, "val_score": val_score, "lr": optimizer.param_groups[0]["lr"]},
                        step=global_step,
                    )

            scheduler.step()

        print("Validation loss: {}".format(val_loss))

        if config["num_classes"] > 1:
            if epoch % config["test_every_n_epocochs"] == 0 and config["test_set"] is not None:
                benchmark_multiclass(net=net, device=device, global_step=global_step)

        state = {
            "num_classes": config["num_classes"],
            "epoch": epoch + 1,
            "step": global_step,
            "img_h": config["img_height"],
            "img_w": config["img_width"],
            "backbone": config["backbone"],
            "dataset_name": config["dataset_name"],
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        # EARLY STOPPING
        epoch_loss /= len(source_train_loader)
        print("Epoch loss: {}".format(epoch_loss))
        if epoch_loss < min_epoch_loss:
            epochs_no_improve = 0
            min_epoch_loss = epoch_loss
        else:
            epochs_no_improve += 1

        if epoch > config["early_stopping_min_epochs"] and epochs_no_improve == config["early_stopping_patience"]:
            print("Early Stopping!")
            break
        #################

        # SAVE BEST CHECKPOINT
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(state, os.path.join(checkpoint_dir, "CP_{}_{}.pth".format(DATASET_TYPE, wandb.run.name)))
            print("*** New min validation loss {}, checkpoint BEST saved!".format(val_loss))

        # SAVE LAST CHECKPOINT
        if save_last_cp:
            torch.save(state, os.path.join(checkpoint_dir, "CP_LAST_{}.pth".format(wandb.run.name)))
            print(f"Checkpoint {epoch + 1} saved !")

        # SAVE EPOCH CHECKPOINT
        if save_cp:
            torch.save(state, os.path.join(checkpoint_dir, "CP_Epoch{}_{}.pth".format(epoch + 1, wandb.run.name)))
            print(f"Checkpoint {epoch + 1} saved !")


if __name__ == "__main__":
    checkpoint_dir = "checkpoints"
    dataset_path = os.path.join(config["dataset_path"], config["dataset_name"])
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device ", device)

    print("model selected: ", MODEL_MAP[config["backbone"]])
    model = MODEL_MAP[config["backbone"]](
        num_classes=config["num_classes"], output_stride=16, pretrained_backbone=config["pretrained_backbone"]
    )
    network.convert_to_separable_conv(model.classifier)

    # set momentum
    for m in model.backbone.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = 0.01

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    if config["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)
    elif config["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10000)
    elif config["scheduler"] == "poly":
        scheduler = PolyLR(optimizer, config["epochs"], power=0.97, min_lr=1e-9)
    else:
        NotImplementedError

    if config["num_classes"] > 1:
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")

    if config["ckpt_pre_train"] is not None and os.path.isfile(config["ckpt_pre_train"]):
        checkpoint = torch.load(config["ckpt_pre_train"], map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        current_epoch = 0
        global_step = 0
        lr = config["lr"]
        model.to(device)

    elif config["ckpt_resume"] is not None and os.path.isfile(config["ckpt_resume"]):
        checkpoint = torch.load(config["ckpt_resume"], map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        current_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]
        lr = optimizer.param_groups[0]["lr"]

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        model.to(device)
        print(
            f"""Resume training from: 
            current epoch:   {current_epoch}
            global step:     {global_step}
        """
        )
    else:
        print("""[!] Retrain""")
        current_epoch = 0
        global_step = 0
        lr = config["lr"]
        model.to(device)

    print("Starting training:")
    for k, v in config.items():
        print(f"\t{k}:   {v}")
    print("")

    try:
        train_net(
            net=model,
            config=config,
            current_epoch=current_epoch,
            global_step=global_step,
            lr=lr,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
