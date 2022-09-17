import os
import time

import torch
from model import UNet
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import myDataset
import transforms


def train_test_transforms(train=True, base_size=565, crop_size=480, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

    min_size = int(0.5 * base_size)
    max_size = int(1.2 * base_size)

    if train:
        train_transforms=transforms.Compose([
            transforms.RandomResize(min_size=min_size, max_size=max_size),
            transforms.RandomHorizontalFlip(flip_prob=hflip_prob),
            transforms.RandomVerticalFlip(flip_prob=vflip_prob),
            transforms.RandomCrop(size=crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return train_transforms
    else:
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return test_transforms


def create_model(num_classes):
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Used {} device'.format(device))

    # 用来保存训练以及验证过程中信息
    results_file = "./results.txt"

    data_path = './'
    # 计算得到的mean std
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    train_dataset = myDataset(data_path,
                              train=True,
                              transforms=train_test_transforms(train=True, base_size=565, crop_size=480, hflip_prob=0.5,
                                                               vflip_prob=0.5,
                                                               mean=mean, std=std))

    val_dataset = myDataset(data_path,
                            train=False,
                            transforms=train_test_transforms(train=False, mean=mean, std=std))

    batch_size = 10
    num_classes = 1 + 1  # (nun_classes + background)
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('nums_workers:{}'.format(num_workers))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes)
    model.to(device)

    epochs = 20
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,lr=0.01, momentum=0.9, weight_decay=1e-4)

    # 学习率 warmup
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs, warmup=True)

    best_dice = 0.   # 评估两个集合的相似度
    start_time = time.time()
    for epoch in range(epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=10)

        confmat, dice = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        if not os.path.exists('./save_weights'):
            os.mkdir('./save_weights')

        if best_dice < dice:
            best_dice = dice

            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": lr_scheduler.state_dict(),
                         "epoch": epoch}

            torch.save(save_file, "save_weights/best_model.pth")

    total_time = time.time() - start_time
    print("training time {}".format(total_time))



if __name__ == '__main__':
    main()
