import argparse

from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm, notebook

from utility.checkpoint import *
from models.resnet50.vggface2 import VGGFace2
from models.se.vggface2_se import VGGFace2SE
from models.bam.vggface2_bam import VGGFace2BAM
from models.cbam.vggface2_cbam import VGGFace2CBAM
from dataloader.faces import Faces


parser = argparse.ArgumentParser(description="Configuration train phase")
parser.add_argument("-a", "--attention", type=str, default="no", choices=["no", "se", "bam", "cbam"], help='Chose the attention module')
parser.add_argument("-bs", "--batch_size", type=int, default=64, help='Batch size to use for training')
parser.add_argument("-o", "--optimizer", type=str, default="adam", choices=["adam", "sgd"], help='Chose the optimizer')
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help='Learning rate to use for training')
parser.add_argument("-e", "--epochs", type=int, default=100, help='Number of epochs')
parser.add_argument("-p", "--patience", type=int, default=10, help='Number of epochs without improvements before reducing the learning rate')
parser.add_argument("-wd", "--weight_decay", type=float, default=0, help='Value of weight decay')
parser.add_argument("-nm", "--momentum", type=float, default=0, help='Value of momentum')
parser.add_argument("-m", "--monitor", type=str, default="acc", choices=["acc", "loss"], help='Chose to monitor the validation accuracy or loss')
parser.add_argument("-d", "--dataset", type=str, default="faces", choices=["demos", "emovo", "demosemovo"], help='Chose the dataset')
parser.add_argument("-cw", "--class_weights", type=bool, default=False, help='Use the class weights in loss function')
parser.add_argument("-s", "--stats", type=str, default="no", choices=["no", "imagenet"], help='Chose the mean and standard deviation')
args = parser.parse_args()

print("Starting training with the following configuration:")
print("Attention module: {}".format(args.attention))
print("Batch size: {}".format(args.batch_size))
print("Optimizer: {}".format(args.optimizer))
print("Learning rate: {}".format(args.learning_rate))
print("Epochs: {}".format(args.epochs))
print("Patience: {}".format(args.patience))
print("Weight decay: {}".format(args.weight_decay))
print("Momentum: {}".format(args.momentum))
print("Class weights: {}".format(args.class_weights))
print("Metric to monitor: {}".format(args.monitor))
print("Dataset: {}".format(args.dataset))
print("Stats: {}".format(args.stats))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.stats == "imagenet":
    # imagenet
    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]

    train_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std),
    ])

    val_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_mean, std=data_std),
    ])
else:
    train_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    val_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

if args.dataset == "faces":
    train_data = Faces(split="train", transform=train_preprocess)
    val_data = Faces(split="val", transform=val_preprocess)
    classes = 6

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

if args.attention == "no":
    model = VGGFace2(pretrained=True, classes=classes).to(device)
elif args.attention == "se":
    model = VGGFace2SE(classes=classes).to(device)
elif args.attention == "bam":
    model = VGGFace2BAM(classes=classes).to(device)
elif args.attention == "cbam":
    model = VGGFace2CBAM(classes=classes).to(device)
else:
    model = VGGFace2(pretrained=False, classes=classes).to(device)

print("Model archticture: ", model)

start_epoch = 0

best_val_loss = 1000000
best_val_acc = 0

criterion = nn.CrossEntropyLoss()

if args.optimizer == "sgd":
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                          momentum=args.momentum, weight_decay=args.weight_decay)
else:
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate,
                           weight_decay=args.weight_decay)

if args.monitor == "loss":
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, verbose=True)
else:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, mode="max", verbose=True)

print("===================================Start Training===================================")

for e in range(start_epoch, args.epochs):
    train_loss = 0
    validation_loss = 0
    train_correct = 0
    val_correct = 0
    is_best = False

    batch_bar = tqdm(total=len(train_loader), desc="Batch", position=0)

    # train the model
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        images, labels = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)
        batch_bar.update(1)

    train_loss = train_loss / len(train_loader)
    train_acc = train_correct.double() / len(train_data)

    # validate the model
    model.eval()
    for batch_idx, batch in enumerate(val_loader):
        images, labels = batch["image"].to(device), batch["label"].to(device)

        with torch.no_grad():
            val_outputs = model(images)
            val_loss = criterion(val_outputs, labels)
            validation_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += torch.sum(val_preds == labels.data)

    validation_loss = validation_loss / len(val_loader)
    val_acc = val_correct.double() / len(val_data)

    if args.monitor == "loss":
        scheduler.step(validation_loss)

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            is_best = True
    else:
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            is_best = True

    checkpoint = {
        'epoch': e + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    save_checkpoint(checkpoint, is_best, "./result/{}/checkpoint".format(args.attention), "./result/{}".format(args.attention))

    if is_best:
        print(
            '\nEpoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Accuracy {:.3f}% \tValidation Accuracy {:.3f}% \t[saved]'
            .format(e + 1, train_loss, validation_loss, train_acc * 100, val_acc * 100))
    else:
        print(
            '\nEpoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Accuracy {:.3f}% \tValidation Accuracy {:.3f}%'
            .format(e + 1, train_loss, validation_loss, train_acc * 100, val_acc * 100))

print("===================================Training Finished===================================")
