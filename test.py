import argparse
import logging
import os
import platform
import torch

if platform.system() == "Linux":
    import shutil

import warnings
from torchvision import transforms
from tqdm import tqdm

from dataloader.EfemFiltered import EfemFiltered
from dataloader.faces import Faces
from models.bam.vggface2_bam import VGGFace2BAM
from models.cbam.vggface2_cbam import VGGFace2CBAM
from models.resnet50.vggface2 import VGGFace2
from models.se.vggface2_se import VGGFace2SE
from utility.checkpoint import load_model
import torch.nn.functional as F
from utility.confusion_matrix import show_confusion_matrix, get_classification_report
import cv2
from PIL import Image




DEFAULT_MODEL = ".\\result\\no\\Emozioni_aniziani_best_model.pt"
warnings.filterwarnings('ignore')
logger = logging.getLogger('mnist_AutoML')

classes = 6
# TODO: imagenet
data_mean = [0.485, 0.456, 0.406]
data_std = [0.229, 0.224, 0.225]

val_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=data_mean, std=data_std),
])

if (torch.cuda.is_available()):
    device = torch.device("cuda")
    print("===================================================")
    print('Cuda available: {}'.format(torch.cuda.is_available()))
    print("GPU: " + torch.cuda.get_device_name(torch.cuda.current_device()))
    print("Total memory: {:.1f} GB".format((float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)))))
    print("===================================================")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print('Cuda not available. Using CPU.')

model = VGGFace2(pretrained=False, classes=classes).to(device)
model = load_model(DEFAULT_MODEL, model, device)
model.eval()

def startanalysis(imgpath):
    global model
    global val_preprocess
    label_mapping = {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "happiness",
        4: "neutral",
        5: "sadness"
    }
    images = cv2.imread(imgpath)
    images = Image.fromarray(images)
    images = val_preprocess(images)
    images = images.unsqueeze(0)
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
        output = F.softmax(output, dim=1)
        valori = output[0].data.numpy().tolist()

        percentuali = []
        for val in valori:
            try:
                float(val)
                percentuali.append(round(float(val) * 100, 3))
            except:
                continue


        emozione = label_mapping[output.data.cpu().numpy().argmax()]
        datas = emozione + "|"
        for p in percentuali:
            datas += str(p) + "|"
        return datas


def normal():
    parser = argparse.ArgumentParser(description="Configuration validation phase")
    parser.add_argument("-a", "--attention", type=str, default="no", choices=["no", "se", "bam", "cbam"],help='Chose the attention module')
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help='Batch size to use for training')
    parser.add_argument("-d", "--dataset", type=str, default="faces", choices=["faces", "efem"], help='Chose the dataset')
    parser.add_argument("-s", "--stats", type=str, default="imagenet", choices=["no", "imagenet"],help='Chose the mean and standard deviation')
    parser.add_argument("-ta", "--validation", type=str, default="Arousal", choices=["emotion_class", "Valenza", "Arousal"], help='Select the type of target')
    parser.add_argument("-m", "--loadModel", type=str, default=DEFAULT_MODEL, choices=[], help='Select the type of target')


    args = parser.parse_args()


    print("Starting validation with the following configuration:")
    print("Attention module: {}".format(args.attention))
    print("Batch size: {}".format(args.batch_size))
    print("Dataset: {}".format(args.dataset))
    print("Validation: {}".format(args.validation))
    print("Load model: {}".format(args.loadModel))
    print("Stats: {}".format(args.stats))

    #TODO: Non dovrebbero servire. Controllare.
    #print("Uses Drive: {}".format(args["uses_drive"]))
    #print("With Augmentation: {}".format(args["withAugmentation"]))
    #print("Workers: {}".format(args['workers']))
    #print("Checkpoint model: {}".format(args.checkpoint))

    if platform.system() == "Linux" and args['uses_drive']:
        print("----------------------------")
        print("** Google Drive Sign In **")
        if not(os.path.exists("../../gdrive/")):
            print("No Google Drive path detected! Please mount it before running this script or disable ""uses_drive"" flag!")
            print("----------------------------")
            exit(0)
        else:
            print("** Successfully logged in! **")
            print("----------------------------")

        if not(os.path.exists("../../gdrive/MyDrive/SysAg2022/{}/{}/{}".format(args["dataset"], args["attention"], args["gender"]))):
           os.makedirs("../../gdrive/MyDrive/SysAg2022/{}/{}/{}".format(args["dataset"], args["attention"], args["gender"]))

        if not(os.path.exists("../../gdrive/MyDrive/SysAg2022/{}/{}/{}".format(args["dataset"], args["attention"], args["gender"]))):
           os.makedirs("../../gdrive/MyDrive/SysAg2022/{}/{}/{}".format(args["dataset"], args["attention"], args["gender"]))

    if not(os.path.exists(os.path.join("result", args.dataset, args.attention))):
           os.makedirs(os.path.join("result", args.dataset, args.attention))

    if(torch.cuda.is_available()):
        device = torch.device("cuda")
        print("===================================================")
        print('Cuda available: {}'.format(torch.cuda.is_available()))
        print("GPU: " + torch.cuda.get_device_name(torch.cuda.current_device()))
        print("Total memory: {:.1f} GB".format((float(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)))))
        print("===================================================")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print('Cuda not available. Using CPU.')

    if args.stats == "imagenet":
        # imagenet
        data_mean = [0.485, 0.456, 0.406]
        data_std = [0.229, 0.224, 0.225]

        val_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_mean, std=data_std),
        ])
    else:
        val_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    #TODO: Dataset Faces
    classes = 6
    if args.dataset == "faces":
        val_data = Faces(split="test", transform=val_preprocess)
        label_mapping = {
            0: "anger",
            1: "disgust",
            2: "fear",
            3: "happiness",
            4: "neutral",
            5: "sadness"
        }
    #TODO: Dataset efem Ergo "dataset di Arianna"
    elif args.dataset == "efem":
        val_data = EfemFiltered(split="test", transform=val_preprocess)
        label_mapping = {
            0: "anger",
            1: "disgust",
            2: "fear",
            3: "happiness",
            4: "neutral",
            5: "sadness"
        }

    if args.attention == "no":
        model = VGGFace2(pretrained=False, classes=classes).to(device)
    elif args.attention == "se":
        model = VGGFace2SE(classes=classes).to(device)
    elif args.attention == "bam":
        model = VGGFace2BAM(classes=classes).to(device)
    elif args.attention == "cbam":
        model = VGGFace2CBAM(classes=classes).to(device)
    else:
        model = VGGFace2(pretrained=False, classes=classes).to(device)

    if args.loadModel:
        print("You specified a pre-loading directory for a model.")
        print("The directory is: {}".format(args.loadModel))
        if os.path.isfile(args.loadModel):
            print("=> Loading model '{}'".format(args.loadModel))
            model = load_model(args.loadModel, model, device)
            print("Custom model loaded successfully")
        else:
            print("=> No model found at '{}'".format(args.loadModel))
            print("Are you sure the directory / model exist?")
            exit(0)
        print("-------------------------------------------------------")
    else:
        print("Default {} model loaded.".format(args.attention))

    # validate the model
    model.eval()




    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

    y_true = []
    y_pred = []
    labels_list = []

    val_correct = 0

    print("\nStarting validation...")
    for batch_idx, batch in enumerate(tqdm(val_loader)):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            val_outputs = model(images)
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += torch.sum(val_preds == labels.data)

            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(val_preds.detach().cpu().numpy().tolist())

    for i in range(len(label_mapping)):
        labels_list.append(label_mapping[i])

    print("Num correct: {}".format(val_correct))
    print("Num samples: {}".format(len(val_data)))

    val_acc = (val_correct.double() / len(val_data)) * 100

    delimiter = "\n===================================================================================\n"
    write = F'Accuracy of the network on the test images: {val_acc:.3f}%'
    print(F'\n{write}')

    classificationReport = get_classification_report(y_true, y_pred, labels_list)
    print(classificationReport)

    # Scriviamo i risultati in un file testuale

    f = open("result/{}/{}/res_validation_{}_{}.txt".format(args.dataset, args.attention, args.dataset, args.attention), "w")

    f.write(write)
    f.write(delimiter)
    f.write("Modello utilizzato: {}".format(args.loadModel))
    f.write(delimiter)
    f.write("val_correct: {val_correct} || val_samples: {val_samples}".format(
        val_correct = val_correct,
        val_samples = len(val_data)))
    f.write(delimiter)
    f.write(classificationReport)
    f.close()

    show_confusion_matrix(y_true, y_pred, labels_list, "result/{}/{}/".format(args.dataset, args.attention))

    if platform.system() == "Linux" and args['uses_drive']:
        shutil.copy("result/{}/{}/{}/res_validation_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]), "../../gdrive/MyDrive/SysAg2022/{}/{}/{}/res_validation_{}_{}_{}.txt".format(args["dataset"], args["attention"], args["gender"], args["attention"], args["dataset"], args["gender"]))
        shutil.copy("result/{}/{}/{}/confusion_matrix.png".format(args["dataset"], args["attention"], args["gender"]), "../../gdrive/MyDrive/SysAg2022/{}/{}/{}/confusion_matrix.png".format(args["dataset"], args["attention"], args["gender"]))

    print("===================================Testing Finished===================================")






