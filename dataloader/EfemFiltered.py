import torch.utils.data as data
import pandas as pd
from PIL import Image


class EfemFiltered(data.Dataset):
    def __init__(self, split='train', transform=None):
        self.transform = transform
        self.split = split

        if self.split == "train":
            self.data = pd.read_csv("")

        elif self.split == "val":
            self.data = pd.read_csv("")

        elif self.split == "test":
            self.data = pd.read_csv("C:\\Users\\Alfonso\\Desktop\\SistemaAgenti\\Datasets\\Dataset Filtrato\\images.csv")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datasetpath = "C:\\Users\\Alfonso\\Desktop\\SistemaAgenti\\Datasets\\Dataset Filtrato\\"
        # TODO: leggere la label associata al file audio nel CSV
        label = self.data.loc[idx, "emozione"]
        label = int(label)

        # TODO: leggere il percorso del file audio dal CSV
        img_name = self.data.loc[idx, "nome file"]

        # TODO: creare lo spettrogramma e salvarlo nell'oggetto img

        img = Image.open(datasetpath + img_name)

        if self.transform is not None:
            img = self.transform(img)

        return {'image': img, 'label': label}


if __name__ == "__main__":

    split = "train"
    demos_train = EfemFiltered(split=split)
    print("Demos {} set loaded".format(split))
    print("{} samples".format(len(demos_train)))

    for i in range(3):
        print(demos_train[i]["label"])
        demos_train[i]["image"].show()

