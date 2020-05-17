import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *


def train():
    torch.cuda.empty_cache()

    epochs = 100
    batch_size = 8
    model_def = "config/yolov3-tiny.cfg"
    data_config = "config/custom.data"
    pretrained_weights = None

    img_size = 416
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)

    # Get data configuration
    data_config = parse_data_config(data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    print(train_path)
    print(valid_path)
    print(class_names)
    # Initiate model
    model = Darknet(model_def).to(device)
    model.apply(weights_init_normal)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True)

    dataloader = torch.utils.data.DataLoader(dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=dataset.collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters())

    b = len(dataloader)
    min_loss = 1000000
    for epoch in range(epochs):
        model.train()
        print(epoch)
        total_loss = 0
        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc = "Epoch# " + str(epoch))):
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
            loss, outputs = model(imgs, targets)
            total_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        ls = total_loss/b
        if ((ls < min_loss) and (epoch % 20 == 0) and (epoch > 20)):
            min_loss = ls
            torch.save(model.state_dict(), "small" + str(int(ls)) +".pth")

    

train()