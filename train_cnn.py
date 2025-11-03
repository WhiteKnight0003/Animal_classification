from dataset import AnimalDataset
from model import EfficientNetClassifier
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Compose, RandomAffine, ColorJitter
import torch
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import os
import numpy as np
import shutil
from torchsummary import summary 
from tqdm.autonotebook import tqdm 
from torch.utils.tensorboard import SummaryWriter 

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(45,45))
    
    plt.imshow(cm, interpolation='nearest', cmap="ocean") 
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def get_args():
    parser = ArgumentParser(description="CNN training")
    parser.add_argument("--epochs","-e", type = int, default=100, help ="Number of epochs" )
    parser.add_argument("--batch_size", "-b",type = int, default=64, help ="batch size" ) 
    parser.add_argument("--image_size","-i", type = int , default= 224, help = "image size")
    parser.add_argument("--root","-r", type = str , default= './data', help = "Root")
    parser.add_argument("--logging","-l", type = str , default= './tensorboard_file')  
    parser.add_argument('--trained_models', '-tr', type=str, default='./trained_models')
    
    parser.add_argument('--checkpoint', '-c', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ =='__main__':
    args = get_args() #  

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("cuda")
    else:
        device = torch.device("cpu")
        print("cpu")
    print(args.batch_size)
    print(args.epochs)

    train_transforms = Compose([ 
        RandomAffine(
            degrees=(-10,10), # xoay 
            translate=(0.1,0.1), # dịch
            scale=(0.9,1.1), # zoom in , zoom out
            shear=10 # xiên
        ),

        # thay đổi màu sắc 
        ColorJitter(
            brightness=0.15,
            contrast=0.4, # độ tương phản 
            saturation= 0.3 , # độ bão hòa
            hue=0.05, # độ nhòe
        ),

        Resize((args.image_size, args.image_size)), 
        ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transforms = Compose([ 
        Resize((args.image_size, args.image_size)), 
        ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    images_list = []
    labels_list = []

    class_folders = sorted([d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_folders)}
    idx_to_class = {i: cls_name for cls_name, i in class_to_idx.items()}

    for class_name in class_folders:
        class_dir = os.path.join(args.root, class_name)
        label = class_to_idx[class_name]
        
        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith(('.png','.jpg','.jpeg')):
                file_path = os.path.join(class_dir, file_name)
                try:
                    image = Image.open(file_path).convert('RGB')
                    images_list.append(image)
                    labels_list.append(label) 
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    train_images, test_images, train_labels, test_labels = train_test_split(
        images_list, labels_list,
        test_size=0.2,       
        random_state=42,
        stratify=labels_list
    )

    train_dataset = AnimalDataset(
        images=train_images, 
        labels=train_labels, 
        transforms=train_transforms
    )

    test_dataset = AnimalDataset(
        images=test_images, 
        labels=test_labels, 
        transforms=test_transforms
    )

    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size=args.batch_size,
        shuffle = True,
        num_workers=8, 
        drop_last=True,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_dataloader = DataLoader( 
        dataset = test_dataset,
        batch_size=args.batch_size,
        shuffle = False,
        num_workers=8,
        drop_last=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging) # xóa hết cả thư mục
    if not os.path.isdir(args.trained_models):
        os.mkdir(args.trained_models) # chưa tồn tại thì tạo

    writer = SummaryWriter(args.logging)


    model = EfficientNetClassifier(num_classes=len(class_folders)).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3, momentum =0.9)

    # nếu model chưa tồn tại thì train từ đầu , nếu đang train dở thì train tiếp
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_acc"]
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else: 
        start_epoch=0
        best_accuracy =0

    num_iter = len(train_dataloader)
    
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour='green') # chọn màu cho thanh tiến trình
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss_value = criterion(outputs, labels)

            progress_bar.set_description("Epoch {}/{}.  Iteration {}/{} . Loss {:.3f}".format(epoch+1, args.epochs, iter+1, num_iter, loss_value)) 

            writer.add_scalar('Train/Loss', loss_value, epoch*num_iter+iter) 

            # Backward pass
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []

        for iter, (images, labels) in enumerate(test_dataloader):
            all_labels.extend(labels)
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                prediction = model(images)
                indices = torch.argmax(prediction.cpu(), dim=1)
                all_predictions.extend(indices)
                loss_value = criterion(prediction, labels)

        all_labels = [label.item() for label in all_labels]
        all_predictions = [pred.item() for pred in all_predictions]
        # Print classification report

        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), class_names=class_to_idx.keys(), epoch=epoch)
        accuracy = accuracy_score(all_labels, all_predictions)
        print("Epoch {}   Accurcy:  {}".format(epoch+1,accuracy))
        
        writer.add_scalar('Val/Accuracy', accuracy,epoch) 

        checkpoint ={
            "epoch": epoch+1, 
            "model": model.state_dict(),
            "best_acc": best_accuracy,
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))

        if accuracy > best_accuracy:
            checkpoint ={
            "epoch": epoch+1, 
            "model": model.state_dict(),
            "best_acc": best_accuracy,
            "optimizer": optimizer.state_dict()
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))
            best_accuracy = accuracy
