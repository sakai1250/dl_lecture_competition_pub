import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.models import BasicConvClassifier
from src.utils import set_seed, LabelSmoothingCrossEntropy

@hydra.main(version_base=None, config_path="configs", config_name="config_pretrain")
def run(args: DictConfig):
    torch.manual_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f'save_dir: {os.path.join(logdir, "model_best.pt")}')
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="Image-classification")

    # ------------------
    #    Dataloader
    # ------------------
    transform = transforms.Compose([
        transforms.Resize((271, 281)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomErasing(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(lambda x: torch.reshape(x, (271, 281)))
    ])

    trainval_dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    n_samples = len(trainval_dataset) # n_samples is 60000
    train_size = int(len(trainval_dataset) * 0.8) # train_size is 48000
    val_size = n_samples - train_size # val_size is 48000

    # shuffleしてから分割.
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ------------------
    #       Model
    # ------------------
    model = BasicConvClassifier(num_classes=len(trainval_dataset.classes), seq_len=trainval_dataset[0][0].shape[1], in_channels=trainval_dataset[0][0].shape[0]).to(args.device)
    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    cross_entropy = LabelSmoothingCrossEntropy()

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(task="multiclass", num_classes=len(trainval_dataset.classes), top_k=1).to(args.device)
      
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for X, y in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X)

            loss = cross_entropy(y_pred, y)
            # loss = F.cross_entropy(y_pred, y) + 0.005 * l1_reg
            # loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            
    
    # # ----------------------------------
    # #  Start evaluation with best model
    # # ----------------------------------
    # model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    # preds = [] 
    # model.eval()
    # for X, _ in tqdm(test_loader, desc="Test"):        
    #     preds.append(model(X.to(args.device)).detach().cpu())
        
    # preds = torch.cat(preds, dim=0).numpy()
    # np.save(os.path.join(logdir, "submission"), preds)
    # cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

if __name__ == "__main__":
    run()