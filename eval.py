import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier, Classifier
from src.utils import set_seed


@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = 'src'
    print(savedir)
    # ------------------
    #    Dataloader
    # ------------------    
    train_set = ThingsMEGDataset("train", args.data_dir)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ------------------
    #       Model
    # ------------------
    len_ids = train_set.get_len_ids()
    model = Classifier(num_classes=test_set.num_classes, seq_len=test_set.seq_len, len_ids=len_ids, in_channels=271).to(args.device)
    model.load_state_dict(torch.load('model_best_main.pt', map_location=args.device))

    # ------------------
    #  Start evaluation
    # ------------------ 
    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):
        X, subject_idxs = X.to(args.device), subject_idxs.to(args.device)  
        preds.append(model(X, subject_idxs).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(savedir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")


if __name__ == "__main__":
    run()