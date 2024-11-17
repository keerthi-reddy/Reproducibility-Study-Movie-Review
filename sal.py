import argparse
import os
import pickle
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import logging
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import tempfile
import shutil
import random
import numpy as np
from torch import nn, Tensor
import math
from torch.optim import AdamW
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score
import warnings

warnings.filterwarnings('always')

class SaliencyDataset(Dataset):

    def __init__(self, split):

        if split == "train":
            self.file_path = "./outputs/scene_classification_data/train.pkl"
        elif split == "test":
            self.file_path = "./outputs/scene_classification_data/test.pkl"
        elif split == "val":
            self.file_path = "./outputs/scene_classification_data/val.pkl"

        with open(self.file_path, "rb") as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)  

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first: bool = False):
        """
        Initialize the PositionalEncoding module.

        Parameters:
        - d_model (int): The dimension of the model (embedding size).
        - dropout (float): The dropout rate to apply to positional encoding.
        - max_len (int): The maximum sequence length.
        - batch_first (bool): Whether the input tensor follows batch-first format (True) or seq-first format (False).
        """

        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)  # Shape: (max_len, 1)
        
        scale_factor = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # Shape: (d_model // 2)
        
        permutation = torch.zeros(max_len, 1, d_model)  # Shape: (max_len, 1, d_model)
        permutation[:, 0, 0::2] = torch.sin(position * scale_factor)
        permutation[:, 0, 1::2] = torch.cos(position * scale_factor)

        if self.batch_first:
            permutation = permutation.permute(1, 0, 2)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('permutation', permutation)  # Shape: (1, max_len, d_model) or (batch_size, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Adds positional encoding to the input tensor x.

        Parameters:
        - x (Tensor): Input tensor of shape (batch_size, seq_len, d_model) if batch_first=True, or 
                      (seq_len, batch_size, d_model) if batch_first=False.

        Returns:
        - Tensor: The input tensor x with added positional encoding.
        """
        if self.batch_first:
            x = x + self.permutation[:, :x.size(1), :]  # Shape of x: (batch_size, seq_len, d_model)
        else:
            x = x + self.permutation[:x.size(0)]  # Shape of x: (seq_len, batch_size, d_model)

        return self.dropout(x)

class SceneSaliency(nn.Module):
    def __init__(self, input_dimensions, nhead, num_layers, output_dim, position_dropout=0.1):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(input_dimensions, position_dropout, batch_first=True)
        self.encoder_layer = TransformerEncoderLayer(d_model=input_dimensions, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(input_dimensions, output_dim)

    def forward(self, scene_embeddings, embed_mask):
        return self.linear(self.transformer_encoder(self.pos_encoder(scene_embeddings), src_key_padding_mask=embed_mask))
    
def collate_batch(batch):
    max_len = -1
    for data in batch:
        if max_len < data["scenes_embeddings"].shape[0]:
            max_len = data["scenes_embeddings"].shape[0]

    scenes = torch.zeros(len(batch), max_len, batch[0]["scenes_embeddings"].shape[-1])
    mask = torch.ones(len(batch), max_len, dtype=torch.bool)
    labels = torch.ones(len(batch), max_len) * -1

    for idx, data in enumerate(batch):
        _embedding = data["scenes_embeddings"]
        scenes[idx, :len(_embedding), :] = _embedding
        mask[idx, :len(_embedding)] = torch.zeros(len(_embedding), dtype=torch.bool)
        labels[idx, :len(_embedding)] = data["labels"]

    return scenes.to(device), mask.to(device), labels.to(device)

def log_metrics(step, metrics):
    logger.info(f"Step {step}: {metrics}")

def logTest(metrics):
    logger.info(f" Final test metrics: {metrics}")

def checkpoint(epoch, model, optimizer, scheduler, directory, filename='checkpoint.pt', max_checkpoints=5):
    # Ensure the directory exists
    print(directory)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    checkpoint_path = os.path.join(directory, filename)
    
    if os.path.exists(checkpoint_path):
        base_name, ext = os.path.splitext(checkpoint_path)

        for i in range(max_checkpoints - 1, 0, -1):
            prev_path = f'{base_name}{i}{ext}'
            if os.path.exists(prev_path):
                os.rename(prev_path, f'{base_name}{i + 1}{ext}')
        
        os.rename(checkpoint_path, f'{base_name}1{ext}')

    torch.save(state, checkpoint_path)

    return checkpoint_path

def save_checkpoint(epoch, model, optimizer, checkpoint_path, scheduler=None, best=False):
    checkpoint_path = checkpoint(epoch, model, optimizer, scheduler, checkpoint_path, max_checkpoints=3)

    if best:
        dirname = os.path.dirname(checkpoint_path)
        basename = os.path.basename(checkpoint_path)
        best_checkpoint_path = os.path.join(dirname, f'best_{basename}')
        shutil.copy2(checkpoint_path, best_checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path, scheduler=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def compute_metrics(pred, target):
    pred = pred.long().detach().cpu()
    target = target.long().detach().cpu()

    precision_score, recall, f1_score, support = precision_recall_fscore_support(target, pred, average='macro')

    accuracy = balanced_accuracy_score(target, pred)

    return {'accuracy': accuracy, 'f1_score': f1_score, 'precision_score': precision_score, 'recall': recall}

def averageScores(metrics):
    accuracy_scores, f1_scores, precision_scores, recall_scores = [], [], [], []
    for idx in metrics:
        accuracy_scores.append(idx['accuracy'])
        f1_scores.append(idx['f1_score'])
        precision_scores.append(idx['precision_score'])
        recall_scores.append(idx['recall'])

    mean_accuracy = np.mean(accuracy_scores)
    mean_f1_score = np.mean(f1_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)

    return {"Average_accuracy":mean_accuracy,"Average_f1_score":mean_f1_score,"Average_precision":mean_precision,"Average_recall":mean_recall}

def train_loop(model, optimizer, criterion, dataloader):
    model.train()
    total_loss = 0
    metrics = []

    for scenes, mask, labels_padded in dataloader:
        # Move data to device
        scenes, mask, labels_padded = scenes.to(device), mask.to(device), labels_padded.to(device)

        # Forward pass
        output_padded = model(scenes, mask)
        
        # Compute masked loss
        loss_mask = ~mask
        #loss = torch.masked_select(criterion(output_padded.squeeze(-1), labels_padded), loss_mask).mean()
        loss_padded = criterion(output_padded.squeeze(-1), labels_padded)

        loss_unpadded = torch.masked_select(loss_padded, loss_mask)
        loss = loss_unpadded.mean()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Predictions and accuracy calculation
        output = torch.masked_select(output_padded.squeeze(-1), loss_mask)
        pred = torch.sigmoid(output) > 0.5
        target = torch.masked_select(labels_padded, loss_mask)

        metrics.append(compute_metrics(pred, target))

    # Return average loss and accuracy metrics
    return total_loss / len(dataloader), metrics

def validation_loop(model, optimizer, criterion, dataloader):
    model.eval()
    total_loss = 0
    metrics = []

    with torch.no_grad():
        for scenes, mask, labels_padded in dataloader:
            # Move data to device
            scenes, mask, labels_padded = scenes.to(device), mask.to(device), labels_padded.to(device)

            # Forward pass
            output_padded = model(scenes, mask)
            
            # Compute masked loss
            loss_mask = ~mask
            loss = torch.masked_select(criterion(output_padded.squeeze(-1), labels_padded), loss_mask).mean()
            total_loss += loss.item()

            # Predictions and accuracy calculation
            output = torch.masked_select(output_padded.squeeze(-1), loss_mask)
            pred = torch.sigmoid(output) > 0.5
            target = torch.masked_select(labels_padded, loss_mask)

            metrics.append(compute_metrics(pred, target))

    return total_loss / len(dataloader), metrics

def train_model(model, optimizer, criterion, args, trainLoader, valLoader):
    progress_bar = tqdm(range(args.epochs))
    best_val_acc, val_avg_metr = -99, 0
    for epoch in range(args.epochs):

        train_loss, train_metrics = train_loop(model, optimizer, criterion, trainLoader)
        #logger.info(f"Step {epoch}: {{'train_loss': train_loss, 'epochs': epoch}}")
        log_metrics(epoch, {'train_loss': train_loss, 'epochs': epoch})

        val_loss, val_metrics = validation_loop(model, optimizer, criterion, valLoader)
        val_average_metric = averageScores(val_metrics)
        log_metrics(epoch, {'val_loss': val_loss, 'epochs': epoch})
        log_metrics(epoch, val_average_metric)
        val_avg_metr = val_average_metric['Average_f1_score']
        
        print(args.checkpoint_path)

        if val_avg_metr>best_val_acc:
            best_val_acc = val_avg_metr
            logger.info(f'saving checkpoint after improving')
            save_checkpoint(epoch, model, optimizer, args.checkpoint_path, scheduler=None, best=True)

        if epoch % 5 == 0:
            logger.info(f'Saving checkpoint at epoch:{epoch}')
            save_checkpoint(epoch, model, optimizer, args.checkpoint_path, scheduler=None, best=False)

        progress_bar.update(1)

    logger.info(f'End of training')

def get_positive_weight(train_dataset):
    # Concatenate all labels into a single list
    labels = [label for data in train_dataset for label in data["labels"]]

    ones = sum(labels)
    zeros = len(labels) - ones

    positive_weight = torch.FloatTensor([zeros / ones]).to(device)
    return positive_weight

def logTest(metrics):
    logger.info(f" Final test metrics: {metrics}")


def load_model(args):
    model = SceneSaliency(input_dimensions=args.input_dimensions, nhead=args.num_head, num_layers=args.num_layers,
                          output_dim=args.output_dim)
    return model

def generate_data(model, dataloader):
    model.eval()
    dataset = []

    with torch.no_grad():
        for scenes, mask, _ in dataloader:
            scenes, mask = scenes.to(device), mask.to(device)
            predictions = model(scenes, mask).squeeze(-1)
            pred_labels = (torch.masked_select(predictions, ~mask) > 0.5).int()

            dataset.append({"prediction_labels": pred_labels})
    return dataset

def savePickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def loadPickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def prepare_data_for_summarization(pred_data, script_data):
    return [
        {
            "script": " ".join(script_data[i]["scenes"][idx] for idx, label in enumerate(pred_data[i]["prediction_labels"].cpu()) if label == 1),
            "summary": script_data[i]["summary"]
        }
        for i in range(len(script_data))
    ]

if __name__ == '__main__':
    
    argument_parser = argparse.ArgumentParser(description="summarization")
    # General settings

    argument_parser.add_argument("--checkpoint_path", type=str, default="./outputs/scene_classification_checkpoints/", help="Path to save checkpoints")
    
    argument_parser.add_argument("--data_output", type=str, default="./outputs/training_data_using_prediction/", help="Path to save pickles for summarization model")

    # Training parameters
    argument_parser.add_argument("--lr", type=float, default=8e-5, help="Learning rate")
    argument_parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    argument_parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    argument_parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    argument_parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classifier predictions")

    # Transformer model parameters
    argument_parser.add_argument("--input_dimensions", type=int, default=1024, help="Transformer input dimensions")
    argument_parser.add_argument("--num_head", type=int, default=16, help="Number of heads in the Transformer")
    argument_parser.add_argument("--num_layers", type=int, default=10, help="Number of layers in the Transformer")
    argument_parser.add_argument("--output_dim", type=int, default=1, help="Transformer output dimension")

    # Experiments
    # argument_parser.add_argument("--lr_scheduler", type=str, default="linear", help="Learning rate scheduler type")
    # argument_parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps for scheduler")
    # argument_parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for regularization")
    # argument_parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    # argument_parser.add_argument("--save_best_only", action="store_true", help="Save only the best model checkpoint")
    # argument_parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
    # argument_parser.add_argument("--augmentation_prob", type=float, default=0.5, help="Probability of data augmentation")
    # argument_parser.add_argument("--model_variant", type=str, default="base", help="Choose model variant")
    # argument_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = argument_parser.parse_args()
    
    seed_value = 1234
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    if not os.path.exists("./outputs/logs/classification/"):
        os.makedirs("./outputs/logs/classification/")

    if not os.path.exists(os.path.join("./outputs/training_data_using_prediction/")):
        os.makedirs(os.path.join("./outputs/training_data_using_prediction/"))
    
    log_dir = "./outputs/logs/classification/log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create logger
    logger = logging.getLogger(__name__)
    # This is the configuration of how the message would be written in the file.
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"{log_dir}/debug_classification.log"),
            logging.StreamHandler()
        ]
    )
    logger.setLevel(logging.INFO)

    trainDataset = SaliencyDataset("train")
    valDataset = SaliencyDataset("val")
    testDataset = SaliencyDataset("test")

    trainLoader = DataLoader(trainDataset, batch_size=8, shuffle=True, collate_fn=collate_batch)
    valLoader = DataLoader(valDataset, batch_size=8, shuffle=False, collate_fn=collate_batch)
    testLoader = DataLoader(testDataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

    model = load_model(args)
    optimizer = AdamW(model.parameters(),  lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=get_positive_weight(trainDataset))
    model.to(device)

    train_model(model, optimizer,criterion,args,trainLoader,valLoader)

    logger.info(f'Loading the last best model and testing')
    model, optimizer, epoch = load_checkpoint(model, optimizer, args.checkpoint_path + "/best_checkpoint.pt")

    test_loss, test_metrics = validation_loop(model, optimizer, criterion, testLoader)
    final_test_metrics = averageScores(test_metrics)
    print("Testing", final_test_metrics)
    logTest(final_test_metrics)

    genTrainLoader = DataLoader(trainDataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
    genValLoader = DataLoader(valDataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
    genTestLoader = DataLoader(testDataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

    logger.info(f'Generating data for summarization')
    trainPred = generate_data(model, genTrainLoader)
    valPred = generate_data(model, genValLoader)
    testPred = generate_data(model, genTestLoader)

    trainSummData = prepare_data_for_summarization(trainPred,trainDataset)
    valSummData = prepare_data_for_summarization(valPred,valDataset)
    testSummData = prepare_data_for_summarization(testPred,testDataset)

    savePickle(os.path.join(args.data_output, "train.pkl"), trainSummData)
    savePickle(os.path.join(args.data_output, "val.pkl"), valSummData)
    savePickle(os.path.join(args.data_output, "test.pkl"), testSummData)

    logger.info(f'Experiment completed')