import os
from transformers import RobertaTokenizer, RobertaModel
import pickle
import torch
from tqdm import tqdm
from datasets import load_dataset
import argparse
from typing import List, Dict, Tuple

class DataProcessor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def savePickle(self, path: str, data: Dict) -> None:
        with open(path, "wb") as f:
            pickle.dump(data, f)
            
    def load_model(self) -> Tuple[RobertaTokenizer, RobertaModel]:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        model = RobertaModel.from_pretrained("roberta-large").to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return tokenizer, model
    
    def loadData(self) -> Tuple[Dict, Dict, Dict]:
        dataset = load_dataset("rohitsaxena/MENSA")   
        return dataset["train"], dataset["validation"], dataset["test"]
    
    def sceneTextToEmbed(self, model: RobertaModel, tokenizer: RobertaTokenizer, data: Dict) -> List[Dict]:
        processed = []
        for idx in tqdm(range(len(data))):
            movieData = data[idx]
            with torch.no_grad():
                encoded_input = tokenizer(movieData["scenes"], return_tensors='pt', padding=True, truncation=True).to(self.device)
                output = model(**encoded_input)
                embeddings = output.last_hidden_state[:, 0, :]
                embeddings = embeddings.detach().cpu()
            movieData["scenes_embeddings"] = embeddings
            movieData["labels"] = torch.tensor(movieData["labels"])
            processed.append(movieData)
        return processed

def main():
    processor = DataProcessor()
    
    parser = argparse.ArgumentParser(description="Evaluate models on Mini-MMLU dataset")
    parser.add_argument(
        "--output_path_root",
        type=str,
        default="./outputs/",
        choices=["gpt4", "gpt-4-turbo", "llama", "claude"],
        help="Path to extract embedding to be used for classification model",
    )
    
    args = parser.parse_args()
    
    print("Extracting embeddings")
    
    trainData, valData, testData = processor.loadData()
    tokenizer, model = processor.load_model()
    
    newTrain = processor.sceneTextToEmbed(model, tokenizer, trainData)
    newVal = processor.sceneTextToEmbed(model, tokenizer, valData)
    newTest = processor.sceneTextToEmbed(model, tokenizer, testData)
    
    output_path = os.path.join(args.output_path_root, "scene_classification_data")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    processor.savePickle(os.path.join(output_path, "train.pkl"), newTrain)
    processor.savePickle(os.path.join(output_path, "val.pkl"), newVal)
    processor.savePickle(os.path.join(output_path, "test.pkl"), newTest)

if __name__ == '__main__':
    main()