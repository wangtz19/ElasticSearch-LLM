from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict
import torch


class BertClassifier():

    def __init__(self, 
                 model_checkpoint: str,
                 tokenizer_checkpoint: str="hfl/chinese-bert-wwm",
                 **kwargs
                ) -> None:
        device = kwargs.get("device", None)
        self.device = torch.device(device if device is not None else \
                                   "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

    def predict(self, text: str) -> Dict:
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True,
                                max_length=512, # 512 is the maximum length of BERT
                                padding="max_length").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits[0]
        pred = torch.argmax(logits).item()
        return {"label": pred}
    
