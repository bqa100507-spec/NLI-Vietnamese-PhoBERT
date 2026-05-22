from torch.utils.data import Dataset
import torch
import numpy as np

class NLIDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label_list=None, max_length=128):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

        if label_list is None:
            self.label_list = np.sort(dataframe.label.unique())
        else:
            self.label_list = label_list

        self.label_dict = {label: idx for idx, label in enumerate(self.label_list)}
        self.num_label = len(self.label_list)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        premise = str(self.df.iloc[idx]['premise'])
        hypothesis = str(self.df.iloc[idx]['hypothesis'])
        label = self.label_dict[self.df.iloc[idx]['label']]

        encoding = self.tokenizer( 
            premise,
            hypothesis,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }