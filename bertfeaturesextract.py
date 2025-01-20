
import torch
# from torch import nn, optim
from transformers import BertTokenizer, BertModel



def extract_awesome_features(texts):
    processed_texts = [text[1] for text in texts]  
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertModel.from_pretrained('bert-base-uncased')
    features_list = []
    batch_size = 32
    for i in range(0, len(processed_texts), batch_size):
        batch_texts = processed_texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cpu')
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        features = torch.mean(embeddings, dim=1)
        features_list.append(features.cpu())
    features = torch.cat(features_list, dim=0)
    features_numpy = features.numpy()
    return features_numpy
    # inputs = tokenizer(processed_texts, padding=True, truncation=True, return_tensors='pt', max_length=1024)

    # with torch.no_grad():
    #     outputs = model(**inputs)
    # embeddings = outputs.last_hidden_state
    # features = torch.mean(embeddings, dim=1)
    # features_numpy = features.numpy()
    # # scaler = MaxAbsScaler()
    # # features_numpy = scaler.fit_transform(features_numpy)
    # return features_numpy