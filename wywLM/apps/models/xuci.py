from ast import Not
import types
from typing import Union, Tuple, List, Iterable, Dict
import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from ...modeling import NNModule, StableDropout


class WeightedLayerPooling(nn.Module):
    """
    Token embeddings are weighted mean of their different hidden layer representations
    """
    def __init__(self, num_hidden_layers: int = 12, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.config_keys = ['word_embedding_dimension', 'layer_start', 'num_hidden_layers']
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
                            else nn.Parameter(torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float))

    def forward(self, word_embeddings):
        all_layer_embedding = torch.stack(word_embeddings)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]  # Start from 4th layers output
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average


class XuciModel(NNModule):
    def __init__(self, model_config, mlm_model, weight_layer_pooler=False, drop_out=None):
        super().__init__()
        self.config = model_config
        if weight_layer_pooler:
            self.pooler = WeightedLayerPooling()
        else:
            self.pooler = None
        self.classifier = nn.Linear(model_config.hidden_size * 3, 2, bias=False)
        drop_out = model_config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        self.apply(self.init_weights)
        self.bert = mlm_model.base_model

    def forward(self, input_ids, compare_pos, position_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # input_ids_a = input_ids[:, 0, ...].squeeze()
        # input_ids_b = input_ids[:, 1, ...].squeeze()
        # attention_mask_a = attention_mask[:, 0, ...].squeeze()
        # attention_mask_b = attention_mask[:, 1, ...].squeeze()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        prediction_scores = outputs['last_hidden_state']
        # sequence_output_b, _ = self.bert(input_ids_b, token_type_ids=token_type_ids, attention_mask=attention_mask_b)
        prediction_scores = prediction_scores.view(-1, 2, prediction_scores.size(1), prediction_scores.size(2))
        sequence_output_a, sequence_output_b = torch.unbind(prediction_scores, dim=1)
        # sequence_output_b = prediction_scores[:, 1, ...].squeeze()
        tokens_a = []
        tokens_b = []
        for so_a, so_b, pos in zip(sequence_output_a, sequence_output_b, compare_pos):
            a = so_a.index_select(0, pos[0][pos[0] > 0]).mean(0)
            b = so_b.index_select(0, pos[1][pos[1] > 0]).mean(0)
            tokens_a.append(a)
            tokens_b.append(b)
        
        tokens_a = torch.stack(tokens_a)
        tokens_b = torch.stack(tokens_b)
        
        # cosine similarity loss
        # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # logits = cos(tokens_a, tokens_b)
        # loss_fn = MSELoss()
        # loss = loss_fn(logits, labels.half())

        # according to SentenceBERT, use u,v,u-v as output
        scores = torch.cat([tokens_a, tokens_b, torch.abs(tokens_a - tokens_b)], -1)
        scores = self.dropout(scores)
        logits = self.classifier(scores).float()

        labels = labels.long().squeeze()

        loss_fn = CrossEntropyLoss() # 
        loss = loss_fn(logits, labels)

        return {
                'logits' : logits,
                'loss' : loss,
                'labels': labels
            }