from venv import create
import types
from loguru import logger
import torch
from torch import nn

def embedding_forward(embedding_obj, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None, **kwargs):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = embedding_obj.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=embedding_obj.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = embedding_obj.word_embeddings(input_ids)

        if embedding_obj.position_embeddings is not None:
            position_embeddings = embedding_obj.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if hasattr(embedding_obj, 'position_biased_input') and embedding_obj.position_biased_input:
            embeddings += position_embeddings
        if hasattr(embedding_obj, 'token_type_embeddings'):
            token_type_embeddings = embedding_obj.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        # if embedding_obj.embedding_size != embedding_obj.config.hidden_size:
        #     embeddings = embedding_obj.embed_proj(embeddings)

        embeddings = embedding_obj.LayerNorm(embeddings)

        if mask is not None:
            if hasattr(embedding_obj, 'padding_idx'):
                padding_idx = embedding_obj.padding_idx
            elif hasattr(embedding_obj, 'config'):
                padding_idx = embedding_obj.config.pad_token_id
            else:
                padding_idx = 0
            mask = input_ids != padding_idx
            mask = mask.to(embeddings.dtype)
            mask = mask.unsqueeze(-1).expand(mask.size(0), mask.size(1), embeddings.size(-1))
            embeddings = embeddings * mask

        embeddings = embedding_obj.dropout(embeddings)
        return embeddings

class MTModel(nn.Module):
    def __init__(self, model_config, mlm_model):
        super().__init__()
        self.mlm_model = mlm_model
        # bind pre-fix attention mask version embedding forward function dynamicly
        self.mlm_model.base_model.embeddings.forward = types.MethodType(embedding_forward, self.mlm_model.base_model.embeddings)
        self.config = model_config
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.tril_matrix = torch.tril(torch.ones((self.config.max_position_embeddings, 
                                                    self.config.max_position_embeddings), dtype=torch.long))

    def create_attention_mask(self, token_type_ids):
        idxs = torch.cumsum(token_type_ids, axis=1)
        mask = idxs[:, None, :] <= idxs[:, :, None]
        mask = mask.byte()
        return mask

    def forward(self, input_ids, position_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        if labels is not None:
            if attention_mask is None:
                attention_mask = self.create_attention_mask(token_type_ids)
            outputs = self.mlm_model(input_ids, token_type_ids=None, attention_mask=attention_mask)
            prediction_scores = outputs['logits'][:, : -1, ...]

            loss = self.crit_mask_lm(prediction_scores.reshape(-1, self.config.vocab_size), \
                                    labels[:, 1:].reshape(-1).long())
            loss = loss.view(input_ids.size(0), -1)
            mask = token_type_ids[:, 1:] * token_type_ids[:, : -1]
            loss = loss * mask
            loss = loss[loss > 0].mean()
            logits = []
            target = []
            for ps, l, m in zip(prediction_scores, labels[:, 1:], mask):
                logits.append(ps[m.bool()].argmax(-1))
                target.append(l[m.bool()])
            return {
                'loss': loss,
                'logits': logits,
                'labels': target
            }