import torch
import os
from collections import OrderedDict
import numpy as np
import tempfile
import numpy as np
import mmap
import pickle
import signal
import sys
import pdb


from ..utils import xtqdm as tqdm

__all__=['ExampleInstance', 'example_to_feature', 'ExampleSet']

class ExampleInstance:
  def __init__(self, segments, label=None,  **kwv):
    self.segments = segments
    self.label = label
    self.__dict__.update(kwv)

  def __repr__(self):
    return f'segments: {self.segments}\nlabel: {self.label}'

  def __getitem__(self, i):
    return self.segments[i]

  def __len__(self):
    return len(self.segments)

class ExampleSet:
  def __init__(self, pairs):
    self._data = np.array([pickle.dumps(p) for p in pairs])
    self.total = len(self._data)

  def __getitem__(self, idx):
    """
    return pair
    """
    if isinstance(idx, tuple):
      idx,rng, ext_params = idx
    else:
      rng,ext_params=None, None
    content = self._data[idx]
    example = pickle.loads(content)
    return example

  def __len__(self):
    return self.total

  def __iter__(self):
    for i in range(self.total):
      yield self[i]

def _truncate_segments(segments, max_num_tokens, rng):
  """
  Truncate sequence pair according to original BERT implementation:
  https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L391
  """
  while True:
  #   if sum(len(s) for s in segments)<=max_num_tokens:
  #     break

    # segments = sorted(segments, key=lambda s:len(s), reverse=True)
    # trunc_tokens = segments[0]

    # assert len(trunc_tokens) >= 1
    if len(segments) <= max_num_tokens:
      break
    if rng.random() < 0.5:
      segments.pop(0)
    else:
      segments.pop()
  return segments

def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
    import random
    if isinstance(tokens_a, str) or isinstance(tokens_b, str):
      print('error')
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (random.random() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()
            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b


def example_to_feature(tokenizer, example, max_seq_len=512, rng=None, mask_generator = None, ext_params=None, label_type='int', **kwargs):
  if not rng:
    rng = random
  max_num_tokens = max_seq_len - len(example.segments) - 1
  segments = _truncate_segments([tokenizer.tokenize(s) for s in example.segments], max_num_tokens, rng)
  tokens = ['[CLS]']
  type_ids = [0]
  for i,s in enumerate(segments):
    tokens.extend(s)
    tokens.append('[SEP]')
    type_ids.extend([i]*(len(s)+1))
  if mask_generator:
    tokens, lm_labels = mask_generator.mask_tokens(tokens, rng)
  token_ids = tokenizer.convert_tokens_to_ids(tokens)
  pos_ids = list(range(len(token_ids)))
  input_mask = [1]*len(token_ids)
  features = OrderedDict(input_ids = token_ids,
      type_ids = type_ids,
      position_ids = pos_ids,
      input_mask = input_mask)
  if mask_generator:
    features['lm_labels'] = lm_labels
  padding_size = max(0, max_seq_len - len(token_ids))
  for f in features:
    features[f].extend([0]*padding_size)
    features[f] = torch.tensor(features[f], dtype=torch.int)
  label_type = torch.int if label_type=='int' else torch.float
  if example.label is not None:
    features['labels'] = torch.tensor(example.label, dtype=label_type)
  return features
