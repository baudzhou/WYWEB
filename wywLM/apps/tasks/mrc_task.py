from collections import OrderedDict
import numpy as np
import os
import random
import torch
import ujson as json
from transformers import AutoModelForMultipleChoice
from ...utils import xtqdm as tqdm
from ...utils import get_logger

from ..models import (
    MultiChoiceModel
)
from ...data import ExampleInstance, ExampleSet, DynamicDataset
from ...data.example import *
from .task import EvalData, Task
from .task_registry import register_task
from .metrics import *

logger = get_logger()


@register_task(name="MRCTask", desc="Token level classification task")
class MRCTask(Task):
    def __init__(self, data_dir, args, **kwargs):
        super().__init__(args, **kwargs)
        self.data_dir = data_dir
        self.label_to_id = {v: k for k, v in enumerate(self.get_labels())}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.model_class = AutoModelForMultipleChoice
        self.custom_model_class = MultiChoiceModel

    def train_data(
        self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs
    ):
        train = self.load_data(
            os.path.join(self.data_dir, "train.json"), max_seq_len=max_seq_len
        )
        examples = ExampleSet(train)
        if dataset_size is None:
            dataset_size = len(examples) * epochs
        return DynamicDataset(
            examples,
            feature_fn=self.get_feature_fn(max_seq_len=max_seq_len, mask_gen=mask_gen),
            dataset_size=dataset_size,
            shuffle=True,
            **kwargs,
        )

    def eval_data(self, max_seq_len=512, dataset_size=None, **kwargs):
        ds = [
            self._data("dev", "dev.json", "dev", max_seq_len=max_seq_len),
            self._data("test", "test.json", "test", max_seq_len=max_seq_len),
        ]

        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            d.data = DynamicDataset(
                d.data,
                feature_fn=self.get_feature_fn(max_seq_len=max_seq_len),
                dataset_size=_size,
                **kwargs,
            )
        return ds

    def test_data(self, max_seq_len=512, dataset_size=None, **kwargs):
        """See base class."""
        ds = [self._data("test", "test.json", "test", max_seq_len=max_seq_len)]

        for d in ds:
            if dataset_size is None:
                _size = len(d.data)
            d.data = DynamicDataset(
                d.data,
                feature_fn=self.get_feature_fn(max_seq_len=max_seq_len),
                dataset_size=_size,
                **kwargs,
            )
        return ds

    def _data(
        self,
        name,
        path,
        type_name="dev",
        ignore_metric=False,
        max_examples=None,
        shuffle=False,
        max_seq_len=512,
    ):
        input_src = os.path.join(self.data_dir, path)
        assert os.path.exists(input_src), f"{input_src} doesn't exists"
        data = self.load_data(
            input_src,
            max_seq_len=max_seq_len,
            max_examples=max_examples,
            shuffle=shuffle,
        )
        examples = ExampleSet(data)
        predict_fn = self.get_predict_fn(examples)
        metrics_fn = self.get_metrics_fn()
        return EvalData(
            name,
            examples,
            metrics_fn=metrics_fn,
            predict_fn=predict_fn,
            ignore_metric=ignore_metric,
            critial_metrics=["accuracy"],
        )

    def get_metrics_fn(self):
        """Calcuate metrics based on prediction results"""

        def metrics_fn(logits, labels):
            # shape = logits.shape
            # logits = np.reshape(logits, (shape[0], -1, 4))
            return OrderedDict(
                accuracy=metric_accuracy(logits, labels),
            )

        return metrics_fn

    def get_predict_fn(self, examples):
        """Calcuate metrics based on prediction results"""

        def predict_fn(logits, output_dir, name, prefix, targets=None):
            output = os.path.join(output_dir, "submit-{}-{}.tsv".format(name, prefix))
            # shape = logits.shape
            # logits = np.reshape(logits, (shape[0], -1, 4))
            preds = np.argmax(logits, axis=-1)
            labels = self.get_labels()
            # with open(output, 'w', encoding='utf-8') as fs:
            #   fs.write('index\tpredictions\n')
            #   for i,(e,p) in enumerate(zip(examples,preds)):
            #     fs.write(f'{labels[e.label]} {labels[p]}\n')
            with open(output, "w", encoding="utf-8") as fs:
                fs.write("targets\tpredictions\n")
                if targets is not None:
                    for i, (e, p) in enumerate(zip(targets, preds)):
                        fs.write(f"{labels[e]}\t{labels[p]}\n")
                else:
                    for i, (e, p) in enumerate(zip(examples, preds)):
                        fs.write(f"{labels[e.label]}\t{labels[p]}\n")

        return predict_fn

    def get_feature_fn(self, max_seq_len=512, mask_gen=None):
        def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
            return self.example_to_feature(
                self.tokenizer,
                example,
                max_seq_len=max_seq_len,
                rng=rng,
                mask_generator=mask_gen,
                ext_params=ext_params,
                **kwargs,
            )

        return _example_to_feature

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def trunc_text(self, d):
        text = d["article"]
        ans_len = len(d["question"])
        max_op_len = 0
        for i in range(4):
            max_op_len = max(len(d[f"option{i}"]), max_op_len)
        # options = d['options']
        art_trunc = text[: 508 - max_op_len - ans_len]
        # while True:
        #     lens = [len(text + op) for op in options]
        #     if max(lens) > 509:
        #       text = text[:-1]
        #     else:
        #         break
        if self.args.s2t:
            from opencc import OpenCC

            cc = OpenCC("s2t")
            art_trunc = cc.convert(art_trunc)
        d["article"] = art_trunc
        return d

    def load_data(self, path, max_seq_len=512, max_examples=None, shuffle=False):
        examples = []
        irc = json.load(open(path, encoding="utf-8"))
        for ircc in irc:
            if self.args.debug and len(examples) >= 3000:
                break
            ircc = self.trunc_text(ircc)
            segments = []
            p = ["[CLS]"] + list(ircc["article"]) + ["[SEP]"]
            p = p + list(ircc["question"]) + ["[SEP]"]
            for c in range(4):
                segments.append(p + list(ircc[f"option{c}"]) + ["[SEP]"])
            assert len(segments) == 4
            examples.append(
                ExampleInstance(segments=segments.copy(), label=int(ircc["answer"]))
            )

        def get_stats(l):
            return f"Max={max(l)}, min={min(l)}, avg={np.mean(l)}"

        ctx_token_size = [max(len(w) for w in e.segments) for e in examples]
        logger.info(
            f"Statistics: {get_stats(ctx_token_size)}, \
                  long={len([t for t in ctx_token_size if t > 500])}/{len(ctx_token_size)}"
        )
        return examples

    def example_to_feature(
        self,
        tokenizer,
        example,
        max_seq_len=512,
        rng=None,
        mask_generator=None,
        ext_params=None,
        label_type="int",
        **kwargs,
    ):
        if not rng:
            rng = random
        features = OrderedDict()
        token_ids = [
            tokenizer.convert_tokens_to_ids(tokens) for tokens in example.segments
        ]
        pad_id = tokenizer.convert_tokens_to_ids(["[PAD]"])
        pos_ids = [list(range(len(token_id))) for token_id in token_ids]
        input_mask = [[1] * len(token_id) for token_id in token_ids]
        padding_size = [max(0, max_seq_len - len(token_id)) for token_id in token_ids]
        features["position_ids"] = [
            pos_id + [0] * ps for pos_id, ps in zip(pos_ids, padding_size)
        ]
        features["attention_mask"] = [
            im + [0] * ps for im, ps in zip(input_mask, padding_size)
        ]
        features["input_ids"] = [
            token_id + pad_id * ps for token_id, ps in zip(token_ids, padding_size)
        ]

        for f in features:
            features[f] = torch.tensor(features[f], dtype=torch.int)
        if (
            example.label is not None
        ):  # and example.label[0]>=0 and example.label[1]>=0:
            features["labels"] = torch.tensor(example.label, dtype=torch.int)
        return features

    def extract_docs(self, path):
        docs = []
        with open(path, "r", encoding="utf-8") as fs:
            doc = []
            sent = []
            for line in fs:
                if line.startswith("-DOCSTART- "):
                    if len(sent) > 0:
                        doc.append(sent)
                    sent = []
                    if len(doc) > 0:
                        docs.append(doc)
                    doc = []
                elif line.strip() == "":
                    if len(sent) > 0:
                        doc.append(sent)
                    sent = []
                else:
                    tabs = line.split(" ")
                    sent.append([tabs[0], tabs[-1].strip()])
        if len(sent) > 0:
            doc.append(sent)
            sent = []
        if len(doc) > 0:
            docs.append(doc)
            doc = []
        logger.info(
            f"Loaded {len(docs)} docs, {sum([len(d) for d in docs])} sentences."
        )
        return docs