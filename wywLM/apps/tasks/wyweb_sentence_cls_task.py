from cProfile import label
from collections import OrderedDict
import numpy as np
import os
import random
import torch
import ujson as json
from transformers import AutoModelForSequenceClassification
from ...utils import xtqdm as tqdm
from ...utils import get_logger

from ...data import ExampleInstance, ExampleSet, DynamicDataset
from ...data.example import *
from .task import EvalData, Task
from .task_registry import register_task
from .metrics import *

logger = get_logger()


@register_task(name="GJCTask", desc="Token level classification task")
class GJCTask(Task):
    def __init__(self, data_dir, tokenizer=None, args=None, **kwargs):
        super().__init__(tokenizer, args, **kwargs)
        self.data_dir = data_dir
        self.label_to_id = {v: k for k, v in enumerate(self.get_labels())}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.model_class = AutoModelForSequenceClassification

    def train_data(
        self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs
    ):
        train = self.load_data(
            os.path.join(self.data_dir, "train.tsv"), max_seq_len=max_seq_len
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
            self._data("dev", "dev.tsv", "dev", max_seq_len=max_seq_len),
            self._data("test", "test.tsv", "test", max_seq_len=max_seq_len),
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
        ds = [self._data("test", "test.tsv", "test", max_seq_len=max_seq_len)]

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
        # assert os.path.exists(input_src), f"{input_src} doesn't exists"
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
        label_cls = list(range(0, 10))

        def metrics_fn(logits, labels):
            return OrderedDict(
                accuracy=metric_accuracy(logits, labels),
                f1=metric_macro_f1(logits, labels, label_cls),
                precision=metric_precision(logits, labels, label_cls),
                recall=metric_recall(logits, labels, label_cls),
            )

        return metrics_fn

    def get_predict_fn(self, examples):
        """Calcuate metrics based on prediction results"""

        def predict_fn(logits, output_dir, name, prefix, targets=None):
            output = os.path.join(output_dir, "submit-{}-{}.tsv".format(name, prefix))
            preds = np.argmax(logits, axis=-1)
            labels = self.get_labels()
            with open(output, "w", encoding="utf-8") as fs:
                fs.write("targets\tpredictions\n")
                if targets is not None:
                    for i, (e, p) in enumerate(zip(targets, preds)):
                        fs.write(f"{labels[e]} {labels[p]}\n")
                else:
                    for i, (e, p) in enumerate(zip(examples, preds)):
                        fs.write(f"{labels[e.label]} {labels[p]}\n")

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
        return ["儒藏", "诗藏", "佛藏", "易藏", "医藏", "艺藏", "子藏", "集藏", "道藏", "史藏"]

    def load_data(self, path, max_seq_len=512, max_examples=None, shuffle=False):
        max_len = max_seq_len - 2
        examples = []
        merged_words = []
        merged_tokens = []
        merged_labels = []
        size = 0
        with open(path, "r", encoding="utf-8") as f:
            for sent in f:
                label, sent = sent.strip().split("\t")
                if self.args.s2t:
                    from opencc import OpenCC

                    cc = OpenCC("s2t")
                    sent = cc.convert(sent)
                label = self.label_to_id[label]

                tokens = self.tokenizer.tokenize(sent)  # for w in sent
                examples.append(
                    ExampleInstance(segments=[tokens], label=label, sentence=sent)
                )
                if self.args.debug and len(examples) >= 3000:
                    break

            def get_stats(l):
                return f"Max={max(l)}, min={min(l)}, avg={np.mean(l)}"

            ctx_token_size = [sum(len(w) for w in e.segments[0]) for e in examples]
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
        max_num_tokens = max_seq_len - 2

        features = OrderedDict()
        tokens = ["[CLS]"] + list(example.segments[0]) + ["[SEP]"]

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        pad_id = tokenizer.convert_tokens_to_ids(["[PAD]"])
        pos_ids = list(range(len(token_ids)))
        input_mask = [1] * len(token_ids)
        # features['token_type_ids'] = None
        features["position_ids"] = pos_ids
        features["attention_mask"] = input_mask
        padding_size = max(0, max_seq_len - len(token_ids))
        features["input_ids"] = token_ids + pad_id * padding_size
        for f in features:
            if len(features[f]) < max_seq_len:
                features[f].extend([0] * padding_size)

        for f in features:
            features[f] = torch.tensor(features[f], dtype=torch.int)
        if (
            example.label is not None
        ):  # and example.label[0]>=0 and example.label[1]>=0:
            features["labels"] = torch.tensor(example.label, dtype=torch.long)
        return features


@register_task(name="FSPCTask", desc="Token level classification task")
class FSPCTask(GJCTask):
    def load_data(self, path, max_seq_len=512, max_examples=None, shuffle=False):
        path = path.replace("tsv", "json")
        examples = []
        raw = json.load(open(path, "r", encoding="utf-8"))

        for doc in raw:
            poem = doc["poem"]
            if self.args.s2t:
                from opencc import OpenCC

                cc = OpenCC("s2t")
                poem = cc.convert(poem)
            sents = poem.split("|")
            poem = poem.replace("|", "，")
            poem += "。"
            sentms = doc["sentiments"]

            tokens = self.tokenizer.tokenize(poem)  # for w in sent
            examples.append(
                ExampleInstance(
                    segments=[tokens], label=int(sentms["holistic"]) - 1, sentence=poem
                )
            )
            for i, t in enumerate(sents):
                examples.append(
                    ExampleInstance(
                        segments=[self.tokenizer.tokenize(t)],
                        label=int(sentms[f"line{i + 1}"]) - 1,
                        sentence=self.tokenizer.tokenize(t),
                    )
                )
            if self.args.debug and len(examples) >= 3000:
                break

        def get_stats(l):
            return f"Max={max(l)}, min={min(l)}, avg={np.mean(l)}"

        ctx_token_size = [sum(len(w) for w in e.segments[0]) for e in examples]
        logger.info(
            f"Statistics: {get_stats(ctx_token_size)}, \
                  long={len([t for t in ctx_token_size if t > 500])}/{len(ctx_token_size)}"
        )
        return examples

    def get_labels(self):
        # 1: negative, 2:implicit negative, 3:neutral, 4:implicit positive and 5:positive
        return ["1", "2", "3", "4", "5"]
