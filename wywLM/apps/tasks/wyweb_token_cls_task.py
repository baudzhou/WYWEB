from collections import OrderedDict
import numpy as np
import os
import pdb
import random
import torch
import ujson as json
from transformers import AutoModelForTokenClassification

from ...utils import get_logger
from ...data import ExampleInstance, ExampleSet, DynamicDataset
from ...data.example import *

from .task import EvalData, Task
from .task_registry import register_task

from seqeval import metrics as seq_metrics

logger = get_logger()


@register_task(name="PUNCTask", desc="Token level classification task")
class PUNCTask(Task):
    def __init__(self, data_dir, tokenizer=None, args=None, **kwargs):
        super().__init__(tokenizer, args, **kwargs)
        self.data_dir = data_dir
        self.label_to_id = {v: k for k, v in enumerate(self.get_labels())}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.model_class = AutoModelForTokenClassification

    def train_data(
        self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs
    ):
        train = self.load_data(
            os.path.join(self.data_dir, "train.txt"), max_seq_len=max_seq_len
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
            self._data("dev", "dev.txt", "dev", max_seq_len=max_seq_len),
            self._data("test", "test.txt", "test", max_seq_len=max_seq_len),
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
        ds = [self._data("test", "test.txt", "test", max_seq_len=max_seq_len)]

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
            critial_metrics=["f1"],
        )

    def get_metrics_fn(self):
        """Calcuate metrics based on prediction results"""

        def metrics_fn(logits, labels):
            preds = np.argmax(logits, axis=-1)
            label_names = self.get_labels()
            y_true = []
            y_pred = []
            for pred, label in zip(preds, labels):
                y_true.append([label_names[l] for l in label if l >= 0])
                y_pred.append([label_names[p] for p, l in zip(pred, label) if l >= 0])
            return OrderedDict(
                accuracy=seq_metrics.accuracy_score(y_true, y_pred),
                f1=seq_metrics.f1_score(y_true, y_pred),
                precision=seq_metrics.precision_score(y_true, y_pred),
                recall=seq_metrics.recall_score(y_true, y_pred),
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
                        fs.write(f"{e}\t{p}\n")
                else:
                    for i, (e, p) in enumerate(zip(examples, preds)):
                        fs.write(f"{e.label}\t{p}\n")

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
        return ["O", "S-。", "S-？", "S-！", "S-，", "S-、", "S-；", "S-：", "S-|"]

    def load_data(self, path, max_seq_len=512, max_examples=None, shuffle=False):
        max_len = max_seq_len - 2
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for sent in f:
                sent, labels = sent.strip().split("\t")
                if len(labels) >= max_len:
                    sent = sent[:max_len]
                    labels = labels[:max_len]
                labels = [lbl if lbl == "O" else "S-" + lbl for lbl in labels]
                labels = [self.label_to_id[label] for label in labels]

                tokens = [self.tokenizer.tokenize(w) for w in sent]
                examples.append(
                    ExampleInstance(segments=[tokens], label=labels, sentence=sent)
                )
                if self.args.debug and len(examples) >= 3000:
                    break

            def get_stats(l):
                return f"Max={max(l)}, min={min(l)}, avg={np.mean(l)}"

            ctx_token_size = [sum(len(w) for w in e.segments[0]) for e in examples]
            logger.info(
                f"Statistics: {get_stats(ctx_token_size)}, long={len([t for t in ctx_token_size if t > 500])}/{len(ctx_token_size)}"
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
        if len(example.label) > len(example.segments[0]) + 1:
            example.label = example.label[: len(example.segments[0]) + 2]
        features = OrderedDict()
        tokens = ["[CLS]"]
        target_labels = [-100]
        type_ids = [0]

        for i, w in enumerate(example.segments[0]):
            tokens.extend(w)
            type_ids.extend([0] * len(w))
            if example.label is not None:
                target_labels.append(example.label[i])
                # target_labels.extend([-1]*(len(w)-1))
        tokens.append("[SEP]")
        if len(example.label) > i + 1:
            target_labels.append(example.label[i + 1])
        if example.label is not None:
            target_labels.extend([-100] * (max_seq_len - len(target_labels)))
        type_ids.append(0)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        pad_id = tokenizer.convert_tokens_to_ids(["[PAD]"])
        pos_ids = list(range(len(token_ids)))
        input_mask = [1] * len(token_ids)
        features["token_type_ids"] = type_ids
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
            features["labels"] = torch.tensor(target_labels, dtype=torch.long)
        return features


@register_task(name="GLNERTask", desc="Token level classification task for GLNER")
class GLNERTask(PUNCTask):
    def train_data(
        self, max_seq_len=512, dataset_size=None, epochs=1, mask_gen=None, **kwargs
    ):
        train = self.load_data(
            os.path.join(self.data_dir, "train.jsonl"), max_seq_len=max_seq_len
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
            self._data("dev", "dev.jsonl", "dev", max_seq_len=max_seq_len),
            self._data("test", "test.jsonl", "test", max_seq_len=max_seq_len),
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
        ds = [self._data("test", "test.jsonl", "test", max_seq_len=max_seq_len)]

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

    def get_labels(self):
        """See base class.
        ['O', 'B-地名', 'I-地名', 'B-人物', 'I-人物', 'B-佛经', 'I-佛经', 'B-宗派', 'I-宗派', 'B-教义', 'I-教义']
        """
        return [
            "O",
            "B-noun_other",
            "I-noun_other",
            "B-noun_bookname",
            "I-noun_bookname",
        ]

    def load_data(self, path, max_seq_len=512, max_examples=None, shuffle=False):
        max_len = max_seq_len - 2
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for sent in f:
                line = json.loads(sent)
                sent = line["text"]
                if self.args.s2t:
                    from opencc import OpenCC

                    cc = OpenCC("s2t")
                    sent = cc.convert(sent)
                labels = line["label"]
                labels_out = ["O"] * len(sent)
                for label in labels:
                    start = label[0]
                    end = label[1]
                    t = label[2]
                    labels_out[start] = "B-" + t
                    for i in range(start + 1, end):
                        labels_out[i] = "I-" + t
                # sent, labels = sent.strip().split('\t')
                if len(labels_out) >= max_len:
                    sent = sent[:max_len]
                    labels_out = labels_out[:max_len]
                labels_out = [self.label_to_id[lbl] for lbl in labels_out]

                tokens = [self.tokenizer.tokenize(w) for w in sent]
                examples.append(
                    ExampleInstance(segments=[tokens], label=labels_out, sentence=sent)
                )
                if self.args.debug and len(examples) >= 3000:
                    break

            def get_stats(l):
                return f"Max={max(l)}, min={min(l)}, avg={np.mean(l)}"

            ctx_token_size = [sum(len(w) for w in e.segments[0]) for e in examples]
            logger.info(
                f"Statistics: {get_stats(ctx_token_size)}, long={len([t for t in ctx_token_size if t > 500])}/{len(ctx_token_size)}"
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
        if len(example.label) > len(example.segments[0]) + 1:
            example.label = example.label[: len(example.segments[0]) + 2]
        features = OrderedDict()
        tokens = ["[CLS]"]
        target_labels = [-100]
        type_ids = [0]

        for i, w in enumerate(example.segments[0]):
            tokens.extend(w)
            type_ids.extend([0] * len(w))
            if example.label is not None:
                target_labels.append(example.label[i])
                # target_labels.extend([-1]*(len(w)-1))
        tokens.append("[SEP]")
        if len(example.label) > i + 1:
            target_labels.append(example.label[i + 1])
        if example.label is not None:
            target_labels.extend([-100] * (max_seq_len - len(target_labels)))
        type_ids.append(0)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        pad_id = tokenizer.convert_tokens_to_ids(["[PAD]"])
        pos_ids = list(range(len(token_ids)))
        input_mask = [1] * len(token_ids)
        features["token_type_ids"] = type_ids
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
            features["labels"] = torch.tensor(target_labels, dtype=torch.long)
        return features


@register_task(name="BNERTask", desc="Token level classification task for BNER")
class BNERTask(GLNERTask):
    def __init__(self, data_dir, tokenizer=None, args=None, **kwargs) -> None:
        super().__init__(data_dir, tokenizer, args, **kwargs)

    def get_labels(self):
        """See base class."""
        return [
            "O",
            "B-地名",
            "I-地名",
            "B-人物",
            "I-人物",
            "B-佛经",
            "I-佛经",
            "B-宗派",
            "I-宗派",
            "B-教义",
            "I-教义",
        ]
