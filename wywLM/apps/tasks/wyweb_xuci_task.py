from collections import OrderedDict
import numpy as np
import os
import random
import torch

from transformers import AutoModel
from ...utils import xtqdm as tqdm
from ...utils import get_logger

from ..models import XuciModel, NNModule
from ...data import ExampleInstance, ExampleSet, DynamicDataset
from ...data.example import *
from .task import EvalData, Task
from .task_registry import register_task

from seqeval import metrics as seq_metrics

__all__ = ["XUCI"]

logger = get_logger()


@register_task(name="XuciTASK", desc="Token level classification task")
class XuciTask(Task):
    def __init__(self, data_dir, tokenizer=None, args=None, **kwargs):
        super().__init__(tokenizer, args, **kwargs)
        self.data_dir = data_dir
        self.model_class = AutoModel

    def get_model_class_fn(self):
        def partial_class(*wargs, **kwargs):
            model, tokenizer, model_config = NNModule.load_model(
                *wargs, self.model_class, **kwargs
            )
            self.tokenizer = tokenizer
            self.model_config = model_config
            model = XuciModel(model_config, model)
            return model

        return partial_class

    def train_data(
        self, max_seq_len=64, dataset_size=None, epochs=1, mask_gen=None, **kwargs
    ):
        train = self.load_data(os.path.join(self.data_dir, "train.tsv"))
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
        assert os.path.exists(input_src), f"{input_src} doesn't exists"
        data = self.load_data(input_src)
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
            if isinstance(logits, torch.Tensor):
                with torch.no_grad():
                    preds = torch.sigmoid(logits)
                    preds = torch.where(preds > 0.5, 1, 0).detach().cpu().numpy()
            else:
                preds = np.argmax(logits, -1)
            return OrderedDict(accuracy=seq_metrics.accuracy_score(preds, labels))

        return metrics_fn

    def get_predict_fn(self, examples):
        """Calcuate metrics based on prediction results"""

        def predict_fn(logits, output_dir, name, prefix, targets=None):
            output = os.path.join(output_dir, "submit-{}-{}.tsv".format(name, prefix))
            if isinstance(logits, torch.Tensor):
                with torch.no_grad():
                    preds = torch.sigmoid(logits).detach().cpu()
                    preds = torch.where(preds > 0.5, 1, 0)
            else:
                # cross entropy
                preds = logits.argmax(-1)

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
            return self.example_to_feature_siamense(
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
        return [0, 1]

    def load_data(self, path, shuffle=False):
        with open(path, "r", encoding="utf-8") as f:
            samples = f.read()
        if self.args.s2t:
            from opencc import OpenCC

            cc = OpenCC("s2t")
            samples = cc.convert(samples)
        samples = samples.split("\n")
        samples = [s.split("\t") for s in samples]
        examples = []
        for sample in samples:
            pos_a = sample[2].split(",")
            pos_a = [int(p) for p in pos_a]
            pos_b = sample[3].split(",")
            pos_b = [int(p) for p in pos_b]
            examples.append(
                ExampleInstance(
                    segments=sample[:2],
                    label=0 if sample[-1] == "f" else 1,
                    compare_pos=[pos_a.copy(), pos_b.copy()],
                )
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
        max_seq_len=64,
        rng=None,
        mask_generator=None,
        ext_params=None,
        label_type="int",
        **kwargs,
    ):
        if not rng:
            rng = random
        max_num_tokens = max_seq_len - 2
        max_compare_pos = 2
        features = OrderedDict()
        tokens = (
            ["[CLS]"]
            + list(example.segments[0])
            + ["[SEP]"]
            + list(example.segments[1])
            + ["[SEP]"]
        )
        ids_a = [
            i + 1
            for i in range(example.compare_pos[0][0], example.compare_pos[0][1] + 1)
        ]
        ids_b = [
            i + len(example.segments[0]) + 2
            for i in range(example.compare_pos[1][0], example.compare_pos[1][1] + 1)
        ]
        ids_a += [0] * (max_compare_pos - len(ids_a))
        ids_b += [0] * (max_compare_pos - len(ids_b))
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        pad_id = tokenizer.convert_tokens_to_ids(["[PAD]"])
        input_mask = [1] * len(input_ids)
        padding_size = max(0, max_seq_len - len(input_ids))
        features["input_ids"] = input_ids + pad_id * padding_size
        features["attention_mask"] = input_mask
        for f in features:
            if len(features[f]) < max_seq_len:
                features[f].extend([0] * padding_size)
        features["compare_pos"] = torch.from_numpy(np.array([ids_a, ids_b]))
        features["labels"] = [example.label]
        for f, v in features.items():
            if not isinstance(v, torch.Tensor):
                features[f] = torch.tensor(features[f], dtype=torch.int)
        return features

    def example_to_feature_siamense(
        self,
        tokenizer,
        example,
        max_seq_len=64,
        rng=None,
        mask_generator=None,
        ext_params=None,
        label_type="int",
        **kwargs,
    ):
        if not rng:
            rng = random
        max_num_tokens = max_seq_len - 2
        max_compare_pos = 2
        features = OrderedDict()
        tokens_a = ["[CLS]"] + list(example.segments[0]) + ["[SEP]"]
        tokens_b = ["[CLS]"] + list(example.segments[1]) + ["[SEP]"]
        ids_a = [
            i + 1
            for i in range(example.compare_pos[0][0], example.compare_pos[0][1] + 1)
        ]
        ids_b = [
            i + 1
            for i in range(example.compare_pos[1][0], example.compare_pos[1][1] + 1)
        ]
        ids_a += [0] * (max_compare_pos - len(ids_a))
        ids_b += [0] * (max_compare_pos - len(ids_b))
        input_ids = [
            tokenizer.convert_tokens_to_ids(tokens_a),
            tokenizer.convert_tokens_to_ids(tokens_b),
        ]
        pad_id = tokenizer.convert_tokens_to_ids(["[PAD]"])
        input_mask = [[1] * len(input_ids[0]), [1] * len(input_ids[1])]
        padding_size_a = max(0, max_seq_len - len(input_ids[0]))
        padding_size_b = max(0, max_seq_len - len(input_ids[1]))
        features["input_ids"] = [
            input_ids[0] + pad_id * padding_size_a,
            input_ids[1] + pad_id * padding_size_b,
        ]
        features["attention_mask"] = [
            input_mask[0] + [0] * padding_size_a,
            input_mask[1] + [0] * padding_size_b,
        ]
        # for f in features:
        #   if len(features[f]) < max_seq_len:
        #     features[f].extend([0] * padding_size)
        features["compare_pos"] = torch.from_numpy(np.array([ids_a, ids_b]))
        features["labels"] = [example.label]
        for f, v in features.items():
            if not isinstance(v, torch.Tensor):
                features[f] = torch.tensor(features[f], dtype=torch.int)
        return features
