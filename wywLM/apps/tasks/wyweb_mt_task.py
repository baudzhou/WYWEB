from cProfile import label
from collections import OrderedDict
import numpy as np
import os
import random
import torch

from sacrebleu.metrics import BLEU, CHRF, TER
from rouge import Rouge
from transformers import AutoModelForMaskedLM

from ...utils import xtqdm as tqdm
from ...utils import get_logger

from ..models import NNModule, MTModel
from ...data import ExampleInstance, ExampleSet, DynamicDataset
from ...data.example import *
from ...data.example import truncate_tokens_pair
from .task import EvalData, Task
from .task_registry import register_task
from .metrics import *

logger = get_logger()


@register_task(name="MTTask", desc="Token level classification task")
class MTTask(Task):
    def __init__(self, data_dir, tokenizer=None, args=None, **kwargs):
        super().__init__(tokenizer, args, **kwargs)
        self.data_dir = data_dir
        self.label_to_id = {v: k for k, v in enumerate(self.get_labels())}
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        self.model_class = AutoModelForMaskedLM

    def get_model_class_fn(self):
        def partial_class(*wargs, **kwargs):
            model, tokenizer, model_config = NNModule.load_model(
                *wargs, self.model_class, **kwargs
            )
            self.tokenizer = tokenizer
            self.model_config = model_config
            model = MTModel(model_config, model)
            return model

        return partial_class

    @classmethod
    def add_arguments(cls, parser):
        """Add task specific arguments
        e.g. parser.add_argument('--data_dir', type=str, help='The path of data directory.')
        """
        parser.add_argument(
            "--MT",
            default=True,
            action="store_true",
            help='If mechine translation task is processing, eval metric should be bleu --MT "*" --help',
        )

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
            critial_metrics=["bleu"],
        )

    def get_metrics_fn(self):
        """Calcuate metrics based on prediction results"""
        bleu = BLEU(tokenize="zh")
        chrf = CHRF()
        ter = TER(asian_support=True)
        rouge = Rouge()

        # tokenizer = self.tokenizer
        def metrics_fn(preds, labels):
            preds = [" ".join(self.tokenizer.convert_ids_to_tokens(p)) for p in preds]
            labels = [" ".join(self.tokenizer.convert_ids_to_tokens(p)) for p in labels]
            return OrderedDict(
                bleu=bleu.corpus_score(preds, [labels]).score,
                chrf2=chrf.corpus_score(preds, [labels]).score,
                ter=ter.corpus_score(preds, [labels]).score,
                rouge=rouge.get_scores(preds, labels, avg=True),
            )

        return metrics_fn

    def get_predict_fn(self, examples):
        """Calcuate metrics based on prediction results"""

        # tokenizer = self.tokenizer
        def predict_fn(preds, output_dir, name, prefix, labels=None):
            output = os.path.join(output_dir, "submit-{}-{}.tsv".format(name, prefix))
            preds = [self.tokenizer.convert_ids_to_tokens(p) for p in preds]
            if labels:
                labels = [self.tokenizer.convert_ids_to_tokens(p) for p in labels]
            with open(output, "w", encoding="utf-8") as fs:
                fs.write("target\tpredictions\n")
                for i, (e, p) in enumerate(zip(labels, preds)):
                    fs.write(f'{"".join(e)}\t{"".join(p)}\n')

        return predict_fn

    def get_feature_fn(self, max_seq_len=512, mask_gen=None):
        def _example_to_feature(example, rng=None, ext_params=None, **kwargs):
            return self.example_to_feature(
                self.tokenizer,
                example,
                max_seq_len=max_seq_len,
                rng=rng,
                tril_matrix=None,
                **kwargs,
            )

        return _example_to_feature

    def get_labels(self):
        """See base class."""
        if self.tokenizer:
            labels = self.tokenizer.vocab
        else:
            labels = []
        return labels

    def load_data(self, path, max_seq_len=512, max_examples=None, shuffle=False):
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            data = f.read().split("\n")
        if self.args.s2t:
            from opencc import OpenCC

            cc = OpenCC("s2t")
            data = [cc.convert(d) for d in data]
        data = [d.split("\t") for d in data]
        for tp in data:
            if len(tp[0]) + len(tp[1]) > max_seq_len - 2:
                continue
            examples.append(ExampleInstance(segments=tp.copy()))

        def get_stats(l):
            return f"Max={max(l)}, min={min(l)}, avg={np.mean(l)}"

        ctx_token_size = [max(len(w) for w in e.segments) for e in examples]
        logger.info(
            f"Statistics: {get_stats(ctx_token_size)}, \
                  long={len([t for t in ctx_token_size if t > 500])}/{len(ctx_token_size)}"
        )
        return examples

    def example_to_feature(
        self, tokenizer, example, max_seq_len=512, rng=None, tril_matrix=None, **kwargs
    ):
        if not rng:
            rng = random
        features = OrderedDict()
        pad_id = tokenizer.convert_tokens_to_ids(["[PAD]"])
        source = list(example.segments[0])
        target = list(example.segments[1])
        truncate_tokens_pair(
            source, target, max_len=max_seq_len - 4, always_truncate_tail=True
        )
        token_ids = ["[CLS]"] + source + ["[SEP]"] + ["[CLS]"] + target + ["[SEP]"]
        token_type_ids = [0] * (len(source) + 2) + [1] * (len(target) + 2)
        padding_size = max(0, max_seq_len - len(token_ids))
        features["input_ids"] = (
            tokenizer.convert_tokens_to_ids(token_ids) + pad_id * padding_size
        )
        features["labels"] = features["input_ids"]
        features["token_type_ids"] = token_type_ids + [0] * padding_size
        for f in features:
            features[f] = torch.tensor(features[f], dtype=torch.int)
        return features
