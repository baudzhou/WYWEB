# WYWEB
An evaluation bentchmark for classical Chinese. This work has been accepted by Findings of ACL 2023.

Classical Chinese is a treasure of the entire human cultural history. We contribute this work with the hope of helping the entire community to be more prosperous. Our work will be an open, community-driven project which improves with the advancement of technology.

We hope more people join in to make this benchmark better and more useful.
# Leader-board
## Online leader-board
See [WYWEB on CADAL](www.cadal.zju.edu.cn/wyweb) for the official leader-board.
## Main

| **Models** | **Avg.** | **PUNC** | **GLNER** | **GJC** | **FSPC** | **TLC** | **Xuci** | **WYWRC** | **IRC** |
|---------------------|-------------------|-------------------|--------------------|------------------|-------------------|------------------|-------------------|--------------------|------------------|
| Human               | 88.0              | 92.4              | 94.3               | 90.3             | 80.0              | 89.0             | 85.3              | 80.0               | 92.3             |
| DeBERTa-base        | **75.9**     | **83.3**     | **86.7**      | **85.2**    | 61.1              | 86.7             | 72.4              | **45.1**      | 86.7             |
| GuwenBERT-base      | 72.9              | 82.5              | 82.8               | 84.8             | 61.3              | 85.1             | 71.7              | 28.0               | 86.8             |
| GuwenBERT-large     | 75.6              | 83.1              | 86.1               | 84.9             | 58.5              | **87.6**    | 73.4              | 44.4               | **87.8**    |
| GuwenBERT-base-fs   | 74.6              | 82.9              | 84.8               | 84.2             | 61.0              | 86.7             | 70.0              | 42.1               | 85.3             |
| RoBERTa-CCBC        | 74.5              | 82.5              | 84.7               | 84.5             | 59.5              | 85.0             | 73.2              | 40.7               | 86.1             |
| RoBERTa-CCLC        | 75.3              | 82.8              | 86.1               | 84.7             | 58.6              | 87.1             | **74.9**     | 41.0               | 86.9             |
| SikuBERT            | 73.7              | 80.8              | 82.8               | 82.2             | 60.9              | 82.4             | 70.4              | 44.0               | 85.8             |
| SikuRoBERTa         | 73.5              | 81.4              | 82.8               | 82.5             | **62.2**    | 83.8             | 68.5              | 41.0               | 85.8             |
| RoBERTa-wwm-ext     | 72.1              | 78.8              | 79.8               | 81.3             | 59.2              | 78.3             | 71.0              | 42.1               | 86.2             |

## WYWMT

| **Model**                                      | **BLEU** | **chrF2** | **TER** | **ROUGE-1** | **ROUGE-2** | **ROUGE-L** |
|---------------------------------------------------------|-------------------|--------------------|---------------------------|----------------------|----------------------|----------------------|
| Human                                                   | 45.6              | 44.2               | 34.4                      | 77.4                 | 50.7                 | 76.2                 |
| guwenbert-base                                          | **40.1**    | **38.1**      | 37.5                      | **72.5**        | **46.0**        | **70.3**       |
| guwenbert-large                                         | 38.8              | 37.2               | 38.1                      | 70.1                 | 43.7                 | 67.7                 |
| guwenbert-base-fs                                       | 36.3              | 35.2               | 39.2                      | 68.3                 | 41.2                 | 65.7                 |
| roberta-CCBC                                            | 39.1              | 37.1               | 36.8                      | 71.4                 | 44.9                 | 69.3                 |
| roberta-CCLC                                            | 39.8              | 38.0               | 36.4                      | 71.6                 | 45.3                 | 69.3                 |
| SikuBERT                                                | 38.8              | 36.2               | 37.9                      | 72.0                 | 45.5                 | 69.8                 |
| SikuRoBERTa                                             | 39.1              | 36.5               | 37.7                      | 72.2                 | 45.7                 | 70.0                 |
| DeBERTa-base                                            | 39.5              | 37.8               | **35.9**             | 71.9                 | 44.2                 | 68.7                 |
| Roberta-wwm-ext                                         | 38.0              | 35.8               | 39.1                      | 69.9                 | 43.2                 | 66.7                 |


# How to test new models?
This is an evaluation benchmark for classical Chinese NLP providing several tasks. Researchers could quickly evaluate pre-trained language models with a few lines of code using the evaluation toolkit.
## Quick Run The Base Line

```
python run.py  \
                --tag wywweb \
                --do_train \
                --max_seq_len 512 \
                --dump 1000 \
                --task_name GJCTask \
                --data_dir data/tasks/gjc \
                --output_dir output/deberta/GJCTask \
                --num_train_epochs 6 \
                --model_dir_or_name bozhou/DeBERTa-base \
                --learning_rate 2e-5 \
                --train_batch_size 48 \
                --fp16 True \
                --workers 4 \
                --warmup 1000 
```
## Test your model and contact us to update the leader board.
- test your model on every task.
- get the best dev set score, use this model to evaluate test set.
- send result of the test set to us.
- maintainers validate the result and then update the leader board.
# Task Description

| **Task** | **Train** | **Dev** | **Test** | **Description** | **Metric** | **Source** |
|-----------------|--------------------|------------------|-------------------|--------------------------|---------------------|---------------------|
| PUNC            | 90k                | 20k              | 20k               | Sequence labeling        | F1                  | Authoritative Texts |
| TLC             | 28k                | 6k               | 6k                | Sentence classification  | Accuracy            | Ancient prose       |
| GJC             | 100k               | 20k              | 20k               | Sentence classification  | Accuracy            | Daizhige            |
| XuCi            | 800                | 200              | 200               | Token similarity         | Accuracy            | Exam papers         |
| WYWRC           | 3k                 | 500              | 500               | Reading comprehension    | Accuracy            | Exam papers         |
| IRC             | 3k                 | 1k               | 1k                | Reading comprehension    | Accuracy            | Exam papers         |
| WYWMT           | 20k                | 3k               | 3k                | Machine Translation      | BLEU                | online              |
| GLNER           | 80k                | 18k              | 18k               | Sequence labeling        | F1                  | \citet{GULIAN2020}  |
| FSPC            | 3000               | 1000             | 1000              | Sentence classification  | Accuracy            | THU-FSPC            |

# Cite us
    @inproceedings{zhou-etal-2023-wyweb,
        title = "{WYWEB}: A {NLP} Evaluation Benchmark For Classical {C}hinese",
        author = "Zhou, Bo  and
          Chen, Qianglong  and
          Wang, Tianyu  and
          Zhong, Xiaomi  and
          Zhang, Yin",
        booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
        month = jul,
        year = "2023",
        address = "Toronto, Canada",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.findings-acl.204",
        doi = "10.18653/v1/2023.findings-acl.204",
        pages = "3294--3319"
    }
