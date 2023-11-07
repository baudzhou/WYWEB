python run.py  \
                --tag wywweb \
                --do_train \
                --max_seq_len 96 \
                --dump 1000 \
                --task_name XuciTASK \
                --data_dir data/tasks/xuci \
                --output_dir output/deberta/XuciTASK \
                --num_train_epochs 20 \
                --model_dir_or_name bozhou/DeBERTa-base \
                --learning_rate 1e-5 \
                --train_batch_size 30 \
                --workers 4 \
                --fp16 True