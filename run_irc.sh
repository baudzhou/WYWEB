python run.py  \
                --tag wywweb \
                --do_train \
                --max_seq_len 256 \
                --dump 1000 \
                --task_name IRCTask \
                --data_dir data/tasks/irc \
                --output_dir output/deberta/IRCTask \
                --num_train_epochs 6 \
                --model_dir_or_name bozhou/DeBERTa-base \
                --learning_rate 1e-5 \
                --train_batch_size 64 \
                --fp16 True \
                --workers 4 \
                --warmup 300
