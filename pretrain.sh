#! /bin/bash

NNODES=4
GPUS_PER_NODE=2
MASTER_PORT=63456
MASTER_ADDR=10.10.10.1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

LOAD_CHECKPOINT_PATH=./checkpoints/gpt2_encdec_4.7B_final_test$DATETIME/
SAVE_CHECKPOINT_PATH=./checkpoints/gpt2_encdec_4.7B_final_test$DATETIME/

TENSORBOARD_PATH=./tensorboard/gpt2_encdec_4.7B_final_test/$DATETIME


VOCAB_FILE=vocab.txt
DETAIL_FILE=detail.txt
LOG_FILE=log.txt
CONFIG_FILE=config.json
DATA_PATH=$(cat data_path_aug.txt)

python -u -m torch.distributed.launch $DISTRIBUTED_ARGS \
        ./pretrain_gpt.py \
        --tokenizer-type EncDecTokenizer \
        --vocab-file $VOCAB_FILE \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 4 \
        --num-layers 40 \
        --hidden-size 3072 \
        --num-attention-heads 24 \
        --seq-length 2048 \
        --max-position-embeddings 2048 \
        --micro-batch-size 4 \
        --global-batch-size 400\
        --train-samples 488322 \
        --rampup-batch-size 8 8 39200\
        --lr-decay-samples 430000 \
        --lr-warmup-samples 7200 \
        --lr 1.0e-04 \
        --min-lr 1.0e-05 \
        --lr-decay-style cosine \
        --log-interval 1 \
        --eval-iters -1 \
        --data-path ${DATA_PATH} \
        --save-interval 500 \
        --split 100,0,0 \
        --clip-grad 1.0 \
        --weight-decay 0.01 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --init-method-std 0.002 \
        --fp16 \
        --DDP-impl local \
        --save $SAVE_CHECKPOINT_PATH \
        --load $LOAD_CHECKPOINT_PATH \
        --checkpoint-activations \
        --checkpoint-num-layers 1 \
        --log-num-zeros-in-grad \
        --log-params-norm \
        --tensorboard-dir $TENSORBOARD_PATH \
        --tensorboard-log-interval 1 \
        --log-path-detail $DETAIL_FILE \
        --log-path $LOG_FILE \
        --num-workers 16 \
        --deepspeed \
        --deepspeed_config $CONFIG_FILE \
        --deepspeed-activation-checkpointing \