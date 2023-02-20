#run_name="lr=0.001_eps=1e-4"
run_name="multilabel_debug"
ADAMEPS=1e-4
seed=42
DATA=/notebooks/data/physionet.org/files/ptb-xl/1.0.3/records100Processed
SAVE=/notebooks/checkpoints/mega/ecg/records100/$run_name
CHUNK=1000

mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

# Mega base
model=mega_ecg_raw_base
python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task ecg --encoder-normalize-before \
    --criterion lra_multilabel_bce --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.001 --adam-betas '(0.9, 0.98)' --adam-eps $ADAMEPS --clip-norm 1.0 \
    --dropout 0.0 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.01 \
    --batch-size 20 --sentence-avg --update-freq 1 --max-update 62500 \
    --lr-scheduler linear_decay --total-num-update 62500 --end-learning-rate 0.0 \
    --warmup-updates 2500 --warmup-init-lr '1e-07' --keep-last-epochs 1 --required-batch-size-multiple 1 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --sentence-class-num 5 --max-positions 1000 --encoder-embed-dim 12 --wandb-project "ECG softmax" \
    #--wandb-id $run_name
    