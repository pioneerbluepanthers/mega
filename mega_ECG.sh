#run_name="lr=0.001_eps=1e-4"

ADAMEPS=1e-8
LR=0.015
ADAMB1=0.89
ADAMB2=0.98
EHD=75 #at most, 240 - reasonable number, 48
FFNEMBED=170
ENCODERLAYER=5 # 3, 4, 5, 6
WD=0.01
run_name="lr=${LR}_B1=${ADAMB1}_eps=${ADAMEPS}_B2=${ADAMB2}_EHD=${EHD}_ENCODERLAYER=${ENCODERLAYER}_FFNEMBED=${FFNEMBED}_WD=${WD}"
#run_name="test1"
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
    --criterion lra_multilabel_bce --best-checkpoint-metric multi_auroc --maximize-best-checkpoint-metric \
    --optimizer adam --lr $LR --adam-betas "(${ADAMB1}, ${ADAMB2})" --adam-eps $ADAMEPS --clip-norm 1.0 \
    --dropout 0.0 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay $WD \
    --batch-size 20 --sentence-avg --update-freq 1 --max-update 62500 \
    --lr-scheduler linear_decay --total-num-update 62500 --end-learning-rate 0.0 \
    --warmup-updates 2500 --warmup-init-lr '1e-07' --keep-last-epochs 1 --required-batch-size-multiple 1 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --sentence-class-num 5 --max-positions 1000 --encoder-embed-dim 12 --wandb-project "ECG multilabel" --encoder-hidden-dim $EHD\
    --encoder-layers $ENCODERLAYER --encoder-ffn-embed-dim $FFNEMBED
    #--wandb-id $run_name
    
    