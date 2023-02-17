seed=42
DATA=/notebooks/data/speech_commands
SAVE=/notebooks/checkpoints/mega/speech
CHUNK=1000

mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

# Mega base
model=mega_sc_raw_base
python -u train.py ${DATA} \
    --seed $seed --ddp-backend c10d --find-unused-parameters \
    -a ${model} --task speech_commands --encoder-normalize-before \
    --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --optimizer adam --lr 0.01 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
    --dropout 0.0 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.01 \
    --batch-size 20 --sentence-avg --update-freq 1 --max-update 250000 \
    --lr-scheduler linear_decay --total-num-update 250000 --end-learning-rate 0.0 \
    --warmup-updates 10000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --required-batch-size-multiple 1 \
    --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0 \
    --sentence-class-num 10 --max-positions 16000 --sc-dropped-rate 0. 
    