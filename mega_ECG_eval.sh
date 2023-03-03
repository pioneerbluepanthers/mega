run_name="initial_run"
seed=42
DATA=/notebooks/data/physionet.org/files/ptb-xl/1.0.3/records100Processed
SAVE=/notebooks/checkpoints/mega/ecg/records100/$run_name
#CHECKPOINT_PATH=$SAVE/checkpoint_best.pt
CHECKPOINT_PATH="/notebooks/checkpoints/mega/ecg/records100/lr=0.015_B1=0.89_eps=1e-8_B2=0.98_EHD=75_ENCODERLAYER=5/checkpoint_best.pt"
CHUNK=1000

mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

# Mega base

python validate.py $DATA --task ecg --batch-size 200 --valid-subset test --path $CHECKPOINT_PATH --criterion lra_multilabel_bce 