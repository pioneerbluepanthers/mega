run_name="initial_run"
seed=42
DATA=/notebooks/data/physionet.org/files/ptb-xl/1.0.3/records100Processed
SAVE=/notebooks/checkpoints/mega/ecg/records100/$run_name
CHECKPOINT_PATH=$SAVE/checkpoint_best.pt
CHUNK=1000

mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

# Mega base

python validate.py $DATA --task ecg --batch-size 60 --valid-subset test --path $CHECKPOINT_PATH