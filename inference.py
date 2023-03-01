#!/usr/bin/env python3 -u
#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from itertools import chain
import logging
import sys
import yaml
import torch
import json
from fairseq import checkpoint_utils, distributed_utils, options, utils
from flask import Flask

app = Flask(__name__)
with open("args.yaml") as args_f:
    args = yaml.unsafe_load(args_f)


#with open("args.json", "w") as args_f:
    #json.dump(args, args_f, indent=2)
#print("Args:", args.__dict__)
utils.import_user_module(args)
#print("Args_max_tokens", args.max_tokens)
#print("Args_max_sentences", args.max_sentences)
assert args.max_tokens is not None or args.max_sentences is not None, \
    'Must specify batch size either with --max-tokens or --max-sentences'

use_fp16 = args.fp16
use_cuda = torch.cuda.is_available() and not args.cpu

if use_cuda:
    torch.cuda.set_device(args.device_id)


# Load ensemble
args.path = "/notebooks/checkpoints/mega/ecg/records100/lr=0.01_B1=0.9_eps=1e-8_B2=0.98/checkpoint_best.pt"
models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
    [args.path],
    suffix=getattr(args, "checkpoint_suffix", ""),
)
model = models[0]

# Move models to GPU
for model in models:
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

target = torch.load("/notebooks/examples/targets.pt", map_location="cuda:0")
src_tokens = torch.load("/notebooks/examples/src_tokens.pt", map_location="cuda:0")
src_lengths = torch.load("/notebooks/examples/src_lengths.pt", map_location="cuda:0")
print("target_device:", target.device)

labels = ["NORM", "MI", "STTC", "CD", "HYP"]
@app.route("/infer")
def infer():
    sample = {
        "id":torch.tensor(0, device=args.device_id),
        "nsentences":1,
        "ntokens":1000, 
        "net_input":{"src_tokens":src_tokens, "src_lengths":src_lengths},
        "target":target
    }
    net_output = model(sample)
    lprobs = model.get_multilabel_probs(net_output)
    probs = torch.exp(lprobs).tolist()[0]
    print("Target:", target)
    return {"probs":{condition: prob for condition, prob in zip(labels, probs)}}