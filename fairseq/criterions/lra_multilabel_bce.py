# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from torchmetrics.classification import MultilabelAUROC
from torchmetrics.classification import MultilabelF1Score
from torchmetrics.classification import MultilabelAccuracy
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import numpy as np

@register_criterion('lra_multilabel_bce')
class LRAMultilabelBCECriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, save_predictions=False, save_path="/notebooks/predictions/run.npy"):
        super().__init__(task)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sentence_avg = sentence_avg
        self.log_threshold = torch.log(torch.tensor(0.5))
        
        self.acc = MultilabelAccuracy(num_labels=5, average="macro").to(self.device)
        
        self.f1 = MultilabelF1Score(num_labels=5,
        average="macro").to(self.device)
        self.auroc = MultilabelAUROC(num_labels=5,
        average="macro").to(self.device)
        self.save_predictions = save_predictions
        self.save_path = save_path
        self.predictions = []
        self.targets = []

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(sample)
        loss, correct = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'ncorrects': correct.data,
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        #print("multilabel_bce")
        lprobs = model.get_multilabel_probs(net_output)
        probs = torch.exp(lprobs)
        if self.save_predictions:
            probs_np = probs.cpu().detach().numpy()
            self.predictions.append(probs_np)
            self.targets.append(sample["target"].cpu().detach().numpy())
        #lprobs = lprobs.view(-1, lprobs.size(-1))
        
        targets = sample["target"]
        #print("targets:", targets)
        loss = F.binary_cross_entropy_with_logits(lprobs, targets, reduction='sum')
        preds = lprobs >= self.log_threshold
        correct = (preds == targets).sum() # TODO: implement AUROCs
        self.acc.update(probs, targets.long())
        self.f1.update(probs, targets.long())
        self.auroc.update(probs, targets.long())
        #print("Self_auroc_compute", self.auroc.compute())
        #print("probs:", probs)
        
        #print("Targets_lra_shape:", targets.shape)
        
        return loss, correct
    
    def on_epoch_end(self):
        acc = self.acc.compute()
        f1 = self.f1.compute()
        auroc = self.auroc.compute()
        metrics.log_scalar('multi_accuracy', acc, round=5)
        metrics.log_scalar('multi_f1', f1, round=5)
        metrics.log_scalar('multi_auroc', auroc, round=5)
        self.acc.reset()
        self.f1.reset()
        self.auroc.reset()
        
        if self.save_predictions:
            targets = np.vstack(self.targets)
            #print("targets_on_epoch_end:", targets.shape)
            np.save("/notebooks/predictions/targets.npy", targets, allow_pickle=True)
            predictions = np.vstack(self.predictions)
            np.save(self.save_path, predictions, allow_pickle=True)
        return {"multi_accuracy":acc.item(), "multi_f1":f1.item(), "multi_auroc":auroc.item()}
        
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        #print("Reduce Metrics")
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if len(logging_outputs) > 0 and 'ncorrects' in logging_outputs[0]:
            ncorrects = sum(log.get('ncorrects', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrects / nsentences, nsentences, round=3)
    
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
