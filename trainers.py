import torch
from torch.nn import functional as F

from trl import DPOTrainer
from utils import *
from typing import Tuple


# DPOTrainer patch to support custom f-divergences
class CustomDPOTrainer(DPOTrainer):

    def __init__(
        self, *args,
        divergence=RKL_divergence,
        kto_divergence=KL_divergence,
        tolerance=1e8, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.divergence = divergence
        self.kto_divergence = kto_divergence
        self.tol = tolerance
        
    def _divergence_call(self, *args, divergence=None, **kwargs):
        if divergence is None:
            divergence = self.divergence
        return torch.clamp(divergence(*args, **kwargs), -self.tol, self.tol)
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_ratios = self._divergence_call(policy_chosen_logps, policy_rejected_logps)
        ref_ratios = self._divergence_call(reference_chosen_logps, reference_rejected_logps)

        logits = pi_ratios - ref_ratios

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (-F.logsigmoid(self.beta * logits))
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto":
            chosen_div = self._divergence_call(
                policy_chosen_logps, reference_chosen_logps, divergence=self.kto_divergence
            ).mean().clamp(min=0)
            rejected_div = self._divergence_call(
                policy_rejected_logps, reference_rejected_logps, divergence=self.kto_divergence
            ).mean().clamp(min=0)

            chosen_logratios = (policy_chosen_logps - reference_chosen_logps)
            rejected_logratios = (policy_rejected_logps - reference_rejected_logps)

            losses = torch.cat((
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_div)),
                1 - F.sigmoid(self.beta * (chosen_div - rejected_logratios))
            ), 0)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo']")

        chosen_rewards = self.beta*self._divergence_call(
            policy_chosen_logps, reference_chosen_logps
        ).detach()
        rejected_rewards = self.beta*self._divergence_call(
            policy_rejected_logps, reference_rejected_logps
        ).detach()    

        return losses, chosen_rewards, rejected_rewards
    