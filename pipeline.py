import json
import pickle
import random
import os
from typing import Tuple, Dict, Union, List, Literal

from transformers import TrainingArguments
from utils import *
from tqdm.auto import tqdm, trange

from collections import defaultdict
import gc
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.nn import functional as F

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import TrainingArguments
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from trl import DPOTrainer


# дефолтные параметры для обучения
# lr выглядит сомнительно, возможно, нужно поменять
# взял те же, которыми дообучали лламу
training_args = TrainingArguments(
    output_dir="experiments",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    evaluation_strategy="epoch",
    fp16=False,
    do_train=True,
    num_train_epochs=10,
    learning_rate=1e-6, # gpt2 learning rate
    warmup_steps=150,
)

# дефолтные параметры для генерации
generation_config = dict(
    top_k=50,
    num_beams=5,
    max_length=100,
    early_stopping=True,
    no_repeat_ngram_size=2,
    renormalize_logits=True
)

# патч для DPOTrainer, нужен новый параметр дивергенции
old_init = DPOTrainer.__init__
def new_init(self, *args, divergence=RKL_divergence, **kwargs):
    old_init(self, *args, **kwargs)
    self.divergence = divergence

    
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
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_ratios = self.divergence(policy_chosen_logps, policy_rejected_logps)
    ref_ratios = self.divergence(reference_chosen_logps, reference_rejected_logps)

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
    else:
        raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo']")

    chosen_rewards = self.beta * self.divergence(
        policy_chosen_logps, reference_chosen_logps
    ).detach()
    rejected_rewards = self.beta * self.divergence(
        policy_rejected_logps, reference_rejected_logps
    ).detach()    

    return losses, chosen_rewards, rejected_rewards

DPOTrainer.__init__ = new_init
DPOTrainer.dpo_loss = dpo_loss


# общий пайплайн для этого задания
class Pipeline:
    
    def __init__(
        self,
        dataset="dataset.hf",
        device="cuda",
        logger=None,
        sft_model_name="lvwerra/gpt2-imdb",
        reward_model_name="lvwerra/distilbert-imdb",
        training_args=training_args,
        generation_config=generation_config
    ):
        """
        Common class for tuning GPT2 models with DPOTrainer
        
        Args:
            dataset: what dataset to use for training/evaluating
            device: torch device, faster with 'cuda' obviously
            logger: pd.DataFrame for storing metrics
            sft_model_name: sft model to load from transformers
            reward_model_name: reward model to load from transformers
            training_args: training parameters for DPOTrainer
            generation_config: generation parameters for GeneratorMixin.generate
        """
        
        # pipeline params
        self.generation_config = generation_config
        self.training_args = training_args
        if logger is None:
            self.logger = pd.DataFrame(columns=["sample", "reward", "diversity"])
        else:
            self.logger = logger
        self.device = device
        self.dataset = DatasetDict.load_from_disk(dataset)
        
        # sft model
        self.sft_tokenizer = GPT2Tokenizer.from_pretrained(
            sft_model_name, padding_side='left',
        )
        self.sft_tokenizer.pad_token = self.sft_tokenizer.eos_token
        self.sft_model = GPT2LMHeadModel.from_pretrained(
            sft_model_name, pad_token_id=self.sft_tokenizer.pad_token_id,
        ).to(self.device) 
        
        # reward model
        self.reward_tokenizer = DistilBertTokenizer.from_pretrained(
            reward_model_name
        )
        self.reward_model = DistilBertForSequenceClassification.from_pretrained(
            reward_model_name
        ).to(self.device)
        self.proba_scaler = nn.Sigmoid()
        
    def generate_from_input(
        self, inp="",
        debug=False,
        naive_guidance=None,
    ):
        """
        Generation pipeline from prompt. Allows empty strings and/or
        naively guided responses with simple positive and negative prompts
        """
        guidance_dict = {
            "positive": f"Good Review: {inp}",
            "negative": f"Negative Review: {inp}",
            None: "..." if inp == "" else inp
        }
        inp = guidance_dict[naive_guidance].rstrip()
        input_ids = self.sft_tokenizer.encode(
            inp, return_tensors='pt',
            add_special_tokens=False,
            padding=True
        ).to(self.device)
        beam_output = self.sft_model.generate(
            input_ids, do_sample=True, **self.generation_config
        ).to(self.device)
        output = self.sft_tokenizer.decode(
            beam_output[0], skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if debug:
            print(output)
        return ".".join(output.split(".")[:-1]) + "."
    
    def generate_samples(self, n_samples, naive_guidance=None):
        """
        Function that generates n_Samples with 
        Pipeline.generate_from_input settings
        """
        
        samples = np.empty(n_samples, dtype=object)
        for i in trange(n_samples, desc="Generating", leave=False):
            samples[i] = self.generate_from_input(
                naive_guidance=naive_guidance,
            )
        return samples
    
    def train(self, **train_kwargs):
        """
        Train function, similar to Trainer with optional DPOTrainer params.
        such as loss_type={'sigmoid', 'hinge'} and
        divergence={'RKL_divergence', 'alpha_divergence', "JS_divergence'},
        more in utils.py
        """
        dpo_trainer = DPOTrainer(
            self.sft_model,
            self.sft_model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            tokenizer=self.sft_tokenizer,
            args=self.training_args,
            **train_kwargs
        )
        dpo_trainer.train()
        
    def _calculate_single_reward(self, text, as_proba=False):
        """Calculate reward with reward_model. Optionally scale to [0, 1]"""
        inputs = self.reward_tokenizer(
            text, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.reward_model(**inputs).logits
        if as_proba:
            logits = self.proba_scaler(logits)
        return logits[:, 1].cpu().numpy()
        
    def calculate_reward(self, dataset):
        return np.vectorize(self._calculate_single_reward)(dataset)
    
    def evaluate(
        self, save_name, save=True, log=True,
        diversity_func=token_entropy, n_samples=1000
    ):
        """Evaluation function. Generate, compute diversity, store if necessary"""
        samples = self.generate_samples(n_samples)
        reward = self.calculate_reward(samples)
        diversity = diversity_func(samples, self.sft_tokenizer)
        if log:
            self.logger.loc[save_name] = (samples, reward, diversity)
        if save:
            np.save(f"samples_{save_name}", samples)
            np.save(f"reward_{save_name}", reward)
            self.logger.reset_index(
                names=["experiment"]
            ).explode(
                ["sample", "reward"]
            ).to_csv(
                f"{save_name}.csv", index=0
            )
        return diversity
    
    def reinit(self, *args, **kwargs):
        self.__init__(*args, **kwargs)
        