import json
import pickle 
import random
import os
from typing import Optional, Tuple, Dict, Union, List, Literal

from transformers import TrainingArguments
from token_data import omit_tokens
from trainers import CustomDPOTrainer
from utils import token_entropy
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


# дефолтные параметры для обучения
# lr выглядит сомнительно, возможно, нужно поменять
# взял те же, которыми дообучали лламу, вроде
training_args = TrainingArguments(
    output_dir="experiment_logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    fp16=False,
    do_train=True,
    num_train_epochs=5,
    learning_rate=1e-5,
    disable_tqdm=True
)

# дефолтные параметры для генерации
generation_config = dict(
    top_k=50,
    num_beams=5,
    max_length=200,
    early_stopping=False,
    no_repeat_ngram_size=2,
    renormalize_logits=True,
    min_new_tokens=5,
    repetition_penalty=1.5,
    temperature=1.5,
)


# общий пайплайн для этого задания
class Pipeline:
    
    def __init__(
        self,
        dataset: str = None,
        device="cuda",
        logger=None,
        sft_model_name="lvwerra/gpt2-imdb",
        reward_model_name="lvwerra/distilbert-imdb",
        rl_trainer=CustomDPOTrainer,
        training_args=training_args,
        generation_config=generation_config
    ):
        """
        Common class for tuning GPT2 models with DPOTrainer
        
        Args:
            dataset: hf dataset for model training, specify for deterministic learning
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
        if dataset is not None:
            self.dataset = DatasetDict.load_from_disk(dataset)
        else:
            self.dataset = None
        self.rl_trainer = rl_trainer
        
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
        
    def sample_from_dataset(self, n_samples, test_size=.2, random_state=None):
        """
        Method for generating dataset from winner-loser common pool.
        Useful for making training more non-deterministic
        """
        if random_state is not None:
            np.random.seed(random_state)
        samples = np.load("src/samples.npz", allow_pickle=True)["arr_0"]
        reward = np.load("src/rewards.npz")["arr_0"]
        dpo_dataset = defaultdict(list)

        winner, loser = reward >= 2.5, reward < 1.75
        winner_samples, loser_samples = samples[winner], samples[loser]
        winner_sample = np.random.choice(winner_samples, size=n_samples)
        loser_sample = np.random.choice(loser_samples, size=n_samples)
        
        prompts = ["This movie is great. "]*n_samples
        dpo_dataset["prompt"] = prompts
        dpo_dataset["chosen"] = winner_sample
        dpo_dataset["rejected"] = loser_sample

        dpo_dataset = Dataset.from_dict(dpo_dataset)
        dpo_dataset = dpo_dataset.map(batched=True).train_test_split(test_size=test_size)

        return dpo_dataset
        
    def generate_from_input(
        self,
        inp: Optional[str] = "",
        debug: Optional[bool] = False,
        naive_guidance: Optional[str] = None,
        sequence_bias: Optional[Dict[str, float]] = None,
    ):
        """
        Generation pipeline from prompt. Allows empty strings and/or
        naively guided responses with simple positive and negative prompts
        """
        guidance_dict = {
            "positive": f"This movie is great. {inp}",
            "negative": f"This movie is bad. {inp}",
            None: "..." if inp == "" else inp
        }
        inp = guidance_dict[naive_guidance].rstrip()
        bad_words_ids = [
            ids[1:] for ids in 
            self.sft_tokenizer(omit_tokens, add_special_tokens=False).input_ids
        ]
        input_ids = self.sft_tokenizer.encode(
            inp, return_tensors='pt',
            add_special_tokens=True,
            add_prefix_space=False,
            padding=True
        ).to(self.device)
        beam_output = self.sft_model.generate(
            input_ids, do_sample=True,
            sequence_bias=sequence_bias,
            bad_words_ids=bad_words_ids,
            **self.generation_config
        ).to(self.device)
        output = self.sft_tokenizer.decode(
            beam_output[0], skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        if debug:
            print(output)
        return ".".join(output.split(".")[:-1]) + "."
    
    def generate_samples(self, n_samples, naive_guidance=None):
        """
        Function that generates n_samples with 
        Pipeline.generate_from_input settings
        """
        samples = np.empty(n_samples, dtype=object)
        for i in trange(n_samples, desc="Generating", leave=False):
            samples[i] = self.generate_from_input(naive_guidance=naive_guidance,)
        return samples
    
    def train(self, n_samples=120, random_state=None, **train_kwargs):
        """
        Train function, similar to Trainer with optional DPOTrainer params.
        such as loss_type={'sigmoid', 'hinge'} and
        divergence={'RKL_divergence', 'alpha_divergence', "JS_divergence'},
        more in utils.py
        """
        if self.dataset is not None:
            dataset == self.dataset
        else:
            dataset = self.sample_from_dataset(n_samples=n_samples, random_state=random_state)
        dpo_trainer = self.rl_trainer(
            self.sft_model,
            self.sft_model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.sft_tokenizer,
            args=self.training_args,
            **train_kwargs
        )
        dpo_trainer.train()
        
    def _calculate_single_reward(self, text, as_proba=False):
        """Calculate reward with reward_model. Optionally scale to [0, 1]"""
        inputs = self.reward_tokenizer(text, return_tensors="pt").to(self.device)
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
        print((samples, reward, diversity))
        print(self.logger)
        if log:
            self.logger.loc[save_name] = (samples, reward, diversity)
        if save:
            np.save(f"src/samples_{save_name}", samples)
            np.save(f"src/reward_{save_name}", reward)
            (self.logger \
             .reset_index(names=["experiment"]) \
             .explode(["sample", "reward"]) \
             .to_csv(f"experiments/{save_name}.csv", index=0))
        return diversity
    
    def reinit(self, *args, **kwargs):
        self.__init__(*args, **kwargs)
        