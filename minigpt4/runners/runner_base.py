"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import webdataset as wds
import wandb
import peft
from peft.peft_model import PeftModelForCausalLM
from minigpt4.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import is_url
from minigpt4.datasets.data_utils import concat_datasets, reorg_datasets_by_split, ChainDataset
from minigpt4.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from minigpt4.processors.blip_processors import Blip2ImageTrainProcessor,BlipCaptionProcessor
from minigpt4.datasets.datasets.video_datasets import Video_validation_Dataset
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION
from tqdm import tqdm
from omegaconf import OmegaConf

from utils import init_logger
import os
program = os.path.basename(__file__)
if os.path.exists(f"../logs/{os.path.splitext(program)[0]}.log"):
    os.remove(f"../logs/{os.path.splitext(program)[0]}.log")
logger = init_logger(program)

@registry.register_runner("runner_base")
class RunnerBase:
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets = datasets

        self._model = model

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0

        # self.setup_seeds()
        self.setup_output_dir()

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)

        return self._device

    @property
    def use_distributed(self):
        return self.config.run_cfg.distributed

    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # move model to device
        # logger.info("self device",self.device)
        # logger.info("self model device",self._model.device)
        
        # logger.info(self._model.device, self.device)

        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            # distributed training wrapper
            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = DDP(
                        self._model, device_ids=[self.config.run_cfg.gpu],find_unused_parameters=False # False
                    )
                    #
            else:
                self._wrapped_model = self._model
        logger.info(f"DDP MODEL DEVICE - {self._wrapped_model.device_ids}")
        return self._wrapped_model

    @property
    def optimizer(self):
        # TODO make optimizer class and configurations
        if self._optimizer is None:
            num_parameters = 0
            p_wd, p_non_wd = [], []
            for n, p in self.model.named_parameters():
                # if p.requires_grad and p.grad is None:
                #     logger.info(f"{n} did not receive gradients")
                if not p.requires_grad:
                    continue  # frozen weights
                # logger.info(n)
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
                num_parameters += p.data.nelement()
            logger.info("number of trainable parameters: %d" % num_parameters)
            optim_params = [
                {
                    "params": p_wd,
                    "weight_decay": float(self.config.run_cfg.weight_decay),
                },
                {"params": p_non_wd, "weight_decay": 0},
            ]
            beta2 = self.config.run_cfg.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run_cfg.init_lr),
                weight_decay=float(self.config.run_cfg.weight_decay),
                betas=(0.9, beta2),
            )

        return self._optimizer

    @property
    def scaler(self):
        amp = self.config.run_cfg.get("amp", False)
        # logger.info("amp", amp)
        # assert False


        if amp:
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()

        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

            # max_epoch = self.config.run_cfg.max_epoch
            max_epoch = self.max_epoch
            # min_lr = self.config.run_cfg.min_lr
            min_lr = self.min_lr
            # init_lr = self.config.run_cfg.init_lr
            init_lr = self.init_lr

            # optional parameters
            decay_rate = self.config.run_cfg.get("lr_decay_rate", None)
            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)
            iters_per_epoch = self.config.run_cfg.get("iters_per_epoch", None)

            if iters_per_epoch is None:
                try:
                    iters_per_epoch = len(self.dataloaders['train'])
                except (AttributeError, TypeError):
                    iters_per_epoch = 10000

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                iters_per_epoch=iters_per_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )

        return self._lr_sched

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:

            # concatenate map-style datasets and chain wds.DataPipe datasets separately
            # training set becomes a tuple (ConcatDataset, ChainDataset), both are
            # optional but at least one of them is required. The resultant ConcatDataset
            # and ChainDataset will be sampled evenly.
            logger.info(
                "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
            )

            batch_sizes = {dataset_name: getattr(self.config.datasets_cfg, dataset_name).batch_size
                           for dataset_name in self.datasets.keys()}

            datasets, batch_sizes = reorg_datasets_by_split(self.datasets, batch_sizes)
            self.datasets = datasets
            # self.datasets = concat_datasets(datasets)
            # logger.info dataset statistics after concatenation/chaining
            for split_name in self.datasets:
                logger.info(f"SPLIT NAME {split_name}")
                if isinstance(self.datasets[split_name], tuple) or isinstance(
                    self.datasets[split_name], list
                ):
                    # mixed wds.DataPipeline and torch.utils.data.Dataset
                    num_records = sum(
                        [
                            len(d)
                            if not type(d) in [wds.DataPipeline, ChainDataset]
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )

                else:
                    if hasattr(self.datasets[split_name], "__len__"):
                        # a single map-style dataset
                        num_records = len(self.datasets[split_name])
                    else:
                        # a single wds.DataPipeline
                        num_records = -1
                        logger.info(
                            "Only a single wds.DataPipeline dataset, no __len__ attribute."
                        )

                if num_records >= 0:
                    logger.info(
                        "Loaded {} records for {} split from the dataset.".format(
                            num_records, split_name
                        )
                    )

            # create dataloaders
            split_names = sorted(self.datasets.keys())
            # datasets = [self.datasets[split] for split in split_names]
            datasets = [self.datasets[split][0] for split in split_names]
            # batch_sizes = [batch_sizes[split] for split in split_names]
            batch_sizes = [batch_sizes[split][0] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            # batch_sizes = [
            #     self.config.run_cfg.batch_size_train
            #     if split == "train"
            #     else self.config.run_cfg.batch_size_eval
            #     for index, split in enumerate(split_names)
            # ]

            # logger.info(split_names)
            # logger.info(f"batch sizes { batch_sizes}")

            collate_fns = []
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                else:
                    collate_fns.append(getattr(dataset, "collater", None))
            

            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config.run_cfg.num_workers,
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
            )
            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}
            logger.info(f"DATALOADERS - {self._dataloaders}")
        return self._dataloaders

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return int(self.config.run_cfg.max_epoch)

    @property
    def log_freq(self):
        log_freq = self.config.run_cfg.get("log_freq", 50)
        return int(log_freq)

    @property
    def init_lr(self):
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run_cfg.min_lr)

    @property
    def accum_grad_iters(self):
        return int(self.config.run_cfg.get("accum_grad_iters", 1))

    @property
    def valid_splits(self):
        valid_splits = self.config.run_cfg.get("valid_splits", [])

        if len(valid_splits) == 0:
            logger.info("No validation splits found.")

        return valid_splits

    @property
    def test_splits(self):
        test_splits = self.config.run_cfg.get("test_splits", [])

        return test_splits

    @property
    def train_splits(self):
        train_splits = self.config.run_cfg.get("train_splits", [])

        if len(train_splits) == 0:
            logger.info("Empty train splits.")

        return train_splits

    @property
    def evaluate_only(self):
        """
        Set to True to skip training.
        """
        return self.config.run_cfg.evaluate

    @property
    def use_dist_eval_sampler(self):
        return self.config.run_cfg.get("use_dist_eval_sampler", True)

    @property
    def resume_ckpt_path(self):
        return self.config.run_cfg.get("resume_ckpt_path", None)

    @property
    def train_loader(self):
        train_dataloader = self.dataloaders["train"]

        return train_dataloader

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))

        output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id
        # output_dir = lib_root / self.config.run_cfg.output_dir
        result_dir = output_dir / "result"
        adapter_output_dir = output_dir / "adapter_output"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)
        adapter_output_dir.mkdir(parents=True, exist_ok=True)
        
        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))
        registry.register_path("adapter_output_dir", str(adapter_output_dir))
        
        self.result_dir = result_dir
        self.output_dir = output_dir
        self.adapter_output_dir = adapter_output_dir

    @staticmethod
    def plot(
        split:str,
        output_dir:str
    )->None:
        import glob
        import matplotlib.pyplot as plt
        import numpy as np
        
        acc = np.array([
            json.load(file_path)[split]['agg_metrics'] 
            for file_path in glob.glob(f"{output_dir}/acc*.json")
        ]).astype(np.float32)
        
        loss = np.array([
            json.load(file_path)['train_loss'] 
            for file_path in glob.glob(f"{output_dir}/loss*.json")
        ]).astype(np.float32)

        plt.plot(np.arange(acc.shape[0]), acc, label='accuracy')
        plt.plot(np.arange(loss.shape[0]), loss, label='loss')
        plt.savefig(f"{output_dir}/plot.png")
    
    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0
        self.log_config()

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for cur_epoch in range(self.start_epoch, self.max_epoch):           
            # validation phase
            # if len(self.valid_splits) > 0:
            #     logger.info("Validation start")
            #     val_log = {}
            #     for split_name in self.valid_splits:
            #         val_log[split_name] = self.eval_epoch(
            #             split_name=split_name,cur_epoch=cur_epoch
            #         )
            #     agg_metrics = val_log[self.valid_splits[-1]]["agg_metrics"]
            #     self._save_checkpoint(cur_epoch, is_best=agg_metrics > best_agg_metric)
            #     early_stopping += 1
            #     if agg_metrics > best_agg_metric:
            #         early_stopping -= 1
            #         best_epoch, best_agg_metric = cur_epoch, agg_metrics
            #         val_log.update({"best_epoch": best_epoch})
                    
            #     self.log_stats(val_log, split_name=self.valid_splits[-1])
            #     wandb.log({"epoch": cur_epoch, "GPT4_Accuracy": val_log[self.valid_splits[-1]]['agg_metrics']})
            #     logger.info("Validation finished") 
                
            # training phase
            if not self.evaluate_only:
                logger.info("Start training")
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)
                
            if not self.evaluate_only:
                self._save_checkpoint(cur_epoch, is_best=False)
             
            if self.evaluate_only:#or float(train_stats["loss"]) < 0.08:
                break 

            if self.config.run_cfg.distributed:
                dist.barrier()

        # testing phase
        test_logs = self.evaluate(cur_epoch=cur_epoch, skip_reload=self.evaluate_only)
        with open(os.path.join(self.output_dir, f"epoch_{cur_epoch + 1}.json"), "w") as f:
            f.write(json.dumpsd(test_logs))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("Training time {}".format(total_time_str))

    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )

            return test_logs

    def train_epoch(self, epoch):
        # train
        self.model.train()

        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get(split_name, None)
        # logger.info(f"EVAL DATALOADER - {split_name} {data_loader}")
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.evaluation(model, data_loader)

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )
            
    def get_validation_loader(self):
        # TODO make the path configurable
        dataset_congif="minigpt4/configs/datasets/video_chatgpt/default.yaml"
        # read the dataset config using omegaconf 
        config = OmegaConf.load(dataset_congif).datasets
        config = config[list(config.keys())[0]]
        vis_processor=Blip2ImageTrainProcessor()
        validation_data = Video_validation_Dataset(vis_processor,
                                videos_path=config.valid['videos_path'],
                                ann_path=config.valid['ann_path'],
                                subtitles_path=config.valid['subtitles_path'],
                                annotations_keys=config.valid['annotations_keys'],
                                add_subtitles=config.valid['add_subtitles'],)
        validation_dataloader = DataLoader(validation_data, batch_size=1, shuffle=False)
        return validation_dataloader
    
    @torch.no_grad()
    def custom_eval_epoch(self, cur_epoch):
        validation_dataloader=self.get_validation_loader()
        model = self.unwrap_dist_model(self.model)
        model.eval()
        conv_temp = CONV_VISION.copy()
        conv_temp.system = ""
        results = []
        for images, texts, gt_answers, lengths,videos_ids in tqdm(validation_dataloader):
            texts = prepare_texts(texts, conv_temp, template='', lengths=lengths)  # warp the texts with conversation template
            models_answers = model.generate(images, texts, max_new_tokens=512, do_sample=False, lengths=lengths,num_beams=1)
            for video_id,model_answer, gt_answer,text in zip(videos_ids,models_answers, gt_answers,texts):
                result = dict()
                result['video_name'] = video_id
                result['Q'] = text.split('\n')[-1].replace('[/INST]','')
                result['A'] = gt_answer
                result['pred'] = model_answer
                results.append(result)
        val_log= self.task.after_evaluation(
                            val_result=results,
                            epoch=cur_epoch,
                        )
        return val_log
    
    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # create a single dataloader for each split
            if isinstance(dataset, ChainDataset) or isinstance(
                dataset, wds.DataPipeline
            ):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                # map-style dataset are concatenated together
                # setup distributed sampler

                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        # e.g. retrieval evaluation
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
                loader = PrefetchLoader(loader)

                if is_train:
                    # logger.info("IS TRAIN")
                    loader = IterLoader(loader, use_distributed=self.use_distributed)

            return loader

        loaders = []

        for dataset, bsz, is_train, collate_fn in zip(
            datasets, batch_sizes, is_trains, collate_fns
        ):
            if isinstance(dataset, list) or isinstance(dataset, tuple):
                if hasattr(dataset[0], 'sample_ratio') and dataset_ratios is None:
                    dataset_ratios = [d.sample_ratio for d in dataset]
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz[i], is_train, collate_fn[i])
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

            loaders.append(loader)

        return loaders

    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        logger.info(f"SAVING CHECKPOINT - {cur_epoch} - {is_best}")
        if isinstance(model_no_ddp.llama_model, peft.peft_model.PeftModelForCausalLM):
            logger.info("Saving adapter model")
            model_no_ddp.llama_model.save_pretrained(self.adapter_output_dir)
            
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logger.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")

        logger.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            logger.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        message = self.unwrap_dist_model(self.model).load_state_dict(state_dict,strict=False)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        logger.info("resume the checkpoint")
        logger.info("Resume checkpoint from {}".format(url_or_filename))

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")
