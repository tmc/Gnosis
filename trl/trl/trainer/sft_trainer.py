# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

import torch
import torch.nn as nn
import transformers
from accelerate import PartialState, logging
from datasets import Dataset, IterableDataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
    is_wandb_available,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.utils import is_peft_available

from ..data_utils import (
    apply_chat_template,
    is_conversational,
    is_conversational_from_value,
    maybe_convert_to_chatml,
    pack_dataset,
    prepare_multimodal_messages,
    truncate_dataset,
)
from ..models import clone_chat_template, get_act_offloading_ctx_manager, prepare_peft_model
from .sft_config import SFTConfig
from .utils import entropy_from_logits, flush_left, generate_model_card, get_comet_experiment_url, pad


if is_peft_available():
    from peft import PeftConfig, PeftModel

if is_wandb_available():
    import wandb

logger = logging.get_logger(__name__)

TListOrMapping = TypeVar("TListOrMapping", list, Mapping)


def remove_none_values(example: TListOrMapping) -> TListOrMapping:
    """
    Recursively removes entries with `None` values from a nested structure (list or dictionary).
    """
    if isinstance(example, list):
        return [remove_none_values(value) if isinstance(value, (dict, list)) else value for value in example]
    elif isinstance(example, Mapping):
        return {
            key: remove_none_values(value) if isinstance(value, (dict, list)) else value
            for key, value in example.items()
            if value is not None
        }
    else:
        raise TypeError("Input must be a list or a dictionary.")


@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    LM collator with support for sequence-level `correctness_label` (B,)  # >>> correctness
    """

    pad_token_id: int
    completion_only_loss: bool = False
    padding_free: bool = False
    return_position_ids: bool = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]

        # Check if we have meaningful seq_lengths from packing (restarting sequences)
        has_packed_position_ids = self.return_position_ids and "seq_lengths" in examples[0] and self.padding_free

        # For packing with position_ids, we should NOT create attention_mask as it causes
        # FlashAttention to ignore position_ids and compute wrong cu_seq_lens from the all-1s mask
        if not has_packed_position_ids:
            attention_mask = [torch.ones_like(input_ids) for input_ids in input_ids]

        if self.return_position_ids:
            if "seq_lengths" in examples[0]:
                position_ids = self.get_position_ids_from_packed_seq_lengths(
                    [example["seq_lengths"] for example in examples]
                )
            else:
                position_ids = [torch.arange(len(ids)) for ids in input_ids]
        if "labels" in examples[0]:
            labels = [torch.tensor(example["labels"]) for example in examples]
        else:
            labels = [torch.tensor(example["input_ids"]) for example in examples]
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = [torch.tensor(example["completion_mask"]) for example in examples]
        if "assistant_masks" in examples[0]:
            assistant_masks = [torch.tensor(example["assistant_masks"]) for example in examples]

        # If padding_free, flatten everything into a single sequence
        output = {}
        if self.padding_free:
            input_ids = [torch.cat(input_ids, dim=0)]
            if not has_packed_position_ids:
                attention_mask = [torch.cat(attention_mask, dim=0)]
            if self.return_position_ids:
                position_ids = [torch.cat(position_ids, dim=0)]
            labels = [torch.cat(labels, dim=0)]
            if self.completion_only_loss and "completion_mask" in examples[0]:
                completion_mask = [torch.cat(completion_mask, dim=0)]
            if "assistant_masks" in examples[0]:
                assistant_masks = [torch.cat(assistant_masks, dim=0)]

        # Pad std fields
        output["input_ids"] = pad(
            input_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        if not has_packed_position_ids:
            output["attention_mask"] = pad(
                attention_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
        if self.return_position_ids:
            output["position_ids"] = pad(
                position_ids, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
        output["labels"] = pad(
            labels, padding_value=-100, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        )
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = pad(
                completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][completion_mask == 0] = -100  # mask everything that is not in the completion
        if "assistant_masks" in examples[0]:
            assistant_masks = pad(
                assistant_masks, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
            )
            output["labels"][assistant_masks == 0] = -100

        # >>> correctness: pass through batch-level correctness labels (no padding)
        if "correctness_label" in examples[0]:
            cls = [float(ex.get("correctness_label", -1)) for ex in examples]
            output["correctness_label"] = torch.tensor(cls, dtype=torch.float32)

        return output

    @staticmethod
    def get_position_ids_from_packed_seq_lengths(batch_seq_lengths: list[list[int]]) -> list[torch.Tensor]:
        # Get lengths per row
        example_lengths = [sum(seq_lengths) for seq_lengths in batch_seq_lengths]
        # Flat list of lengths
        batch_seq_lengths = torch.tensor(
            [seq_length for seq_lengths in batch_seq_lengths for seq_length in seq_lengths]
        )
        position_ids = torch.ones(sum(example_lengths), dtype=batch_seq_lengths.dtype)
        position_ids[0] = 0
        # Reset position ids to 0 at the start of each sequence
        position_ids[batch_seq_lengths[:-1].cumsum(0)] = -(batch_seq_lengths[:-1] - 1)
        position_ids = position_ids.cumsum(0)
        # Split back into one tensor per example
        return list(position_ids.split(example_lengths))


@dataclass
class DataCollatorForVisionLanguageModeling(DataCollatorMixin):
    """
    VLM collator with support for sequence-level `correctness_label` (B,)  # >>> correctness
    """

    processor: ProcessorMixin
    max_length: Optional[int] = None
    completion_only_loss: bool = False
    pad_to_multiple_of: Optional[int] = None
    dataset_text_field: str = "text"
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        if "messages" in examples[0] or self.dataset_text_field in examples[0]:
            if self.completion_only_loss:
                raise ValueError(
                    "The `completion_only_loss` argument is not supported for language modeling datasets."
                )
            return self._collate_language_modeling(examples)
        elif "prompt" in examples[0] and "completion" in examples[0]:
            return self._collate_prompt_completion(examples)
        else:
            raise KeyError(f"Unexpected input keys in examples: {list(examples[0].keys())}.")

    def _collate_language_modeling(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        images = [example["images"] for example in examples]

        if "messages" in examples[0]:  # conversational case
            for example in examples:
                prepare_multimodal_messages(example["messages"], len(example["images"]))
            messages = [example["messages"] for example in examples]
            texts = self.processor.apply_chat_template(messages)
        elif self.dataset_text_field in examples[0]:  # standard case
            texts = [example[self.dataset_text_field] for example in examples]
        else:
            raise KeyError("The input examples must contain either 'messages' or 'text'.")

        output = self.processor(
            images=images,
            text=texts,
            padding=True,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=self.max_length is not None,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            add_special_tokens=False,
        )
        labels = output["input_ids"].clone()
        labels[output["attention_mask"] == 0] = -100
        output["labels"] = labels

        # >>> correctness
        if "correctness_label" in examples[0]:
            cls = [float(ex.get("correctness_label", -1)) for ex in examples]
            output["correctness_label"] = torch.tensor(cls, dtype=torch.float32)

        return output

    def _collate_prompt_completion(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        if self.pad_to_multiple_of is not None:
            raise NotImplementedError("pad_to_multiple_of not yet implemented for VLM prompt-completion.")
        images = [example["images"] for example in examples]
        if is_conversational(examples[0]):  # conversational case
            for example in examples:
                prepare_multimodal_messages(example["prompt"] + example["completion"], len(example["images"]))
            examples = [apply_chat_template(example, self.processor) for example in examples]

        prompts = [example["prompt"] for example in examples]
        completions = [example["completion"] for example in examples]

        processed_prompts = self.processor(
            images=images,
            text=prompts,
            padding=True,
            padding_side="left",
            return_tensors=self.return_tensors,
            add_special_tokens=False,
        )
        processed_completions = self.processor(
            text=completions,
            padding=True,
            padding_side="right",
            return_tensors=self.return_tensors,
            add_special_tokens=False,
        )

        # Concatenate prompts and completions
        prompt_ids, completion_ids = processed_prompts["input_ids"], processed_completions["input_ids"]
        prompt_mask, completion_mask = processed_prompts["attention_mask"], processed_completions["attention_mask"]
        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
        completion_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)

        # Flush left to reduce padding
        attention_mask, input_ids, completion_mask = flush_left(attention_mask, input_ids, completion_mask)

        # Truncate if necessary
        if self.max_length is not None:
            input_ids = input_ids[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]
            completion_mask = completion_mask[:, : self.max_length]

        # Create labels and mask padding tokens
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if self.completion_only_loss:
            labels[completion_mask == 0] = -100

        # Build the output dictionary
        output = processed_prompts  # contains the images tensor(s)
        output["input_ids"] = input_ids
        output["attention_mask"] = attention_mask
        output["labels"] = labels

        # >>> correctness
        if "correctness_label" in examples[0]:
            cls = [float(ex.get("correctness_label", -1)) for ex in examples]
            output["correctness_label"] = torch.tensor(cls, dtype=torch.float32)

        return output


class SFTTrainer(Trainer):
    """
    Trainer for Supervised Fine-Tuning (SFT), extended to support a sequence-level correctness head.

    Add this to your SFTConfig if you want to train the head only:
        train_only_correctness_head: bool = True  # optional, default False
    """

    _tag_names = ["trl", "sft"]

    def __init__(
        self,
        model: Union[str, nn.Module, PreTrainedModel],
        args: Optional[Union[SFTConfig, TrainingArguments]] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        optimizer_cls_and_kwargs: Optional[tuple[type[torch.optim.Optimizer], dict[str, Any]]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        formatting_func: Optional[Callable[[dict], str]] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = SFTConfig(f"{model_name}-SFT")
        elif isinstance(args, TrainingArguments) and not isinstance(args, SFTConfig):
            dict_args = args.to_dict()
            dict_args["hub_token"] = args.hub_token  # to_dict hides the hub_token
            dict_args.pop("push_to_hub_token")
            args = SFTConfig(**dict_args)

        # Model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            dtype = model_init_kwargs.get("dtype")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass
            elif isinstance(dtype, str) and dtype in ["bfloat16", "float16", "float32"]:
                dtype = getattr(torch, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError("Invalid `dtype` in SFTConfig.")
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            model = architecture.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                logger.warning("`model_init_kwargs` will be ignored because model is already instantiated.")

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model_id)

        # Handle pad token
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
            self._is_vlm = True
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
            self._is_vlm = False
        else:
            raise TypeError("processing_class must be a tokenizer or a processor")

        if args.eos_token is not None:
            eos_token = args.eos_token
            eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
            if eos_token_id is None:
                raise ValueError(f"eos_token '{eos_token}' not found in vocab of {processing_class.__class__.__name__}")
            tokenizer.eos_token_id = eos_token_id

        if args.chat_template_path is not None:
            if os.path.isfile(args.chat_template_path) and args.chat_template_path.endswith((".jinja", ".j2")):
                with open(args.chat_template_path, encoding="utf-8") as chat_template_file:
                    processing_class.chat_template = chat_template_file.read()
                added_tokens = []
            else:
                model, processing_class, added_tokens = clone_chat_template(
                    model, processing_class, args.chat_template_path
                )
        else:
            added_tokens = []

        # Sanity rules for VLMs
        if self._is_vlm and args.packing:
            raise ValueError("Packing is not supported for VLMs; set `packing=False`.")
        if self._is_vlm and args.padding_free:
            raise ValueError("Padding-free training is not supported for VLMs.")
        if self._is_vlm and args.assistant_only_loss:
            raise ValueError("assistant_only_loss is not supported for VLMs.")

        # PEFT wrapping
        if peft_config is not None:
            if added_tokens:
                if peft_config.trainable_token_indices is None:
                    peft_config.trainable_token_indices = {"embed_tokens": added_tokens}
                elif "embed_tokens" not in peft_config.trainable_token_indices:
                    peft_config.trainable_token_indices["embed_tokens"] = added_tokens
                else:
                    peft_config.trainable_token_indices["embed_tokens"].extend(added_tokens)
                if peft_config.modules_to_save is None or "lm_head" not in peft_config.modules_to_save:
                    logger.warning(
                        "Chat template added tokens but 'lm_head' not in PEFT modules_to_save; adding it to avoid gen issues."
                    )
                    if peft_config.modules_to_save is None:
                        peft_config.modules_to_save = ["lm_head"]
                    else:
                        peft_config.modules_to_save.append("lm_head")

        self.num_virtual_tokens = 0
        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            model = prepare_peft_model(model, peft_config, args)
            if model.active_adapter in model.peft_config:
                peft_model_config = model.peft_config[model.active_adapter]
                self.num_virtual_tokens = getattr(peft_model_config, "num_virtual_tokens", 0)

        # Data collator selection
        self.padding_free = args.padding_free or (args.packing and args.packing_strategy == "bfd")
        use_flash_attention = model.config._attn_implementation in [
            "flash_attention_2",
            "flash_attention_3",
            "kernels-community/vllm-flash-attn3",
        ]
        if self.padding_free:
            if data_collator is not None:
                raise ValueError("Custom collator not supported with padding-free.")
            if args.packing and args.packing_strategy == "wrapped":
                logger.warning("Using padding_free=True with 'wrapped' packing is not recommended.")
            if not use_flash_attention:
                logger.warning(
                    "Padding-free without FlashAttention can be unstable. Prefer attn_implementation='flash_attention_2'."
                )
            if args.per_device_train_batch_size == 1 and not args.packing:
                logger.warning("Batch size 1 with padding-free removes most benefits; consider >= 2.")

        # Decide completion-only
        dataset_sample = next(iter(train_dataset))
        if args.completion_only_loss is None:
            self.completion_only_loss = "prompt" in dataset_sample and "completion" in dataset_sample
        else:
            self.completion_only_loss = args.completion_only_loss

        if data_collator is None and not self._is_vlm:
            pad_token = args.pad_token or tokenizer.pad_token or tokenizer.eos_token
            pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
            if pad_token_id is None:
                raise ValueError(f"pad_token '{pad_token}' not found in vocab")
            data_collator = DataCollatorForLanguageModeling(
                pad_token_id=pad_token_id,
                completion_only_loss=self.completion_only_loss,
                padding_free=self.padding_free,
                return_position_ids=use_flash_attention,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )
        elif data_collator is None and self._is_vlm:
            data_collator = DataCollatorForVisionLanguageModeling(
                processor=processing_class,
                max_length=args.max_length,
                completion_only_loss=self.completion_only_loss,
                pad_to_multiple_of=args.pad_to_multiple_of,
                dataset_text_field=args.dataset_text_field,
            )

        if args.packing and args.packing_strategy == "bfd" and not use_flash_attention:
            logger.warning("Packing without FA2/FA3 can cause contamination; prefer FA2/FA3 or disable packing.")
        if args.assistant_only_loss and not is_conversational(dataset_sample):
            raise ValueError("assistant_only_loss=True requires a conversational dataset.")

        # Dataset preparation (skip for VLM or if requested)
        skip_prepare_dataset = (
            args.dataset_kwargs is not None and args.dataset_kwargs.get("skip_prepare_dataset", False) or self._is_vlm
        )
        if not skip_prepare_dataset:
            if self.completion_only_loss and formatting_func:
                raise ValueError(
                    "Provided `formatting_func` while `completion_only_loss=True`. Apply formatting upfront or disable."
                )
            train_dataset = self._prepare_dataset(
                train_dataset, processing_class, args, args.packing, formatting_func, "train"
            )
            if eval_dataset is not None:
                packing = args.packing if args.eval_packing is None else args.eval_packing
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, packing, formatting_func, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(
                        eval_dataset, processing_class, args, packing, formatting_func, "eval"
                    )

        # >>> freeze-head: optionally freeze everything but stop_head
        if getattr(args, "train_only_correctness_head", False):
            for p in model.parameters():
                p.requires_grad_(False)
            if not hasattr(model, "stop_head"):
                raise AttributeError("train_only_correctness_head=True but model has no `stop_head`")
            for p in model.stop_head.parameters():
                p.requires_grad_(True)
            # set frozen modules to eval() to disable their dropout
            for name in ("model", "lm_head", "attn_extractor", "conf_extractor", "hid_extractor"):
                m = getattr(model, name, None)
                if m is not None:
                    m.eval()
            model.stop_head.train()
            trainables = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            logger.info(f"[freeze] trainable params: {trainables:,}/{total:,} ({100*trainables/total:.5f}%)")

        # Initialize metrics, tokens
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Activation offloading context
        if self.args.activation_offloading:
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = contextlib.nullcontext()

        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: SFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Clean Nones (arrow backends often inject them)
        if isinstance(dataset, Dataset):  # IterableDataset does not support with_transform
            dataset = dataset.with_transform(remove_none_values)

        # Already tokenized?
        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        # Build map kwargs
        map_kwargs = {}
        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            if formatting_func is not None and is_processed:
                logger.warning(
                    "Dataset already processed (has `input_ids`) but formatting_func was provided; ignoring formatter."
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                dataset = dataset.map(_func, batched=False, **map_kwargs)

            if not is_processed:
                # Convert to ChatML if needed
                first_example = next(iter(dataset))
                if is_conversational_from_value(first_example):
                    if isinstance(dataset, Dataset):
                        map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
                    column_names = next(iter(dataset)).keys()
                    dataset = dataset.map(
                        maybe_convert_to_chatml,
                        remove_columns="conversations" if "conversations" in column_names else None,
                        **map_kwargs,
                    )

                # Add EOS if needed
                first_example = next(iter(dataset))
                if not is_conversational(first_example):
                    if isinstance(dataset, Dataset):
                        map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                    def add_eos(example, eos_token):
                        if "text" in example and not example["text"].endswith(eos_token):
                            example["text"] = example["text"] + eos_token
                        elif "completion" in example and not example["completion"].endswith(eos_token):
                            example["completion"] = example["completion"] + eos_token
                        return example

                    dataset = dataset.map(
                        add_eos,
                        fn_kwargs={"eos_token": processing_class.eos_token},
                        remove_columns="messages" if "messages" in column_names else None,
                        **map_kwargs,
                    )

                # Tokenize (+ carry correctness_label through)  # >>> correctness
                if isinstance(dataset, Dataset):
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                def tokenize(example, processing_class, dataset_text_field, assistant_only_loss):
                    output = {}
                    # (a) prompt-completion
                    if "prompt" in example:
                        if is_conversational(example):
                            prompt_ids = processing_class.apply_chat_template(
                                example["prompt"],
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            prompt_completion_processed = processing_class.apply_chat_template(
                                example["prompt"] + example["completion"],
                                return_dict=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            prompt_completion_ids = prompt_completion_processed["input_ids"]
                            if "assistant_masks" in prompt_completion_processed:
                                output["assistant_masks"] = prompt_completion_processed["assistant_masks"]
                        else:
                            prompt_ids = processing_class(text=example["prompt"])["input_ids"]
                            prompt_completion_ids = processing_class(text=example["prompt"] + example["completion"])[
                                "input_ids"
                            ]
                        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
                            logger.warning("Mismatch between tokenized prompt and start of tokenized prompt+completion.")

                        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
                        output["input_ids"] = prompt_completion_ids
                        output["completion_mask"] = completion_mask
                    # (b) language modeling
                    else:
                        if is_conversational(example):
                            processed = processing_class.apply_chat_template(
                                example["messages"],
                                return_dict=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            if "assistant_masks" in processed and 1 not in processed["assistant_masks"]:
                                raise RuntimeError(
                                    "assistant_only_loss=True but example has no assistant tokens; check chat template."
                                )
                            output = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}
                        else:
                            output = {"input_ids": processing_class(text=example[dataset_text_field])["input_ids"]}

                    # >>> correctness: carry sequence label if present
                    if "correctness_label" in example:
                        output["correctness_label"] = example["correctness_label"]
                    return output

                dataset = dataset.map(
                    tokenize,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "dataset_text_field": args.dataset_text_field,
                        "assistant_only_loss": args.assistant_only_loss,
                    },
                    **map_kwargs,
                )

            # Pack or truncate
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"

                columns = ["input_ids"]
                if "completion_mask" in dataset.column_names:
                    columns.append("completion_mask")
                if "assistant_masks" in dataset.column_names:
                    columns.append("assistant_masks")
                # >>> correctness: keep the sequence labels through packing
                if "correctness_label" in dataset.column_names:
                    columns.append("correctness_label")

                dataset = dataset.select_columns(columns)
                dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)

            elif args.max_length is not None:
                if isinstance(dataset, Dataset):
                    map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
                dataset = truncate_dataset(dataset, args.max_length, map_kwargs)

            # Liger kernel minimal columns
            if args.use_liger_kernel:
                collator_expected_keys = {"input_ids", "seq_lengths", "completion_mask", "assistant_masks"}
                # >>> correctness: allow correctness_label to survive
                if "correctness_label" in dataset.column_names:
                    collator_expected_keys.add("correctness_label")
                dataset = dataset.select_columns(collator_expected_keys.intersection(dataset.column_names))

        return dataset

    def _set_signature_columns_if_needed(self):
        # Keep correctness labels in the signature so Trainer doesn't drop them.  # >>> correctness
        if self._signature_columns is None:
            if self._is_vlm:
                self._signature_columns = ["messages", "prompt", "completion", "images", "correctness_label"]
            else:
                self._signature_columns = [
                    "input_ids",
                    "labels",
                    "seq_lengths",
                    "completion_mask",
                    "assistant_masks",
                    "correctness_label",  # <<<
                ]

    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     """
    #     Compute training loss, plus metrics. Loss is taken from the model's `forward`.
    #     If your model is the Qwen3 subclass with a correctness head, its `forward(training)`
    #     should return the BCE loss over correctness when `correctness_label` is provided.
    #     """
    #     mode = "train" if self.model.training else "eval"

    #     # Ensure no caching in training
    #     inputs["use_cache"] = False

    #     # >>> correctness: assert labels if training only the head
    #     if getattr(self.args, "train_only_correctness_head", False) and "correctness_label" not in inputs:
    #         raise RuntimeError(
    #             "train_only_correctness_head=True but 'correctness_label' missing from the batch."
    #         )

    #     # (loss, outputs) from the model; model is expected to compute the correct loss
    #     (loss, outputs) = super().compute_loss(
    #         model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
    #     )

    #     # Entropy metric (skip if using liger which returns no logits)
    #     if not self.args.use_liger_kernel and hasattr(outputs, "logits") and outputs.logits is not None:
    #         with torch.no_grad():
    #             per_token_entropy = entropy_from_logits(outputs.logits)
    #             if "attention_mask" in inputs:
    #                 attention_mask = inputs["attention_mask"]
    #                 virtual_attention_mask = torch.ones(
    #                     attention_mask.size(0), self.num_virtual_tokens, device=attention_mask.device
    #                 )
    #                 attention_mask = torch.cat((virtual_attention_mask, attention_mask), dim=1)
    #                 entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
    #             elif "position_ids" in inputs:
    #                 entropy = torch.mean(per_token_entropy)
    #             else:
    #                 raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
    #             entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
    #         self._metrics[mode]["entropy"].append(entropy)

    #     if mode == "train":
    #         if "attention_mask" in inputs:
    #             num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
    #         elif "position_ids" in inputs:
    #             local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
    #             num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
    #         else:
    #             raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
    #         self._total_train_tokens += num_tokens_in_batch
    #     self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

    #     # # Token accuracy (for visibility), if logits available
    #     # if "labels" in inputs and not self.args.use_liger_kernel and hasattr(outputs, "logits"):
    #     #     with torch.no_grad():
    #     #         shift_logits = outputs.logits[..., :-1, :].contiguous()
    #     #         shift_labels = inputs["labels"][..., 1:].contiguous()
    #     #         shift_logits = shift_logits[:, self.num_virtual_tokens :, :]
    #     #         predictions = shift_logits.argmax(dim=-1)
    #     #         mask = shift_labels != -100
    #     #         correct_predictions = (predictions == shift_labels) & mask
    #     #         total_tokens = mask.sum()
    #     #         correct_tokens = correct_predictions.sum()
    #     #         correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
    #     #         total_tokens = self.accelerator.gather_for_metrics(total_tokens)
    #     #         total_sum = total_tokens.sum()
    #     #         accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
    #     #         self._metrics[mode]["mean_token_accuracy"].append(accuracy)

    #     return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss, plus metrics. Loss is taken from the model's `forward`.
        If your model is the Qwen3 subclass with a correctness head, its `forward(training)`
        should return the BCE loss over correctness when `correctness_label` is provided.
        """
        mode = "train" if self.model.training else "eval"

        # Ensure no caching in training
        inputs["use_cache"] = False

        # >>> correctness: assert labels if training only the head
        if getattr(self.args, "train_only_correctness_head", False) and "correctness_label" not in inputs:
            raise RuntimeError(
                "train_only_correctness_head=True but 'correctness_label' missing from the batch."
            )


        # # --- DEBUG: log decoded prompt+completion once in a while ---
        # if getattr(self.args, "debug_log_prompt_completion", True) and self.accelerator.is_main_process:
        #     try:
        #         # Pick the first item in the batch
        #         ids = inputs["input_ids"][0].detach().cpu()
        #         # Determine effective length: prefer attention_mask, else labels!=-100 as a fallback
        #         if "attention_mask" in inputs:
        #             L = int(inputs["attention_mask"][0].sum().item())
        #         else:
        #             L = int((inputs.get("labels", inputs["input_ids"])[0] != -100).sum().item())
        #         ids = ids[:L]

        #         # Find where completion begins (if we have completion_mask)
        #         comp_start = None
        #         if "completion_mask" in inputs:
        #             cm = inputs["completion_mask"][0][:L].detach().cpu()
        #             nz = (cm == 1).nonzero(as_tuple=True)[0]
        #             if len(nz) > 0:
        #                 comp_start = int(nz[0].item())

        #         # Get a tokenizer (handles both tokenizer and processor.tokenizer for VLMs)
        #         tok = getattr(self.processing_class, "tokenizer", None) or self.processing_class
        #         decoded = tok.batch_decode([ids.tolist()], skip_special_tokens=False)[0]

        #         # Pretty marker for the boundary if known
        #         if comp_start is not None:
        #             # Re-decode segmented parts to avoid index->char mismatch
        #             dec_prompt     = tok.decode(ids[:comp_start].tolist(), skip_special_tokens=False)
        #             dec_completion = tok.decode(ids[comp_start:].tolist(), skip_special_tokens=False)
        #             logger.info(
        #                 f"[debug] input_len={L} comp_start={comp_start}\n"
        #                 f"--- PROMPT ---\n{dec_prompt}\n"
        #                 # f"--- COMPLETION ---\n{dec_completion}\n"
        #             )
        #         else:
        #             logger.info(f"[debug] input_len={L} (no completion_mask)\n{decoded}\n")

        #     except Exception as e:
        #         logger.warning(f"[debug] prompt+completion logging failed: {e}")


        # (loss, outputs) from the model; model is expected to compute the correct loss
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # # Entropy metric (skip if using liger which returns no logits)
        # if not self.args.use_liger_kernel and hasattr(outputs, "logits") and outputs.logits is not None:
        #     with torch.no_grad():
        #         per_token_entropy = entropy_from_logits(outputs.logits)
        #         if "attention_mask" in inputs:
        #             attention_mask = inputs["attention_mask"]
        #             virtual_attention_mask = torch.ones(
        #                 attention_mask.size(0), self.num_virtual_tokens, device=attention_mask.device
        #             )
        #             attention_mask = torch.cat((virtual_attention_mask, attention_mask), dim=1)
        #             entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
        #         elif "position_ids" in inputs:
        #             entropy = torch.mean(per_token_entropy)
        #         else:
        #             raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
        #         entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
        #     self._metrics[mode]["entropy"].append(entropy)

        if mode == "train":
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # # === C orrectness-head accuracy (GRPO-style hygiene) ====================
        # # Extract stop_prob (preferred) / correctness_prob / correctness_logit from outputs
        # # and compare against correctness labels (allow -1 => skip).
        # if ("correctness_label" in inputs) or ("correctness_labels" in inputs):
        #     with torch.no_grad():
        #         # ---- labels (+ shape hygiene, allow singular or plural key)
        #         raw_labels = inputs.get("correctness_label", None)
        #         if raw_labels is None:
        #             raw_labels = inputs.get("correctness_labels")
        #         labels = raw_labels
        #         if labels is None:
        #             return (loss, outputs) if return_outputs else loss

        #         # squeeze trailing singleton dimension if present
        #         if labels.dim() == 2 and labels.size(-1) == 1:
        #             labels = labels.squeeze(-1)
        #         labels = labels.to(dtype=torch.float32)

        #         # ---- pick prob source from outputs
        #         probs = None
        #         if hasattr(outputs, "stop_prob") and (outputs.stop_prob is not None):
        #             probs = outputs.stop_prob
        #         elif hasattr(outputs, "correctness_prob") and (outputs.correctness_prob is not None):
        #             probs = outputs.correctness_prob
        #         elif hasattr(outputs, "correctness_logit") and (outputs.correctness_logit is not None):
        #             probs = torch.sigmoid(outputs.correctness_logit)

        #         if probs is not None:
        #             # shape hygiene for probs
        #             if probs.dim() == 2 and probs.size(-1) == 1:
        #                 probs = probs.squeeze(-1)

        #             # ---- sanitize & clamp in FP32 (no in-place)
        #             probs = probs.to(torch.float32)
        #             probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
        #             probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)

        #             # ---- flatten & align lengths to avoid mask shape mismatches
        #             labels_f = labels.reshape(-1)
        #             probs_f = probs.reshape(-1)
        #             L = min(labels_f.numel(), probs_f.numel())
        #             if L == 0:
        #                 return (loss, outputs) if return_outputs else loss

        #             labels_f = labels_f[:L]
        #             probs_f = probs_f[:L]

        #             # skip rows with -1 labels
        #             keep = labels_f.ne(-1.0)
        #             if not torch.any(keep):
        #                 return (loss, outputs) if return_outputs else loss

        #             # restrict to valid rows
        #             probs_v  = probs_f[keep]
        #             labels_v = labels_f[keep]
        #             labels_v = torch.nan_to_num(labels_v, nan=0.0)
        #             labels_v = torch.clamp(labels_v, 0.0, 1.0)

        #             # ---- binarize @ 0.5 and compute accuracy
        #             preds_v  = (probs_v >= 0.5).to(dtype=labels_v.dtype)
        #             acc_val  = (preds_v == labels_v).float().mean()

        #             # let pending CUDA errors surface, but don't die if they do
        #             try:
        #                 torch.cuda.synchronize()
        #             except Exception:
        #                 pass

        #             # ---- CPU-safe gather (mean across processes)
        #             acc_cpu = acc_val.detach().to("cpu")
        #             try:
        #                 acc_world = self.accelerator.gather_for_metrics(acc_cpu).mean().item()
        #             except Exception:
        #                 acc_world = acc_cpu.item()

        #             self._metrics[mode].setdefault("correctness_acc", []).append(acc_world)
        #             pm = probs_f.detach().float().cpu()
        #             self._metrics[mode].setdefault("probs_mean", []).append(pm.mean().item())
        #             self._metrics[mode].setdefault("probs_max",  []).append(pm.max().item())
        #             self._metrics[mode].setdefault("probs_min",  []).append(pm.min().item())

        # return (loss, outputs) if return_outputs else loss

        # === Correctness-head accuracy (GRPO-style hygiene) ====================
        # Extract stop_prob (preferred) / correctness_prob / correctness_logit from outputs
        # and compare against correctness labels (allow -1 => skip).
        if ("correctness_label" in inputs) or ("correctness_labels" in inputs):
            with torch.no_grad():
                # ---- labels (+ shape hygiene, allow singular or plural key)
                raw_labels = inputs.get("correctness_label", None)
                if raw_labels is None:
                    raw_labels = inputs.get("correctness_labels")
                labels = raw_labels
                if labels.dim() == 2 and labels.size(-1) == 1:
                    labels = labels.squeeze(-1)
                labels = labels.to(dtype=torch.float32)

                # skip rows with -1 labels
                keep = (labels != -1)
                
                if torch.any(keep):
                    # ---- pick prob source from outputs
                    probs = None
                    if hasattr(outputs, "stop_prob") and (outputs.stop_prob is not None):
                        probs = outputs.stop_prob
                    elif hasattr(outputs, "correctness_prob") and (outputs.correctness_prob is not None):
                        probs = outputs.correctness_prob
                    elif hasattr(outputs, "correctness_logit") and (outputs.correctness_logit is not None):
                        probs = torch.sigmoid(outputs.correctness_logit)

                    # print(labels, probs)
                    if probs is not None:
                        # shape hygiene
                        if probs.dim() == 2 and probs.size(-1) == 1:
                            probs = probs.squeeze(-1)

                        # ---- sanitize & clamp in FP32 (no in-place)
                        probs = probs.to(torch.float32)
                        probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
                        probs = torch.clamp(probs, 1e-6, 1.0 - 1e-6)

                        # restrict to valid rows
                        probs_v  = probs[keep]
                        labels_v = labels[keep]
                        labels_v = torch.nan_to_num(labels_v, nan=0.0)
                        labels_v = torch.clamp(labels_v, 0.0, 1.0)

                        # ---- binarize @ 0.5 and compute accuracy
                        preds_v  = (probs_v >= 0.5).to(dtype=labels_v.dtype)
                        acc_val  = (preds_v == labels_v).float().mean()

                        # let pending CUDA errors surface, but don't die if they do
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass

                        # ---- CPU-safe gather (mean across processes)
                        acc_cpu = acc_val.detach().to("cpu")
                        try:
                            acc_world = self.accelerator.gather_for_metrics(acc_cpu).mean().item()
                        except Exception:
                            acc_world = acc_cpu.item()

                        self._metrics[mode].setdefault("correctness_acc", []).append(acc_world)
                        pm = probs.detach().float().cpu()
                        self._metrics[mode].setdefault("probs_mean", []).append(pm.mean().item())
                        self._metrics[mode].setdefault("probs_max",  []).append(pm.max().item())
                        self._metrics[mode].setdefault("probs_min",  []).append(pm.min().item())

        return (loss, outputs) if return_outputs else loss

    # Override training step to add activation offloading context.
    def training_step(self, *args, **kwargs):
        with self.maybe_activation_offload_context:
            return super().training_step(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs.update(metrics)
        super().log(logs, start_time)
        self._metrics[mode].clear()

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        # normalize `tags` to a mutable set
        if tags is None:
            tags = set()
        elif isinstance(tags, str):
            tags = {tags}
        else:
            tags = set(tags)

        if hasattr(self.model.config, "unsloth_version"):
            tags.add("unsloth")

        if "JOB_ID" in os.environ:
            tags.add("hf_jobs")

        tags.update(self._tag_names)

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=list(tags),
            wandb_url=wandb.run.url if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="SFT",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
