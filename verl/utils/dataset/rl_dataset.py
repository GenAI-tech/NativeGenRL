# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from omegaconf import ListConfig
import os
from typing import List, Union, Optional
import copy
import pandas as pd
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from datasets import load_dataset, load_from_disk, concatenate_datasets


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 image_key='images',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 filter_overlong_prompts=False):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
                tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                                                                 axis=1)]

            print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.image_key)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if is_multi_modal:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()

class JanusTextOnlyRLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """
    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key=None,
                 image_key=None,
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir=None,
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 filter_overlong_prompts=False,
                 system_prompt="",
                 prompt_template=None,
                 cot_generate=False,
                 ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts
        self.system_prompt = system_prompt
        self.cot_generate = cot_generate
        self.prompt_template = prompt_template
        self.data_source_list = []
        if self.prompt_template is None:
            self.prompt_template = "A photo of {}."
        
        self.prompts = []
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = os.path.expanduser(parquet_file)
            file_name = os.path.basename(self.parquet_files[i]).replace('.txt', '')
            if self.parquet_files[i].endswith('.txt'):
                with open(self.parquet_files[i], 'r') as f:
                    prompts = f.readlines()
                for prompt in prompts:
                    if self.filter_overlong_prompts:
                        if len(tokenizer.encode(prompt, add_special_tokens=False))+10 > self.max_prompt_length: # +5 for <user>, <assistant>, <img_start>, etc..
                            continue
                    self.prompts.append(prompt.strip())
                    self.data_source_list.append(file_name)
                        
                # self.prompts.extend([prompt.strip() for prompt in prompts])
                # self.data_source_list.extend([file_name] * len(prompts))
            else:
                raise ValueError(f"Unsupported file format: {self.parquet_files[i]}")
        
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = {}

        chat = [
            {
                "role": "<|User|>",
                "content": self.prompt_template.format(self.prompts[item]),
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=chat,
            sft_format=self.processor.sft_format,
            system_prompt=self.system_prompt,
        )
        
        if not self.cot_generate:
            prompt = sft_format + self.processor.image_start_tag
        else:
            prompt = sft_format
        
        raw_prompt = prompt

        is_multi_modal = False
        assert not is_multi_modal, "JanusTextOnlyRLHFDataset only supports t2i data"
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        input_ids = input_ids[0]
        attention_mask = attention_mask[0]
        sentence_start_token, image_start_token = self.tokenizer.encode(self.processor.image_start_tag)
        input_ids = torch.cat([torch.LongTensor([self.tokenizer.pad_token_id]), input_ids])
        attention_mask = torch.cat([torch.LongTensor([0]), attention_mask])

        num_pad = torch.sum(input_ids == self.tokenizer.pad_token_id, dim=-1)
        last_pad_idx = num_pad - 1
        
        input_ids[last_pad_idx] = sentence_start_token
        attention_mask[last_pad_idx] = 1
        
        position_ids = compute_position_id_with_mask(attention_mask)
        
        row_dict['input_ids'] = input_ids
        row_dict['attention_mask'] = attention_mask
        row_dict['position_ids'] = position_ids
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        row_dict['data_source'] = self.data_source_list[item]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = self.prompts[item]

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
    
    
class DummyJanusDPORLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """
    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 image_key='images',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 filter_overlong_prompts=False,
                 system_prompt="",
                 prompt_template=None,
                 cot_generate=False,
                 ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts
        self.system_prompt = system_prompt
        self.cot_generate = cot_generate
        self.prompt_template = prompt_template
        self.dummy_prompts = [
            'One apple and two bananas on a plate',
            'A plate on the left of the cup',
            'A lamp on the left side of the bed, and a clock on the right',
            'A red apple on the left and a green apple on the right, both on a plate with a blue rim',
            'a plant in front of a jar',
            'a person above a leaf',
            'three suitcases and one bottle'
            'one screen and two watchs',
            'A dog lying on a rug in front of a fireplace',
            'two wines and two eggs',
            'A white mug in front of a green notebook on a desk',
            'a green bed and a khaki dish',
            'a brown bag and a pink car',
            'A pencil and an eraser beside an open notebook',
            'A cat sitting on the windowsill with a plant to its left',
            'A white book on top of a black book, which is on top of a red book',
            'A teddy bear on the right side of the pillow on a bed'
        ]
        
    def __len__(self):
        return 32

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = {}

        chat = [
            {
                "role": "<|User|>",
                "content": self.prompt_template.format(self.dummy_prompts[item%len(self.dummy_prompts)]),
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=chat,
            sft_format=self.processor.sft_format,
            system_prompt=self.system_prompt,
        )
        if not self.cot_generate:
            prompt = sft_format + self.processor.image_start_tag
        else:
            prompt = sft_format
        
        raw_prompt = prompt

        is_multi_modal = False
        assert not is_multi_modal, "JanusRLHFDataset only supports t2i data"
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        input_ids = input_ids[0]
        attention_mask = attention_mask[0]
        sentence_start_token, image_start_token = self.tokenizer.encode(self.processor.image_start_tag)
        input_ids = torch.cat([torch.LongTensor([self.tokenizer.pad_token_id]), input_ids, torch.LongTensor([image_start_token])])
        attention_mask = torch.cat([torch.LongTensor([0]), attention_mask, torch.LongTensor([1])])

        num_pad = torch.sum(input_ids == self.tokenizer.pad_token_id, dim=-1)
        last_pad_idx = num_pad - 1
        
        input_ids[last_pad_idx] = sentence_start_token
        attention_mask[last_pad_idx] = 1
        
        position_ids = compute_position_id_with_mask(attention_mask)
        
        row_dict['input_ids'] = input_ids
        row_dict['attention_mask'] = attention_mask
        row_dict['position_ids'] = position_ids
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()

class InterleaveDataset():
    def __init__(self, data_dir:str, split:str=None):
        self.data_dir = data_dir
        self.datasets = {}
        self.dataset_names = []  # keep track of the order of datasets
        self.test_datasets = {}
        self.split = split
        self.debug = False
        
        self.load_datasets()
        self.lengths = {name: len(dataset) for name, dataset in self.datasets.items()}
        self.total_length = sum(self.lengths.values())
        self.dataset_start_idx = {name: sum([self.lengths[n] for n in self.dataset_names[:i]]) for i, name in enumerate(self.dataset_names)}
        
    def load_single_dataset(self, data_path):
        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"
            
        def load(data_path):
            try:
                dataset = load_dataset(data_path, split=data_split)
                print(f"loading dataset: {data_path}")
            except Exception as e:
                print(f"loading dataset from disk: {data_path}")
                dataset = load_from_disk(data_path)
            return dataset
        
        if 'batch_0' in os.listdir(data_path): # if the dataset is splited into batches
            datasets = []
            for batch in os.listdir(data_path):
                path = os.path.join(data_path, batch)
                datasets.append(load(path))
            dataset = concatenate_datasets(datasets)
        else:
            dataset = load(data_path)
        
        return dataset
    
    def split_for_test(self):
        test_datasets = {}
        for dataset_name in self.dataset_names:
            dataset = self.datasets[dataset_name]
            dataset = dataset.train_test_split(
                test_size=0.9,
                shuffle=False,
                seed=0,
            )
            test_datasets[dataset_name] = dataset['test']
            self.datasets[dataset_name] = dataset['train']
            if self.debug:
                print("Debugging dataset")
                test_datasets[dataset_name] = test_datasets[dataset_name].select(range(8))
                self.datasets[dataset_name] = self.datasets[dataset_name].select(range(8))
        return self.datasets, test_datasets
        
    def load_datasets(self):
        if '@' in self.data_dir:
            data_path = self.data_dir.split('@')[0]
            name = os.path.basename(data_path)
            self.datasets[name] = self.load_single_dataset(data_path)
            self.dataset_names = [name]
            return
        
        sub_dirs = os.listdir(self.data_dir)
        json_files = [f for f in sub_dirs if f.endswith('.json')]
        if len(json_files) > 0:
            self.datasets[self.data_dir] = self.load_single_dataset(self.data_dir)
            self.dataset_names = [os.path.basename(self.data_dir)]
        else:
            for dataset_name in os.listdir(self.data_dir):
                data_path = os.path.join(self.data_dir, dataset_name)
                self.datasets[dataset_name] = self.load_single_dataset(data_path)
                self.dataset_names.append(dataset_name)
            if self.split is not None:
                self.datasets, self.test_datasets = self.split_for_test()
            if self.split == 'test':
                self.datasets = self.test_datasets
            
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, index):
        for dataset_name in self.dataset_names:
            if index < self.lengths[dataset_name]:
                item = self.datasets[dataset_name][index]
                item['data_source'] = dataset_name
                return item
            else:
                index -= self.lengths[dataset_name]
        raise IndexError("Index out of range")

class OmniTextOnlyRLHFDataset(Dataset):
    """
    Dataset for OmniGen2 text-to-image generation using RLHF
    We assume the dataset contains prompts for text-to-image generation
    """
    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key=None,
                 image_key=None,
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir=None,
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 filter_overlong_prompts=False,
                 system_prompt="You are a helpful assistant that generates high-quality images based on user instructions.",
                 prompt_template=None,
                 cot_generate=False,
                 ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts
        self.system_prompt = system_prompt
        self.cot_generate = cot_generate
        self.prompt_template = prompt_template
        self.data_source_list = []
        if self.prompt_template is None:
            self.prompt_template = "{}."
        
        self.prompts = []
        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = os.path.expanduser(parquet_file)
            file_name = os.path.basename(self.parquet_files[i]).replace('.txt', '')
            if self.parquet_files[i].endswith('.txt'):
                with open(self.parquet_files[i], 'r') as f:
                    prompts = f.readlines()
                for prompt in prompts:
                    if self.filter_overlong_prompts:
                        if len(tokenizer.encode(prompt, add_special_tokens=False))+10 > self.max_prompt_length: # +10 for system prompt, etc.
                            continue
                    self.prompts.append(prompt.strip())
                    self.data_source_list.append(file_name)
            else:
                raise ValueError(f"Unsupported file format: {self.parquet_files[i]}")
        
    def __len__(self):
        return len(self.prompts)

    def apply_chat_template(self, instruction, system_prompt):
        """Apply chat template following OmniGen2 format"""
        prompt = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": instruction},
        ]
        instruction = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        return instruction

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = {}

        # Format the prompt using the template
        formatted_prompt = self.prompt_template.format(self.prompts[item])
        
        # Apply chat template for OmniGen2
        # prompt = self.apply_chat_template(formatted_prompt, self.system_prompt)
        prompt = formatted_prompt  # no chat template for OmniGen2, since the model pipeline adds chat template internally
        
        # For text generation, we don't add image tokens initially
        # The model will generate them as part of the response
        if self.cot_generate:
            # For CoT generation, keep the prompt as is
            final_prompt = prompt
        else:
            # For direct image generation, we may need to add special formatting
            final_prompt = prompt
        
        raw_prompt = final_prompt

        is_multi_modal = False
        assert not is_multi_modal, "OmniTextOnlyRLHFDataset only supports text-to-image data"
        
        # TODO this needs to change to OmniGen2 token process, including mllm embedding
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=final_prompt,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        input_ids = input_ids[0]
        attention_mask = attention_mask[0]
        
        position_ids = compute_position_id_with_mask(attention_mask)
        
        row_dict['input_ids'] = input_ids
        row_dict['attention_mask'] = attention_mask
        row_dict['position_ids'] = position_ids
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        row_dict['data_source'] = self.data_source_list[item]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = self.prompts[item]

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()


class DummyOmniDPORLHFDataset(Dataset):
    """
    Dummy dataset for OmniGen2 text-to-image generation using RLHF for testing purposes
    """
    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 image_key='images',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 filter_overlong_prompts=False,
                 system_prompt="You are a helpful assistant that generates high-quality images based on user instructions.",
                 prompt_template=None,
                 cot_generate=False,
                 ):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts
        self.system_prompt = system_prompt
        self.cot_generate = cot_generate
        self.prompt_template = prompt_template
        if self.prompt_template is None:
            self.prompt_template = "A photo of {}."
            
        self.dummy_prompts = [
            'One apple and two bananas on a plate',
            'A plate on the left of the cup',
            'A lamp on the left side of the bed, and a clock on the right',
            'A red apple on the left and a green apple on the right, both on a plate with a blue rim',
            'a plant in front of a jar',
            'a person above a leaf',
            'three suitcases and one bottle'
            'one screen and two watchs',
            'A dog lying on a rug in front of a fireplace',
            'two wines and two eggs',
            'A white mug in front of a green notebook on a desk',
            'a green bed and a khaki dish',
            'a brown bag and a pink car',
            'A pencil and an eraser beside an open notebook',
            'A cat sitting on the windowsill with a plant to its left',
            'A white book on top of a black book, which is on top of a red book',
            'A teddy bear on the right side of the pillow on a bed'
        ]
        
    def __len__(self):
        return 32

    def apply_chat_template(self, instruction, system_prompt):
        """Apply chat template following OmniGen2 format"""
        prompt = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": instruction},
        ]
        instruction = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        return instruction

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = {}

        # Format the prompt using the template
        formatted_prompt = self.prompt_template.format(self.dummy_prompts[item % len(self.dummy_prompts)])
        
        # Apply chat template for OmniGen2
        prompt = self.apply_chat_template(formatted_prompt, self.system_prompt)
        
        # For text generation, we don't add image tokens initially
        # The model will generate them as part of the response
        if self.cot_generate:
            # For CoT generation, keep the prompt as is
            final_prompt = prompt
        else:
            # For direct image generation, we may need to add special formatting
            final_prompt = prompt
        
        raw_prompt = final_prompt

        is_multi_modal = False
        assert not is_multi_modal, "DummyOmniDPORLHFDataset only supports text-to-image data"
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=final_prompt,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        input_ids = input_ids[0]
        attention_mask = attention_mask[0]
        
        position_ids = compute_position_id_with_mask(attention_mask)
        
        row_dict['input_ids'] = input_ids
        row_dict['attention_mask'] = attention_mask
        row_dict['position_ids'] = position_ids
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = formatted_prompt

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()