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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video, fetch_image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from torchvision import io
from . import torch_functional as VF


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], resized_height: Optional[int]=None, resized_width: Optional[int]=None,
    min_pixels: Optional[int]=None, max_pixels: Optional[int]=None
) -> ImageObject:
    ele = {"image": image}
    if resized_height is not None and resized_width is not None:
        ele["resized_height"] = resized_height
        ele["resized_width"] = resized_width
    if min_pixels is not None:
        ele["min_pixels"] = min_pixels
    if max_pixels is not None:
        ele["max_pixels"] = max_pixels

    return fetch_image(ele, image_patch_size=16)


def process_video(
    video: str, nframes: Optional[int] = None, resized_height: Optional[int] = None, resized_width: Optional[int] = None,
    min_pixels: Optional[int] = None, max_pixels: Optional[int] = None, video_fps: Optional[float] = None, return_fps: bool = False, return_video_metadata: bool = False
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vi, au, inf = io.read_video(video, pts_unit='sec', output_format='TCHW')
    total_frames = vi.shape[0]
    vision_info = {"video": video}
    if nframes is not None:
        vision_info["nframes"] = min(nframes, total_frames)
    elif video_fps is not None:
        vision_info["fps"] = video_fps
    if resized_height is not None and resized_width is not None:
        vision_info["resized_height"] = resized_height
        vision_info["resized_width"] = resized_width
    if min_pixels is not None and max_pixels is not None:
        vision_info["min_pixels"] = min_pixels
        vision_info["max_pixels"] = max_pixels

    return fetch_video(vision_info, image_patch_size=16, return_video_sample_fps=return_fps, return_video_metadata=return_video_metadata)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        nframes: Optional[int] = None,
        resized_height: Optional[int] = 28,
        resized_width: Optional[int] = 28,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.nframes = nframes
        self.resized_height = resized_height
        self.resized_width = resized_width

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        # Sample 100 examples for debugging
        self.dataset = self.dataset.shuffle(seed=42).select(range(100))
        
        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        # Filter out examples with missing video/image paths
        initial_dataset_size = len(self.dataset)
        self.dataset = self.dataset.filter(
            self._filter_missing_paths,
            desc="Filtering missing video/image paths",
            num_proc=filter_overlong_prompts_workers,
        )
        filtered_dataset_size = len(self.dataset)
        num_filtered = initial_dataset_size - filtered_dataset_size
        print(f"Filtered {num_filtered} samples with missing video/image paths. "
                    f"Dataset size: {initial_dataset_size} -> {filtered_dataset_size}")

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )
            filtered_overlong_dataset_size = len(self.dataset)
            num_filtered_overlong = filtered_dataset_size - filtered_overlong_dataset_size
            print(f"Filtered {num_filtered_overlong} samples with overlong prompts. "
                  f"Dataset size: {filtered_dataset_size} -> {filtered_overlong_dataset_size}")


    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        # Check if example[self.prompt_key] is str or list
        if isinstance(example[self.prompt_key], list):
            prompt_str = example[self.prompt_key][0]["content"]
        elif isinstance(example[self.prompt_key], str):
            prompt_str = example[self.prompt_key]
        else:
            raise ValueError(f"Unsupported prompt type: {type(example[self.prompt_key])}")

        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key not in example and self.video_key not in example:
            return [{"role": "user", "content": prompt_str}]
        
        content_list = []
        if self.image_key in example and self.video_key in example and len(example[self.image_key]) > 0 and len(example[self.video_key]) > 0:
            # for i, content in enumerate(prompt_str.split("<video>\n<image>")):
            #     if i != 0:
            #         content_list.append({"type": "video"})
            #         content_list.append({"type": "image"})
            #     if content:
            #         content_list.append({"type": "text", "text": content})
            for i, content in enumerate(prompt_str.split("<video>\n<image>")):
                if i != 0:
                    content_list.append({"type": "video", "video": example[self.video_key][0], "resized_height": self.resized_height, "resized_width": self.resized_width, "fps": self.video_fps})
                    content_list.append({"type": "image", "image": example[self.image_key][0], "resized_height": self.resized_height, "resized_width": self.resized_width})
                if content:
                    content_list.append({"type": "text", "text": content})
        elif self.image_key in example and len(example[self.image_key]) > 0:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})
        elif self.video_key in example and len(example[self.video_key]) > 0:
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

        return [{"role": "user", "content": content_list}]

    def _filter_missing_paths(self, example: dict[str, Any]) -> bool:
        """
        Filter out examples where video or image paths don't exist.
        Returns True if the example should be kept, False if it should be skipped.
        """
        # Check video paths if video_key exists
        if self.video_key in example and len(example[self.video_key]) > 0:
            for video_path in example[self.video_key]:
                if isinstance(video_path, str):
                    full_path = video_path
                    if self.image_dir is not None:
                        full_path = os.path.join(self.image_dir, video_path)
                    if not os.path.exists(full_path):
                        print(f"Video path does not exist: {full_path}")
                        return False

        # Check image paths if image_key exists
        if self.image_key in example and len(example[self.image_key]) > 0:
            for image_path in example[self.image_key]:
                if isinstance(image_path, str):
                    full_path = image_path
                    if self.image_dir is not None:
                        full_path = os.path.join(self.image_dir, image_path)
                    if not os.path.exists(full_path):
                        print(f"Image path does not exist: {full_path}")
                        return False

        return True

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example and self.video_key in example and len(example[self.image_key]) > 0 and len(example[self.video_key]) > 0:
            # Handle both images and videos
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            videos = example[self.video_key]

            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, resized_height=self.resized_height, resized_width=self.resized_width, min_pixels=self.min_pixels, max_pixels=self.max_pixels))

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, nframes=self.nframes, resized_height=self.resized_height, resized_width=self.resized_width, min_pixels=self.min_pixels, max_pixels=self.max_pixels, video_fps=self.video_fps))

            model_inputs = self.processor(
                text=[prompt], images=processed_images, videos=processed_videos, add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.image_key in example and len(example[self.image_key]) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, resized_height=self.resized_height, resized_width=self.resized_width, min_pixels=self.min_pixels, max_pixels=self.max_pixels))

            model_inputs = self.processor(text=[prompt], images=processed_images, add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example and len(example[self.video_key]) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, nframes=self.nframes, resized_height=self.resized_height, resized_width=self.resized_width, min_pixels=self.min_pixels, max_pixels=self.max_pixels, video_fps=self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        if self.image_key in example and self.video_key in example and len(example[self.image_key]) > 0 and len(example[self.video_key]) > 0:            
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            processed_images, processed_videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True, image_patch_size=16)
            
            if processed_videos is not None:
                processed_videos, video_metadatas = zip(*processed_videos)
                processed_videos, video_metadatas = list(processed_videos), list(video_metadatas)
            else:
                video_metadatas = None
            video_fps_list = video_kwargs.get('fps', None)

            model_inputs = self.processor(text=[prompt], images=processed_images, videos=processed_videos, video_metadata=video_metadatas, return_tensors="pt", add_special_tokens=False, do_resize=True, padding=True, **video_kwargs)
            
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            
            images = example[self.image_key]
            videos = example[self.video_key]
            example["multi_modal_data"] = {"images": images, "videos": videos}
            
        elif self.image_key in example and len(example[self.image_key]) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, resized_height=self.resized_height, resized_width=self.resized_width, min_pixels=self.min_pixels, max_pixels=self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example and len(example[self.video_key]) > 0:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(video, nframes=self.nframes, resized_height=self.resized_height, resized_width=self.resized_width, min_pixels=self.min_pixels, max_pixels=self.max_pixels, video_fps=self.video_fps, return_fps=True)
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        example["messages"] = messages
        return example