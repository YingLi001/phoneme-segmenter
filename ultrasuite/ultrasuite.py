'''
@Course name: HDR
@Project   : SMAAT
@File      : ultrasuite.py
@Author    : Ying Li
@Year      : 2024
'''
# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

# Lint as: python3
from glob import glob
import re
import random

"""UltraSuite automatic speech recognition dataset."""

import os
from pathlib import Path

import datasets
from datasets.tasks import AutomaticSpeechRecognition

_CITATION = """\
@article{eshky2019ultrasuite,
  title={UltraSuite: a repository of ultrasound and acoustic data from child speech therapy sessions},
  author={Eshky, Aciel and Ribeiro, Manuel Sam and Cleland, Joanne and Richmond, Korin and Roxburgh, Zoe and Scobbie, James and Wrench, Alan},
  journal={arXiv preprint arXiv:1907.00835},
  year={2019}
}
"""

_DESCRIPTION = """\
UltraSuite is a collection of ultrasound and acoustic speech data from child speech therapy sessions. 
The current release includes three datasets, one from typically developing children and two from speech disordered children. 
UltraSuite also includes a set of annotations, some manual and some automatically produced, and tools to process, transform and visualise the data.
More info on UltraSuite dataset can be found here:
https://ultrasuite.github.io/
"""

_HOMEPAGE = "https://ultrasuite.github.io/"


class ULTRASUITEASRConfig(datasets.BuilderConfig):
    """BuilderConfig for ULTRASUITEASR."""

    def __init__(self, **kwargs):
        """
        Args:
          data_dir: `string`, the path to the folder containing the files in the
            downloaded .tar
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          **kwargs: keyword arguments forwarded to super.
        """
        super(ULTRASUITEASRConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)


class ULTRASUITEASR(datasets.GeneratorBasedBuilder):
    """ULTRASUITEASR dataset."""

    BUILDER_CONFIGS = [ULTRASUITEASRConfig(name="clean", description="'Clean' speech.")]

    @property
    def manual_download_instructions(self):
        return (
            "To use UltraSuite you have to download it manually. "
            "Please download the dataset from https://ultrasuite.github.io/download/ \n"
            "Then extract all files in one folder and load the dataset with: "
            "`datasets.load_dataset('ultrasuite', data_dir='path/to/folder/folder_name')`"
        )

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "file": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "text": datasets.Value("string"),
                    "phonetic_detail": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "stop": datasets.Value("int64"),
                            "utterance": datasets.Value("string"),
                        }
                    ),
                }
            ),
            supervised_keys=("file", "text"),
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))

        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"{data_dir} does not exist. Make sure you insert a manual dir via `datasets.load_dataset('ultrasuite', data_dir=...)` that includes files unzipped from the UltraSuite zip. Manual download instructions: {self.manual_download_instructions}"
            )

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"split": "TRAIN", "data_dir": data_dir}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"split": "TEST", "data_dir": data_dir}),
        ]

 
    def _generate_examples(self, split, data_dir):
        
        """Generate examples from UltraSuite archive_path based on the test/train csv information."""
        # Iterating the contents of the data to extract the relevant information
        wav_paths = sorted(Path(data_dir).glob(f"**/{split}/**/**/*.wav"))
        wav_paths = wav_paths if wav_paths else sorted(Path(data_dir).glob(f"**/{split.upper()}/**/**/*.WAV"))
        s = set()
        for key, wav_path in enumerate(wav_paths):
            # extract transcript
            txt_path = with_case_insensitive_suffix(wav_path, ".txt")
            with txt_path.open(encoding="utf-8") as op:
                
                transcript = " ".join(op.readline().split())

            # extract phonemes
            phn_path = with_case_insensitive_suffix(wav_path, ".phn")
            with phn_path.open(encoding="utf-8") as op:
                    phonemes = [
                        {
                            "start": i.split(" ")[0],
                            "stop": i.split(" ")[1],
                            "utterance": " ".join(i.split(" ")[2:]).strip(),
                        }  
                        for i in op.readlines()
                    ]
            example = {
                "file": str(wav_path),
                "audio": str(wav_path),
                "text": transcript,
                "phonetic_detail": phonemes,
            }

            yield key, example


def with_case_insensitive_suffix(path: Path, suffix: str):
    path = path.with_suffix(suffix.lower())
    path = path if path.exists() else path.with_suffix(suffix.upper())
    return path
