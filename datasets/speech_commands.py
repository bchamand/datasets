import os
from typing import Callable, Tuple

import torch
import torch.utils.data as data

import torchaudio

from avreader import load_audio
from downloader import download_and_extract_archive

__all__ = ["SpeechCommands", "SpeechCommandsV2", "SpeechCommandsV1"]


class SpeechCommands(data.Dataset):
    """[summary]

    Parameters
    ----------
    root : str
        Root directory of the Speech Commands Dataset.
    split : str, optional (default="train")
        The dataset split, supports ``train`` or ``val``

    transform : Callable[[torch.Tensor], torch.Tensor], optional (default=None)
        [description]
    target_transform : Callable[[torch.Tensor], torch.Tensor], optional (default=None)
        [description]
    download : bool, optional (default=False)
        [description]

    Attributes
    ----------
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).

    """

    base_folder = "speech_commands/v0.02"
    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    md5 = "6b74f3901214cb2c2934e98196829835"
    filename = url.rpartition("/")[2]

    splits = ("train", "val", "test")

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Callable[[torch.Tensor], torch.Tensor] = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] = None,
        download: bool = False,
    ) -> None:
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        self.classes, self.class_to_idx = self._find_classes(os.path.join(self.root, self.base_folder))

        self.data, self.targets = self._make_dataset(
            os.path.join(os.path.abspath(self.root), self.base_folder), split, self.classes, self.class_to_idx
        )

    # @property
    # def class_to_idx(self) -> Dict[str, int]:
    #     return {_class: i for i, _class in enumerate(self.classes)}


    def _find_classes(self, path: str) -> Tuple[list, dict]:
        classes = [
            d.name
            for d in os.scandir(path)
            if not d.name.startswith(".") and not d.name.startswith("_") and d.is_dir()
        ]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(
        self, path: str, split: str, classes: str, class_to_idx: dict
    ) -> Tuple[list, list]:
        # load the test and validation lists
        with open(os.path.join(path, "testing_list.txt")) as fd:
            test_files = [os.path.join(path, fpath) for fpath in fd.read().strip().splitlines()]
        with open(os.path.join(path, "validation_list.txt")) as fd:
            eval_files = [os.path.join(path, fpath) for fpath in fd.read().strip().splitlines()]

        # load data and labels depending on the split part
        if split == "test":
            data = test_files
            targets = [class_to_idx[os.path.dirname(fpath).rpartition("/")[2]] for fpath in test_files]
        elif split == "val":
            data = eval_files
            targets = [class_to_idx[os.path.dirname(fpath).rpartition("/")[2]] for fpath in eval_files]
        else:
            data = []
            targets = []
            test_eval_files = test_files + eval_files
            for classe in self.classes:
                for entry in os.scandir(os.path.join(path, classe)):
                    if (
                        not entry.name.startswith(".")
                        and entry.is_file()
                        and entry.name.endswith(".wav")
                    ):
                        data.append(entry.path)
            # removes the test and validation file paths to keep only the
            # training file paths
            data = list(set(data) - set(test_eval_files))
            targets = [class_to_idx[os.path.dirname(fpath).rpartition("/")[2]] for fpath in data]

        return data, torch.tensor(targets)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_path, target = self.data[index], int(self.targets[index])
        audio, fs = load_audio(audio_path)

        if self.transform is not None:
            audio = self.transform(audio)

        if self.target_transform is not None:
            targer = self.target_transform(target)

        return audio, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_exists(self) -> bool:
        # can be more robust
        return os.path.exists(os.path.join(self.root, self.base_folder))

    def download(self) -> None:
        if self._check_exists():
            print("Files already downloaded and verified")

        download_and_extract_archive(
            self.url,
            self.root,
            filename=self.base_folder,
            remove_finished=True,
        )

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = [f"Number od datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append("Transforms:")
        body.append("Target transforms:")
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)



class SpeechCommandsV1(SpeechCommands):
    url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
    md5 = "3cd23799cb2bbdec517f1cc028f8d43c"

SpeechCommandsV2 = SpeechCommands
