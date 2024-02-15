import json
from typing import Iterable

from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
from src.processor.blip_processors import Blip2ImageTrainProcessor,BlipCaptionProcessor
import webdataset as wds
from typing import List
import logging
import random as rnd
import random

image_processor = Blip2ImageTrainProcessor()
text_processor = BlipCaptionProcessor()

class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        # print("ann paths", ann_paths)
        for ann_path in ann_paths:
            # print("ann_path", ann_path)
            ann = json.load(open(ann_path, "r"))
            if isinstance(ann, dict):
                self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])
                # self.annotation.extend(json.load(open(ann_path, "r")))
            else:
                self.annotation.extend(json.load(open(ann_path, "r")))
    
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)
    
class CCSBUDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "labels": self.text_processor(sample[1]["caption"]),
            "image": sample[0]
        }

class ChainDataset(wds.DataPipeline):
    r"""Dataset for chaining multiple :class:`DataPipeline` s.

    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """
    def __init__(self, paths) -> None:
        super().__init__()
        self.datasets = [CCSBUDataset(image_processor,text_processor,path).inner_dataset for path in paths]
        self.prob = []
        self.names = []
        for dataset in self.datasets:
            if hasattr(dataset, 'name'):
                self.names.append(dataset.name)
            else:
                self.names.append('Unknown')
            if hasattr(dataset, 'sample_ratio'):
                self.prob.append(dataset.sample_ratio)
            else:
                self.prob.append(1)
                logging.info("One of the datapipeline doesn't define ratio and set to 1 automatically.")
        
        self.prob = [1,1]

    def __iter__(self):
        datastreams = [iter(dataset) for dataset in self.datasets]
        while True:
            select_datastream = random.choices(datastreams, weights=self.prob, k=1)[0]
            yield next(select_datastream)