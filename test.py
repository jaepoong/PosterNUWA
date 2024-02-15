from torch.utils.data import DataLoader
from src.dataset.caption_dataset import ChainDataset,CCSBUDataset
from src.processor.blip_processors import Blip2ImageTrainProcessor,BlipCaptionProcessor

from accelerate import Accelerator
from accelerate.utils import set_seed
paths = ["/data1/poong/data/cc_sbu_laion/cc_sbu_dataset/01255.tar","/data1/poong/data/cc_sbu_laion/laion_dataset/01255.tar"]
image_processor = Blip2ImageTrainProcessor()
text_processor = BlipCaptionProcessor()
#dataset = ChainDataset(paths)
#loader = DataLoader(dataset,batch_size=2)
dataset = CCSBUDataset(image_processor,text_processor,paths[0])
loader = DataLoader(dataset.inner_dataset,batch_size=2)

accelerator = Accelerator(dispatch_batches=False,
                          mixed_precision="fp16")
data_loader = accelerator.prepare(loader)
device = accelerator.device
print(device)
for step in range(int(1)):
    sample = next(iter(data_loader))
    sample['image'] = sample['image'].to(device)
    print(sample['image'].device)

accelerator.save_state("checkpoints")