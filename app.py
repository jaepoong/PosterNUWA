import gradio as gr
import torch
from torch.utils.data import DataLoader
from src.dataset.custom_dataset import RawFileDataset
from src.processor.blip_processors import Blip2ImageTrainProcessor
import os
from src.model.minigpt4 import MiniGPT4


from PIL import Image
from layout_generation.word import TextGenerationModel
from layout_generation.layout import layout_generator
from layout_generation.render import text_renderer

#file="data/cgl_dataset/for_posternuwa/html_format_only_layout_img_instruction_int/val_llama_numerical.jsonl" # validation path
#dataset = RawFileDataset(file,img_file_path="data/cgl_dataset/box_inpainintg_layout_cgl")

#dataloader= DataLoader(dataset,batch_size=1,shuffle=True)

#image_processor = Blip2ImageTrainProcessor()

device = "cuda:6" if torch.cuda.is_available() else "cpu"

model = MiniGPT4(lora_r=64)
model.device = device
state = torch.load("log_dir/train_stage2/checkpoints/checkpoint-1/pytorch_model.bin",map_location="cpu")
model.load_state_dict(state)
model = model.to(device)
model.requires_grad_ = False
model.eval()

text_model = TextGenerationModel()
text_model.requires_grad_ = False
text_model.set_seed(42)

def fn(image, prompt, category, brand, product) :
    image = image.resize((513, 750))
    # stage 1
    with torch.no_grad() :
        phrase = text_model.generate_ad_phrases(prompt, category, brand, product)
    
    texts = [phrase, brand, product]
    # stage 2
    layout = layout_generator(image, texts, model)
    # stage 3
    output = text_renderer(image, layout, texts)

    return output

demo = gr.Interface(fn, 
    inputs=[gr.components.Image(source='upload', type='pil', label="background"),
            gr.components.Text(label="prompt"),
            gr.components.Text(label="카테고리"),
            gr.components.Text(label="브랜드"),
            gr.components.Text(label="상품"),
            ], 
    outputs=[gr.components.Image(label='output')]
)
demo.launch(share=True)