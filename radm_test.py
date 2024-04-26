from src.model.minigpt4 import MiniGPT4
from src.processor.blip_processors import Blip2ImageTrainProcessor,Blip2ImageEvalProcessor,DinoImageProcessor
import torch
import json
import os
from PIL import Image, ImageFilter
from html_to_ui import get_bbox
from generation import *

from helper.global_var import *

def build_input(vals=None,text=None,canvas_width=513, canvas_height=750,task="unconditional") :
    """
        text = list of string
        vals = list of layout (category, xl,yl,xr,yr)
    """
    html_template = TEMPLATE_FORMAT.get("html_format")
    bbox_template = TEMPLATE_FORMAT.get("bbox_format")
    #instruction
    if text:
        task_instruction = TASK_INSTRUCTION.get("cgl")
        instruction = TEXT_INSTRUCTION.get(task)
        instruction = task_instruction + instruction
        
        t = []
        for te in text:
            t.append(te)
        text = " & ".join(text)
        
    else:
        task_instruction = TASK_INSTRUCTION.get("cgl")
        instruction = INSTRUCTION.get(task)
        instruction = task_instruction + instruction
    MASK= "<M>"
    def _build_rect(category='background', x=None, y=None, w=None, h=None):
        
        if not x:
            x="<M>"
        if not y:
            y="<M>"
        if not w:
            w="<M>"
        if not h:
            h = "<M>"
        
        rect = bbox_template.format(c=category, x=x, y=y, w=w, h=h)
        return rect #f'<rect data-category=\"{category}\", x=\"{x}\", y=\"{y}\", width=\"{w}\", height=\"{h}\"/>\n'
    
    
    contents = []
    if vals:
        for category, x, y, w, h in vals :
            contents.append(_build_rect(category, x, y, w, h))
            content = "\n".join(contents)
    else:
        content = ""
    bbox_html = html_template.format(W=canvas_width, H=canvas_height, content = content)
    #str_output = f'{instruction} (in html format):\n###bbox html:  <body> <svg width=\"{canvas_width}\" height=\"{canvas_height}\">{rects} </svg> </body>'
    if text:
        str_output = instruction.format(text= text, bbox_html = bbox_html)
    else:
        str_output = instruction.format(bbox_html = bbox_html)
    return str_output

device = f"cuda:{2}" if torch.cuda.is_available() else "cpu"
vit_model_name = "dino_v2"
llama_model = "models/codeLlama-7b-hf"
base_model = "log_dir/train_stage2_with_augment_dino_codellama/checkpoints/checkpoint-16/pytorch_model.bin"
model = MiniGPT4(lora_r=64,low_resource=False,vit_model = vit_model_name,llama_model = llama_model)
model.load_state_dict(torch.load(base_model,map_location="cpu"))
model = model.to(device)
model.device = device
model.half()
model.eval()

image_processor = DinoImageProcessor()

import json
with open("/data1/poong/tjfwownd/PosterNUWA/data/cgl_dataset/radm_dataset/RADM_augmentation_test/test_translated_text.json","r") as f:
    test_translated = json.load(f)
translated_dict = {}
for test in test_translated:
    key = list(test.keys())[0]
    value = list(test.values())[0]
    translated_dict[key]=value
    
import matplotlib.pyplot as plt
import os
DATASET_COLOR = {
    1: '#929F29',   
    2: '#1FA39A',  
    3: '#987FF2',      
    4: '#F56881',    
    5: "#0000FF"      
}
def sort_key(x):
    if x[0] == 3:
        return -1
    else:
        return x[0]
img_paths = os.listdir("/data1/poong/tjfwownd/PosterNUWA/data/cgl_dataset/radm_dataset/RADM_dataset/images/test") #list(translated_dict.keys())  #os.listdir("/data1/poong/tjfwownd/PosterNUWA/data/cgl_dataset/radm_dataset/RADM_dataset/images/test")
si = 0
output_file = f"/data1/poong/tjfwownd/PosterNUWA/log_dir/Paper_samples/Generated_Sample_RADM_test/PosterLlama_non_text/radious_2/{si}"
os.makedirs(os.path.join(output_file,"generated_sample"),exist_ok=True)


#uncond_inst = 'I want to generate layout in poster design format.plaese generate the layout html according to the image I provide (in html format):\n###bbox html: <body> <svg width="513" height="750">  </svg> </body> '
#inst = [uncond_inst]*len(img_paths)

boxx = []
clx = []
imgs = []
save_img=True
from tqdm import tqdm
import random
used_json=[]
for path in tqdm(img_paths):
    img = Image.open(os.path.join("/data1/poong/tjfwownd/PosterNUWA/data/cgl_dataset/radm_dataset/RADM_dataset/images/test",path)).resize((513,750))
    filtered_img = img.filter(ImageFilter.GaussianBlur(radius = 2))
    processed_images = image_processor(filtered_img).unsqueeze(0)

    #texts_base = translated_dict[path]
    #texts = [item for item in texts_base if item != '']
    #N = random.randint(1,len(texts))
    #texts = random.sample(texts,N)
    
    inst = [build_input(None,None)]#[build_input(None,texts)]
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda"):
            samp = model.generate(processed_images, inst,max_new_tokens=512,do_sample=False,temperature=0.6,top_p=0.9)
    
    bboxes,clses = get_bbox(samp[0])
    #used_json.append({path : texts})
    if save_img:
        drawbbox = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x2 += x1
            y2 += y1
            bbox  = [x1, y1, x2, y2]
            drawbbox.append(np.array(bbox))
        drawn_outline = img.copy()
        drawn_fill = img.copy()
        draw_ol = ImageDraw.ImageDraw(drawn_outline)
        draw_f = ImageDraw.ImageDraw(drawn_fill)

        sorted_pairs = sorted(zip(clses, drawbbox), key=sort_key)
        sorted_classes, sorted_bboxes = zip(*sorted_pairs)

        clses = list(sorted_classes)
        drawbbox = list(sorted_bboxes)

        for b,l in zip(drawbbox,clses):
            draw_ol.rectangle([b[0],b[1],b[2],b[3]], outline=DATASET_COLOR[l], width=2)
        for b,l in zip(drawbbox,clses):
            draw_f.rectangle([b[0],b[1],b[2],b[3]], fill=DATASET_COLOR[l])
        drawn_outline = drawn_outline.convert("RGBA")
        drawn_fill = drawn_fill.convert("RGBA")

        drawn_fill.putalpha(int(256 * 0.4))
        img = Image.alpha_composite(drawn_outline, drawn_fill)
        #layout_images.append(img)
        #display(img)
        img.convert("RGB").save(os.path.join(output_file,"generated_sample",path))

    boxx.append(np.array(drawbbox).tolist())
    clx.append(np.array(clses).tolist())
    imgs.append(path)

#if  save_img:
#    display(img)
    
with open(os.path.join(output_file,"box.json"), "w") as f:
    json.dump(boxx,f,indent=2)

with open(os.path.join(output_file,"clses.json"),"w") as f:
    json.dump(clx,f,indent=2)

with open(os.path.join(output_file,"text_order.json"),"w") as f:
    json.dump(imgs,f,indent=2)
    
with open(os.path.join(output_file,"used_text.json"),"w") as f:
    json.dump(used_json,f,indent=2)