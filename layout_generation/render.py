from PIL import Image, ImageDraw, ImageFont

def text_renderer(image, layout, texts) :

    draw = ImageDraw.Draw(image)
    color = (0, 255, 0)
    
    for text, bbox in zip(texts, layout) :
        draw.rectangle(bbox, outline=color, width=3)
        text_pos = (bbox[0], bbox[1])

        font = ImageFont.truetype("/data1/poong/PosterNUWA/layout_generation/CJ+ONLYONE+Bold.ttf", 12)
        w, h = font.getsize(text)
        font = ImageFont.truetype("/data1/poong/PosterNUWA/layout_generation/CJ+ONLYONE+Bold.ttf", min(int(h * (bbox[2]-bbox[0]) / w), bbox[3] - bbox[1]))
        w, h = font.getsize(text)
        
        text_pos = ((bbox[2]+bbox[0]-w)//2, (bbox[3]+bbox[1]-h)//2)
        draw.text(text_pos, text, font=font, fill=(0,0,0))

    return image