from generation import generation_code
from generation import build_input
"""
def build_input(canvas_width, canvas_height, vals) :
    '''
    vals : [(category, x, y, w, h), ...]
    '''
    def _build_rect(category='background', x=0, y=0, w=0, h=0) :
        return f'<rect data-category=\"{category}\", x=\"{x}\", y=\"{y}\", width=\"{w}\", height=\"{h}\"/>\n'
    
    INSTRUCTION = [
        "I want to generate layout in poster design format.please generate the layout html according to the categories and image I provide",
        "I want to generate layout in poster design format.please generate the layout html according to the categories and size and image I provide",
        "I want to generate layout in poster design format.please recover the layout html according to the bbox, categories and size according to the image I provide"
    ]

    rects = ''
    for category, x, y, w, h in vals :
        rects += _build_rect(category, x, y, w, h)
    
    prompt = random.choice(INSTRUCTION)
    str_output = f'{prompt} (in html format):\n###bbox html:  <body> <svg width=\"{canvas_width}\" height=\"{canvas_height}\">{rects} </svg> </body>'

    return bytes(str_output, 'utf-8')"""

from bs4 import BeautifulSoup

def layout_generator(image, texts, model) :
    
    w, h = image.size
    vals = [('text', None, None, None, None)for _ in range(len(texts))]
    
    str_output = build_input(w, h, vals) 
    
    # TODO : layout generator inference code (str_output)
    output,img = generation_code(image,[str_output],model)
    
    #output = '<body> <svg width="513" height="750"> <rect data-category="logo", x="166", y="31", width="183", height="58"/>\n<rect data-category="text", x="61", y="115", width="395", height="49"/>\n<rect data-category="background", x="69", y="185", width="377", height="53"/>\n<rect data-category="text", x="124", y="194", width="261", height="36"/>\n<rect data-category="background", x="172", y="254", width="171", height="48"/>\n<rect data-category="text", x="190", y="264", width="132", height="29"/> </svg> </body>'

    parsed_html = BeautifulSoup(output[0])

    layout = list()
    for rect in parsed_html.body.find_all('rect') :
        category = rect['data-category'] 
        if category == 'text' :
            x, y, w, h = int(rect['x']), int(rect['y']), int(rect['width']), int(rect['height'])
            bbox = (x, y, x+w, y+h)
            layout.append(bbox)

    return layout