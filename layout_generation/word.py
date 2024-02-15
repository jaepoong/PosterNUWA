import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import os

import re

from transformers import BitsAndBytesConfig



class TextGenerationModel:

    def __init__(self, model_name="LDCC/LDCC-Instruct-Llama-2-ko-13B-v1.4"):

        # Configure for single GPU usage

        os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # Assuming you want to use 'cuda:0'

        torch.backends.cuda.max_split_size_mb = 1024

        quantization_config = BitsAndBytesConfig(

                load_in_4bit=True,

                bnb_4bit_compute_dtype=torch.bfloat16,

                bnb_4bit_use_double_quant=True,

                bnb_4bit_quant_type='nf4',

            )



        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        

        # Load the model without specifying device, as 'accelerate' handles this

        self.model = AutoModelForCausalLM.from_pretrained(

            model_name,

            quantization_config=quantization_config,

            torch_dtype=torch.float16,

            low_cpu_mem_usage=True,

        )

        self.model.eval()



        # Initialize the pipeline without specifying the device

        self.pipe = pipeline(

            'text-generation', 

            model=self.model,

            tokenizer=self.tokenizer

        )



    def set_seed(self, seed_value):

        """

        Set a seed for reproducibility.

        """

        torch.manual_seed(seed_value)

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(seed_value)



    def generate_ad_phrases(self, prompt, category, brand, product, max_new_tokens=32, temperature=0.8, top_p=0.9):

        """

        Generate an advertising phrase and extract the portion enclosed in quotation marks.

        """

        context_template = """다음 정보를 기반으로 창의적인 광고 문구를 만들어주세요:

        - 카테고리 '{}', 브랜드 '{}', 그리고 제품 '{}'에 맞는 매력적인 광고 문구.

        - 소비자의 관심을 끌 수 있는 문구로, 브랜드와 제품의 특징을 강조해 주세요.



        광고 문구는 간결하면서도 효과적이어야 합니다. 토큰 수는 한 개당 32 이내로 제한합니다."""

        context = context_template.format(category, brand, product)

        prompt = f"### 질문: {prompt}\n\n### 맥락: {context}\n\n### 답변:"



        result = self.pipe(

            prompt, 

            do_sample=True, 

            max_new_tokens=max_new_tokens,

            temperature=temperature,

            top_p=top_p,

            return_full_text=False,

            eos_token_id=2,

        )



        generated_phrase = result[0]['generated_text'] if result else ""

        

        # Extract the content within quotation marks

        match = re.search(r'"([^"]*)"', generated_phrase)

        match2 = re.search(r'"([^"]*)', generated_phrase)

        return match.group(1) if match else print("output 길이가 너무 김: "+match2.group(1))

