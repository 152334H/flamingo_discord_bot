from model import OpenFlamingo

import torch
import requests
from io import BytesIO
from pathlib import Path
from typing import Union, Optional
from PIL import Image
from pydantic import BaseModel, HttpUrl

from dotenv import dotenv_values
ENV = dotenv_values(".env")

OF_model = OpenFlamingo(ENV['LLAMA_PATH'])

class ImageTextPair(BaseModel):
    text: str
    image_src: Union[HttpUrl, Path, bytes]
    image: Optional[Image.Image] = None
    thumb: Optional[Image.Image] = None
    processed: Optional[torch.Tensor] = None

    def __init__(self, **data):
        super().__init__(**data)

        if isinstance(self.image_src, str) and self.image_src.startswith("http"):
            self.image = Image.open(requests.get(self.image_src, stream=True).raw)
        elif isinstance(self.image_src, Path):
            self.image = Image.open(self.image_src)
        elif isinstance(self.image_src, bytes):
            self.image = Image.open(BytesIO(self.image_src))
        else:
            raise ValueError("Invalid image source")
        self.thumb = self.image.copy()
        self.thumb.thumbnail((256, 256))

        self.processed = OF_model.image_processor(self.image).unsqueeze(0)

    class Config: # for image/processed
        arbitrary_types_allowed = True


class ICLRequest(BaseModel):
    examples: list[ImageTextPair]
    query: ImageTextPair
    #m
    def handle(self, **hf_kwargs):
        imgs = [ex.processed for ex in self.examples]
        imgs.append(self.query.processed)
        texts = [ex.text for ex in self.examples]
        texts.append(self.query.text)

        gen_text = OF_model.process_and_generate(imgs, texts, **hf_kwargs)
        return self.query.text + gen_text

CAPTION_basepath = Path('./ICL/captioning/')
CAPTION_examples = [
    {
        "image_src": CAPTION_basepath / "000000039769.jpg",
        "text": "<image>An image of two cats.",
    }, {
        "image_src": CAPTION_basepath / "000000028137.jpg",
        "text": "<image>An image of a bathroom sink.",
    }
]

def generate_prompt(text, resp=None):
    s = f"""### Instruction:
    {text}
    ### Response:"""
    if resp is not None:
        s += '\n'
        s += resp
    return s
INSTRUCT_examples = [
    {
        "image_src": CAPTION_basepath / "000000039769.jpg",
        "text": generate_prompt("Caption this image: <image>", "An image of two cats."),
    }, {
        "image_src": CAPTION_basepath / "000000028137.jpg",
        "text": generate_prompt("Caption this image: <image>", "An image of a bathroom sink."),
    }
]
CAPTION_req_example = {
    "examples": CAPTION_examples,
    "query": {
        "image_src": CAPTION_basepath / "000000028352.jpg",
        "text": "<image>An image of",
    }
}

