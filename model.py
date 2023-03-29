from typing import List

import torch
from PIL import Image

from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms

class OpenFlamingo:
    def __init__(self, llama_path, device="cuda"):
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=llama_path,
            tokenizer_path=llama_path,
            cross_attn_every_n_layers=4,
            inference=True,
            precision='fp16',
            device=device,
            checkpoint_path=hf_hub_download("openflamingo/OpenFlamingo-9B", "checkpoint.pt"),
        )
        self.device = device

    def preprocess_images(self, processed: list[torch.Tensor]):
        # TODO consider preprocess first
        """
        Step 2: Preprocessing images
        Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
         batch_size x num_media x num_frames x channels x height x width. 
         In this case batch_size = 1, num_media = 3, num_frames = 1 
         (this will always be one expect for video which we don't support yet), 
         channels = 3, height = 224, width = 224.
        """
        #vision_x = [self.image_processor(img).unsqueeze(0) for img in images]
        vision_x = torch.cat(processed, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        return vision_x.to(self.device).half()

    def preprocess_text(self, texts: List[str]):
        """
        Step 3: Preprocessing text
        Details: In the text we expect an <image> special token to indicate where an image is.
         We also expect an <|endofchunk|> special token to indicate the end of the text 
         portion associated with an image.
        """
        self.tokenizer.padding_side = "left"
        #["<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"],
        assert all('<image>' in s for s in texts)
        formatted_text = '<|endofchunk|>'.join(texts)
        lang_x = self.tokenizer(formatted_text, return_tensors="pt")
        return lang_x.to(self.device)

    def generate_text(self, vision_x, lang_x, max_new_tokens=20, num_beams=3, temperature=1.0, top_k=0, top_p=1.0, length_penalty=1.0, do_sample=False, early_stopping=False):
        generated_text = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            length_penalty=length_penalty,
            do_sample=do_sample,
            early_stopping=early_stopping,
        )
        return self.tokenizer.decode(generated_text[0][len(lang_x["input_ids"][0]):])

    def process_and_generate(self, processed_images: list[torch.Tensor], texts: List[str], **hf_kwargs):
        vision_x = self.preprocess_images(processed_images)
        lang_x = self.preprocess_text(texts)
        generated_text = self.generate_text(vision_x, lang_x, **hf_kwargs)
        return generated_text

