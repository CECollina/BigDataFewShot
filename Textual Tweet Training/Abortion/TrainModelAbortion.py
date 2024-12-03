from transformers import LlamaForVision, LlamaProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch

#Load pre-trained LLaMA Vision model:
model = LlamaForVision.from_pretrained("facebook/llama-3.2-vision")

#Load processor:
processor = LlamaProcessor.from_pretrained("facebook/llama-3.2-vision")
