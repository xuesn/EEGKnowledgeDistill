

from transformers import CLIPProcessor, CLIPModel

clip_vit_base_path =  '/data/snxue/clip-vit-base-patch32'
model_base = CLIPModel.from_pretrained(clip_vit_base_path, local_files_only=True)
processor_base = CLIPProcessor.from_pretrained(clip_vit_base_path, local_files_only=True)

clip_vit_large_path = '/data/snxue/clip-vit-large-patch14'
model_large = CLIPModel.from_pretrained(clip_vit_large_path, local_files_only=True)
processor_large = CLIPProcessor.from_pretrained(clip_vit_large_path, local_files_only=True)

# clip-vit-base-patch32
#   (text_model): CLIPTextTransformer(
#   (vision_model): CLIPVisionTransformer(
#   (visual_projection): Linear(in_features=768, out_features=512, bias=False)
#   (text_projection): Linear(in_features=512, out_features=512, bias=False)

# clip-vit-large-patch14
#   (text_model): CLIPTextTransformer(
#   (vision_model): CLIPVisionTransformer(
#   (visual_projection): Linear(in_features=1024, out_features=768, bias=False)
#   (text_projection): Linear(in_features=768, out_features=768, bias=False)



# 数据预处理：在加载模型后，需要对输入图像进行预处理。可以使用CLIPProcessor对图像和文本进行处理，生成模型所需的输入格式：

from PIL import Image
import requests
    
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)    
image = Image.open("/data/snxue/visual_embedding的图片/72pic/11.png")
inputs = processor_base(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

'''
>>> inputs.keys()
dict_keys(['input_ids', 'attention_mask', 'pixel_values'])
>>> bbb=inputs.pixel_values
>>> bbb
tensor([[[[0.0763, 0.0763, 0.0763,  ..., 0.0763, 0.0763, 0.0763],
          [0.0763, 0.0763, 0.0763,  ..., 0.0763, 0.0763, 0.0763],
          [0.0763, 0.0763, 0.0763,  ..., 0.0763, 0.0763, 0.0763],
          ...,
          [0.0763, 0.0763, 0.0763,  ..., 0.0763, 0.0763, 0.0763],
          [0.0763, 0.0763, 0.0763,  ..., 0.0763, 0.0763, 0.0763],
          [0.0763, 0.0763, 0.0763,  ..., 0.0763, 0.0763, 0.0763]],

         [[0.1689, 0.1689, 0.1689,  ..., 0.1689, 0.1689, 0.1689],
          [0.1689, 0.1689, 0.1689,  ..., 0.1689, 0.1689, 0.1689],
          [0.1689, 0.1689, 0.1689,  ..., 0.1689, 0.1689, 0.1689],
          ...,
          [0.1689, 0.1689, 0.1689,  ..., 0.1689, 0.1689, 0.1689],
          [0.1689, 0.1689, 0.1689,  ..., 0.1689, 0.1689, 0.1689],
          [0.1689, 0.1689, 0.1689,  ..., 0.1689, 0.1689, 0.1689]],

         [[0.3399, 0.3399, 0.3399,  ..., 0.3399, 0.3399, 0.3399],
          [0.3399, 0.3399, 0.3399,  ..., 0.3399, 0.3399, 0.3399],
          [0.3399, 0.3399, 0.3399,  ..., 0.3399, 0.3399, 0.3399],
          ...,
          [0.3399, 0.3399, 0.3399,  ..., 0.3399, 0.3399, 0.3399],
          [0.3399, 0.3399, 0.3399,  ..., 0.3399, 0.3399, 0.3399],
          [0.3399, 0.3399, 0.3399,  ..., 0.3399, 0.3399, 0.3399]]]])
>>> bbb.shape
torch.Size([1, 3, 224, 224])
'''

# 模型推理：将预处理后的数据输入到模型中，进行推理：

outputs = model_base(**inputs)
logits_per_image = outputs.logits_per_image # 这是图像-文本相似度得分
probs = logits_per_image.softmax(dim=1) # 可以通过softmax获取标签概率


len(outputs) == 6
outputs.loss
outputs.logits_per_image
outputs.logits_per_text
text_embeds = outputs.text_embeds  # text_embeds.shape: 512/768
image_embeds = outputs.image_embeds  # image_embeds.shape: 512/768
vision_model_output = outputs.vision_model_output




