# import os
# import cv2


# base_dir = '/home/vsfh/dataset/flip/down_mask'
# out_path = '/home/vsfh/dataset/mask/a_down'
# img_list = os.listdir(base_dir)

# for img in img_list:
#     in_dir = os.path.join(base_dir, img)
#     a = cv2.imread(in_dir)
#     out_dir = os.path.join(out_path, img)
#     cv2.imwrite(out_dir, a[:,::-1,:])
    
    


from torch.utils.data import DataLoader
from datasets import load_dataset, Image
import torch
from torchvision import transforms
dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
transform = transforms.Compose(
                    [
                        transforms.Resize([256,256]),
                        transforms.ToTensor()
                        
                    ]
            )
def trans(examples):
    examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
    return examples

dataset = dataset.with_transform(trans)
print(dataset[0])
def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example["pixel_values"]))
        labels.append(example["text"])
        
    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return {"pixel_values": pixel_values, "labels": labels}
dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4)
next(iter(dataloader))