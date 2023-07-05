from criteria.lpips.lpips import LPIPS
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from argparse import ArgumentParser

class GTResDataset(Dataset):

	def __init__(self, root_path, gt_dir=None, transform=None, transform_train=None):
		self.pairs = []
		for f in os.listdir(root_path):
			image_path = os.path.join(root_path, f, 'sfhRes/seg1.jpg')
			gt_path = os.path.join(root_path, f, 'TeethEdge.png')

			self.pairs.append([image_path, gt_path, None])
		self.transform = transform
		self.transform_train = transform_train

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, index):
		from_path, to_path, _ = self.pairs[index]
		from_im = Image.open(from_path).convert('RGB')
		to_im = Image.open(to_path).convert('RGB')

		if self.transform:
			to_im = self.transform(to_im)
			from_im = self.transform(from_im)

		return from_im, to_im

def parse_args():
	parser = ArgumentParser(add_help=False)
	parser.add_argument('--mode', type=str, default='l2', choices=['lpips', 'l2'])
	parser.add_argument('--data_path', type=str, default='/mnt/e/data/smile/to_b/test_63/test_03_26')
	parser.add_argument('--gt_path', type=str, default='gt_images')
	parser.add_argument('--workers', type=int, default=4)
	parser.add_argument('--batch_size', type=int, default=4)
	args = parser.parse_args()
	return args


def run(args):

	transform = transforms.Compose([transforms.Resize((256, 256)),
									transforms.ToTensor()])

	print('Loading dataset')
	dataset = GTResDataset(root_path=args.data_path,
	                       gt_dir=args.gt_path,
						   transform=transform)

	dataloader = DataLoader(dataset,
	                        batch_size=args.batch_size,
	                        shuffle=False,
	                        num_workers=int(args.workers),
	                        drop_last=True)

	if args.mode == 'lpips':
		loss_func = LPIPS(net_type='alex')
	elif args.mode == 'l2':
		loss_func = torch.nn.MSELoss()
	else:
		raise Exception('Not a valid mode!')
	loss_func.cuda()

	global_i = 0
	scores_dict = {}
	all_scores = []
	for result_batch, gt_batch in tqdm(dataloader):
		for i in range(args.batch_size):
			loss = float(loss_func(result_batch[i:i+1].cuda(), gt_batch[i:i+1].cuda()))
			all_scores.append(loss)
			im_path = dataset.pairs[global_i][0]
			scores_dict[os.path.basename(im_path)] = loss
			global_i += 1

	all_scores = list(scores_dict.values())
	mean = np.mean(all_scores)*1000
	std = np.std(all_scores)*100000000000
	result_str = 'Average loss is {:.2f}+-{:.2f}'.format(mean, std)
	print('Finished with ', args.data_path)
	print(result_str)
    
if __name__ == '__main__':
	args = parse_args()
	run(args)
    