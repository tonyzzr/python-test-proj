#!/usr/bin/env python3
"""Simple PyTorch GPU training benchmark (MNIST-like).

This script trains a small ConvNet on either MNIST (download) or synthetic data
and reports time/epoch and images/sec. Synthetic mode is the fastest way to
benchmark raw GPU training speed (no IO or CPU preprocessing bottlenecks).

Usage examples:
  python pytorch-test.py --synthetic --batch-size 512 --num-batches 200
  python pytorch-test.py --dataset mnist --batch-size 128 --epochs 3
"""

import time
import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class SmallConvNet(nn.Module):
	def __init__(self, in_channels=1, num_classes=10):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
		self.pool = nn.MaxPool2d(2)
		# After two conv layers and one MaxPool2d(2) the spatial size is 14x14
		# for 28x28 input, so flattened features = 64 * 14 * 14
		self.fc1 = nn.Linear(64 * 14 * 14, 128)
		self.fc2 = nn.Linear(128, num_classes)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x


class SyntheticDataset(Dataset):
	"""Dataset that returns random images and labels for benchmarking.

	Images are floats in [0,1) and labels are integers [0, num_classes).
	"""

	def __init__(self, num_samples, shape=(1, 28, 28), num_classes=10, dtype=torch.float32):
		self.num_samples = num_samples
		self.shape = shape
		self.num_classes = num_classes
		self.dtype = dtype

	def __len__(self):
		return self.num_samples

	def __getitem__(self, idx):
		img = torch.rand(self.shape, dtype=self.dtype)
		label = torch.randint(0, self.num_classes, (1,)).item()
		return img, label


def train_epoch(model, loader, optimizer, criterion, device, sync_cuda=True, log_interval=50):
	model.train()
	running_loss = 0.0
	total_samples = 0
	start_time = time.perf_counter()

	for batch_idx, (data, target) in enumerate(loader, 1):
		data = data.to(device, non_blocking=True)
		target = target.to(device, non_blocking=True)

		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * data.size(0)
		total_samples += data.size(0)

		if batch_idx % log_interval == 0:
			print(f"  batch {batch_idx}, avg loss {(running_loss/total_samples):.4f}")

	if device.type == 'cuda' and sync_cuda:
		torch.cuda.synchronize()
	elapsed = time.perf_counter() - start_time
	avg_loss = running_loss / total_samples if total_samples else 0.0
	return elapsed, avg_loss, total_samples


def build_dataloader(args):
	if args.synthetic:
		# Synthetic dataset: num_batches * batch_size samples
		num_samples = args.num_batches * args.batch_size
		dataset = SyntheticDataset(num_samples, shape=(1, 28, 28), num_classes=10)
		loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
		return loader

	# Try to import torchvision and provide MNIST fallback
	try:
		from torchvision import datasets, transforms
	except Exception as e:
		print("torchvision not available; fallback to synthetic. To use MNIST install torchvision.")
		return build_dataloader(argparse.Namespace(**{**vars(args), 'synthetic': True}))

	transform = transforms.Compose([transforms.ToTensor()])
	if args.dataset.lower() == 'mnist':
		ds = datasets.MNIST(root=args.data_dir or './data', train=True, download=True, transform=transform)
	else:
		# default to MNIST shape but allow other torchvision datasets by name in future
		ds = datasets.MNIST(root=args.data_dir or './data', train=True, download=True, transform=transform)

	loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
	return loader


def parse_args():
	p = argparse.ArgumentParser(description='PyTorch GPU training speed benchmark (MNIST-like)')
	p.add_argument('--synthetic', action='store_true', help='Use synthetic random data (fast, recommended for raw GPU benchmarking)')
	p.add_argument('--dataset', type=str, default='mnist', help='Dataset to use (mnist)')
	p.add_argument('--batch-size', type=int, default=256)
	p.add_argument('--epochs', type=int, default=5)
	p.add_argument('--warmup-epochs', type=int, default=0, help='Number of initial epochs to run as warmup and exclude from timing')
	p.add_argument('--num-batches', type=int, default=200, help='When using synthetic data, number of batches per epoch')
	p.add_argument('--lr', type=float, default=0.01)
	p.add_argument('--workers', type=int, default=4)
	p.add_argument('--no-cuda-sync', dest='sync_cuda', action='store_false', help='Disable torch.cuda.synchronize() for timing (not recommended)')
	p.add_argument('--data-dir', type=str, default=None, help='Directory to store dataset')
	return p.parse_args()


def main():
	args = parse_args()

	use_cuda = torch.cuda.is_available()
	device = torch.device('cuda' if use_cuda else 'cpu')

	print(f"PyTorch {torch.__version__}")
	print(f"CUDA available: {use_cuda}")
	if use_cuda:
		try:
			print(f"CUDA version: {torch.version.cuda}")
			print(f"cuDNN version: {torch.backends.cudnn.version()}")
			print(f"GPU: {torch.cuda.get_device_name(0)}")
		except Exception:
			pass

	# build model
	model = SmallConvNet(in_channels=1, num_classes=10).to(device)
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
	criterion = nn.CrossEntropyLoss()

	# build data loader
	loader = build_dataloader(args)

	print("Starting benchmark")
	total_images = 0
	total_time = 0.0

	for epoch in range(1, args.epochs + 1):
		is_warmup = epoch <= args.warmup_epochs
		tag = "(warmup)" if is_warmup else ""
		print(f"Epoch {epoch}/{args.epochs} {tag}")

		elapsed, avg_loss, samples = train_epoch(model, loader, optimizer, criterion, device, sync_cuda=args.sync_cuda)
		imgs_per_sec = samples / elapsed if elapsed > 0 else 0.0
		print(f"  epoch time: {elapsed:.4f}s, imgs: {samples}, imgs/sec: {imgs_per_sec:.2f}, avg loss: {avg_loss:.4f}")

		# accumulate only if not a warmup epoch
		if not is_warmup:
			total_images += samples
			total_time += elapsed

	if total_time > 0:
		print(f"Timed Overall (excluding warmup): images {total_images}, total_time {total_time:.4f}s, avg imgs/sec {total_images/total_time:.2f}")
	else:
		print("No timed epochs were run (total_time == 0). Increase --epochs or reduce --warmup-epochs.")


if __name__ == '__main__':
	main()
