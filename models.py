import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
__all__ = ['DSN']
class DSN(nn.Module):
	"""Deep Summarization Network"""
	def __init__(self,in_dim =19,hid_dim=256, num_layers=1):
		super(DSN, self).__init__()
		# in_dim = in_dim*in_dim # for semantic segementation
		in_dim = in_dim 
		self.lstm = nn.LSTM(in_dim, hid_dim, num_layers=1, bidirectional=True, batch_first=True)
		self.fc2 = nn.Linear(hid_dim*2,hid_dim*2)
		self.fc1 = nn.Linear(hid_dim*2,1)
		

	def forward(self, x,in_dim=19):
		
		# x=torch.reshape(x,(x.shape[0],x.shape[1],in_dim*in_dim)) # for semantic segmentation
		x = x.float()
		x,_=self.lstm(x)
		x= self.fc2(x)
		x= self.fc1(x)
		p = F.sigmoid(x)
		return p