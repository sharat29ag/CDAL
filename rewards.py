import torch
import sys
import torch.nn.functional as F
import tqdm
import numpy as np

def KL_classification(a,b):
	a=F.softmax(a)
	b=F.softmax(b)
	kl1=a*torch.log(a/b)
	kl2=b*torch.log(b/a)
	kl= -0.5*(torch.sum(kl1)) - 0.5*(torch.sum(kl2))
	return abs(kl)

def KL_object(a,b):
	
	kl1=a*torch.log(a/b)
	kl2=b*torch.log(b/a)
	kl= -0.5*(torch.sum(kl1)) - 0.5*(torch.sum(kl2))
	return abs(kl)
	
def KL_segment(ac,bc,nc):## nc = number of classes
	kl_classes=[]
	reward_kl=0.0
	for i in range(nc):
		a=ac[i,:]
		b=bc[i,:]
		kl1=a*torch.log(a/b)
		kl2=b*torch.log(b/a)
		kl= -0.5*(torch.sum(kl1)) - 0.5*(torch.sum(kl2))
		if(kl == kl and not torch.isinf(kl)):
			kl_classes.append(abs(kl))
	if(len(kl_classes)!=0):
		reward_kl=sum(kl_classes)
	if reward_kl == 0.0:
		return torch.tensor(0.0)
	reward_kl = reward_kl.float()
	return reward_kl

def CD(seq,pick_idxs,nc):
	reward_kl=[]
	for k in pick_idxs:
		for l in pick_idxs:
			reward_kl.append(KL_object(seq[k,:],seq[l,:])) # change function according to the task.
	reward_kl=torch.stack(reward_kl)
	reward_kl=torch.mean(reward_kl)
	return reward_kl


def V_rep(_seq,pick_idxs):
	n=_seq.shape[0]
	dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n)
	dist_mat = dist_mat + dist_mat.t()
	dist_mat.addmm_(1, -2, _seq, _seq.t())
	dist_mat = dist_mat[:,pick_idxs.copy()]
	dist_mat = dist_mat.min(1, keepdim=True)[0]
	# reward_rep = torch.exp(torch.FloatTensor([-dist_mat.mean()]))[0] 
	reward_rep = torch.exp(-dist_mat.mean()/50)
	return reward_rep


def compute_reward(seq, actions,probs,nc,picks,use_gpu=False):
	_seq = seq.detach()
	_actions = actions.detach()
	pick_idxs = _actions.squeeze().nonzero().squeeze()
	probs=probs.detach().cpu().numpy().squeeze()
	top_pick_idxs=probs.argsort()[-1*picks:][::-1]
	_seq = _seq.squeeze()
	n = _seq.size(0) 
	
	reward_kl=CD(_seq,top_pick_idxs.squeeze(),nc)
	rep_reward=V_rep(_seq,top_pick_idxs.squeeze())
	reward=rep_reward*0.5 + reward_kl*1.5
	return reward,top_pick_idxs
