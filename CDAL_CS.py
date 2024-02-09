import numpy as np
import random,os,glob,datetime
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import pdist
import torch.nn.functional as F
import torch
from random import randrange
from tqdm import tqdm 
import argparse

class kCenterGreedy():
	def __init__(self, X, metric='kl'):
		self.features =X
		self.metric = metric
		self.min_distances = None
		self.n_obs = X.shape[0]
		self.already_selected = []
	def update_distances(self, cluster_centers, only_new=True, reset_dist=False):

		if reset_dist:
			self.min_distances = None
		if only_new:
			cluster_centers = [d for d in cluster_centers
						 if d not in self.already_selected]
		if cluster_centers:
			x = self.features[cluster_centers]
			dist = pairwise_distances(self.features,x,metric=kl)  ## change to euclidean for Coreset
			if self.min_distances is None:
				self.min_distances = np.min(dist, axis=1).reshape(-1,1)
			else:
				self.min_distances = np.minimum(self.min_distances, dist)

	def select_batch(self, already_selected, N):
		try:
			# print('Calculating distances...')
			self.update_distances(already_selected, only_new=False, reset_dist=True)
		except:
			print('Using flat_X as features.')
			self.update_distances(already_selected, only_new=True, reset_dist=False)
		new_batch = []
		for idx in range(N):
			if self.already_selected is None or idx == 0:
				ind = np.random.choice(np.arange(self.n_obs))
			else:
				ind = np.argmax(self.min_distances)
			assert ind not in already_selected
			self.update_distances([ind], only_new=True, reset_dist=False)
			new_batch.append(ind)
		# print('Maximum distance from cluster centers is %0.2f'% max(self.min_distances))
		self.already_selected = already_selected
		return new_batch

def kl(ac,bc):
	nc = 19
	kl_classes=[]
	ac=np.reshape(ac,(nc,nc))
	bc=np.reshape(bc,(nc,nc))
	for i in range(nc):
		a=ac[i,:]
		b=bc[i,:]
		kl1=a*np.log(a/b)
		kl2=b*np.log(b/a)
		kl= -0.5*(np.sum(kl1)) - 0.5*(np.sum(kl2))
		if(kl == kl and not np.isinf(kl)):
			kl_classes.append(kl)

	if(len(kl_classes) != 0):
		reward_kl=sum(kl_classes)/len(kl_classes)
	else:
		reward_kl = 0
	return abs(reward_kl)

def parse_args():
	parser = argparse.ArgumentParser(description='Run CDAL_CS selection')
	parser.add_argument('--budget',type=int, help='budget size')
	parser.add_argument('--file_path',help='path to save selected IDs name')
	parser.add_argument('--feature',help='path to features')
	parser.add_argument('--labeled_ids',help='txt file of labeled IDs')
	parser.add_argument('--nc',help='number of class')
	args = parser.parse_args()
	return args


def main():

	args = parse_args()
	features_path = args.feature
	features=[]
	unlabeled_names=[]
	
	for f in tqdm(sorted(glob.glob(features_path+'/*'))):
		features.append(torch.load(f).detach().cpu().numpy())
		unlabeled_names.append(f)
	temp = []
	for i in unlabeled_names:
		temp.append(i.split('/')[-1].split('.')[0])
		
	feature=np.stack(features)
	dist = len(feature)

	f1 = open(args.labeled_ids)
	l = f1.readlines()
	al_selected = []
	for i in l:
		i = i.strip()
		al_selected.append(i.split('/')[-1].split('.')[0])
	
	already_selected = []
	for i in range(len(temp)):
		if temp[i] in al_selected:
			already_selected.append(i)

	print('Number of Labeled IDs:'+str(len(np.unique(already_selected))))
	
	target_pick = args.budget

	feature=np.reshape(feature,(dist,args.nc*args.nc))

	selected_names = []
	kc = kCenterGreedy(feature)
	select = np.sort(kc.select_batch(already_selected,target_pick))

	for s in select:
		selected_names.append(unlabeled_names[s])

	file_name = 'selected.txt'
	path = args.file_path

	f = open(path+file_name,'w')
	for s in selected_names:
		f.write(s.split('/')[-1].split('.')[0]+'\n')
	print('Selection done')
	
		
if __name__ == '__main__':
    main()


