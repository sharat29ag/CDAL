import os
import numpy as np
import glob
import torch
import torch.nn.functional as F

features_path='./features/*'
os.system('rm -r ./features2')
os.system('mkdir ./features2')

features_path2='./features2/'

for idx,f in enumerate(sorted(glob.glob(features_path))):
	feature=np.load(f)
	feature=F.softmax(torch.tensor(feature),dim=1).numpy()
	w=-1*np.nansum(feature*np.log(feature),axis=1)
	feature=np.average(feature,axis=0,weights=w)
	np.save(features_path2+f.split('/')[-1],feature)

