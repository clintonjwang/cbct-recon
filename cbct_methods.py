import config
import copy
import importlib
import mahotas.features as mah
import niftiutils.masks as masks
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import niftiutils.registration as reg
import numpy as np
import random
import math
from math import pi, radians, degrees
import pandas as pd
import glob
import shutil
import os
import time
from os.path import *
from scipy.ndimage.morphology import binary_closing, binary_opening, binary_dilation
from skimage.morphology import ball, label

###########################
### Preprocessing
###########################

def check_multi_tumors(lesion_id, target_dir):
	P = get_paths_dict(lesion_id, target_dir)
	
	mask_path = P['mrbl']['tumor']
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1 and tumor_vols[0] < 10*tumor_vols[1]:
		print(lesion_id, "has multiple tumors on BL MR")
		
	mask_path = P['mr30']['tumor']
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1 and tumor_vols[0] < 10*tumor_vols[1]:
		print(lesion_id, "has multiple tumors on 30d MR")
		
	mask_path = P['ct24']['tumor']
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1 and tumor_vols[0] < 10*tumor_vols[1]:
		print(lesion_id, "has multiple tumors on 24h CT")

def restrict_masks(lesion_id, target_dir):
	P = get_paths_dict(lesion_id, target_dir)
	
	mask_path = P['mrbl']['tumor']
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1:# and tumor_vols[0] < 10*tumor_vols[1]:
		for fn in glob.glob(mask_path+"*"):
			shutil.copy(fn, join(dirname(fn),"ZZbackup"+basename(fn)))
		masks.restrict_mask_to_largest(mask_path, img_path=P['mrbl']['art'])
		
	mask_path = P['mr30']['tumor']
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1:
		for fn in glob.glob(mask_path+"*"):
			shutil.copy(fn, join(dirname(fn),"ZZbackup"+basename(fn)))
		masks.restrict_mask_to_largest(mask_path, img_path=P['mr30']['art'])
		
	mask_path = P['ct24']['tumor']
	tumor_vols = masks.get_mask_disjoint_vols(mask_path)
	if len(tumor_vols) > 1:
		for fn in glob.glob(mask_path+"*"):
			shutil.copy(fn, join(dirname(fn),"ZZbackup"+basename(fn)))
		masks.restrict_mask_to_largest(mask_path, img_path=P['ct24']['img'])

def spherize(lesion_id, target_dir, R=1.):
	importlib.reload(reg)
	def ball_ct_batch():
		reg.transform_region(P['ct24']['img'], xform_path, crops, pads, [R]*3, P['ball']['ct24']['img'],
								 intermed_shape=ball_shape)
		#if exists(P['ct24']['midlip']+".off"):
		#	reg.transform_mask(P['ct24']['midlip'], P['ct24']['img'], xform_path,
		#						 crops, pads, [R]*3, P['ball']['ct24']['midlip'], intermed_shape=ball_shape)
		#	if exists(P['ct24']['highlip']+".off"):
		#		reg.transform_mask(P['ct24']['highlip'], P['ct24']['img'], xform_path,
		#							 crops, pads, [R]*3, P['ball']['ct24']['highlip'], intermed_shape=ball_shape)

	def ball_mr_batch(mod):
		reg.transform_region(P[mod]['art'], xform_path, crops, pads, [R]*3, P['ball'][mod]['art'], intermed_shape=ball_shape)
		
		if exists(P['ball'][mod]['enh']+".off"):
			reg.transform_mask(P[mod]['enh'], P[mod]['art'], xform_path,
								 crops, pads, [R]*3, P['ball'][mod]['enh'], intermed_shape=ball_shape)

	P = get_paths_dict(lesion_id, target_dir)
	
	ctmask,ctd = masks.get_mask(P['ct24']['tumor'], img_path=P['ct24']['img'])
	mrmask,mrd = masks.get_mask(P['mrbl']['tumor'], img_path=P['mrbl']['art'])
	ctmask = hf.crop_nonzero(ctmask)[0]
	mrmask = hf.crop_nonzero(mrmask)[0]
	CT = np.max([ctmask.shape[i] * ctd[i] for i in range(3)])
	MRBL = np.max([mrmask.shape[i] * mrd[i] for i in range(3)])
	
	mrmask,mrd = masks.get_mask(P['mr30']['tumor'], img_path=P['mr30']['art'])
	mrmask = hf.crop_nonzero(mrmask)[0]
	MR30 = np.max([mrmask.shape[i] * mrd[i] for i in range(3)])
	
	if CT > MRBL and CT > MR30:
		xform_path, crops, pads = reg.get_mask_Tx_shape(P['ct24']['img'], P['ct24']['tumor'], mask_path=P['ball']['mask'])
		ball_shape = masks.get_mask(P['ball']['mask'])[0].shape
		ball_ct_batch()

		xform_path, crops, pads = reg.get_mask_Tx_shape(P['mrbl']['art'], P['mrbl']['tumor'], ball_mask_path=P['ball']['mask'])
		ball_mr_batch('mrbl')

		xform_path, crops, pads = reg.get_mask_Tx_shape(P['mr30']['art'], P['mr30']['tumor'], ball_mask_path=P['ball']['mask'])
		ball_mr_batch('mr30')
		
	elif MRBL > MR30:
		xform_path, crops, pads = reg.get_mask_Tx_shape(P['mrbl']['art'],
											P['mrbl']['tumor'], mask_path=P['ball']['mask'])
		ball_shape = masks.get_mask(P['ball']['mask'])[0].shape
		ball_mr_batch('mrbl')
		
		xform_path, crops, pads = reg.get_mask_Tx_shape(P['ct24']['img'], P['ct24']['tumor'], ball_mask_path=P['ball']['mask'])
		ball_ct_batch()

		xform_path, crops, pads = reg.get_mask_Tx_shape(P['mr30']['art'], P['mr30']['tumor'], ball_mask_path=P['ball']['mask'])
		ball_mr_batch('mr30')
		
	else:
		xform_path, crops, pads = reg.get_mask_Tx_shape(P['mr30']['art'], P['mr30']['tumor'], mask_path=P['ball']['mask'])
		ball_shape = masks.get_mask(P['ball']['mask'])[0].shape
		ball_mr_batch('mr30')
		
		xform_path, crops, pads = reg.get_mask_Tx_shape(P['mrbl']['art'], P['mrbl']['tumor'], ball_mask_path=P['ball']['mask'])
		ball_mr_batch('mrbl')
		
		xform_path, crops, pads = reg.get_mask_Tx_shape(P['ct24']['img'], P['ct24']['tumor'], ball_mask_path=P['ball']['mask'])
		ball_ct_batch()

def reg_to_ct24(lesion_id, target_dir, D=[1.,1.,2.5], padding=.2):
	importlib.reload(reg)
	P = get_paths_dict(lesion_id, target_dir)

	ct24, ct24_dims = hf.nii_load(P['ct24']['img'])
	fmod='ct24'

	mod = 'mrbl'
	xform_path, crops, pad_m = reg.get_mask_Tx(P[fmod]['img'], P[fmod]['tumor'],
					P[mod]['art'], P[mod]['tumor'], padding=padding, D=D)
	
	crop_ct24 = hf.crop_nonzero(ct24, crops[1])[0]
	t_shape = crop_ct24.shape
	reg.transform_region(P[mod]['art'], xform_path, crops, pad_m, ct24_dims,
				P['ct24Tx'][mod]['art'], target_shape=t_shape, D=D);
	reg.transform_region(P[mod]['sub'], xform_path, crops, pad_m, ct24_dims,
				P['ct24Tx'][mod]['sub'], target_shape=t_shape, D=D);
	#reg.transform_region(P[mod]['equ'], xform_path, crops, pad_m, ct24_dims,
	#			P['ct24Tx'][mod]['equ'], target_shape=t_shape, D=D);
	if exists(P[mod]['enh']+".off"):
		reg.transform_mask(P[mod]['enh'], P[mod]['art'], xform_path, crops, pad_m, ct24_dims,
				P['ct24Tx'][mod]['enh'], target_shape=t_shape, D=D);

	hf.save_nii(crop_ct24, P['ct24Tx']['crop']['img'], ct24_dims)
	M = masks.get_mask(P['ct24']['tumor'], ct24_dims, ct24.shape)[0]
	M = hf.crop_nonzero(M, crops[1])[0]
	masks.save_mask(M, P['ct24Tx']['crop']['tumor'], ct24_dims)

	mod = 'mr30'
	xform_path, crops, pad_m = reg.get_mask_Tx(P[fmod]['img'], P[fmod]['tumor'],
					P[mod]['art'], P[mod]['tumor'], padding=padding, D=D)
	
	reg.transform_region(P[mod]['art'], xform_path, crops, pad_m, ct24_dims,
				P['ct24Tx'][mod]['art'], target_shape=t_shape, D=D);
	if exists(P[mod]['enh']+".off"):
		reg.transform_mask(P[mod]['enh'], P[mod]['art'], xform_path, crops, pad_m, ct24_dims,
				P['ct24Tx'][mod]['enh'], target_shape=t_shape, D=D);


###########################
### File I/O
###########################

def check_paths(lesion_id, target_dir):
	P = get_paths_dict(lesion_id, target_dir)
	for path in [P['mask'], P['nii'], P['ct24']['img'], P['ct24']['tumor'], P['ct24']['liver'], \
			P['mrbl']['art'], P['mrbl']['pre'], P['mrbl']['sub'], \
			P['mrbl']['tumor'], P['mrbl']['liver'], \
			P['mr30']['art'], P['mr30']['pre'], \
			P['mr30']['tumor'], P['mr30']['liver']]:
		if not exists(path) and not exists(path+".ics"):
			print(path, "does not exist!")
			raise ValueError(path)

	#if P['mrbl']['enh'] and P['mrbl']['nec']:
	#if P['mr30']['enh'] and P['mr30']['nec']:

	P['ball']['ct24']['img'] = join(nii_dir, "ct24_ball.nii")
	ball_mribl_path = join(nii_dir, "mribl_ball.nii")
	ball_mri30d_path = join(nii_dir, "mri30d_ball.nii")
	ball_mask_path = join(mask_dir, "ball_mask")
	P['ball']['mrbl']['enh'] = join(mask_dir, "ball_mribl_enh_mask")
	P['ball']['mr30']['enh'] = join(mask_dir, "ball_mri30d_enh_mask")

	midlip_mask_path = join(mask_dir, "mid_lipiodol")
	ball_midlip_mask_path = join(mask_dir, "ball_mid_lipiodol")
	highlip_mask_path = join(mask_dir, "high_lipiodol")
	ball_highlip_mask_path = join(mask_dir, "ball_lipiodol")

	paths += [P['ball']['ct24']['img'], ball_mribl_path, ball_mri30d_path, \
			ball_mask_path, P['ball']['mrbl']['enh'], P['ball']['mr30']['enh'], \
			midlip_mask_path, ball_midlip_mask_path, \
			highlip_mask_path, ball_highlip_mask_path]

	#if flag:
	#	paths[-1] = None
	return paths

def get_paths_dict(lesion_id, target_dir):
	if not exists(join(target_dir, lesion_id)):
		raise ValueError(lesion_id, "does not exist!")

	mask_dir = join(target_dir, lesion_id, "masks")
	nii_dir = join(target_dir, lesion_id, "nii_files")
	if not exists(nii_dir):
		os.makedirs(nii_dir)

	P = {'mask':mask_dir, 'nii':nii_dir, 'mrbl':{}, 'ct24':{}, 'mr30':{},
		'ball':{'mrbl':{}, 'ct24':{}, 'mr30':{}}, 'mr30Tx':{'mrbl':{}, 'ct24':{}},
			'ct24Tx':{'mrbl':{}, 'crop':{}, 'mr30':{}}}

	P['ct24']['img'] = join(nii_dir, "ct24.nii.gz")
	P['ct24']['tumor'] = glob.glob(join(mask_dir, "tumor*24h*.ids"))
	P['ct24']['liver'] = glob.glob(join(mask_dir, "wholeliver_24hCT*.ids"))
	P['ct24']['lowlip'] = join(mask_dir, "lipiodol_low")
	P['ct24']['midlip'] = join(mask_dir, "lipiodol_mid")
	P['ct24']['highlip'] = join(mask_dir, "lipiodol_high")
	if len(P['ct24']['tumor']) > 0:
		P['ct24']['tumor'] = P['ct24']['tumor'][0]
	else:
		raise ValueError('tumor')
	if len(P['ct24']['liver']) > 0:
		P['ct24']['liver'] = P['ct24']['liver'][0]
	else:
		raise ValueError('liver')

	P['mrbl']['art'] = join(target_dir, lesion_id, "MRI-BL", "mrbl_art.nii.gz")
	P['mrbl']['pre'] = join(target_dir, lesion_id, "MRI-BL", "mrbl_pre.nii.gz")
	P['mrbl']['sub'] = join(target_dir, lesion_id, "MRI-BL", "mrbl_sub.nii.gz")
	P['mrbl']['equ'] = join(target_dir, lesion_id, "MRI-BL", "mrbl_equ.nii.gz")
	P['mr30']['art'] = join(target_dir, lesion_id, "MRI-30d", "mr30_art.nii.gz")
	P['mr30']['pre'] = join(target_dir, lesion_id, "MRI-30d", "mr30_pre.nii.gz")
	P['mr30']['sub'] = join(target_dir, lesion_id, "MRI-30d", "mr30_sub.nii.gz")
	P['mr30']['equ'] = join(target_dir, lesion_id, "MRI-30d", "mr30_equ.nii.gz")

	P['mrbl']['tumor'] = join(mask_dir, "tumor_BL_MRI")
	P['mrbl']['liver'] = join(mask_dir, "mribl_liver")
	P['mrbl']['enh'] = join(mask_dir, "enh_bl")
	P['mrbl']['nec'] = join(mask_dir, "nec_bl")
	P['mr30']['tumor'] = join(mask_dir, "tumor_30dFU_MRI")
	P['mr30']['liver'] = join(mask_dir, "mri30d_liver")
	P['mr30']['enh'] = join(mask_dir, "enh_30d")
	P['mr30']['nec'] = join(mask_dir, "nec_30d")


	if not exists(join(nii_dir, "reg")):
		os.makedirs(join(nii_dir, "reg"))
	if not exists(join(mask_dir, "reg")):
		os.makedirs(join(mask_dir, "reg"))

	P['ball']['mask'] = join(mask_dir, "reg", "ball_mask")
	P['ball']['mrbl']['art'] = join(nii_dir, "reg", "ball_mrbl.nii")
	P['ball']['mrbl']['enh'] = join(mask_dir, "reg", "ball_mrbl_enh_mask")
	P['ball']['mr30']['art'] = join(nii_dir, "reg", "ball_mr30.nii")
	P['ball']['mr30']['enh'] = join(mask_dir, "reg", "ball_mri30d_enh_mask")
	P['ball']['ct24']['img'] = join(nii_dir, "reg", "ball_ct24.nii")
	P['ball']['ct24']['lowlip'] = join(mask_dir, "reg", "ball_lowlip")
	P['ball']['ct24']['midlip'] = join(mask_dir, "reg", "ball_midlip")
	P['ball']['ct24']['highlip'] = join(mask_dir, "reg", "ball_highlip")

	P['mr30Tx']['mrbl']['art'] = join(nii_dir, "reg", "mr30Tx-mrbl-art.nii")
	P['mr30Tx']['mrbl']['enh'] = join(mask_dir, "reg", "mr30Tx-mrbl-enh")
	P['mr30Tx']['ct24']['img'] = join(nii_dir, "reg", "mr30Tx-ct24-img.nii")
	P['mr30Tx']['ct24']['midlip'] = join(mask_dir, "reg", "mr30Tx-ct24-midlip")
	P['mr30Tx']['ct24']['highlip'] = join(mask_dir, "reg", "mr30Tx-ct24-highlip")

	P['ct24Tx']['mrbl']['art'] = join(nii_dir, "reg", "ct24Tx-mrbl-art.nii")
	P['ct24Tx']['mrbl']['sub'] = join(nii_dir, "reg", "ct24Tx-mrbl-sub.nii")
	P['ct24Tx']['mrbl']['equ'] = join(nii_dir, "reg", "ct24Tx-mrbl-equ.nii")
	P['ct24Tx']['mrbl']['enh'] = join(mask_dir, "reg", "ct24Tx-mrbl-enh")
	P['ct24Tx']['mr30']['art'] = join(nii_dir, "reg", "ct24Tx-mr30-art.nii")
	P['ct24Tx']['mr30']['enh'] = join(mask_dir, "reg", "ct24Tx-mr30-enh")
	P['ct24Tx']['crop']['img'] = join(nii_dir, "reg", "ct24Tx-crop-img.nii")
	P['ct24Tx']['crop']['tumor'] = join(mask_dir, "reg", "ct24Tx-crop-tumor")
	P['ct24Tx']['crop']['midlip'] = join(mask_dir, "reg", "ct24Tx-crop-midlip")
	P['ct24Tx']['crop']['highlip'] = join(mask_dir, "reg", "ct24Tx-crop-highlip")

	return P

###########################
### Segmentation methods
###########################

def seg_lipiodol(P, thresholds=[75,160,215]):
	#P = get_paths_dict(lesion_id, target_dir)
	
	img, dims = hf.nii_load(P['ct24']['img'])

	low_mask = copy.deepcopy(img)
	low_mask = low_mask > thresholds[0]
	low_mask = binary_closing(binary_opening(low_mask, structure=np.ones((3,3,2))), structure=np.ones((2,2,1)))#2,2,1
	mid_mask = copy.deepcopy(img)
	mid_mask = mid_mask > thresholds[1]
	mid_mask = binary_closing(mid_mask)
	high_mask = copy.deepcopy(img)
	high_mask = high_mask > thresholds[2]

	mask,_ = masks.get_mask(P['ct24']['liver'], dims, img.shape)
	low_mask = low_mask*mask
	mid_mask = mid_mask*mask
	high_mask = high_mask*mask
	
	masks.save_mask(low_mask, P['ct24']['lowlip'], dims, save_mesh=True)
	masks.save_mask(mid_mask, P['ct24']['midlip'], dims, save_mesh=True)
	masks.save_mask(high_mask, P['ct24']['highlip'], dims, save_mesh=True)

	return low_mask, mid_mask, high_mask

def seg_tumor_ct(P, threshold=150, num_tumors=1):
	ct_img, D = hf.nii_load(P["ct24"]["img"])
	mask, _ = masks.get_mask(P["ct24"]["liver"], D, ct_img.shape)

	tumor_mask = (mask > 0) * ct_img > threshold
	tumor_mask = binary_closing(tumor_mask)
	
	if num_tumors == 1:
		tumor_labels, num_labels = label(tumor_mask, return_num=True)
		label_sizes = [np.sum(tumor_labels == label_id) for label_id in range(1,num_labels+1)]
		biggest_label = label_sizes.index(max(label_sizes))+1
		tumor_mask[tumor_labels != biggest_label] = 0
	
	B3 = ball(3)
	B3 = B3[:,:,[0,2,3,4,6]]
	tumor_mask = binary_opening(binary_closing(mask1, B3), B3)
	
	masks.save_mask(tumor_mask, P["ct24"]["tumor"], D)

	return tumor_mask

def seg_target_lipiodol(P, thresholds=[75,160,215], num_tumors=1):
	raise ValueError("Not ready")
	ct_img, D = hf.nii_load(P["ct24"]["img"])
	mask, _ = masks.get_mask(seg, D, ct_img.shape)
			
	img = (mask > 0) * ct_img
	#save_folder = 
	info[dirname(seg)] = (ct_img, D, mask)

	if P["ct24"]["tumor"] is None:
		tumor_mask = img > thresholds[1]
		tumor_mask = binary_closing(tumor_mask)
	else:
		tumor_mask = masks.get_mask(P["ct24"]["tumor"], D, ct_img.shape)
	
	mask1 = copy.deepcopy(img)
	mask1 = mask1 > thresholds[1]
	B2 = ball(2)
	B2 = B2[:,:,[0,2,4]]
	mask1 = binary_dilation(mask1, np.ones((3,3,1)))
	
	if num_tumors == 1:
		tumor_labels, num_labels = label(mask1, return_num=True)
		label_sizes = [np.sum(tumor_labels == label_id) for label_id in range(1,num_labels+1)]
		biggest_label = label_sizes.index(max(label_sizes))+1
		mask1[tumor_labels != biggest_label] = 0
	
	mask1 = binary_opening(binary_closing(mask1, structure=B2, iterations=2), structure=B2, iterations=2)
	
	if num_tumors == 1:
		tumor_labels, num_labels = label(mask1, return_num=True)
		label_sizes = [np.sum(tumor_labels == label_id) for label_id in range(1,num_labels+1)]
		biggest_label = label_sizes.index(max(label_sizes))+1
		mask1[tumor_labels != biggest_label] = 0
	
	target_mask = mask1 * tumor_mask
	nontarget_mask = (1-mask1) * tumor_mask
	
	masks.save_mask(target_mask, join(P["mask"], "target_lip"), D, save_mesh=True)
	masks.save_mask(nontarget_mask, join(P["mask"], "nontarget_lip"), D, save_mesh=True)

def seg_liver_mr(mri_path, save_path, model, tumor_mask_path=None):
	mri_img, mri_dims = hf.nii_load(mri_path)
	seg_liver_mri(mri_img, save_path, mri_dims, model, tumor_mask_path)

def seg_liver_mri(mri_img, save_path, mri_dims, model, tumor_mask_path=None):
	"""Use a UNet to segment liver on MRI"""

	C = config.Config()
	#correct bias field!

	orig_shape = mri_img.shape

	x = mri_img
	x -= np.amin(x)
	x /= np.std(x)

	crops = list(map(int,[.05 * x.shape[0], .95 * x.shape[0]] + \
					[.05 * x.shape[1], .95 * x.shape[1]] + \
					[.05 * x.shape[2], .95 * x.shape[2]]))
	
	x = x[crops[0]:crops[1], crops[2]:crops[3], crops[4]:crops[5]]
	scale_shape = x.shape
	x = tr.rescale_img(x, C.dims)
	
	y = model.predict(np.expand_dims(x,0))[0]
	liver_mask = (y[:,:,:,1] > y[:,:,:,0]).astype(float)
	liver_mask = tr.rescale_img(liver_mask, scale_shape)#orig_shape)
	liver_mask = np.pad(liver_mask, ((crops[0], orig_shape[0]-crops[1]),
									 (crops[2], orig_shape[1]-crops[3]),
									 (crops[4], orig_shape[2]-crops[5])), 'constant')
	liver_mask = liver_mask > .5

	B3 = ball(3)
	B3 = B3[:,:,[0,2,3,4,6]]
	#B3 = ball(4)
	#B3 = B3[:,:,[1,3,5,6,8]]
	liver_mask = binary_opening(binary_closing(liver_mask, B3, 1), B3, 1)

	labels, num_labels = label(liver_mask, return_num=True)
	label_sizes = [np.sum(labels == label_id) for label_id in range(1,num_labels+1)]
	biggest_label = label_sizes.index(max(label_sizes))+1
	liver_mask[labels != biggest_label] = 0

	if tumor_mask_path is not None:
		tumor_mask, _ = masks.get_mask(tumor_mask_path, mri_dims, mri_img.shape)
		liver_mask[tumor_mask > tumor_mask.max()/2] = liver_mask.max()

	masks.save_mask(liver_mask, save_path, mri_dims, save_mesh=True)

def seg_liver_ct(ct_path, save_path, model, tumor_mask_path=None):
	"""Use a UNet to segment liver on CT"""

	C = config.Config()
	ct_img, ct_dims = hf.nii_load(ct_path)

	orig_shape = ct_img.shape

	x = tr.apply_window(ct_img)
	x -= np.amin(x)

	crops = list(map(int,[.05 * x.shape[0], .95 * x.shape[0]] + \
					[.05 * x.shape[1], .95 * x.shape[1]] + \
					[.05 * x.shape[2], .95 * x.shape[2]]))
	
	x = x[crops[0]:crops[1], crops[2]:crops[3], crops[4]:crops[5]]
	scale_shape = x.shape
	x = tr.rescale_img(x, C.dims)
	
	y = model.predict(np.expand_dims(x,0))[0]
	liver_mask = (y[:,:,:,1] > y[:,:,:,0]).astype(float)
	liver_mask[x < 30] = 0
	liver_mask = tr.rescale_img(liver_mask, scale_shape)
	liver_mask = np.pad(liver_mask, ((crops[0], orig_shape[0]-crops[1]),
									 (crops[2], orig_shape[1]-crops[3]),
									 (crops[4], orig_shape[2]-crops[5])), 'constant')

	B3 = ball(3)
	B3 = B3[:,:,[0,2,3,4,6]]
	liver_mask = binary_opening(binary_closing(liver_mask, B3, 1), B3, 1)

	labels, num_labels = label(liver_mask, return_num=True)
	label_sizes = [np.sum(labels == label_id) for label_id in range(1,num_labels+1)]
	biggest_label = label_sizes.index(max(label_sizes))+1
	liver_mask[labels != biggest_label] = 0

	if tumor_mask_path is not None:
		tumor_mask, _ = masks.get_mask(tumor_mask_path, ct_dims, ct_img.shape)
		liver_mask[tumor_mask > tumor_mask.max()/2] = liver_mask.max()
	
	masks.save_mask(liver_mask, save_path, ct_dims, save_mesh=True)