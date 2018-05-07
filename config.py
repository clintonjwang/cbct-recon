"""
Author: Clinton Wang, E-mail: `clintonjwang@gmail.com`, Github: `https://github.com/clintonjwang/lipiodol`
"""

class Config:
	def __init__(self):
		self.proj_dims = [64,64,32]
		self.world_dims = [64,64,64]
		#self.aug_factor = 100

		self.npy_dir = r"D:\CBCT\Train\NPYs"
		self.nii_dir = r"D:\CBCT\Train\NIFTIs"
		self.dcm_dir = r"D:\CBCT\Train\DICOMs"
		self.train_dir = r"D:\CBCT\Train"