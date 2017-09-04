# simple face region extractor.
# last modified : 2017.09.04, nashory
# requirements : dlib library (http://dlib.net/)

import os, sys
from os.path import join
import dlib
import glob
from PIL import Image
from skimage import io
import numpy as np
import json
import argparse
import mcutils as utils
import random



def main(params):
	# parsing parameters.
	input = params['input']
	output = params['output']
	imsize = params['imsize']
	resize = params['resize']
	detect = params['detect']


	# refresg and make directories.
	utils.refresh_directory(output)

	
	print 'extract face regions...'
	detector = dlib.get_frontal_face_detector()

	cnt = 0
	valid_ext = ['.jpg', '.png']
	for filename in glob.glob(os.path.join(input, '*')):
		flist = os.path.splitext(filename)
		fname = utils.get_folder_name(flist[0])
		fext = flist[1]
		
		if fext.lower() not in valid_ext:
			continue
	
		
		im = Image.open(filename)
		if fext.lower() == '.png':
			image = Image.open(filename)
			image = im.convert('RGB')
			im = np.fromstring(image.tobytes(), dtype=np.uint8)
			im = im.reshape((image.size[1], image.size[0], 3)) 
		else:
			im = Image.open(filename)

		dets = detector(im, 1)
		#crop = im[dets[0].left():dets[0].right(), dets[0].top():dets[0].bottom()]
		crop = im[dets[0].top():dets[0].bottom(), dets[0].left():dets[0].right()]

		# save image.
		io.imsave(os.path.join(output, fname+'.jpg'), crop)
		
		# logging.
		cnt = cnt +1
		print '[' + str(cnt) + '] ' + 'saved @ ' + os.path.join(output, fname+'.jpg') 
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	# options
	parser.add_argument('--input', default='/home1/irteam/nashory/data/FERG_DB_256/img_only')
	parser.add_argument('--output', default='/home1/irteam/nashory/data/domain/domain_B')
	parser.add_argument('--imsize', default=96, type=int, help='image width/height for resize.')
	parser.add_argument('--resize', default=True, type=bool, help='true: resize. false: no resize.')
	parser.add_argument('--detect', default=False, type=bool, help='true: face detection. false: no face detection.')
	
	args = parser.parse_args()
	params = vars(args)		# convert to ordinary dict
	print 'parsed input parameters : '
	print json.dumps(params, indent = 4)
	main(params)



