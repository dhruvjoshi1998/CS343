def conv2nn(imgs, kers, bias, verbose=True):
	'''General 2D convolution operation suitable for a convolutional layer of a neural network.
	Uses 'same' boundary conditions.

	Parameters:
	-----------
	imgs: ndarray. Input IMAGES to be filtered. shape=(BATCH_SZ, n_chans, img_y, img_x)
		where batch_sz is the number of images in the mini-batch
		n_chans is 1 for grayscale images and 3 for RGB color images
	kers: ndarray. Convolution kernels. shape=(n_kers, N_CHANS, ker_sz, ker_sz)
		NOTE: Each kernel should be square and [ker_sz < img_y] and [ker_sz < img_x]
	bias: ndarray. Bias term used in the neural network layer. Shape=(n_kers,)
		i.e. there is a single bias per filter. Convolution by the c-th filter gets the c-th
		bias term added to it.
	verbose: bool. I suggest making helpful print statements showing the shape of various things
		as you go. Only execute these print statements if verbose is True.

	What's new (vs conv2):
	-----------
	- Multiple images (mini-batch support)
	- Kernels now have a color channel dimension too
	- Collapse (sum) over color channels when computing the returned output images
	- A bias term

	Returns:
	-----------
	output: ndarray. imgs filtered with all kers. shape=(BATCH_SZ, n_kers, img_y, img_x)

	Hints:
	-----------
	- You may need additional loop(s).
	- Summing inside your loop can be made simpler compared to conv2.
	- Adding the bias should be easy.
	'''
	batch_sz, n_chans, img_y, img_x = imgs.shape
	n_kers, n_ker_chans, ker_x, ker_y = kers.shape

	# Flip the Kernel
	kers_flipped = kers[:, :, ::-1, ::-1]

	# Add Zero Padding to all imgs in batch
	p = math.ceil((ker_x -1)/2)

	cols = np.zeros((batch_sz, n_chans, img_y, p))
	padded_img = np.append(cols, imgs, axis=3)
	padded_img = np.append(padded_img, cols, axis=3)

	rows = np.zeros((batch_sz, n_chans, p, padded_img.shape[3]))
	padded_img = np.append(rows, padded_img, axis=2)
	padded_img = np.append(padded_img, rows, axis=2)


	# Create Output Images
	filteredImg = np.zeros([batch_sz, n_kers, img_y, img_x])

	# Apply Conv
	for l in range(batch_sz): 
		for k in range(n_kers):
			for m in range(n_ker_chans):
				for j in range(img_y):
					for i in range(img_x):
						filteredImg[l, k, j, i] = np.sum(np.multiply(padded_img[l, :, j:j+ker_y, i:i+ker_x], kers_flipped[k,m,:,:]))
			filteredImg[l,k,:,:] = filteredImg[l,k,:,:] + bias[k].astype(np.int)

	return filteredImg