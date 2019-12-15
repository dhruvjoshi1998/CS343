'''word2vec.py
Implements the Skip-gram Word2vec neural network
CS343: Neural Networks
YOUR NAMES HERE
Project 5: Word Embeddings and SOMs
'''
import re
import numpy as np
import network
import layer
from scipy._lib.six import xrange


def tokenize_words(text):
	'''Transforms a string sentence into words.

	Parameters:
	-----------
	text: string. Sentence of text.

	Returns:
	-----------
	list of strings. Words in the sentence `text`.

	This method is pre-filled for you (shouldn't require modification).
	'''
	# Define words as lowercase text with at least one alphabetic letter
	pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
	return pattern.findall(text.lower())


def one_hot(y, num_classes):
	'''One-hot codes the output classes for a mini-batch

	This method is pre-filled for you (shouldn't require modification).

	Parameters:
	-----------
	y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
	num_classes: int. Number of unique output classes total

	Returns:
	-----------
	y_one_hot: ndarray. One-hot coded class assignments.
		e.g. if y = [0, 2, 1] and num_classes = 4 we have:
		[[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
	'''
	y_one_hot = np.zeros([len(y), num_classes])
	y_one_hot[np.arange(len(y_one_hot)), y] = 1
	return y_one_hot


def make_corpus(data, min_sent_size=5):
	'''Make the text corpus.
	Transforms text documents (list of strings) into a list of list of words (both Python lists).
	The format is [[<sentence>], [<sentence>], ...], where <sentence> = [<word>, <word>, ...].

	For the IMDb data, this transforms a list of reviews (each is a single string) into a list of
	sentences, where each sentence is represented as a list of string words. So the elements of the
	resulting list are the i-th sentence overall; we lose information about which review the
	sentence comes from.

	Parameters:
	-----------
	data: list of strings.
	min_sent_size: int. Don't add sentences LESS THAN this number of words to the corpus (skip over them).
		This is important because it removes empty sentences (bad parsing) and those with not enough
		word context.

	Returns:
	-----------
	corpus: list of lists (sentences) of strings (words in each sentence)

	TODO:
	- Split each review into sentences based on periods.
	- Tokenize the sentence into individual word strings (via tokenenize_words())
	- Only add a list of words to the corpus if the length is at least `min_sent_size`.
	'''
	corpus = []
	for review in data:
		sentences = review.split('.')
		for sentence in sentences:
			words = tokenize_words(sentence)
			if (len(words)>=5):
				corpus.append(words)
	return corpus


def find_unique_words(corpus):
	'''Define the vocabulary in the corpus (unique words)

	Parameters:
	-----------
	corpus: list of lists (sentences) of strings (words in each sentence).

	Returns:
	-----------
	unique_words: list of unique words in the corpus.

	TODO:
	- Find and return a list of the unique words in the corpus.
	'''
	corpus_np = [word for sentence in corpus for word in sentence]    
	sorted_unique_words, indices = np.unique(corpus_np, return_index=True)
	words = [corpus_np[index] for index in sorted(indices)]
	return words

def make_word2ind_mapping(vocab):
	'''Create dictionary that looks up a word index (int) by its string.
	Indices for each word are in the range [0, vocab_sz-1].

	Parameters:
	-----------
	vocab: list of strings. Unique words in corpus.

	Returns:
	-----------
	dictionary with key,value pairs: string,int
	'''
	return dict(zip(vocab, range(len(vocab))))


def make_ind2word_mapping(vocab):
	'''Create dictionary that uses a word int code to look up its word string
	Indices for each word are in the range [0, vocab_sz-1].

	Parameters:
	-----------
	vocab: list of strings. Unique words in corpus.

	Returns:
	-----------
	dictionary with key,value pairs: int,string
	'''
	return dict(zip(range(len(vocab)), vocab))


def make_target_context_word_lists(corpus, word2ind, vocab_sz, context_win_sz=2):
	'''Make the target word list (training data) and context word list ("classes")

	Parameters:
	-----------
	corpus: list of lists (sentences) of strings (words in each sentence).
	word2ind: Dictionary mapping word string -> int code index. Range is [0, vocab_sz-1] inclusive.
	context_win_sz: How many words to include before/after the target word in sentences for context.

	Returns:
	-----------
	target_words_onehot: Python list of ndarrays.
		Each ndarray is the i-th one-hot coded target word in corpus.
		len(outer list) = N training samples. shape(each inner ndarray) = (1, vocab_sz)
		NOTE: The 1 is needed as a placeholder for the batch dimension used by the neural net.
	context_words_int: Python list of ndarrays.
		Each ndarray contains int code indices for words surrounding the i-th target word in the
		sentence.
		len(outer list) = N training samples.
		shape(each inner ndarray) = (#context_words,).
		NOTE: #context_words is a variable value in the range [context_win_sz, 2*context_win_sz].
		It is not always the same because of sentence boundary effects. This is why we're using a
		list of ndarrays (not simply one multidimensional ndarray).

	HINT:
	- Search in a window `context_win_sz` words before after the current target in its sentence.
	Add int code indices of these context words to a ndarray and add this ndarray to the
	`context_words_int` list.
		- Only add context words if they are valid within the window. For example, only int codes of
		words on the right side of the first word of a sentence are added for that target word.

	Example:
	corpus = [['with', 'all', 'this', 'stuff', ...], ...]
	target_words_onehot = [array([[1., 0., 0., ..., 0., 0., 0.]]),
						   array([[0., 1., 0., ..., 0., 0., 0.]]),
						   array([[0., 0., 1., ..., 0., 0., 0.]]),
						   array([[0., 0., 0., ..., 0., 0., 0.]]),...]
	context_words_int =   [array([1, 2]),
						   array([0, 2, 3]),
						   array([0, 1, 3, 4]),
						   array([1, 2, 4, 5]),...]

	'''
	
	context_words_int = []
	for sentence in corpus:
		for i, word in enumerate(sentence):
			curr_context_words = [word2ind[sentence[val]] for val in range(i-context_win_sz,i+context_win_sz+1) if val >=0 and val <= (len(sentence)-1) and val!=i]
			context_words_int.append(np.array(curr_context_words))
	int_coded_corpus = [word2ind[word] for sentence in corpus for word in sentence]
	target_words_onehot = [one_hot([int_code], vocab_sz) for int_code in int_coded_corpus]
	return target_words_onehot, context_words_int

class Skipgram(network.Network):
	'''
	Network architecture is: Input -> Dense (linear act) -> Dense (softmax act)

	Shapes of layer inputs/wts:
	input:         (1, vocab_sz)
	hidden wts:    (vocab_sz, embedding_sz)
	output_wts:    (embedding_sz, vocab_sz)
	output netAct: (1, vocab_sz)

	embedding_sz is the embedding dimenson for words (number of dimensions of word vectors).
	vocab_sz is the number of unique words in the corpus. This is the number of input and output units
		Intuition for output layer activations:
		"which word(s) in the vocab likely surround the current target word?"
	'''

	def __init__(self, input_shape=(1, 1016), dense_interior_units=(10,), n_classes=1016,
				 wt_scale=1e-3, reg=0, verbose=True):
		'''
		Parameters:
		-----------
		input_shape: tuple. Shape of a SINGLE input sample. shape=(1, vocab_size)
			The 1 is hard-coded. We aren't adding mini-batch support, so we are always processing
			one target word at a time — hence, our batch dimension is 1.
			Adding mini-batch support is an extension.
		dense_interior_units: tuple. Number of hidden units in each dense layer (not counting output layer).
			Same as embedding dimension / embedding_sz.
		n_classes: int. Number of classes in the input.
			This is also the number of units in the Output Dense layer. Same as `vocab_sz`.
		wt_scale: float. Global weight scaling to use for all layers with weights
		reg: float. Regularization strength
		verbose: bool. Do we want to term network-related debug print outs on?
			NOTE: This is different than per-layer verbose settings, which you can turn on manually on below.

		TODO:
		1. Assemble the layers of the network and add them (in order) to `self.layers`.
		2. You will soon handle the "stacked" output layer via a modified softmax activation function.
		   Make the output layer activation function be this function: 'softmax_embedding'.
		3. Remember to define self.wt_layer_inds as the list indicies in self.layers that have weights.
		'''
		super().__init__(reg, verbose)

		_, vocab_sz = input_shape

		D1 = layer.Dense(number=1,name="Hidden",units=dense_interior_units[0],n_units_prev_layer=vocab_sz,wt_scale=wt_scale,activation="linear",reg=reg,verbose=False)
		self.layers.append(D1)
		D2 = layer.Dense(number=1,name="Output",units=vocab_sz,n_units_prev_layer=dense_interior_units[0],wt_scale=wt_scale,activation="softmax_embedding",reg=reg,verbose=False)
		self.layers.append(D2)
		self.wt_layer_inds = [0,1]

	def fit(self, targets_train, contexts_train, n_epochs=10, print_every=10):
		'''Trains the Skip-gram neural network on target and context word data

		Parameters:
		-----------
		targets_train: list of ndarrays. len=num training samples, sub ndarray shape=(1, vocab_sz).
			Training target word data one-hot coded.
		contexts_train: list of ndarrays.
			len=num training samples, sub ndarray shape=(win_sz=<c<2*win_sz).
			Training context word data one-hot coded. sub ndarray shape[0] is variable due to border
			effects at start/end of sentences in corpus.
		n_epochs: int. Number of training epochs.
		print_every: int. How many training EPOCHS we wait before printing the current epoch loss.

		Returns:
		-----------
		self.loss_history. Python list of float. len=n_epochs
			The i-th entry is the MEAN loss value computed across all iterations in the i-th epoch.

		TODO: Update this method's implementation.
		- Remove mini-batch support. Assume that wts will be updated after every training sample is
			processed (stochastic gradient descent).
		- On each training iteration, get the i-th target one-hot vector and associated context word
		indices. This is your "x" and "y". Do the forward/backward pass and wt update like usual.
		- self.loss_history: Only add loss values at the end of an epoch. Make the loss value that you
		add be the MEAN loss value across all iterations in one epoch.
		- Remove support for accuracy/validation checking. This isn't needed for basic Skip-gram.
		'''

		iter_per_epoch = len(targets_train)
		n_iter = n_epochs * iter_per_epoch
		sum_of_loss_per_epoch = 0

		print(f'Starting to train ({n_epochs} epochs)...')

		# FILL IN CODE HERE
		for j in range(1, n_epochs+1):
			for i in range(iter_per_epoch):
				xi = targets_train[i]
				yi = contexts_train[i]

				# compute net act for each layer and pass it forward
				inputs = targets_train[i].copy()
				for layer in self.layers:
					inputs = layer.forward(inputs)

				# compute loss of iteration and add it to the sum of losses
				loss = self.layers[-1].loss(contexts_train[i])
				sum_of_loss_per_epoch += loss

				# use loss to call backward passing in the d_upstream
				d_upstream = loss
				for layer in reversed(self.layers):
					#return dprev_net_act, self.d_wts, self.d_b
					d_upstream = layer.backward(d_upstream, contexts_train[i])[0]

				# update the weights of the layers
				for layer in self.layers:
					layer.update_weights()

			# Put this in your traing loop
			if (j) % print_every == 0:
				print(f'Finished epoch {j}/{n_epochs}. Epoch Loss: {loss/iter_per_epoch:.3f}')
				print(self.loss_history[-1])
			# at the end of an epoch save avg loss and reset sum
			# save average loss of the epoch
			self.loss_history.append(sum_of_loss_per_epoch/iter_per_epoch)
			# reset loss of epoch
			sum_of_loss_per_epoch = 0


		return self.loss_history

	def get_word_vector(self, word2ind, word):
		'''Get a single word embedding vector from a trained network

		   Parameters:
		   -----------
		   word2ind: Dictionary. Maps word strings -> int code indices in vocab
		   word: Word for which we want to return its word embedding vector

		   Returns:
		   -----------
		   ndarray. Word embedding vector. shape=(embedding_sz,)
		   This is the wt vector from the 1st net layer at the index specified by the word's int-code.
		'''
		if word not in word2ind:
			raise ValueError(f'{word} not in word dictionary!')
		return self.layers[1].wts[:,word2ind[word]].T

	def get_all_word_vectors(self, word2ind, wordList):
		'''Get all word embedding vectors for the list of words `wordList` from the trained network

		   Parameters:
		   -----------
		   word2ind: Dictionary. Maps word strings -> int code indices in vocab
		   wordList: List of strings. Words for which we want to return their word embedding vectors

		   Returns:
		   -----------
		   ndarray. Word embedding vectors. shape=(len(wordList), embedding_sz)
			This is the wt vectors from the 1st net layer at the index specified by each word's int-code.
		'''
		return self.layers[1].wts.T
