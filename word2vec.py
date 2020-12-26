import argparse
from os import path
import pickle as pkl
from tqdm import tqdm
from random import shuffle
import numpy as np
from numpy import exp, sum as nsum, matmul, zeros, empty, sqrt
from numpy import outer, log, nan, dot, concatenate as concat, ones
from numpy.random import randint, choice, uniform, seed
from numpy.linalg import norm

def subSample(sentences,minCount=3,t=0.0001):	

	# set dropout probability
	freq, tot = dict(), 0
	for s in sentences:
		for w in s:
			if w not in freq:
				freq[w] = 1
			else:
				freq[w] +=1
			tot+=1

	relfreq = {w:freq[w]/tot for w in freq}
	if t>0:
		Pd = {w:max(1-sqrt(t/relfreq[w]),0) for w in relfreq}
	else:
		pass

	# prune sentences
	subSampledSents, subTot = list(), 0
	for s in tqdm(sentences):
		tempS = list()

		if t>0:	# valid subsampling
			for w in s:
				if freq[w] > minCount and choice([True,False],1,p=[1-Pd[w],Pd[w]]):	
					tempS.append(w)
		else:	# subsampling turned OFF	# only minFreq
			for w in s:
				if freq[w] > minCount:
					tempS.append(w)

		if len(tempS) > 2:
			subSampledSents.append(tempS)
			subTot += len(tempS)
	
	print('>>> Initial: sentences='+str(len(sentences))+' words='+str(tot))
	print('>>> Final: sentences='+str(len(subSampledSents))+' words='+str(subTot))
	return subSampledSents,freq

def numericalize(sentences,vocab):
	wordKeys = {w:i for i,w in enumerate(vocab)}
	indKeys = {i:w for i,w in enumerate(vocab)}
	numS = list()
	for s in tqdm(sentences):
		numS.append([wordKeys[w] for w in s])
	return numS,wordKeys,indKeys

def sigmoid(x):
	return 1/(1+exp(-x))

def clog(x):	# custom log
	return log(x+1e-15)

def validDim(wL,wR,size):
	if wL.shape[1] != size:
		return False,'>>> error: wL dim shape mismatch'
		exit()
	elif wR.shape[0] != size:
		return False,'>>> error: wR dim shape mismatch'
		exit()
	elif wL.shape[0] != wR.shape[1]:
		return False,'>>> error: w:L-R vocab shape mismatch'
		exit()
	else:
		return True,''

def configVocab(vocab):
	vocab = {w: f for w, f in sorted(vocab.items(), key=lambda item: item[1],reverse=True)}
	return vocab

def updateLr(res,lossList):
	if not res.sg and not res.ns:	# cbow
		res.lr *= 0.95
	if res.sg and res.ns:	# sgns
		res.lr *= 0.66
	return True,res
	# if len(lossList)<2:
	# 	return True, res
	# better = lossList[-1] <= lossList[-2]
	# if res.sg and res.ns:
	# 	res.lr *= 0.66
	# else:
	# 	if better:
	# 		res.lr *= 0.95
	# 	else:
	# 		res.lr *= 0.5
	# print('>>> learning rate adjusted to:',res.lr)
	# return better, res

def saveModel(wL,wR,vocab,epoch,res,loss,lossList,init=False):

	if init:	# save init model (before training)
		model = {'wL':wL,'wR':wR,'vocab':vocab}
		finalOutPath = res.outPath+'_init.vec'
		with open(finalOutPath,'wb') as outfile:
		    pkl.dump(model,outfile)
		print(">>> _init_ model saved as: '"+finalOutPath+"'")
		return None

	isBest = lossList[-1] <= min(lossList)	# is best model
	if res.sl:	# to save loss
		fpath = res.outPath+'.loss.pkl' 
		with open(fpath,'wb') as outfile:
			pkl.dump(lossList,outfile)
		print('>>> loss saved in '+fpath)

	metadata = {'size':res.size,
				'window':res.window,
				'epochs':str(epoch+1)+'/'+str(res.epochs),
				'lr':res.lr,
				'train_method':'skip-gram' if res.sg else 'cbow',
				'optim_method':'neg-sampl' if res.ns else 'full-sftmx',
				'pkl_count': None if res.inPath else res.pkl_count,
				'inpFile': None if not res.inPath else res.inPath,
				'outFile': res.outPath,
				'best':isBest}
	model = {'wL':wL,'wR':wR,'vocab':vocab,'metadata':metadata}

	# save model
	#-------------------------
	if res.sb and isBest:	# save if best model yet
		finalOutPath = res.outPath+'_best.vec'
		with open(finalOutPath,'wb') as outfile:
		    pkl.dump(model,outfile)
		print(">>> (sota) model saved as: '"+finalOutPath+"'")

	if (res.fr and epoch%res.fr==res.fr-1) or epoch==res.epochs-1:	# periodic or last epoch
		dlen = len(str(res.epochs))
		finalOutPath = res.outPath+'_e'+str(epoch+1).zfill(dlen)+'.vec'
		with open(finalOutPath,'wb') as outfile:
		    pkl.dump(model,outfile)
		print(">>> model saved as: '"+finalOutPath+"'")

def setWordProb(vocab,wordKeys):
	wordProb = zeros(len(vocab))
	for w in vocab:
		wordProb[wordKeys[w]] = vocab[w]**(3/4)
	wordProb/=nsum(wordProb)
	return wordProb

def cbow(sentences,vocab,wordProb,res,wL,wR):

	valid,mess = validDim(wL,wR,res.size)
	if not valid:
		print(mess)
		return wL, wR

	indKeys = {ind:w for ind,w in enumerate(vocab.keys())}
	wordKeys = {w:ind for ind,w in enumerate(vocab.keys())}

	vlen, nlen = wL.shape[0], wL.shape[1]	# V, N
	if not res.ns:	# full softmax
		loss = 0	# loss per epoch
		lossList = list()	# stores loss per epoch
		for epoch in range(res.epochs):
			print(">>> epoch",str(epoch+1)+'/'+str(res.epochs),": started")
			for sent in tqdm(sentences):
				slen = len(sent)
				for i in range(slen):

					inpContext, tarWord = sent[max(0,i-res.window):min(slen,i+res.window)], sent[i]
					c = len(inpContext)

					# Forward  
					# h = (1/c)*nsum(wL[inpContext,:],axis=0)
					h = zeros(nlen)
					for w in inpContext:
						h += wL[w,:]
					h /= c
					U = dot(h.T,wR).T	# Vx1
					eU = exp(U)
					seU = nsum(eU)
					y = eU/seU 	# softmax

					# Score
					e = y
					# e = y.copy()
					e[tarWord]-=1	# e[j] = y[j]-t[j]

					# Backprop
					dl_dwR = outer(h,e)	# NxV
					dl_dwL = dot(wR,e.T)	# Nx1

					for w in inpContext:
						# wL[w] -= (res.lr/c) * dl_dwL
						wL[w] -= (res.lr) * dl_dwL

					# wR[:,tarWord] += res.lr * dl_dwR[:,tarWord]
					wR -= res.lr * dl_dwR
					# wR[:,tarWord] -= 0.1 * dl_dwR[:,tarWord]

					E = -U[tarWord] + clog(seU)
					loss += E

					# camera = False
					# if indKeys[tarWord] == 'camera':
					# 	camera = True
					# 	arr = y.copy()
					# 	predWords = arr.argsort()[-5:][::-1]
					# 	print('tarWord: camera')
					# 	print('context:',[indKeys[w] for w in inpContext])
					# 	print('predWords:',[indKeys[w] for w in predWords])
					# 	camV = wL[wordKeys['camera']]
					# 	lensV = wL[wordKeys['lens']]
					# 	camV,lensV = camV/norm(camV),lensV/norm(lensV)
					# 	print('cam-lens similarity:',dot(camV,lensV))
					# 	print('==================================================================')

			print('>>> epoch loss:',loss/len(sentences))
			lossList.append(loss/len(sentences))
			# save model
			saveModel(wL,wR,vocab,epoch,res,loss,lossList)
			better,res = updateLr(res,lossList)
			loss = 0	# reset

		return wL,wR

	else:	# negative sampling
		pass

def skip_gram(sentences,vocab,wordProb,res,wL,wR):

	valid,mess = validDim(wL,wR,res.size)
	if not valid:
		print(mess)
		return wL, wR

	vlen, nlen = wL.shape[0], wL.shape[1]	# V, N

	if not res.ns:	# full softmax
		loss = 0	# loss per epoch
		lossList = list()	# stores loss per epoch
		for epoch in range(res.epochs):
			wL_old, wR_old = wL.copy(), wR.copy()
			print(">>> epoch",str(epoch+1)+'/'+str(res.epochs),": started")
			for sent in tqdm(sentences):
				slen = len(sent)
				for i in range(slen):
					inpWord, tarContext = sent[i], sent[max(0,i-res.window):min(slen,i+res.window)]
					
					# Forward
					h = wL[inpWord,:]	# Nx1
					U = dot(h,wR)	# Vx1
					# print(U)
					eU = exp(U)
					# print(eU)
					seU = nsum(eU)
					if seU == 0:
						print('arre!')
						continue
					# print(seU)
					y = eU/seU	# softmax

					t = zeros(vlen); # target	# Vx1
					for w in tarContext:
						t[w]+=1
					e = len(tarContext)*y - t 	# Vx1

					dl_dwR = outer(h,e)	# NxV
					dl_dwL = dot(wR,e.T)	# 1x1

					wL[inpWord,:] -= res.lr*dl_dwL
					wR -= res.lr*dl_dwR

					E = -nsum([U[w] for w in tarContext]) + len(tarContext)*clog(seU)
					loss += E

					# # Forward
					# h,t,c = wL[inpWord,:], zeros(vlen), len(tarContext)
					# for i in tarContext:
					# 	t[i] += 1
					# # Score
					# U = matmul(wR.T,h)
					# E = 0
					# for w in tarContext:
					# 	E += -U[w]
					# E += c*clog(nsum(U))	# loss per example
					# loss += E
					# eU = exp(U)
					# EI = c*(eU/nsum(eU))-t
					# # Backprop
					# wL[inpWord] -= res.lr*matmul(wR,EI)
					# wR -= res.lr*outer(h,EI)

			print('>>> epoch loss:',loss/len(sentences))
			lossList.append(loss/len(sentences))
			# save model
			saveModel(wL,wR,vocab,epoch,res,loss,lossList)
			better,res = updateLr(res,lossList)

			loss = 0	# reset

		return wL,wR

	else:	# negative sampling
		loss = 0	# loss per epoch
		lossList = list()	# stores loss per epoch
		for epoch in range(res.epochs):
			print(">>> epoch",str(epoch+1)+'/'+str(res.epochs),": started")
			for sent in tqdm(sentences):
				slen = len(sent)
				for i in range(slen):
					inpWord, tarContext = sent[i], sent[max(0,i-res.window):min(slen,i+res.window)]

					negContext = list()
					temp = choice(vlen,size=res.nc,p=wordProb)	# negative context
					for w in temp:
						if w==inpWord or w in tarContext:	# avoid positive samples
							continue
						else:
							negContext.append(w)	

					totContext = tarContext.copy()	
					totContext.extend(negContext)	# concatenated cummulative context

					# Forward
					h = wL[inpWord,:]	# Nx1	
					U = sigmoid(dot(h,wR[:,totContext]))	# len(totContext)x1

					# Score
					y = concat((ones(len(tarContext)),zeros(len(negContext))))
					e = U - y 	# error

					# Backprop
					dl_dwL = dot(e,wR[:,totContext].T)	# Nx1
					dl_dwR = outer(e,h)	# len(totContext)xN
					wL[inpWord] -= res.lr * dl_dwL
					for i in range(len(totContext)):
						wR[:,totContext[i]] -= res.lr * dl_dwR[i]

					E = nsum(y*clog(U) + (1-y)*clog(1-U))
					loss -= E

			print('>>> epoch loss:',loss/len(sentences))
			lossList.append(loss/len(sentences))
			# save model
			saveModel(wL,wR,vocab,epoch,res,loss,lossList)
			better,res = updateLr(res,lossList)

			loss = 0	# reset

		return wL,wR

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-s','--size', action='store', dest='size', type=int, default=50,
	                    help='dimensionality of embeddings (def=50)')
	parser.add_argument('-w','--window', action='store', dest='window', type=int, default=5,
	                    help='window size for training [w_w] (def=5)')
	parser.add_argument('-e','--epochs', action='store', dest='epochs', type=int, default=1,
	                    help='number of epochs for training (def=1)')
	parser.add_argument('-l','--lrate', action='store', dest='lr', type=float, default=0.01,
	                    help='learning rate for training (def=0.01)')
	parser.add_argument('-sg', action='store_true', dest='sg', default=False,
	                    help='training mode: skip-gram (def=False -> cbow)')
	parser.add_argument('-ns', action='store_true', dest='ns', default=False,
	                    help='optimization mode: neg-sampling (def=False -> full-sftmx)')
	parser.add_argument('-nc','--neg_count', action='store', dest='nc', type=int, default=10,
	                    help='number of neg samples (if neg sampling) (def=10)')
	parser.add_argument('-fr','--frequency', action='store', dest='fr', type=int, default=None,
	                    help='save model after every fr epochs (def=None)')
	parser.add_argument('-i','--input', action='store', dest='inPath', type=str, default=None,
	                    help='input pickle file path (def=None)')
	parser.add_argument('-o','--output', action='store', dest='outPath', type=str, default='w2vModel',
	                    help="output file prefix for model (def='w2vModel')")
	parser.add_argument('-sl', action='store_true', dest='sl', default=False,
	                    help='saves epoch losses in <prefix>.loss.pkl (def=False)')
	parser.add_argument('-sb', action='store_true', dest='sb', default=False,
	                    help='saves best model to <prefix>_best.vec (def=False)')
	parser.add_argument('-sp', action='store_true', dest='sp', default=False,
	                    help='saves initial model to <prefix>_init.vec (def=False)')
	parser.add_argument('--pkl_count', action='store', dest='pkl_count', type=int, default=1,
	                    help='number of sent.pkl files as input (def=1)')
	parser.add_argument('-t','--sub', action='store', dest='subSamp', type=float, default=0.0001,
	                    help='learning rate for training (def=0.0001)')
	parser.add_argument('-mc','--minC', action='store', dest='minCount', type=float, default=3,
	                    help='minimum lexical frequency for pruning (def=3)')

	res = parser.parse_args()

	# seed(0)
	# input raw sentences
	sentences = list()
	if not res.inPath:	# input pkl file not specified	
		if not path.exists('data/sent'):
			print('>>> No sent files found. Run preprocessing.sh first')
			exit()
		for i in range(res.pkl_count):
		    with open('data/sent/sent_'+str(i).zfill(4)+'.pkl','rb') as infile:	# 4 -> 0000 - 9999(1689)
		        sentences += pkl.load(infile)


	else:	# explicit pickle file specified	# eg: 'reviews_Electronics_5_20k.pkl'
		if not path.isfile(res.inPath):
			print(">>> cannot access '"+res.inPath+"': No such file found")
			exit()
		else:
			with open(res.inPath,'rb') as infile:
				sentences = pkl.load(infile)
	print('>>> sentence files loaded')

	# data pruning
	shuffle(sentences)
	sentences,vocab = subSample(sentences,res.minCount,res.subSamp)	# remove uniform or infrequent words
	print('>>> sentences subsampled')

	# model configuration
	if res.fr != None:
		res.fr = int(res.fr)
	if res.fr != None and (res.fr < 1 or res.fr > 1000):
		print(">>> error: -fr must be in range [1-1000] (if active)")
		exit()
	wL, wR = uniform(-1., 1, (len(vocab), res.size)), uniform(-1., 1., (res.size,len(vocab)))
	# with open('wlwr.pkl','wb') as infile:
	# 	pkl.dump((wL,wR),infile)
	vocab = configVocab(vocab)
	print('>>> model configured')

	# numericalizing data
	sentences, wordKeys, indKeys = numericalize(sentences,vocab)	# turn word strings to int
	print('>>> sentences numericalized')

	# setting up word probabilities
	wordProb = setWordProb(vocab,wordKeys)

	if res.sp:
		saveModel(wL,wR,vocab,None,res,0,[],init=True)

	# model training
	print(">>> training started")
	if res.sg:	# skip-gram
		wL, wR = skip_gram(sentences,vocab,wordProb,res,wL,wR)
	else:	# cbow
		wL, wR = cbow(sentences,vocab,wordProb,res,wL,wR)

	print(">>> training complete")

