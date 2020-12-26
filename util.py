from numpy import dot, zeros
from numpy.linalg import norm 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle as pkl

class Word2Vec:

	def __init__(self,fpath):
		model = self.load(fpath)
		self.wL = model['wL']
		self.wR = model['wR']
		self.vocab = model['vocab']
		self.metadata = model['metadata']
		self.KeyedVectors = self.setKeyedVectors()

	def setKeyedVectors(self):
		KeyedVectors = dict()
		for i,w in enumerate(self.vocab):
			KeyedVectors[w] = self.wL[i,:]
		return KeyedVectors

	def load(self,fpath):
		'''
			load and return .vec file (from word2vec.py)
		'''
		model = None
		with open(fpath,'rb') as infile:
			model = pkl.load(infile)
		return model

	def similarity(self,a,b):
		'''
			return cosine similarity b/w scaled vec of 'a' & 'b'
		'''
		if a not in self.vocab:
			print('Key-error:',"'"+str(a)+"'",'not in vocab')
			return None
		if b not in self.vocab:
			print('Key-error:',"'"+str(b)+"'",'not in vocab')
			return None
		vA,vB = self.KeyedVectors[a], self.KeyedVectors[b]
		d,nA,nB = dot(vA,vB),max(1e-5,norm(vA)),max(1e-5,norm(vB))
		return abs(d/(nA*nB))

	def mostSimilar(self,a,n=10):
		'''
			return list(w,sim) of top n words similar to 'a'
		'''
		if a not in self.vocab:
			print('Key-error:',"'"+str(a)+"'",'not in vocab')
			return None
		vA = self.KeyedVectors[a]
		simScore = dict()
		for w in self.vocab:
			simScore[w] = self.similarity(a,w)
		simScore = sorted(simScore.items(), key=lambda item: item[1],reverse=True)
		return simScore[:n]

	def analogy(self,a,b,c,n=10):
		if a not in self.vocab:
			print('Key-error:',"'"+str(a)+"'",'not in vocab')
			return None
		if b not in self.vocab:
			print('Key-error:',"'"+str(b)+"'",'not in vocab')
			return None
		if c not in self.vocab:
			print('Key-error:',"'"+str(c)+"'",'not in vocab')
			return None
		vA,vB,vC = self.KeyedVectors[a], self.KeyedVectors[b], self.KeyedVectors[c]
		vA,vB,vC = norm(vA), norm(vB), norm(vC)
		vD = vB - vA + vC 	# target
		simScore = dict()
		for w in self.vocab:
			simScore[w] = self.similarity(a,w)
		simScore = sorted(simScore.items(), key=lambda item: item[1],reverse=True)
		return simScore[:n]

	def plotMostSimilar(self,a,n=10,outPath=None):
	    labels = []
	    tokens = []

	    wordList = [w for w,sim in self.mostSimilar(a,n)]

	    for word in wordList:
	        tokens.append(self.KeyedVectors[word])
	        labels.append(word)
	    
	    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
	    new_values = tsne_model.fit_transform(tokens)

	    x = []
	    y = []
	    for value in new_values:
	        x.append(value[0])
	        y.append(value[1])
	        
	    plt.figure(figsize=(16, 16)) 
	    for i in range(len(x)):
	        plt.scatter(x[i],y[i])
	        plt.annotate(labels[i],
	                     xy=(x[i], y[i]),
	                     xytext=(5, 2),
	                     textcoords='offset points',
	                     ha='right',
	                     va='bottom')
	    if outPath:
	    	plt.savefig(outPath)
	    else:
		    plt.show()




