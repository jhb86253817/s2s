import theano
import numpy
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
from numpy import linalg as LA


def load_sent_vecs(folder):
    with open(folder, 'r') as f:
        sent_vecs = json.loads(f.read())
    return sent_vecs

def load_sents(folder):
    with open(folder, 'r') as f:
        sents = [s.strip() for s in f]
    return sents

def visualize(sent_vecs, sents):
    pca = PCA(n_components=2)
    pca.fit(sent_vecs)
    sent_proj = pca.transform(sent_vecs)
    plt.scatter(sent_proj[:,0], sent_proj[:,1])
    for sent,x,y in zip(sents, sent_proj[:,0], sent_proj[:,1]):
        plt.annotate(sent, xy=(x,y))
    plt.show()


#def visualize(embedding, start_index, length):
#    #x = numpy.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
#    #print x
#    embedding = embedding[start_index:start_index+length,:]
#    pca = PCA(n_components=2)
#    pca.fit(embedding)
#    embedding_proj = pca.transform(embedding)
#    plt.scatter(embedding_proj[:,0], embedding_proj[:,1])
#    labels = xrange(start_index, start_index+length)
#    with open('index2word.json', 'rb') as f:
#        index2word = json.loads(f.read())
#    labels = [index2word[str(label)] for label in labels]
#    for label,x,y in zip(labels, embedding_proj[:,0], embedding_proj[:,1]):
#        plt.annotate(label, xy=(x,y))
#    plt.show()

if __name__ == "__main__":
    sent_vecs = load_sent_vecs('../model/s2s/sent_vecs.json')
    sents = load_sents('arxiv_cs_clean_toy.txt')
    visualize(sent_vecs, sents)
    #norms = [LA.norm(w) for w in embedding]
    #embedding = [w/norm for (w,norm) in zip(embedding,norms)]
    #embedding = numpy.array(embedding)
    #start_index = 0 
    #length = 100
    #visualize(embedding, start_index, length)

