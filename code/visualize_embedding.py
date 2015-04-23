import theano
import numpy
import os
#import matplotlib.pyplot as plt
#from sklearn.decomposition import PCA
import json
from numpy import linalg as LA

def load_embedding(nw, nh, folder):
    wxg = theano.shared(name='wxg',
                       value=0.02 * numpy.random.randn(nw, nh)
                       .astype(theano.config.floatX))

    wxg.set_value(numpy.load(os.path.join(folder, wxg.name+'.npy')))
    show_embedding = theano.function(inputs=[], outputs=[wxg])
    return show_embedding()[0]

def load_sent_vecs(folder):
    with open(folder, 'r') as f:
        sent_vecs = json.loads(f.read())
    print len(sent_vecs)
    print sent_vecs[0]

def visualize(embedding, start_index, length):
    #x = numpy.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
    #print x
    embedding = embedding[start_index:start_index+length,:]
    pca = PCA(n_components=2)
    pca.fit(embedding)
    embedding_proj = pca.transform(embedding)
    plt.scatter(embedding_proj[:,0], embedding_proj[:,1])
    labels = xrange(start_index, start_index+length)
    with open('index2word.json', 'rb') as f:
        index2word = json.loads(f.read())
    labels = [index2word[str(label)] for label in labels]
    for label,x,y in zip(labels, embedding_proj[:,0], embedding_proj[:,1]):
        plt.annotate(label, xy=(x,y))
    plt.show()

if __name__ == "__main__":
    load_sent_vecs('../model/s2s/sent_vecs.json')
    #norms = [LA.norm(w) for w in embedding]
    #embedding = [w/norm for (w,norm) in zip(embedding,norms)]
    #embedding = numpy.array(embedding)
    #start_index = 0 
    #length = 100
    #visualize(embedding, start_index, length)

