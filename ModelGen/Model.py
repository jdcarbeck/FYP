import gensim
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from gensim.models import TfidfModel

from pprint import pprint

import pyLDAvis.gensim

class Model:
    def __init__(self, texts, topics=10):
        self.texts = texts
        self.dct = corpora.Dictionary(texts)
        self.boc = [self.dct.doc2bow(text) for text in texts]
        self.gen_model(topics)
        
        # tfidf = TfidfModel(corpus)
        # self.boc = []
        # for doc in corpus:
        #     self.boc.append(tfidf[doc])
        # print(len(self.boc))

    def topic_dist(self, unseen_text, show=False):
        unseen_corp = self.dct.doc2bow(unseen_text)
        if show:
            for index, score in sorted(self.lda_model[unseen_corp][0], key=lambda tup: -1*tup[1]):
                print("TopicNum: {}\tScore: {}\t Topic: {}".format(index, score, self.lda_model.print_topic(index, 5)))
        return self.lda_model[unseen_corp]

    def gen_model(self, topics):
        print('Creating Model with {} topics...\n'.format(topics))
        # mallet_path = './mallet-2.0.8/bin/mallet'
        # self.lda_model = gensim.models.wrappers.LdaMallet(mallet_path=mallet_path,corpus=self.boc,num_topics=topics,id2word=self.dct)
        self.lda_model = LdaModel(corpus=self.boc, 
                                  id2word=self.dct,
                                  num_topics=topics,
                                  random_state=100,
                                  update_every=1,
                                  passes=3,
                                  alpha='auto'
                                )
    
    def compute_coherence_values(self, limit, start=2, step=3):
        coherence_values = {}
        for number_of_topics in range(start, limit, step):
            self.gen_model(number_of_topics)
            coherence_values[str(number_of_topics)] = self.coherence_val()
        return coherence_values

    def coherence_val(self):
        model = CoherenceModel(model=self.lda_model, texts=self.texts, dictionary=self.dct, coherence='c_v')
        coherence_val = model.get_coherence()
        return coherence_val

    def print_model(self):
        model = self.lda_model
        pprint(model.print_topics())

    def show_model(self):
        vis = pyLDAvis.gensim.prepare(self.lda_model, self.boc, self.dct)
        pyLDAvis.show(vis)

def loadModel(path):
    pass