import pulp as p
from .Corpus import Corpus
import math
import pprint as pp
from numpy import dot
from numpy.linalg import norm
import re

class Summary:
    def __init__(self, doc, corpus: Corpus, min_len=100):
        self.corpus = corpus
        self.doc = [sent for sent in doc if len(sent) >= min_len]
        self.__sent_term_weighting()

    def __sent_term_weighting(self):
        sent_con_freq = {}
        freq_concepts = {}
        for sent in self.doc:
            freq_concepts[sent] = {}
            concepts_in_sent = self.corpus.sen2con[sent]
            for concept in concepts_in_sent:
                if concept in freq_concepts[sent]:
                    freq_concepts[sent][concept] += 1
                else:
                    freq_concepts[sent][concept] = 1
                if concept in sent_con_freq:
                    if sent not in sent_con_freq[concept]:
                        sent_con_freq[concept].append(sent)
                else:
                    sent_con_freq[concept] = [sent]
        
        term_sent_weights = {}
        for sent in self.doc:
            term_sent_weights[sent] = {}
            for con in sent_con_freq:
                term_freq = 0
                if con in freq_concepts[sent]:
                    term_freq = freq_concepts[sent][con]
                inverse_sent_freq = math.log10(len(freq_concepts.keys())/len(sent_con_freq[con]))

                term_sent_weights[sent][con] = (term_freq * inverse_sent_freq)

        for con in sent_con_freq:
            sent_con_freq[con] = len(sent_con_freq[con])
        
        self.con_freq = sent_con_freq
        self.term_sent_weights = term_sent_weights


    def doc_summary(self, sen_len=10000, alpha=0.5):
        sentences = self.doc

        lp_problem = p.LpProblem('problem', p.LpMaximize)
        x_ij = p.LpVariable.dicts('xij', [(sentences[i],sentences[j]) for i in range(0, len(sentences)-2) for j in range(i+1, len(sentences)-1)], cat='Binary')
        constriant_len = p.lpSum([(len(sentences[i]) + len(sentences[j]))*(x_ij[(sentences[i],sentences[j])]) for i in range(0, len(sentences)-2) for j in range(i+1, len(sentences)-1)]) <= sen_len
        # for i in sentences:
        #     for j in sentences:
        #         lp_problem += len(i) + len(j) <= sen_len
        
        # objective = p.LpAffineExpression( \
        #     (alpha * ((self.cos_sim(document, i) + self.cos_sim(document, j) - self.cos_sim(i, j))*(x_i[i] * x_j[j])) for i in sentences for j in sentences) \
        #         + \
        #     ((1 - alpha) * ((self.ngd_sim(document, i) + self.ngd_sim(document, j) - self.ngd_sim(i, j))*(x_i[i] * x_j[j])) for i in sentences for j in sentences)
        # )
        # objective = p.LpSum(([(self.cos_sim(i) + self.cos_sim(j) - self.cos_sim(i, item2=j))*(x_ij[(i,j)])) for i in sentences for j in sentences)
        objective = p.lpSum([(self.cos_sim(sentences[i]) + self.cos_sim(sentences[j]) - self.cos_sim(sentences[i],item2=sentences[j])) * x_ij[(sentences[i],sentences[j])] for i in range(0, len(sentences)-2) for j in range(i+1, len(sentences)-1)])
        lp_problem += constriant_len
        lp_problem += objective

        # lp_problem.writeLP("CheckLpProgram.lp")
        maxium_summary_sent = []

        lp_problem.solve()
        for v in lp_problem.variables():
            if v.varValue > 0:
                maxium_summary_sent.append(v.name)

        return self.sentence_extract(maxium_summary_sent)

    def cos_sim(self, item1, item2=None):
        item2_vec = []
        item1_vec = list((self.term_sent_weights[item1]).values())
        if item2 is None:
            item2_vec = list(self.con_freq.values())
        else:
            item2_vec = list((self.term_sent_weights[item2]).values())
        numerator = dot(item1_vec, item2_vec)
        denominator = (norm(item1_vec)*norm(item2_vec))
        similarity = numerator/denominator
        return similarity

    def ngd_sim(self, item1, item2=None):
        pass

    def sentence_extract(self, list_vars):
        sentences = []
        for var in list_vars:
            found = re.findall('"[^"]+"', var)
            for obj in found:
                if obj not in sentences:
                    obj = re.sub('_', ' ', obj)
                    obj = re.sub('\\\\', '', obj)
                    sentences.append(obj)
        return sentences
