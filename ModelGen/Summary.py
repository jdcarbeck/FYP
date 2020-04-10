import pulp as p
from .Corpus import Corpus
import math
import pprint as pp
from numpy import dot
from numpy.linalg import norm
import re
import threading
from multiprocessing import Process
from multiprocessing import Manager

class Summary:
    def __init__(self, doc, corpus: Corpus, min_len=20):
        self.corpus = corpus
        sentences = [sent for sent in doc if len(sent) > min_len]
        self.doc = list(dict.fromkeys(sentences))
        self.__sent_term_weighting()

        sentences_pairs = []
        for i in range(0, len(sentences)-2):
            for j in range(i+1, len(sentences)-1):
                sentences_pairs.append((sentences[i], sentences[j]))

        # split sentence pairs and then start a process that will
        sentences_chunks = self.chunkIt(sentences_pairs, 8)

        manager = Manager()
        cos_dict = manager.dict()
        ngd_dict = manager.dict()
        p_cos = Process(target=self.cos_sim_model, args=(sentences, self.term_sent_weights, self.con_freq, cos_dict,))

        processes = []
        print("Num of chunks: ", len(sentences_chunks))
        for chunk in sentences_chunks:
            p = Process(target=self.ngd_sim_model, args=(chunk, len(sentences_pairs), self.sent_con_freq, ngd_dict,))
            processes.append(p)
            p.start()

        p_cos.start()

        p_cos.join()
        for p in processes:
            p.join()

        self.cos_sim_dict = cos_dict
        self.ngd_sim_dict = ngd_dict
        print("done")
        
    def chunkIt(self, seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0
        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg
        return out
    
    def cos_sim_model(self, sentences, term_sent_weights, con_freq, return_dict):
        model = {}
        for i in range(0, len(sentences)-2):
            print("done cos: ", i/(len(sentences)-2))
            for j in range(i+1, len(sentences)-1):
                model[sentences[i] + sentences[j]] = self.sen_cos_sim(sentences, i, j, term_sent_weights, con_freq)
        return_dict = model
    
    def ngd_sim_model(self, sentences_pairs, n, sent_con_freq, return_dict):
        i = 1
        for sentence1, sentence2 in sentences_pairs:
            print("done ngd: ", i/len(sentences_pairs))
            return_dict[sentence1 + sentence1] = self.sen_ngd_sim(sentence1, sentence2, n, sent_con_freq)
            i += 1

    def find_cos_sim(self, i,j, sentences):
        return self.cos_sim_dict[sentences[i] + sentences[j]]

    def find_ngd_sim(self, i, j, sentences):
        return self.ngd_sim_dict[sentences[i] + sentences[j]]

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


        con_freq = {}
        for con in sent_con_freq:
            con_freq[con] = len(sent_con_freq[con])
        
        self.sent_con_freq = sent_con_freq
        self.con_freq = con_freq
        self.term_sent_weights = term_sent_weights


    def doc_summary(self, sen_len=300, alpha=0.8):
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
        objective = p.lpSum([ \
            (alpha * (self.find_cos_sim(i, j, sentences) * x_ij[(sentences[i],sentences[j])])) \
            + \
            ((1-alpha) * (self.find_ngd_sim(i, j, sentences) * x_ij[(sentences[i],sentences[j])]))
                for i in range(0, len(sentences)-2) for j in range(i+1, len(sentences)-1)])
        lp_problem += constriant_len
        lp_problem += objective

        # lp_problem.writeLP("CheckLpProgram.lp")
        maxium_summary_sent = []

        lp_problem.solve()


        summary_sent = {}
        variables_dict = {}
        variables_list = lp_problem.variables()
        for v in variables_list:
            variables_dict[v.name] = v.varValue

        # for key in variables_dict.keys():
        #     if variables_dict[key] > 0:
        #         print(key)


        variables_keys = list(variables_dict.keys())

        index = 0
        for i in range(0, len(sentences)-2):
            for j in range(i+1, len(sentences)-1):
                key = variables_keys[index]

                if variables_dict[key] > 0:
                    summary_sent[sentences[i]] = ""
                    summary_sent[sentences[j]] = ""
                index+=1

        return "\n".join(list(summary_sent.keys()))


        # for v in lp_problem.variables():
        #     if v.varValue > 0:
        #         maxium_summary_sent.append(v.name)

        # return self.sentence_extract(maxium_summary_sent)

    def sen_cos_sim(self, sentences, i, j, term_sent_weights, con_freq):
        cos_sim = (self.cos_sim(sentences[i], term_sent_weights, con_freq ) + self.cos_sim(sentences[j], term_sent_weights, con_freq) - self.cos_sim(sentences[i], term_sent_weights, con_freq, item2=sentences[j]))
        return cos_sim

    def sen_ngd_sim(self, sentence1, sentence2, n, sent_con_freq):
        ngd_sim = (self.ngd_sim(sentence1, n, sent_con_freq) + self.ngd_sim(sentence2, n, sent_con_freq) - self.ngd_sim(sentence1, n, sent_con_freq, item2=sentence2))
        return ngd_sim


    def cos_sim(self, item1, term_sent_weights, con_freq, item2=None):
        item2_vec = []
        item1_vec = list((term_sent_weights[item1]).values())
        if item2 is None:
            item2_vec = list(con_freq.values())
        else:
            item2_vec = list((term_sent_weights[item2]).values())
        numerator = dot(item1_vec, item2_vec)
        denominator = (norm(item1_vec)*norm(item2_vec))
        similarity = numerator/denominator
        return similarity

    def ngd_sim(self, item1, n, sent_con_freq, item2=None,):
        # self.corpus.sen2con[sent]
        # slef.corpus.con2sen[con] to get number of sentences that contain that concept
        # intersect the two terms lists to get the number of co-occurances and combine the score
        item2_con = []
        item1_con = self.corpus.sen2con[item1]
        if item2 is None:
            item2_con = list(self.sent_con_freq.keys())
        else:
            item2_con = self.corpus.sen2con[item2]

        ngd_sum = 0
        for term1 in item1_con:
            for term2 in item2_con:
                ngd = self.ngd_term(term1, term2, n, sent_con_freq)
                ngd_sim = math.exp(-ngd)
                ngd_sum += ngd_sim
        
        ngd = (ngd_sum/(len(item1_con) * len(item2_con)))
        return ngd

        # sentences contain termk and setneces that contain both termk and terml

    def ngd_term(self, t1, t2, n, sent_con_freq):
        t1_sen_count = len(sent_con_freq[t1])
        t2_sen_count = len(sent_con_freq[t2])
        t1_t2_sen_count = len([con for con in sent_con_freq[t1] if con in sent_con_freq[t2]])
        t1_log = math.log10(t1_sen_count)
        t2_log = math.log10(t2_sen_count)
        t1_t2_log = 0
        if(t1_t2_sen_count > 0):
            t1_t2_log = math.log10(t1_t2_sen_count)
        numerator = max(t1_log,t2_log) - t1_t2_log
        denominator = math.log10(n) - min(t1_log, t2_log)
        return (numerator/denominator)

    def sentence_extract(self, list_vars):
        sentences = {}
        for var in list_vars:
            found = re.findall('"[^"]+"', var)
            for obj in found:
                obj = re.sub('_', ' ', obj)
                obj = re.sub('\\\\', '', obj)
                sentences[obj] = ""
        sentences_list = list(sentences.keys())
        return " ".join(sentences_list)
