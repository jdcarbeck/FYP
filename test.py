import glob
import json
import random
import os
import shutil
from rouge import Rouge
from pprint import pprint
from progress.bar import ChargingBar


from ModelGen.Corpus import Corpus
from ModelGen.Summary import Summary


def test_single_doc(path, n_files=20):
    test_result_path = "./Test_Results/Single_Doc_Results"
    files = load_n_files(path, n=n_files)
    # files = load_all_files(path)
    
    results = {}

    bar = ChargingBar('Single Doc Testing', max=len(list(files.keys())))
    for file in list(files.keys()):
        model_sum = files[file]['abstract']
        bar.next()
        if len(" ".join(model_sum)) > 100:
            system_sum = single_doc_summary(files[file]['article'], len(" ".join(model_sum)))
            
            base = os.path.basename(file)
            file_id = os.path.splitext(base)[0]
            
            system_sum_name = 'Result.' + file_id + ".txt"
            model_sum_name = 'Result.A.'+ file_id + ".txt"

            rouge = Rouge()
            scores = rouge.get_scores(" ".join(system_sum), " ".join(model_sum))
            
            results[file_id] = scores
            # keys_1 = ['rouge_1', 'rouge_2', 'rouge_l']
            # keys_2 = ['f','p','r']
            # for i, obj in enumerate(scores):
            #     for j in keys_2:
            #         print(obj[i])
            #         results[keys_1[i]][j] += obj[i]

            system_sum_path = os.path.join(test_result_path, system_sum_name)
            model_sum_path = os.path.join(test_result_path, model_sum_name)

            # save model summary
            with open(model_sum_path, 'w') as file:
                for sent in model_sum:
                    file.write(sent + "\n")
            
            with open(system_sum_path, 'w') as file:
                for sent in system_sum:
                    file.write(sent + "\n")
    bar.finish()

    with open(os.path.join(test_result_path, "Results.json"), "w") as result_file:
        json.dump(results, result_file)
            

def single_doc_summary(text_arry: [str], length):
    corpus = Corpus(text_arry)
    document = list(corpus.sen2con.keys())
    summary = Summary(document, corpus,)
    return summary.doc_summary(sen_len=length,alpha=0.5)


def load_n_files(path, n=20):
    files = load_all_files(path)
    files_list = list(files.keys())
    chosen_files = []
    for i in range(0, n):
        choice = random.choice(files_list)
        while choice in chosen_files:
            choice = random.choice(files_list)
        chosen_files.append(choice)
    
    n_files = {}
    for file in chosen_files:
        n_files[file] = files[file]
    return n_files

def load_all_files(path):
    data_obj = {}
    file_names = glob.glob(path + "/*.json")
    bar = ChargingBar('Loading Files', max=len(file_names))
    for name in file_names:
        bar.next()
        with open(name) as json_file:
            data = json.load(json_file)
            data_obj[name] = data
    bar.finish()
    return data_obj

def test_multi_doc(path, test_result_path, n_files=20):
    src, tgt = load_multinews(path, n=n_files)
    results = {}
    n = len(src)
    test_data = zip(src[:n], tgt[:n])
    bar = ChargingBar('Multi Doc Testing', max=n)
    for i, (src, tgt) in enumerate(test_data):
        model_sum = tgt
        system_sum = multi_doc_summary(src.split('.'), length=len(model_sum))
        
        if (system_sum != []) and (model_sum != []):
            rouge = Rouge()
            scores = rouge.get_scores(" ".join(system_sum), model_sum)
            
            results[i] = scores

            system_sum_name = "Result." + str(i) + ".txt" 
            model_sum_name = "Result.A." + str(i) + ".txt"
            system_sum_path = os.path.join(test_result_path, system_sum_name)
            model_sum_path = os.path.join(test_result_path, model_sum_name)
            
            with open(system_sum_path, "w") as system:
                system.write(" ".join(system_sum))

            with open(model_sum_path, "w") as model:
                model.write(model_sum)
        bar.next()
    
    bar.finish()

    with open(os.path.join(test_result_path, "Results.json"), "w") as result_file:
        json.dump(results, result_file)

def multi_doc_summary(src_doc: [str],length):
    corpus = Corpus(src_doc)
    document = list(corpus.sen2con.keys())
    summary = Summary(document, corpus)
    return summary.doc_summary(sen_len=length,alpha=0.5)


def load_multinews(path, n=20):
    system_path = os.path.join(path, "test.txt.src")
    model_path = os.path.join(path, "test.txt.tgt")
    
    with open(system_path) as src:
        src_content = src.readlines()
    src_content = [x.strip() for x in src_content] 

    with open(model_path) as tgt:
        tgt_content = tgt.readlines()
    tgt_content = [x.strip() for x in tgt_content]
    
    return src_content[:n], tgt_content[:n]


def avg_rouge(path):
    n = 0
    average = {
        'rouge-1': {"f": 0.0, "p": 0.0, "r": 0.0},
        'rouge-2': {"f": 0.0, "p": 0.0, "r": 0.0},
        'rouge-l': {"f": 0.0, "p": 0.0, "r": 0.0},
    }
    file = os.path.join(path, "Results.json")
    with open(file) as results:
        data = json.load(results)
        for _, score in data.items():
            rouge_scores = score[0]
            n += 1

            for key in rouge_scores:
                for val_key in rouge_scores[key]:
                    average[key][val_key] += rouge_scores[key][val_key]

        for key in average:
            for val_key in average[key]:
                average[key][val_key] = (average[key][val_key]/len(data.keys()))

    pprint(average)



def run(test_single=False, test_multi=False, avg=False):
    singl_doc_test_data = "./Test_Data/CNN_DM"
    multi_doc_test_data = "./Test_Data/MultiNews"
    single_doc_results = "./Test_Results/Single_Doc_Results"
    multi_doc_results = "./Test_Results/Multi_Doc_Results"

    print("Running Rogue Tests")

    if test_single:
        # Remove exsiting files in the single doc results 
        files = glob.glob(single_doc_results + "/*.txt")
        for f in files:
            os.remove(f)
        test_single_doc(singl_doc_test_data, n_files=100)

    if test_multi:
        # Remove exsiting files in the multi doc results
        files = glob.glob(multi_doc_results + "/*.txt")
        for f in files:
            os.remove(f)
        test_multi_doc(multi_doc_test_data, multi_doc_results, n_files=100)

    if avg:
        single_results = avg_rouge(single_doc_results)
        multi_resutlts = avg_rouge(multi_doc_results)

run(test_single=False, test_multi=False, avg=True)
