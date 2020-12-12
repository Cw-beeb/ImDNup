from __future__ import print_function
from Bio import SeqIO
import numpy as np
from glove import Glove
from glove import Corpus
import pickle
from keras.preprocessing.text import Tokenizer

import argparse
import pprint
import gensim
from glove import Glove
from glove import Corpus


path_1_H = "E:/he_xiao_ti_ding_wei/Datasets/Setting1/Homo_Sapiens/nucleosomes_vs_linkers_sapiens.fas"
path_1_D = "E:/he_xiao_ti_ding_wei/Datasets/Setting1/Drosophila/nucleosomes_vs_linkers_melanogaster.fas"
path_1_E = "E:/he_xiao_ti_ding_wei/Datasets/Setting1/Elegans/nucleosomes_vs_linkers_elegans.fas"
path_1_Y = "E:/he_xiao_ti_ding_wei/Datasets/Setting1/Yeast/nucleosomes_vs_linkers_yeast.fas"
##############################################################################################
path_2_D_5U = "E:/he_xiao_ti_ding_wei/Datasets/Setting2/Drosophila/5UTR/nucleosomes_vs_linkers_drosophila_5u.fas"
path_2_D_L = r"E:\he_xiao_ti_ding_wei\Datasets\Setting2\Drosophila\Longest_chr\nucleosomes_vs_linkers_drosophila_lc.fas"
path_2_D_P = r"E:\he_xiao_ti_ding_wei\Datasets\Setting2\Drosophila\Promoter\nucleosomes_vs_linkers_drosophila_prom.fas"
path_2_H_5U = "E:/he_xiao_ti_ding_wei/Datasets/Setting2/Homo_Sapiens/5UTR/nucleosomes_vs_linkers_human_5u.fas"
path_2_H_L = r"E:\he_xiao_ti_ding_wei\Datasets\Setting2\Homo_Sapiens\Longest_chr\nucleosomes_vs_linkers_human_lc.fas"
path_2_H_P = r"E:\he_xiao_ti_ding_wei\Datasets\Setting2\Homo_Sapiens\Promoter\nucleosomes_vs_linkers_human_prom.fas"
path_2_Y_P = r"E:\he_xiao_ti_ding_wei\Datasets\Setting2\Yeast\Promoter\nucleosomes_vs_linkers_yeast_prom.fas"
path_2_Y_W = r"E:\he_xiao_ti_ding_wei\Datasets\Setting2\Yeast\Whole_genome\nucleosomes_vs_linkers_yeast_wg.fas"
#######################################################
#阅读fasta格式文件
#######################################################
def readfasta(path,hebin=False):
    fastaSequences = SeqIO.parse(open(path),'fasta')
    #print(fastaSequences)
    nucList = []
    linkList = []
    for fasta in fastaSequences:
        #print(fasta)
        name, sequence = fasta.id, str(fasta.seq)
        if "nucleosomal" in name:
            nucList.append(sequence.upper())
        else:
            linkList.append(sequence.upper())
    print("Nucleosomi: "+str(len(nucList)))
    print("Linker: "+str(len(linkList)))
    #print(len(nucList[4]))
    if hebin == False:
        return nucList,linkList
    else:
        return nucList+linkList

#print(nucList)
# for line in nucList:
#     for line1 in line:
#         if (line1 not in ['A','T','G','C']):
#             print(0)
##################################################################
#seq2ngram：将序列变成K-mer
################################################################
def seq2ngram(List_in, k, s):

    List_out = []
    print ('need to n-gram %d lines' % len(List_in))
    #print(List_in)
    for num, line in enumerate(List_in):
        List_out1 = []
        if num % 500 == 0:
            print ('%d lines to n-grams' % num)

        l = len(line) # length of line

        for i in range(0,l,s):
            if i+k >= l+1:
                break

            List_out1.append(line[i:i+k])

        List_out.append(List_out1)

    return List_out

def concentrate(k,s):
    out_all = []
    for i in [path_1_H,path_1_D,path_1_E,path_1_Y,path_2_D_5U,path_2_D_L,path_2_D_P,path_2_H_5U,path_2_H_L,path_2_H_P,path_2_Y_P,path_2_Y_W]:

        out_all_tem = seq2ngram(readfasta(i,hebin=True), k, s)

        out_all = out_all + out_all_tem
    return out_all
###############################################################
#组建词库
###############################################################

def make_kmer_list(k, alphabet):
    # Base case.
    if (k == 1):
        return (alphabet)

    # Handle k=0 from user.
    if (k == 0):
        return ([])

    # Error case.
    if (k < 1):
        sys.stderr.write("Invalid k=%d" % k)
        sys.exit(1)

    # Precompute alphabet length for speed.
    alphabet_length = len(alphabet)

    # Recursive call.
    return_value = []
    for kmer in make_kmer_list(k - 1, alphabet):
        for i_letter in range(0, alphabet_length):
            return_value.append(kmer + alphabet[i_letter])

    return (return_value)
###################################################################
#局部词典
###################################################################
def ju_bu_ci_dian(path,k,s):
    nuc, link = readfasta(path, hebin=False)
    nuc = seq2ngram(nuc, k, s)
    link = seq2ngram(link, k, s)
    all1 = nuc+link
    ju_bu_list = []
    for line1 in all1:
        for line2 in line1:
            ju_bu_list.append(line2)
    list_all = set(ju_bu_list)
    list_all = list(list_all)
    return list_all
# a = ju_bu_ci_dian(path_1_H,3,2)
# print(a)
# print(len(a))
#################################################################
#kmer局部词典嵌入
##################################################################
def kmer2number_1(path,k,s):
    nuc, link = readfasta(path, hebin=False)
    nuc = seq2ngram(nuc, k, s)
    link = seq2ngram(link, k, s)
    voke = ju_bu_ci_dian(path,k,s)

    nuc_number = []
    for line in nuc:
        nuc_tem = []
        for line1 in line:
            nuc_tem.append(voke.index(line1))
        nuc_number.append(nuc_tem)

    link_number = []
    for line in link:
        link_tem = []
        for line1 in line:
            link_tem.append(voke.index(line1))
        link_number.append(link_tem)
    return nuc_number, link_number


##################################################################
#kmer编码，然后做词袋，丧失了时间顺序关系。
##################################################################

def fa2matrix(k,alphabet,path):
    fastaSequences = SeqIO.parse(open(path),'fasta')
    #print(fastaSequences)
    nucList = []
    linkList = []
    mx = make_kmer_list(k, alphabet) #创建词库
    print(len(mx))
    for fasta in fastaSequences:
        sequence1 = []
        #print(fasta)
        name, sequence = fasta, str(fasta.seq)
        if "nucleosomal" in name:
            for line in mx:
                sequence1.append(sequence.count(line))
            nucList.append(sequence1)
        else:
            for line in mx:
                sequence1.append(sequence.count(line))
            linkList.append(sequence1)

    print("Nucleosomi: "+str(len(nucList)))
    print("Linker: "+str(len(linkList)))
    #print(len(nucList[4]))
    return nucList,linkList,len(mx)
#######################################################################
#将序列进行kmer编码，再转化为整数，这样就可以进行word2vec
#######################################################################
def kmer2number(path,k,s,alphaber):
    nuc, link = readfasta(path, hebin=False)
    nuc = seq2ngram(nuc, k, s)
    link = seq2ngram(link, k, s)
    voke = make_kmer_list(k, alphaber)

    nuc_number = []
    for line in nuc:
        nuc_tem = []
        for line1 in line:
            nuc_tem.append(voke.index(line1))
        nuc_number.append(nuc_tem)

    link_number = []
    for line in link:
        link_tem = []
        for line1 in line:
            link_tem.append(voke.index(line1))
        link_number.append(link_tem)
    return nuc_number,link_number
##############################################################
#one-hot模型
##############################################################
def hot_encode(path):
    nuclist = []
    linklist = []
    dict_nuc = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3
    }
    nuc,link = readfasta(path, hebin=False)
    for sequence in nuc:

        seq_encoded = np.zeros((len(sequence),4))

        i = 0
        for l in sequence:
            if(l.upper() in dict_nuc.keys()):
                seq_encoded[i][dict_nuc[l.upper()]] = 1
                i = i+1
            else:
                return []
        nuclist.append(seq_encoded)

    for sequence in link:

        seq_encoded = np.zeros((len(sequence), 4))

        i = 0
        for l in sequence:
            if (l.upper() in dict_nuc.keys()):
                seq_encoded[i][dict_nuc[l.upper()]] = 1
                i = i + 1
            else:
                return []
        linklist.append(seq_encoded)

    #nuclist = np.array(nuclist)
    #linklist = np.array(linklist)

    return nuclist,linklist
#############################################################
#随机编码
#############################################################
def sui_ji_shu(path):
    nuc,link = readfasta(path, hebin=False)
    dic_1 = {"A": 2,
            "C": 4,
            "G": 7,
            "T": 9}
    nuclist = []
    linklist = []
    for line1 in nuc:
        nuc_tem = []
        for line2 in line1:
            if (line2.upper() in dic_1.keys()):
                nuc_tem.append(dic_1[line2])
            else:
                return []
        nuclist.append(nuc_tem)

    for line1 in link:
        link_tem = []
        for line2 in line1:
            if (line2.upper() in dic_1.keys()):
                link_tem.append(dic_1[line2])
            else:
                return []
        linklist.append(link_tem)
    return nuclist,linklist



# a ,b= sui_ji_shu(path_1_H)
# print(a)
# a = np.array(a)
# print(a.shape)


##############################################################
#划分datasets和lables
##############################################################
def datasets_labels(a,b):
    a = np.array(a,dtype=np.float32)
    b = np.array(b,dtype=np.float32)
    labels = np.concatenate(
        (np.ones((len(a), 1), dtype=np.float32), np.zeros((len(b), 1), dtype=np.float32)),
        axis=0)
    dataset = np.concatenate((a, b), 0)
    input_length = len(dataset[0])
    return dataset,labels,input_length





















#print(make_kmer_list(5, 'ATCG'))


#print(len(concentrate()[5]))
#print(concentrate())

#print(np.array(concentrate()).shape)

##################################################################################
# corpus_model = Corpus()
# corpus_model.fit(concentrate(7,1), window=50)
# corpus_model.save('corpus7_1.model')
#
#
#
#
# corpus_model = Corpus.load('corpus7_1.model')
# glove = Glove(no_components=6, learning_rate=0.05)
# glove.fit(corpus_model.matrix, epochs=88,
#            no_threads=1, verbose=True)
# glove.add_dictionary(corpus_model.dictionary)
# #
# glove.save('glove7_1.model')
# glove = Glove.load('glove7_1.model')
# # print(glove.word_vectors.shape)
# print(glove)
#
# print(glove.word_vectors[glove.dictionary['GGGGG']])
##################################################################################


