  #!/usr/bin/env python
import code
import numpy as np
np.random.seed(123)
import os
import pandas
import sqlite3
import srs
import sys
sys.setrecursionlimit(5000)
import json
import itertools
from operator import itemgetter
import random
random.seed(123)
import cPickle as pickle
import hierarchical_model_features_and_language
from hierarchical_model_features_and_language import Vocab, pad_into_variable_tensor, pad_into_matrix, HierarchicalModel
import theano
import theano.tensor as T
from lstm import LSTM
from score_hypos import adjust_f1_score
import gensim

import sklearn.metrics

# read in the inconfig
inconfig, outconfig = sys.argv[1:]
inconfig = os.path.abspath(inconfig)

config = srs.Config(inconfig)
database_dir = config.get('database_dir')

subreddit = config.get('subreddit')

database_filename = os.path.join(database_dir, subreddit + '.db')

features_dir = config.get('features_dir')
features_dir = os.path.join(features_dir, subreddit)

feature_files = config.get('feature_files')
feature_files = feature_files.split(' ')

feature_groups = config.get('feature_groups')
feature_groups = json.loads(feature_groups)

conn = sqlite3.connect(database_filename)
c = conn.cursor()

query = """SELECT id, author, parent_id, link_id, timestamp FROM comments"""

c.execute(query)
data = pandas.DataFrame([row for row in c],
                        columns=['id', 'author', 'parent', 'post', 'time'])
query = """SELECT author_id, author_label, post_id FROM author_labels"""
c.execute(query)
author_labels = pandas.DataFrame([row for row in c],
                        columns=['author', 'score', 'post'])

data = pandas.merge(left=data, right=author_labels, left_on=['post','author'], right_on=['post','author'])

data['post'] = data['post'].apply(lambda x: x[3:])
data['parent'] = data['parent'].apply(lambda x: x[3:])

# labels based on the Hao's 7-label system
labels_file = config.get('labels_file').replace('subreddit', subreddit)
labels = pandas.read_csv(labels_file, sep='\t', usecols=['id', 'label'])

data = pandas.merge(left=data, right=labels, left_on='id', right_on='id')

query = """SELECT id, tokens FROM toktext"""
c.execute(query)
tokens = pandas.DataFrame([row for row in c],
                        columns=['id', 'tokens'])

data = pandas.merge(left=data, right=tokens, left_on='id', right_on='id')

query = """SELECT id, author, created FROM posts"""

c.execute(query)
post_data = pandas.DataFrame([row for row in c],
                        columns=['id', 'author', 'time'])

post_data = pandas.merge(left=post_data, right=tokens, left_on='id', right_on='id')

partition_file = os.path.join(features_dir, config.get('partition_file'))
partition = pandas.read_csv(partition_file, sep='\t')

col_names = partition.columns.tolist()
col_names[0] = 'post'
partition.columns = col_names

data = pandas.merge(left=data, right=partition, left_on='post', right_on='post')

pruned_file = config.get('pruned_file').replace('subreddit', subreddit)
pruned = pandas.read_csv(pruned_file, sep='\t', usecols=['post_id', 'author_id', 'keep'])

col_names = ['post', 'author', 'keep_pruned']
pruned.columns = col_names
pruned['post'] = pruned['post'].apply(lambda x: x[3:])

data = pandas.merge(left=data, right=pruned, how='outer', left_on=['post', 'author'], right_on=['post', 'author'])
# fill [deleted] authors as keep = 0
data = data.fillna(0)

for f in feature_files:
    feat_file = config.get(f)
    feat_file = os.path.join(features_dir, feat_file)
    feat = pandas.read_csv(feat_file, sep='\t')
    
    data = pandas.merge(left=data, right=feat, left_on='id', right_on='id', how='left')

feature_names = []
for g in feature_groups:
    group_names = feature_groups[g]
    group_names = [x.replace('subreddit', subreddit) for x in group_names]
    group_names = [x for x in group_names if x in data.columns.tolist()]
    feature_groups[g] = group_names
    feature_names.extend(group_names)
    
for feat in feature_names:
    data.ix[data[feat].isnull(), feat] = 0

vocab = Vocab()

def flatten(text):
    a = json.loads(text)
    return [w for p in a['tokens'] for s in p for w in s]

def add_vocab(tokens):
    vocab.add_words(tokens)

updated_keep_file = config.get('keep_updated_file').replace('subreddit', subreddit)
if not os.path.isfile(updated_keep_file):
    print 'Calculating updated pruning'
    def keep_parent(groupdb, parent):
    
        if (groupdb.ix[groupdb.id == parent, 'keep'] == 0).any():
            groupdb.loc[groupdb.id == parent, 'keep'] = 1
            grandparent = groupdb.ix[groupdb.id == parent, 'parent'].max()
            groupdb = keep_parent(groupdb, grandparent)
            return
        else:
            return

    data['keep'] = data['keep_pruned']
    for group,comments in data.groupby('post'):
        print group
        comments.sort('time', inplace=True, ascending=False)
        num_changes = len(comments)
        for _,comment in comments.iterrows():
            if comment.keep == 1 and comment.parent.startswith('c'):
                keep_parent(comments, comment.parent)
        data.update(comments)
    # save for next time since calculating takes a lot of time
    data.to_csv(subreddit + '_updated_pruning.csv', sep='\t', columns=['id', 'post','author','keep','keep_pruned'], index=False)

else:
    updated_keep = pandas.read_csv(updated_keep_file, sep='\t', usecols=['id', 'keep'])
    data = pandas.merge(left=data, right=updated_keep, left_on='id', right_on='id')
    
data['tokens'] = data.tokens.apply(flatten)
post_data['tokens'] = post_data.tokens.apply(flatten)

data.set_index('id', drop=False, inplace=True)
traindata = data[data.partition >= 4]
devdata = data[(data.partition < 4) & (data.partition >=2)]

traindata.tokens.apply(add_vocab)
vocab.cut(10)
vocab.add_words(["EMPTY_COMMENT"])

code.interact(local=locals())

def word2index(tokens):
    idx = [vocab.word2index[w] for w in tokens if w in vocab.word2index]
    if len(idx) == 0:
        idx = [vocab.word2index["EMPTY_COMMENT"]]
    if len(idx) > 100:
        idx = idx[:100]
    return idx


traindata['idx'] = traindata.tokens.apply(word2index)
devdata['idx'] = devdata.tokens.apply(word2index)
post_data['idx'] = post_data.tokens.apply(word2index)


def calc_bias(node, children, bias):
    b_h = 0
    b_v = 0
    if node in children:
        for child in children[node][::-1]:
            c_bias = calc_bias(child[1], children, bias)
            b_h += 1 + c_bias[0]
            b_v += c_bias[1]
        bias[node] = (b_h, b_v + 1)
        return (b_h, b_v + 1)
    else:
        bias[node] = (b_h, b_v)
        return (0, 0)
    
def get_threads(datadb, postdb, test=False):
    threads = []
    feats = []
    data_taps = []
    data_children_taps = []
    data_labels = []
    data_ids = []
    max_tree_size = 0
    
    for group,comments in datadb.groupby('post'):
        thread = []
        feat = []
        data_id = []
        # 0 means "None", it will point to default value, 1 - post, 2 etc. - comments
        taps = [[0,0,0,0,0,0]]
        labels = [0]
        post = postdb[postdb.id == group]
        comments = comments.sort('time', inplace=False)
        feat.append([0] * len(feature_names))
        thread.extend(post.idx.tolist())
#        comments_idx = comments.idx.tolist()
#        thread.extend(comments_idx[:min(len(comments_idx), 300)])
        i = 1
        parents = {group: 1}
        parents_ids = [None, group]
        children = {}
        kept_children = {}
        pruned_older_siblings = {}
        for _,comment in comments.iterrows():
            if i > 300:
                break
            if comment.keep == 0:
                if comment.parent in children:
                    children[comment.parent].append((0, comment.id, comment.keep))
                else:
                    children[comment.parent] = [(0, comment.id, comment.keep)]
                if comment.parent in kept_children:
                    older_sib = kept_children[comment.parent][-1][1]
                    if older_sib in pruned_older_siblings:
                        pruned_older_siblings[older_sib].append(comment.id)
                    else:
                        pruned_older_siblings[older_sib] = [comment.id]
                continue
            if comment.keep == 1: 
                data_id.append(comment.id)
                feat.append(comment[feature_names].tolist())
                thread.append(comment.idx)
            tap = [0,0,0,0,0,0]
            # time taps are relative to across the siblings
            if comment.parent in parents:
                tap[0] = parents[comment.parent]
                n_bias = 0
                if comment.parent in children:
                    for c in children[comment.parent][::-1]:
                         if c[-1] == 1:
                            tap[1] = c[0]
                            break
                         else:
                            n_bias += 1
                    children[comment.parent].append((i + 1, comment.id, comment.keep))
                else:
                    children[comment.parent] = [(i + 1, comment.id, comment.keep)]
                if comment.parent in kept_children:
                    kept_children[comment.parent].append((i + 1, comment.id))
                else:
                    kept_children[comment.parent] = [(i + 1, comment.id)]

            tap[2] = n_bias
            taps.append(tap)
            parents[comment.id] = i + 1
            parents_ids.append(comment.id)
            labels.append(comment.label)
            i += 1

        children_taps = []
        siblings = {}
        l = len(taps)

        bias = {}
        _ = calc_bias(comment.post, children, bias)                
            
        for i,tap in enumerate(taps[::-1]):
            c_tap = [0,0,0,0,0,0]
            h_bias = [0,0]
            t_bias = [0,0]
            c_id = parents_ids[l - i]
            if c_id in children:
                for a in children[c_id]:
                    if a[-1] == 1:
                        c_tap[0] = a[0]
                        break
                    else:
                        try:
                            h_bias[0] += bias[a[1]][0]
                            h_bias[1] += bias[a[1]][1]
                        except:
                            pass
            if c_id in pruned_older_siblings:
                for s in pruned_older_siblings[c_id]:
                    t_bias[0] += bias[s][0]
                    t_bias[1] += bias[s][1]
            c_tap[2:4] = h_bias
            c_tap[4:] = t_bias
            if tap[1] != 0:
                siblings[tap[1]] = l - i
            if l - i in siblings:
                c_tap[1] = siblings[l - i]
            children_taps.append(c_tap)
        children_taps = children_taps[::-1]
                    
        feats.append(feat)
        threads.append(thread)
        data_taps.append(taps)
        data_labels.append(labels)
        data_children_taps.append(children_taps)
        data_ids.append(data_id)
    return threads, feats, data_taps, data_children_taps, data_labels, data_ids

print 'Processing training'
train_threads, train_feats, train_taps, train_children, train_labels, train_ids = get_threads(traindata, post_data)

#shuffle_freq = 100
#lengths = np.array([len(x) for x in train_threads])
#ids = range(len(train_threads))
#random.shuffle(ids)
#lengths = lengths[ids]
#pairs = zip(ids, lengths)
#for i in range(len(train_threads) / shuffle_freq):
#    ids[i*shuffle_freq:(i+1)*shuffle_freq] = zip(*sorted(pairs[i*shuffle_freq:(i+1)*shuffle_freq],
#                                                    key=lambda x: x[1]))[0]
#if len(train_threads) % shuffle_freq != 0:
#    ids[(i+1)*shuffle_freq:] = zip(*sorted(pairs[(i+1)*shuffle_freq:], 
#                                                    key=lambda x: x[1]))[0]
#train_threads = list(np.array(train_threads)[ids])
#train_feats = list(np.array(train_feats)[ids])
#train_taps = list(np.array(train_taps)[ids])
#train_children = list(np.array(train_children)[ids])
#train_labels = list(np.array(train_labels)[ids])
#train_ids = list(np.array(train_ids)[ids])


print 'Processing dev'
dev_threads, dev_feats, dev_taps, dev_children, dev_labels, dev_ids = get_threads(devdata, post_data, test=True)

batch_size = int(config.get('batch_size'))
hidden_dim = int(config.get('hidden_dim'))
input_dim = int(config.get('input_dim'))
tuning_th = float(config.get('tuning_th'))
stop_epoch = int(config.get('tuning_epoch_cut'))

model_name = config.get('model_name').replace('subreddit', subreddit).replace('model','model_input_bow_plus_feat_h%d_emb' % hidden_dim)

dev_results_name = model_name.replace('model', 'dev_output').replace('pickle','txt')

devdata['pred'] = 0

dev_to_write = []
with open(os.path.join('SRS-GO/data/',dev_results_name), 'w') as dev_output:
    dev_output.write('')

model_params = [hidden_dim, input_dim,
                max(data['label']) + 1, batch_size]

model = HierarchicalModel(hidden_size=model_params[0],
                          input_size=model_params[1],
                          output_size=model_params[2],
                          batch_size=model_params[3],
                          vocab_sizes =[len(vocab), len(feature_names)],
                          celltype=LSTM)

new_vars = [x.get_value() for x in model.params]


word2vec_file = config.get('word2vec_file').replace('DIMNUM',str(input_dim)).replace('subreddit', subreddit)
with open(word2vec_file, 'rb') as inf:
    word2vec = pickle.load(inf)

assert np.shape(word2vec.syn0)[1] == input_dim
init = [word2vec[w] if w in word2vec.vocab else new_vars[0][i] for i,w in enumerate(vocab.index2word)]
new_vars[0][:len(vocab),:] = init
copy_params = theano.clone(model.params, replace=zip(model.params, new_vars))
model.set_params(copy_params)

best_score = [0, 0, 0] # score, time

for e in range(30):
    dev_to_write.append('Epoch: %d\n' % e)
    
    # early stop - has no update
    if e - best_score[2] > stop_epoch:
        break
    if e == 1 and best_score[0] < tuning_th:
        tuning_success = False
        break
    
    for i in range(len(train_threads) / batch_size):        
        batch, lengths = pad_into_variable_tensor(train_threads[i * batch_size:(i + 1) * batch_size])
        feat_batch, _ = pad_into_variable_tensor(train_feats[i * batch_size:(i + 1) * batch_size])
        batch = np.concatenate([batch, batch != 0, feat_batch], axis=2)
        taps_batch, _ = pad_into_variable_tensor(train_taps[i * batch_size:(i + 1) * batch_size])
        children_taps_batch, _ = pad_into_variable_tensor(train_children[i * batch_size:(i + 1) * batch_size])
        labels_batch, _ = pad_into_matrix(train_labels[i * batch_size:(i + 1) * batch_size])
        #error = model.update_fun(batch, taps_batch, children_taps_batch, lengths, labels_batch)
        error = model.train(batch, taps_batch, children_taps_batch, lengths, labels_batch)
        #code.interact(local=locals())
        if i % 20 == 0: 
            print 'testing on dev: ', i
            iteration = e * len(train_threads) / batch_size + i
            dev_preds = []
            for j in range(len(dev_threads) / batch_size):
                batch, lengths = pad_into_variable_tensor(dev_threads[j * batch_size:(j + 1) * batch_size])
                feat_batch, _ = pad_into_variable_tensor(dev_feats[j * batch_size:(j + 1) * batch_size])
                # add mask
                batch = np.concatenate([batch, batch != 0, feat_batch], axis=2)
                taps_batch, _ = pad_into_variable_tensor(dev_taps[j * batch_size:(j + 1) * batch_size])
                children_taps_batch, _ = pad_into_variable_tensor(dev_children[j * batch_size:(j + 1) * batch_size])
                preds = np.argmax(model.pred_fun(batch, taps_batch, children_taps_batch), axis=2)
                # we don't calculate the label for the post
                preds_cut = [x[1:lengths[l]] for l,x in enumerate(preds)]
                preds_cut_sq = [item for x in preds_cut for item in x]
#                id_cut = dev_ids[j * batch_size:(j+1) * batch_size]
#                id_cut_sq = [item for x in id_cut for item in x]
                dev_preds.extend(preds_cut_sq)
                #devdata.loc[id_cut_sq, 'pred'] = preds_cut_sq
            
            j += 1
            
            if not len(dev_threads) % batch_size == 0:
                batch, lengths = pad_into_variable_tensor(dev_threads[j * batch_size:])
                feat_batch, _ = pad_into_variable_tensor(dev_feats[j * batch_size:])
                # add mask
                batch = np.concatenate([batch, batch != 0, feat_batch], axis=2)
                taps_batch, _ = pad_into_variable_tensor(dev_taps[j * batch_size:])
                children_taps_batch, _ = pad_into_variable_tensor(dev_children[j * batch_size:])
                preds = np.argmax(model.pred_fun(batch, taps_batch, children_taps_batch), axis=2)

                preds_cut = [x[1:lengths[l]] for l,x in enumerate(preds)]
                preds_cut_sq = [item for x in preds_cut for item in x]
#                id_cut = dev_ids[j * batch_size:]
#                id_cut_sq = [item for x in id_cut for item in x]
                dev_preds.extend(preds_cut_sq)
#                devdata.loc[id_cut_sq, 'pred'] = preds_cut_sq

            print 'before integrating back'
            id_cut_sq = [item for x in dev_ids for item in x]
            devdata.loc[id_cut_sq, 'pred'] = dev_preds

            mask_deleted = (devdata.author == '[deleted]')
            active_devdata = devdata.ix[~mask_deleted]
            score = adjust_f1_score(active_devdata['label'], active_devdata['pred'], average='weighted')

            dev_to_write.append('''iter: %d, acc: %.3f\n''' % (iteration, score))
            if score > best_score[0]:
                print score
                best_score = [score, iteration, e]
                new_vars = [x.get_value() for x in model.params]
                copy_params = theano.clone(model.params, replace=zip(model.params, new_vars))
                with open(os.path.join('SRS-GO/data/scratch_local_ttmp',model_name), 'wb') as outf:
                    pickle.dump(model_params, outf)
                    pickle.dump(vocab, outf)
                    pickle.dump(copy_params, outf)

    with open(os.path.join('SRS-GO/data/', dev_results_name), 'a') as dev_output:
        dev_output.write(''.join(dev_to_write))
    dev_to_write = []

# write out the outconfig file  
with open(outconfig, 'w') as f:
  f.write('INCLUDE {0}\n'.format(inconfig))
