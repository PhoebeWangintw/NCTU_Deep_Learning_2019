import numpy as np
from string import ascii_lowercase
from torch.utils import data
import torch
import random

class build_index(torch.utils.data.Dataset):
    def __init__(self, f_name):
        self.char_to_idx = dict()
        self.idx_to_char = dict()
        self.char_to_idx['SOS'] = 0
        self.char_to_idx['EOF'] = 1
        self.char_to_idx['PAD'] = 2
        self.idx_to_char[0] = 'SOS'
        self.idx_to_char[1] = 'EOF'
        self.idx_to_char[2] = 'PAD'
        
        for idx, c in enumerate(ascii_lowercase):
            self.char_to_idx[c] = idx + 3
            self.idx_to_char[idx + 3] = c
            
        if f_name is None:
            self.max_len = 16
            return
        
        f = open(f_name, "r")
        # 0 for simple present, 1 for third person, 2 for present progressing, 3 for simple past
        self.vocs = []
        self.sp_vocs = []
        self.input_labels = []
        self.output_labels = []
        self.max_len = 0
        
        for line in f:
            split_voc = line[:-1].split(' ')
            indices = []
            for v in split_voc:
                if len(v) > self.max_len:
                    self.max_len = len(v)
                indices.append(self.seq_to_idx(v))
            for i in range(4):
                self.sp_vocs.append(indices[i])
                self.input_labels.append(i)
                self.vocs.append(indices[i])
                self.output_labels.append(i)
        
        # sort according to sequence character length
#         self.sp_vocs, self.input_labels, self.vocs, self.output_labels = zip(*sorted(zip(self.sp_vocs, self.input_labels, self.vocs, self.output_labels), key=lambda seqs:len(seqs[0]), reverse=True))        
#         self.sp_vocs, self.input_labels, self.vocs, self.output_labels = list(self.sp_vocs), list(self.input_labels), list(self.vocs), list(self.output_labels)
        self.sp_vocs_len = [len(s) for s in self.sp_vocs]
        self.max_len += 1  # add eof

        for i in range(len(self.sp_vocs)):
            self.sp_vocs[i] = self.do_padding(self.sp_vocs[i], self.max_len)
            self.vocs[i] = self.do_padding(self.vocs[i], self.max_len)  

        self.input_labels = self.one_hot(self.input_labels, max(self.input_labels))
        self.output_labels = self.one_hot(self.output_labels, max(self.output_labels))

        self.sp_vocs = torch.LongTensor(self.sp_vocs)
        self.sp_vocs_len = torch.LongTensor(self.sp_vocs_len)
        self.input_labels = torch.FloatTensor(self.input_labels)
        self.output_labels = torch.FloatTensor(self.output_labels)
        self.vocs = torch.LongTensor(self.vocs)

        p = [x for x in range(len(self.sp_vocs))]
        random.shuffle(p) 
        self.sp_vocs, self.input_labels, self.vocs, self.output_labels = self.sp_vocs[p], self.input_labels[p], self.vocs[p], self.output_labels[p]
      
    def __len__(self):
        return len(self.vocs)
    
    def __getitem__(self, idx):
        return self.sp_vocs[idx], self.sp_vocs_len[idx], self.input_labels[idx], self.vocs[idx], self.output_labels[idx]
    
    def seq_to_idx(self, seq):
        indices = []
        for c in seq:
            indices.append(self.char_to_idx[c])
        indices.append(self.char_to_idx['EOF'])
        
        return indices
    
    def idx_to_seq(self, indices):
        seq = []
        for idx in indices:
            if self.idx_to_char[idx] == 'EOF':
                break
            seq.append(self.idx_to_char[idx])
        
        return seq
    
    def idx_to_word(self, indices):
        seq = ""
        for idx in indices:
            if self.idx_to_char[idx] == 'EOF':
                break
            seq += self.idx_to_char[idx]
        
        return seq



    def do_padding(self, seq, max_len):
        paddings = (max_len - len(seq))
        
        if paddings > 0:
            seq.extend([self.char_to_idx['PAD']] * paddings)
        return seq
    
    def one_hot(self, idx, n):
        cs = []
        n += 1
        for i in idx:
            c = [0.0] * n
            c[i] = 1.0
            cs.append(c)
        return cs
       
    def get_test(self):
        raw_input_words = ["abandon", "abet", "begin", "expend", "sent", "split", "flared", "functioning", "functioning", "healing"]
        raw_output_words = ["abandoned", "abetting", "begins", "expends", "sends", "splitting", "flare", "function", "functioned", "heals"]
        input_words = []
        output_words = []
        input_labels = [0, 0, 0, 0, 3, 0, 3, 2, 2, 2]
        output_labels = [3, 2, 1, 1, 1, 2, 0, 0, 3, 1]
        
        raw_input_words, input_labels, raw_output_words, output_labels = zip(*sorted(zip(raw_input_words, input_labels, raw_output_words, output_labels), key=lambda seqs:len(seqs[0]), reverse=True))
        raw_input_words, input_labels, raw_output_words, output_labels = list(raw_input_words), list(input_labels), list(raw_output_words), list(output_labels)
        input_vocs_len = [len(s) for s in raw_input_words]
        
        for i in range(len(raw_input_words)):
            iw = [s for s in raw_input_words[i]]
            ow = [s for s in raw_output_words[i]]
            input_words.append(self.seq_to_idx(iw))
            output_words.append(self.seq_to_idx(ow))
            
        for i in range(len(input_words)):
            input_words[i] = self.do_padding(input_words[i], self.max_len)
            output_words[i] = self.do_padding(output_words[i], self.max_len)  
        
        input_labels = self.one_hot(input_labels, max(input_labels))
        output_labels = self.one_hot(output_labels, max(output_labels))
        
        input_words = torch.LongTensor(input_words)
        input_vocs_len = torch.LongTensor(input_vocs_len)
        input_labels = torch.FloatTensor(input_labels)
        output_labels = torch.FloatTensor(output_labels)
        output_words = torch.LongTensor(output_words)
        
        return input_words, input_labels, input_vocs_len, output_words, output_labels