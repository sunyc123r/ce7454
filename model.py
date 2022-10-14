import clip
import json
import torch
import torch.nn as nn
from time import time
import torch.nn.functional as F
from math  import sqrt


class MLP2(nn.Module):

    def __init__(self,in_dim,drop=0.1):
        super(MLP2, self).__init__()
        self.dropout = nn.Dropout(p=drop)
        self.linear1=nn.Linear(in_dim,int(in_dim/2))
        self.linear2=nn.Linear(int(in_dim/2),int(in_dim/4))
        self.linear3=nn.Linear(int(in_dim/4),56)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)

        self.relu=nn.ReLU(0.1)

    def forward(self,input):
       # input=torch.cat([a,b],dim=1)
        out=self.linear1(input)
        out=self.relu(out)
        out=self.dropout(out)
        out=self.linear2(out)
        out=self.relu(out)
        out=self.dropout(out)
        out=self.linear3(out)
        return out

class Clip_Rel(nn.Module):

    def __init__(self,device,drop=0.1):
        super(Clip_Rel, self).__init__()
        self.device=device
    
        self.lm,self.preprocess = clip.load('ViT-L/14')
    
        self.lm=self.lm.to(self.device)
        self.label_encode() # encode the prompt filled with 56 classes
        self.embed_dim=768
        self.mlp2=MLP2(2*self.embed_dim,drop)

    def label_encode(self):
        with open('./data/psg/psg_cls_advanced.json') as f:
            dataset = json.load(f)
            self.classes=dataset['thing_classes']+dataset['stuff_classes']
            self.relation_classes=dataset['predicate_classes']

        self.label=['A picture of a '+str(i) for i in self.classes]
        text=clip.tokenize(self.label).to(self.device)
        with torch.no_grad():
            self.text_features=self.lm.encode_text(text).float().to(self.device)
            self.text_features/=self.text_features.norm(dim=-1,keepdim=True)

    def semantic_encoder(self,labels):

        label=['A picture contains{},{} and {}'.format(label[0],label[1],label[2]) for label in labels]
        text=clip.tokenize(label).to(self.device)
        with torch.no_grad():
            label_features=self.lm.encode_text(text).float().to(self.device)
            label_features/=label_features.norm(dim=-1,keepdim=True)
        return label_features

    def forward(self,x):

        ## object detector
        with torch.no_grad():
            image_features=self.lm.encode_image(x).float().to(self.device)
            image_features=image_features/image_features.norm(dim=-1,keepdim=True)
            similarity=torch.mm(image_features,self.text_features.T)
            prob,label=similarity.topk(3,dim=1)

        semantic_feature=self.semantic_encoder(label) 
        out=self.mlp2(torch.cat([semantic_feature,image_features],dim=1))

        return out
