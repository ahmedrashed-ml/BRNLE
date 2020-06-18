import numpy as np
import pandas as pd
import time
from collections import deque

import tensorflow as tf
tf.reset_default_graph()
from six import next
from tensorflow.core.framework import summary_pb2
from sklearn import preprocessing
import sys
import matplotlib.pyplot as plt
import random
import pickle
import math
from scipy.sparse import csr_matrix
from scipy.sparse import identity
from keras import backend as K
from sklearn.metrics.pairwise import cosine_similarity


DIM=700
USER_NUM = 3890
ITEM_NUM = 50
EPOCH_MAX =316
BATCH_SIZE = 1
NEGSAMPLES=1
UseUserData=1
UseItemData=1
UseValidation=0
UserReg=0.0
ItemReg=0.0
PERC=0.9
UL2Weight=0.0125
IL2Weight=0.0005
#np.random.seed(3)
#tf.set_random_seed(3)
LEARNRATE=0.00003
DEVICE = "/gpu:0"

def load_data(filename):
    try:
        with open(filename, "rb") as f:
            x= pickle.load(f)
    except:
        x = []
    return x

def save_data(data,filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def generate_MainMatrix(dfgp):
    df = dfgp
    rows = len(df)
    dictUsers={}
    ##Filling all PosTrain
    print('Filling all PosTrain')
    for index, row in df.iterrows():
      userid=int(row['userid'])
      itemid=int(row['gp'])
      if userid in dictUsers:
        dictUsers[userid][0].append(row['gp'])
      else:
        dictUsers[userid]={0:list(),1:list()}
        dictUsers[userid][0].append(row['gp'])

    print('Filling Neg Instance ')
    ## Filling Neg Instance          
    for userid in dictUsers: 
      for i in range(ITEM_NUM):
        if i not in dictUsers[userid][0] :
          dictUsers[userid][1].append(i)
      
    return dictUsers    

def generate_AuxMatrix(dfedg):
    df = dfedg
    rows = len(df)
    dictUsers={}
    ##Filling all PosTrain
    print('Filling all PosTrain')
    for index, row in df.iterrows():
      userid=int(row['userid1'])
      userid2=int(row['userid2'])
      if userid in dictUsers:
        dictUsers[userid][0].append(row['userid2'])
      else:
        dictUsers[userid]={0:list(),1:list()}
        dictUsers[userid][0].append(row['userid2'])

      if userid2 in dictUsers:
        dictUsers[userid2][0].append(row['userid1'])
      else:
        dictUsers[userid2]={0:list(),1:list()}
        dictUsers[userid2][0].append(row['userid1'])

    print('Filling Neg Instance ')
    ## Filling Neg Instance          
    for userid in dictUsers: 
      for i in range(USER_NUM):
        if i not in dictUsers[userid][0] :
          dictUsers[userid][1].append(i)
      
    return dictUsers  

def GetMainTrainSample(DictUsers,BatchSize=1):
  global TESTNodes
  global TRAINNodes
  trainInst=list()
  numusers=BatchSize
  #print(numusers)
  for i in range(numusers):
    batchusers=random.choice(TRAINNodes) #random.randint(0,USER_NUM-1)
    while len(DictUsers[batchusers][0])==0:
      batchusers=random.choice(TRAINNodes)

    trainInst.append([batchusers,np.random.choice(DictUsers[batchusers][0], 1)[0],np.random.choice(DictUsers[batchusers][1], 1)[0]])
    
  trainInst=np.asarray(trainInst)  
  return trainInst

def GetAuxTrainSample(DictUsers,BatchSize=1):
  trainInst=list()
  numusers=BatchSize
  #print(numusers)
  for i in range(numusers):
    batchusers=random.choice(list(DictUsers.keys())) #random.randint(0,USER_NUM-1)
    while len(DictUsers[batchusers][0])==0:
      batchusers=random.choice(list(DictUsers.keys()))

    trainInst.append([batchusers,np.random.choice(DictUsers[batchusers][0], 1)[0],np.random.choice(DictUsers[batchusers][1], 1)[0]])
    
  trainInst=np.asarray(trainInst)  
  return trainInst

def GetTestData(DictUsers):
  global TESTNodes
  testInst=list()
  for i in range(len(TESTNodes)):
    batchusers=TESTNodes[i]
    for j in range(len(DictUsers[batchusers][0])):
      testInst.append([batchusers,DictUsers[batchusers][0][j],1])
    for j in range(len(DictUsers[batchusers][1])):
      testInst.append([batchusers,DictUsers[batchusers][1][j],0])

  testInst=np.asarray(testInst)  
  return testInst

def GetTrainData(DictUsers):
  global TRAINNodes
  trainInst=list()
  for i in range(len(TRAINNodes)):
    batchusers=TRAINNodes[i]
    for j in range(len(DictUsers[batchusers][0])):
      trainInst.append([batchusers,DictUsers[batchusers][0][j],1])
    for j in range(len(DictUsers[batchusers][1])):
      trainInst.append([batchusers,DictUsers[batchusers][1][j],0])

  trainInst=np.asarray(trainInst)  
  return trainInst


def GetMainTrainObs(DictUsers):
  global TRAINNodes
  obs=0
  for i in range(len(TRAINNodes)):
    batchusers=TRAINNodes[i]
    for j in range(len(DictUsers[batchusers][0])):
      obs=obs+1
  return obs  

def GetAuxTrainObs(DictUsers):
  obs=0
  for i in range(USER_NUM):
    for j in range(len(DictUsers[i][0])):
      obs=obs+1
  return obs  

def EvalMatrix(UGPredMatrix,UGTrueMatrix):
  UGPredMatrixTmp=np.array(UGPredMatrix, copy=True)  
  UGTrueMatrixTmp=np.array(UGTrueMatrix, copy=True)  
  UGPredMatrixNewListMod=list()
  UGPredMatrixNewListReal=list()
  UGTrueMatrixNewListTrue=list()
  TotalUsersAUC=0
  ValidUsersCount=0

  for i in range(USER_NUM):

    RawPredicted=np.array(UGPredMatrixTmp[i], copy=True) 
    RawTrue=np.array(UGTrueMatrixTmp[i], copy=True)  
    ModPredicted=[0 for y in range(ITEM_NUM)]
    RealPredicted= [0 for y in range(ITEM_NUM)]
    TopN=np.sum(RawTrue)
    if(TopN>0):
    #print(' group  ',i, ' topn ',TopN)
      for j in range(TopN):
        #print(RawPredicted)
        #print(RawTrue)
        maxIndex=np.argmax(RawPredicted)
        maxValue=RawPredicted[maxIndex]
        #print(maxIndex,'   ',maxValue)
        RawPredicted[maxIndex]=-9999999
        ModPredicted[maxIndex]=1
        RealPredicted[maxIndex]=maxValue
      TopNPred=np.sum(ModPredicted)
      UGPredMatrixNewListMod.append(np.array(ModPredicted, copy=True) )
      UGPredMatrixNewListReal.append(np.array(RealPredicted, copy=True) )
      UGTrueMatrixNewListTrue.append(np.array(RawTrue, copy=True) )
      ValidUsersCount=ValidUsersCount+1
      


  UGPredMatrixNewMod=np.array(UGPredMatrixNewListMod)
  UGPredMatrixNewReal=np.array(UGPredMatrixNewListReal)
  UGPredMatrixNewTrue=np.array(UGTrueMatrixNewListTrue)
  #print(UGPredMatrixNewMod.shape)
  #print(UGPredMatrixNewTrue.shape)
  UGPredMatrixT=np.transpose(UGPredMatrixNewMod)
  UGTrueMatrixT=np.transpose(UGPredMatrixNewTrue)

  TotalTP=0
  TotalFP=0
  TotalTN=0
  TotalFN=0
  GroupsF1Scores=list()
  GroupsF1Scores2=list()
  GroupsPrecScores=list()
  GroupsRecScores=list()
  for i in range(ITEM_NUM):
    #print(TopN)
    #print(TopNPred)
    ModPredicted=np.array(UGPredMatrixT[i], copy=True) 
    RawTrue=np.array(UGTrueMatrixT[i], copy=True) 
    TP=0
    FP=0
    TN=0
    FN=0
    F1=0
    for j in range(len(ModPredicted)):

      if(ModPredicted[j]==1 and  ModPredicted[j]==RawTrue[j]):
        TP=TP+1
        TotalTP=TotalTP+1
        #print(j,' TP ',ModPredicted[j],' ',RawTrue[j])
      if(ModPredicted[j]==1 and  ModPredicted[j]!=RawTrue[j]):
        FP=FP+1
        TotalFP=TotalFP+1
        #print(j,' FP ',ModPredicted[j],' ',RawTrue[j])
      if(ModPredicted[j]==0 and  ModPredicted[j]==RawTrue[j]):
        TN=TN+1
        TotalTN=TotalTN+1
      if(ModPredicted[j]==0 and  ModPredicted[j]!=RawTrue[j]):
        FN=FN+1
        TotalFN=TotalFN+1
    #if(TP!=0 or FP!=0 or FN!=0):
    F1=(2*TP)/(2*TP+FP+FN)
    Prec=0
    Recall=0
    if(TP!=0 or FP!=0):
      Prec=TP/(TP+FP)
    if(TP!=0 or FN!=0):
      Recall=TP/(TP+FN)
    GroupsF1Scores2.append(F1)
    GroupsPrecScores.append(Prec)
    GroupsRecScores.append(Recall)      
    GroupsF1Scores.append(F1)

  MicroF1Score=(2*TotalTP)/(2*TotalTP+TotalFP+TotalFN) 
  #print(TotalTP,' ',TotalFP,' ',TotalFN,' ',TotalTN)
  MicroPrecScore=TotalTP/(TotalTP+TotalFP)
  MicroRecScore=TotalTP/(TotalTP+TotalFN)


  #print(MicroPrecScore)
  #print(MicroRecScore)
  MacroF1Score= np.mean(GroupsF1Scores)
  MacroF1Score2= np.mean(GroupsF1Scores2)
  #print(len(GroupsF1Scores2))
  #print(GroupsPrecScores)
  #print(GroupsRecScores)
  MacroPrecScore=np.mean(GroupsPrecScores)
  MacroRecScore=np.mean(GroupsRecScores)
  allauc=0#roc_auc_score(UGPredMatrixNewTrue, UGPredMatrixNewMod)

  datadict ={"MicroF1Score":MicroF1Score,
               "MacroF1Score":MacroF1Score,
               "MacroF1Score2":MacroF1Score2,
               "MicroPrecScore":MicroPrecScore,
               "MicroRecScore":MicroRecScore,
               "MacroPrecScore":MacroPrecScore,
               "MacroRecScore":MacroRecScore}


  return datadict

def inferenceDense(phase,input_batch,relflag, embedding_size=5, device="/cpu:0"):
    global USER_NUM
    global ITEM_NUM
    with tf.device(device):

      indx1,indx2,indx3=tf.split(input_batch, [1,1,1], 1)
      
      indx1=tf.reshape(indx1,[-1])
      indx2=tf.reshape(indx2,[-1])
      indx3=tf.reshape(indx3,[-1])

      inputsT=tf.one_hot(indx1, USER_NUM)
      ###Aux
      inputsAux2=tf.one_hot(indx2, USER_NUM)
      inputsAux3=tf.one_hot(indx3, USER_NUM)

      ###Main
      inputsMain2=tf.one_hot(indx2, ITEM_NUM)
      inputsMain3=tf.one_hot(indx3, ITEM_NUM)



      hidden_layer1U=tf.get_variable('hidden_layer1U', [USER_NUM, 700], initializer=tf.random_normal_initializer(stddev=0.01))
      hidden_layer1UBias=tf.get_variable('hidden_layer1UBias', [1, 700], initializer=tf.constant_initializer(0))

      hidden_layer1I=tf.get_variable('hidden_layer1I', [ITEM_NUM, 700], initializer=tf.random_normal_initializer(stddev=0.01))
      hidden_layer1IBias=tf.get_variable('hidden_layer1IBias', [1, 700], initializer=tf.constant_initializer(0))

      mainUser=tf.nn.crelu(tf.add(tf.matmul(inputsT,hidden_layer1U),hidden_layer1UBias))
      
      posUser=tf.nn.crelu(tf.add(tf.matmul(inputsAux2,hidden_layer1U),hidden_layer1UBias))
      negUser=tf.nn.crelu(tf.add(tf.matmul(inputsAux3,hidden_layer1U),hidden_layer1UBias))

      posItem=tf.nn.crelu(tf.add(tf.matmul(inputsMain2,hidden_layer1I),hidden_layer1IBias))
      negItem=tf.nn.crelu(tf.add(tf.matmul(inputsMain3,hidden_layer1I),hidden_layer1IBias))


      Prediction=tf.reduce_sum(tf.multiply(mainUser,posItem),axis=1)

      BPR1=tf.reduce_sum(tf.multiply(mainUser, tf.subtract(posUser, negUser)), keep_dims=True)
      BPR2=tf.reduce_sum(tf.multiply(mainUser, tf.subtract(posItem, negItem)), keep_dims=True)

      BPR1sig=tf.sigmoid(BPR1)
      BPR2sig=tf.sigmoid(BPR2)

      BPR1sigClipped=tf.clip_by_value(BPR1sig, 1e-20, 1e+20)
      BPR2sigClipped=tf.clip_by_value(BPR2sig, 1e-20, 1e+20)

      BPR1log=tf.reduce_mean(tf.log(BPR1sigClipped))
      BPR2log=tf.reduce_mean(tf.log(BPR2sigClipped))

      l2_normR1U = tf.nn.l2_loss(mainUser)#tf.reduce_sum(tf.multiply(u_embR1u, u_embR1u))
      l2_normR1I = tf.nn.l2_loss(posUser)#tf.reduce_sum(tf.multiply(u_embR1i, u_embR1i))
      l2_normR1J = tf.nn.l2_loss(negUser)#tf.reduce_sum(tf.multiply(u_embR1j, u_embR1j))


      l2_normR2U = tf.nn.l2_loss(mainUser)#tf.reduce_sum(tf.multiply(u_embR2u, u_embR2u))
      l2_normR2I = tf.nn.l2_loss(posItem)#tf.reduce_sum(tf.multiply(i_embR2i, i_embR2i))
      l2_normR2J = tf.nn.l2_loss(negItem)#tf.reduce_sum(tf.multiply(i_embR2j, i_embR2j))



      cost1= 0.5*BPR1log  -UL2Weight*l2_normR1U -UL2Weight*l2_normR1I  -UL2Weight*l2_normR1J 

      cost2= 0.5*BPR2log  -UL2Weight*l2_normR2U -IL2Weight*l2_normR2I  -IL2Weight*l2_normR2J 



      costr = tf.cond(relflag < 2, lambda: cost1, lambda: cost2)
      costrneg=tf.negative(costr)
      cost= costrneg

      train_step=tf.train.GradientDescentOptimizer(0.02).minimize(cost)

    return train_step,cost,Prediction,inputsT,inputsMain2



def Run():
    global TRAINNodes
    global AuxDict
    global MainDict
    global TESTDATA
    global TRAINDATA

    input_batch = tf.placeholder(tf.int32, shape=[None,3], name="id_user")
    phase = tf.placeholder(tf.bool, name='phase')
    relflag = tf.placeholder(tf.int32, name="relflag")

    global_step = tf.contrib.framework.get_or_create_global_step()    
    train_step,cost,Prediction,mainUser,posItem = inferenceDense(phase,input_batch,relflag, embedding_size=DIM,device=DEVICE)
    

    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0

    with tf.Session() as sess:
        sess.run(init_op)
        ObsCount1=GetAuxTrainObs(AuxDict)
        ObsCount2=GetMainTrainObs(MainDict)
        evaluate=True
        evaluateTrain=True
        errorsAux= deque(maxlen=ObsCount1)
        errorsMain = deque(maxlen=ObsCount2)
        print("Start ",ObsCount1,'  ',ObsCount2)
        for i in range(EPOCH_MAX):
          #############################
          for l in range(ObsCount1):
            trainbatch=GetAuxTrainSample(AuxDict)

            predAux,costAux,_=sess.run([Prediction,cost,train_step],feed_dict={input_batch:trainbatch,relflag:1,phase:True })
            errorsAux.append(costAux)
 
          ##############################
          for l in range(ObsCount2):
            trainbatch=GetMainTrainSample(MainDict)

            predMain,costMain,_=sess.run([Prediction,cost,train_step],feed_dict={input_batch:trainbatch,relflag:2,phase:True })
            errorsMain.append(costMain)
 
          ##############################

          if evaluate:
                FinalPreds = list()
                Finallabels = list()
                FinalPairs = list()

                pairs,labels=np.split(TESTDATA, [2],axis=1)
                testbatch=TESTDATA

                preds,t1,t2=sess.run([Prediction,mainUser,posItem],feed_dict={input_batch:testbatch,relflag:2,phase:False })
                FinalPreds=preds
                Finallabels=labels.reshape(-1)
                FinalPairs=pairs
                

                UGEvalMatrix = [[0 for x in range(ITEM_NUM)] for y in range(USER_NUM)] 
                UGTrueMatrix =[[0 for x in range(ITEM_NUM)] for y in range(USER_NUM)] 
                for k in range(len(FinalPreds)):
                    user=FinalPairs[k][0]
                    group=FinalPairs[k][1]
                    UGEvalMatrix[user][group]=FinalPreds[k]
                    UGTrueMatrix[user][group]=Finallabels[k]

                CorrectEval=EvalMatrix(UGEvalMatrix,UGTrueMatrix)

                MicroF1Score=CorrectEval['MicroF1Score']
                MacroF1Score=CorrectEval['MacroF1Score']
                MacroF1Score2=CorrectEval['MacroF1Score2']

                print('Test metrics :',i,' Micro=  ',MicroF1Score,' Macro=  ',MacroF1Score)
          ##############################

          if evaluateTrain:
                FinalPreds = list()
                Finallabels = list()
                FinalPairs = list()

                pairs,labels=np.split(TRAINDATA, [2],axis=1)
                trainbatch=TRAINDATA

                preds,t1,t2=sess.run([Prediction,mainUser,posItem],feed_dict={input_batch:trainbatch,relflag:2,phase:False })
                FinalPreds=preds
                Finallabels=labels.reshape(-1)
                FinalPairs=pairs
                

                UGEvalMatrix = [[0 for x in range(ITEM_NUM)] for y in range(USER_NUM)] 
                UGTrueMatrix =[[0 for x in range(ITEM_NUM)] for y in range(USER_NUM)] 
                for k in range(len(FinalPreds)):
                    user=FinalPairs[k][0]
                    group=FinalPairs[k][1]
                    UGEvalMatrix[user][group]=FinalPreds[k]
                    UGTrueMatrix[user][group]=Finallabels[k]

                CorrectEval=EvalMatrix(UGEvalMatrix,UGTrueMatrix)

                MicroF1Score=CorrectEval['MicroF1Score']
                MacroF1Score=CorrectEval['MacroF1Score']
                MacroF1Score2=CorrectEval['MacroF1Score2']
                MacroF1Score2=CorrectEval['MacroF1Score2']

                print('Train metrics :',i,' Micro=  ',MicroF1Score,' Macro=  ',MacroF1Score)
                print("Train loss :",i,'  Main cost :',np.mean(errorsMain),'  Aux cost :',np.mean(errorsAux))
                print('###############')          

    return 


## Data Genneration Step (Uncomment this part if you want to generate new data split)
'''
dfgroups=pd.read_csv("../Data/group-edges.csv", sep=',', engine='python')
dfedges=pd.read_csv("../Data/edges.csv", sep=',', engine='python')

UsEncoder= preprocessing.LabelEncoder()
GpEncoder= preprocessing.LabelEncoder()

dfgroups.userid= UsEncoder.fit_transform(dfgroups.userid)
dfedges.userid1= UsEncoder.transform(dfedges.userid1)
dfedges.userid2= UsEncoder.transform(dfedges.userid2)

dfgroups.gp= GpEncoder.fit_transform(dfgroups.gp)

TESTNodes=np.random.choice(dfgroups.userid.unique(), math.floor(len(dfgroups.userid.unique())*PERC),replace=False)
TRAINNodes=np.setdiff1d(dfgroups.userid.unique(), TESTNodes)

MainDict= generate_MainMatrix(dfgroups)
AuxDict= generate_AuxMatrix(dfedges)

save_data(MainDict,"../Data/MainDict901.dat")
save_data(AuxDict,"../Data/AuxDict901.dat")
save_data(TRAINNodes,"../Data/TRAINNodes901.dat")
save_data(TESTNodes,"../Data/TESTNodes901.dat")
'''
##



MainDict=load_data("../Data/MainDict901.dat")
AuxDict=load_data("../Data/AuxDict901.dat")
TRAINNodes=load_data("../Data/TRAINNodes901.dat")
TESTNodes=load_data("../Data/TESTNodes901.dat")
print(len(TRAINNodes))
print(len(TESTNodes))
TESTDATA=GetTestData(MainDict)
TRAINDATA=GetTrainData(MainDict)
print(TRAINDATA.shape)
print(TESTDATA.shape)

Run()
#SampleMain=GetMainTrainSample(MainDict)
#SampleAux=GetAuxTrainSample(AuxDict)
