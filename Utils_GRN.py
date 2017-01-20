import sys
#sys.path
#sys.path.append('D:\Program Files (x86)\Graphviz2.38\\bin')
#sys.path
import numpy as np
import pandas as pd
import random

from graphviz import Digraph
import pydotplus
import networkx as nx

import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler


#%matplotlib inline
#initialization()

###########--------------------  Initialization 

def initialization():
    global PopSize,NumGen,max_neurons,min_neurons,obsN,max_input,con_rate
    global crossover_rate,mutation_rate,elit_rate,tournrate,B_max_w,w_rate,n_exp,n_syn,runs
    
    PopSize=50
    NumGen=500

    max_neurons=4      #maximum number of neurons
    min_neurons=1
    n_exp=1 ## number of experiments
    obsN=5          #ecoli=8, synthetic=15 # n number of observed variables like number of genes
    max_input=max_neurons-1+obsN
    
    con_rate=2#2/obsN #.2       # ratio of number nodes that should be connected to neurons
    B_max_w=0.5          # connection weight initial boundary
    w_rate=0.1
    mutation_rate =0.1 # 0.1mutate input nodes, output nides, weights 
    elit_rate=0.1
    crossover_rate=.5
    tournrate=0.1
    
    n_syn=10 ## number of generated synthetic GRNs under same noise
    runs=10  # number of repeats of GA, among them select the minimum fitness
    
    
    
##########------------------------------------------------------------
def parallel_calc(num_cores,Data):
    genes=np.arange(0,obsN)+max_neurons-1
    result=Parallel(n_jobs=num_cores) (delayed(GRN)(target,Data) for target in genes)
    return result
    
def GRN(target,Data):
    initialization()
    #Data1=pd.read_excel("ecoli_ex1.xls", header=None)    ## ecoli
    #Data1.drop(Data1.columns[0], axis=1)
    #Data=pd.DataFrame.transpose(Data1)
    
    #Data=pd.read_excel("cell.xls", header=None)          ## synthetic 15 genes, no delay
    
    #Data = pd.read_csv('insilico_size10_1_dream4_timeseries.tsv',sep='\t')
    #del Data['Time']
    
    #Data= pd.read_csv('Data_si1.csv',delimiter=",",header=None)
    
    
    Sc=MinMaxScaler(feature_range=(-.7, .7))
    Data_normed=Sc.fit_transform(Data)#.values[:].reshape(-1, 1)    
    

        
    ##########-------------------    
    Wmut=int(NumGen)/2
    elit_n=int(np.floor(elit_rate*PopSize))
    newsize=PopSize+elit_n
    cross=int(PopSize/2)  

    ##%%%%%%%%%%%% For each target gene repeat the GA
    bestfit=np.empty(runs)
    Finalnet=[]
    for run in range(runs):
        Pop=[GenerateIndividual(target) for i in range(0,PopSize)]

        Fitness=np.zeros(PopSize)
        for net in range(0,PopSize):
            Fitness[net]=CalcFitness(Pop[net],Data_normed,target,n_exp)   

        #######------------------------
        #MinFit=np.empty(NumGen)
        #MaxFit=np.empty(NumGen)
        #AvgFit=np.empty(NumGen)
        #UNI=np.empty(NumGen)

        for j in range(0,NumGen):      
            ind_best=Fitness.argsort()[:elit_n].copy()      
            elit_ind=[Pop[int(a)].copy() for a in ind_best].copy()

            selectedpop=[]
            for i in range(0,PopSize):
                tournindex=tournamentSelection(Fitness,PopSize,tournrate)
                selectedpop.append(Pop[tournindex].copy())

            ####----------------------------  
            crosspop=[]
            random.shuffle(selectedpop)

            S=np.empty(PopSize)
            for i in range(PopSize):
                S[i]=selectedpop[i][0]
            sorted_=S.argsort()
            selectedpop_=[selectedpop[int(a)] for a in sorted_].copy() 

            if crossover_rate!=0:
                for i in range(0,cross):  
                    if random.random() < crossover_rate and selectedpop_[2*i][0]==selectedpop_[2*i+1][0]: 
                        CrossChild1,CrossChild2= Crossover(selectedpop_[2*i].copy(),selectedpop_[2*i+1].copy(),crossover_rate)
                    else:
                        CrossChild1=selectedpop_[2*i].copy()
                        CrossChild2=selectedpop_[2*i+1].copy()
                    crosspop.append(CrossChild1)
                    crosspop.append(CrossChild2)  
            else:
                crosspop=selectedpop_.copy()

            ###------------------

            newpop=[]
            if j <=Wmut:
                for i in range(0,PopSize):
                    newpop.append(Mutation(crosspop[i].copy(),mutation_rate,w_rate,target))
            else:
                for i in range(0,PopSize): #mutation just on the weight
                    newpop.append(WMutation(crosspop[i].copy(),mutation_rate,w_rate))

            newpop.extend(elit_ind) 

            #######-----------

            newFitness=np.zeros(newsize)
            for net in range(0,newsize):
                newFitness[net]=CalcFitness(newpop[net],Data_normed,target,n_exp) 


            ###########--------------------

            index_fittest=newFitness.argsort()[:PopSize].copy()
            Pop=[newpop[int(a)].copy() for a in index_fittest] .copy()
            Fitness=newFitness[index_fittest].copy()

            ###----------    
            #MinFit[j]=min(Fitness)
            #MaxFit[j]=max(Fitness)
            #AvgFit[j]=np.mean(Fitness)
            #UNI[j]=len(np.unique(Fitness))  


        ############### End of GA
        x_select=np.argmin(Fitness)
        bestfit[run]=min(Fitness)
        Finalnet.append(Pop[x_select])
        
    bestfit_select=np.argmin(bestfit)
    Finalnet_select=Finalnet[bestfit_select]
    
    #y_hat=Evaluate(Finalnet,Data_normed,target,n_exp,Sc)
    #Data_normed_G=Data_normed.copy()
    #Data_normed_G[:,target-max_neurons+1]=y_hat
    #Data_G=Sc.inverse_transform(Data_normed_G)#.reshape(-1,1)
    #y_hat=Data_G[:,target-max_neurons+1]     
    

    #fig1 = plt.figure()
    #ax1 = fig1.add_subplot(111)
    #ax1.plot(MinFit)
    #plt.hold('on')
    #ax1.plot(AvgFit)
    #plt.hold('on')
    #ax1.plot(MaxFit)
    #plt.ylabel('Fitness g' + str(target-max_neurons+2))
    #plt.xlabel('Iterations')
    #plt.legend()
    #plt.savefig('evolution1.png')
    #plt.savefig('Evolution_g' + str(target-max_neurons+2) + '.png')    
    

    #fig2 = plt.figure()
    #ax2 = fig2.add_subplot(111)
    #line_up,=ax2.plot(Data[target-max_neurons+1],label='g' + str(target-max_neurons+2))
    #plt.hold(True)
    #line_down,=ax2.plot(y_hat,label='Predict')
    #handles, labels = ax2.get_legend_handles_labels()
    #plt.ylabel('Gene expression' + str(target-max_neurons+2))
    #plt.legend(handles=[line_up, line_down])
    #plt.savefig('Expression_g' + str(target-max_neurons+2) + '.png')
    

    #Gragh_pannet(Finalnet_select,target)     
    
    return Finalnet_select  #,min(Fitness)]
        
    
    
###########--------------------  Population  
def GenerateIndividual(target_code):
    NumNeurons=random.randint(min_neurons,max_neurons)
    contextN=NumNeurons-1
    Individual=np.array(NumNeurons)
    
    for i in range(NumNeurons): 
        if contextN !=0 and i!=contextN:   # net with context nodes, inputs of the neurons except the last one
            #NumInputs=random.randint(1,np.ceil((NumNeurons-1+obsN-1)*con_rate))
            NumInputs=random.randint(1,con_rate)
            connections,weights=np.array([]),np.array([])
            connections=np.append(connections,i)
            weights=np.append(weights,B_max_w * np.random.uniform(0, 1, 1))
            
            A=np.arange(0,max_input)
            A=np.delete(A,target_code)
            for link in range(max_neurons-2,NumNeurons-2,-1):# No extra context nodes
                A=np.delete(A,link)
            A=np.delete(A,i)

            for k in range(NumInputs-1): 
                index=random.randint(0,len(A)-1)
                node=A[index]
                connections=np.append(connections,node)
                A=np.delete(A,index)
                

                if node<=(contextN-1):
                    weights=np.append(weights,B_max_w * np.random.uniform(0, 1, 1))
                else: 
                    weights=np.append(weights,B_max_w * np.random.uniform(-1, 1, 1))
         
            ######--------------------------------
        elif  contextN !=0 and i==contextN: # net with context nodes, generate inputs of the last neuron
            #NumInputs=random.randint(1,np.ceil((obsN-1)*con_rate))
            NumInputs=random.randint(1,con_rate)
            connections,weights=np.array([]),np.array([])
            A=np.arange(max_input)
            
            A=np.delete(A,target_code)
            for link in range(max_neurons-2,-1,-1):  ## no context node feeding into the last neuron 
                A=np.delete(A,link)
            
            for k in range(NumInputs): 
                index=random.randint(0,len(A)-1)
                node=A[index]
                connections=np.append(connections,node)
                A=np.delete(A,index)              
                
                weights=np.append(weights,B_max_w * np.random.uniform(-1, 1, 1))
                """
                if node<=(contextN-1):
                    weights=np.append(weights,B_max_w * np.random.uniform(0, 1, 1)) 
                else: 
                    weights=np.append(weights,B_max_w * np.random.uniform(-1, 1, 1))
                """         
        elif contextN==0:# net includes no context nodes,
            #NumInputs=random.randint(1,np.ceil((obsN-1)*con_rate))
            NumInputs=random.randint(1,con_rate)
            connections,weights=np.array([]),np.array([])                       
            A=np.arange(max_input)
            
            A=np.delete(A,target_code)
            for link in range(max_neurons-2,NumNeurons-2,-1):
                A=np.delete(A,link)
                                   
            for k in range(NumInputs):
                index=random.randint(0,len(A)-1)
                node=A[index]
                connections=np.append(connections,node)
                A=np.delete(A,index)
                
                weights=np.append(weights,B_max_w * np.random.uniform(-1, 1, 1))
        
        Individual=np.hstack( [Individual,NumInputs,connections,weights])
        #print(Individual)

    #Gene_depression=B_max_w * np.random.uniform(0, 1,1)
    #Individual=np.hstack([Individual,Gene_depression])                          
                                  
    return Individual







###########--------------------    Fitness
def CalcFitness(Individual,TrainingData,target,n_exp):#,X,RxpNum
    X=0.01#*np.ones(max_input)
    T=int(len(TrainingData)/n_exp)   
    NumNeuron=int(Individual[0])
    target_g=target-max_neurons+1
    A=np.zeros([NumNeuron,max_input])

    k=1
    s_w=0
    K=0
    for i in range(0,NumNeuron):
        NumInputs=int(Individual[k])       
        Inputnodes=[int(x) for x in Individual[k+1:k+1+NumInputs] ]
        InputWeights=Individual[k+1+NumInputs :k+1+ 2 *NumInputs  ]
        
        leafs=[node for node in Inputnodes if node>=(NumNeuron-1)]
        s_w=s_w+sum(abs(InputWeights))
        K=K+len(leafs)#NumInputs ## number of connections
        k=k+1 +2 *NumInputs
        A[i,Inputnodes]=InputWeights

    #landa=Individual[-1]

    #XNext=X*np.ones(NumNeuron)#copy()
    
    
    ##########-----------------------------
    s=0
    for exp in range(0,n_exp):
        XNext=X*np.ones(max_input)#copy()
        XNext[max_neurons-1:]=TrainingData[exp*(T),:].copy()
        for t in range(1,T):
            x=np.dot(A,XNext)
            #update=np.piecewise(x, [x < 0, x >= 0], [-1, 1])
            update=np.tanh(x)
            #update[0]=update[0]*(1-landa)
            x_hat=update[0].copy()
            
            #if t==NumNeuron-1:
                #s=0

            s=s +  (x_hat-TrainingData[exp*T+t][target_g])**2    ## MSE  
            #s=s +  np.abs(x_hat-TrainingData[exp*T+t][target_g])     ## MAE
            XNext[max_neurons-1:]=TrainingData[exp*T+t,:].copy()
            XNext[0:NumNeuron-1]=update[1:]

    error=(s/(T-1)).copy()                    
    #Fitness=Fitness#+0.1*s_w  
    AIC= (T-1)*np.log(error)+2*K  ### AIC
    BIC=(T-1)*np.log(error)+np.log(T-1)*K
    if K>=(T-1):
        AICc=10^2
    else:
        AICc=AIC+2*K*(K+1)/(T-K-2)
    Fitness=AIC#error#AICc#*2**(K)#AIC
    return Fitness   





###########--------------------    tournamentSelection
def tournamentSelection(Fitness,PopSize,tournrate):
    tournsize=int(np.ceil(PopSize*tournrate))
    indices = range(PopSize)
    selected_indices=random.sample(indices,tournsize)
    selected_fitness=Fitness[selected_indices]
    index_min=np.argmin(selected_fitness)
    return index_min 




###########--------------------    Crossover
def Crossover(Parent1,Parent2,crossover_rate):   
    IndexNeuron1=random.randint(0,Parent1[0]-1)
    IndexNeuron2=IndexNeuron1

    Partitions1=[]
    Partitions2=[]

    k=1
    for i in range(0,int(Parent1[0])):
        NumInputs=int(Parent1[k])
        Partitions1.append(Parent1[k:k+1 +2 *NumInputs])
        k=k+1 +2 *NumInputs

    k=1
    for i in range(0,int(Parent2[0])):
        NumInputs=int(Parent2[k])
        Partitions2.append(Parent2[k:k+1 +2 *NumInputs])
        k=k+1 +2 *NumInputs
    #######----------------------------------
    
    x=Partitions1[IndexNeuron1].copy()
    y=Partitions2[IndexNeuron2].copy()

    Partitions1[IndexNeuron1]=y.copy()
    Partitions2[IndexNeuron2]=x.copy()
    #######-----------------------------------
    
    Child1=np.array([Parent1[0]])
    for i in range(0,int(Parent1[0])):
        Child1=np.hstack([Child1,Partitions1[i]])
    #Child1=np.hstack([Child1,Parent1[-1]])

    Child2=np.array([Parent2[0]])
    for i in range(0,int(Parent2[0])):
        Child2=np.hstack([Child2,Partitions2[i]])
    #Child2=np.hstack([Child2,Parent2[-1]])            
            
    return Child1,Child2




###########--------------------   Mutation


def Mutation(Parent,mutation_rate,w_rate,target_code): ## write the code for mutation on number of inputs
    NumNeurons=int(Parent[0])
    contextN=NumNeurons-1
    #decay=Parent[-1]
    
    Parent_new=np.array([NumNeurons])
    k_old=1
    for neuron in range(NumNeurons): # For each neuron in the network
        NumInputs=int(Parent[k_old])
        Inputnodes=[int(x) for x in Parent[k_old+1:k_old+1+NumInputs] ]
        InputWeights=Parent[k_old+1+NumInputs :k_old+1+ 2 *NumInputs  ]
        
        A=np.arange(max_input)
        A=np.delete(A,target_code)
        
        if neuron!=NumNeurons-1:   ### for neurons excluding the last one
            start=1
            for link in range(max_neurons-2,NumNeurons-2,-1):  #### No extra context nodes
                A=np.delete(A,link)
        else:                      ### last neuron
            start=0
            for link in range(max_neurons-2,-1,-1):          ### No context nodes
                A=np.delete(A,link)
                
        #############-----------------------------------------------
        for link in range(start,NumInputs):
            if random.random() < mutation_rate:     #mutate input nodes
                remain_A=[]
                remain_A=list(set(A)-set(Inputnodes))   #candidates for rewiring
                

                if len(remain_A)>0:
                    index=random.randint(0,len(remain_A)-1)
                    node=remain_A[index]
                    Inputnodes[link]=node
                    if random.random() < mutation_rate: #mutate input wights
                        if node<=(contextN-1):       
                            InputWeights[link]=max(0.01,np.random.normal(InputWeights[link],w_rate*B_max_w))
                        else:
                            InputWeights[link]=np.random.normal(InputWeights[link],w_rate*B_max_w*2)
                    else:  
                        if node<=(contextN-1): 
                            InputWeights[link]=abs(InputWeights[link]).copy()

                            
                            
        k_old=k_old+1 +2 *NumInputs 
        Parent_new=np.append(Parent_new,NumInputs ) 
        Parent_new=np.append(Parent_new,Inputnodes )
        Parent_new=np.append(Parent_new,InputWeights )
        
    ##########------------------------------------  add or delete links
    
    Parent_new2=np.array([NumNeurons])
    k_old=1
    for neuron in range(NumNeurons): # For each neuron in the network
        NumInputs=int(Parent_new[k_old])
        NumInputs_new=NumInputs
        Inputnodes=[int(x) for x in Parent_new[k_old+1:k_old+1+NumInputs] ]
        InputWeights=Parent_new[k_old+1+NumInputs :k_old+1+ 2 *NumInputs  ]
        
        A=np.arange(max_input)
        A=np.delete(A,target_code)
        
        if neuron!=NumNeurons-1:   ### for neurons excluding the last one
            start=1
            for link in range(max_neurons-2,NumNeurons-2,-1):  #### No extra context nodes
                A=np.delete(A,link)
        else:                      ### last neuron
            start=0
            for link in range(max_neurons-2,-1,-1):          ### No context nodes
                A=np.delete(A,link)    
        #########----------------------
        
        if random.random() < mutation_rate:     #mutate on number of input nodes
            remain_A=[]
            remain_A=list(set(A)-set(Inputnodes))   #candidates for rewiring
                
            if len(remain_A)>0:
                add=np.random.randint(2) ## 0 and 1 delete and add link respectively 
            else:     
                add=0
            ###########----------------   
            
            if add==1:  ##add link
                NumInputs_new=NumInputs+1
                index=random.randint(0,len(remain_A)-1)
                node=remain_A[index]
                Inputnodes=np.append(Inputnodes,node)
                    
                if node<=(contextN-1):
                    weight=B_max_w * np.random.uniform(0, 1, 1)
                else: 
                    weight=B_max_w * np.random.uniform(-1, 1, 1)
                    
                InputWeights=np.append(InputWeights,weight)
                #print(Inputnodes,InputWeights)
            else:  ### delete link
                if NumInputs-1>start:
                    index=random.randint(start,NumInputs-1)
                    #print(Inputnodes,InputWeights,index)

                    Inputnodes=np.delete(Inputnodes,index)  
                    InputWeights=np.delete(InputWeights,index)
                    #print(Inputnodes,InputWeights)
                    NumInputs_new=NumInputs-1
      
        k_old=k_old+1 +2 *NumInputs 
        Parent_new2=np.append(Parent_new2,NumInputs_new ) 
        Parent_new2=np.append(Parent_new2,Inputnodes )
        Parent_new2=np.append(Parent_new2,InputWeights )                
           
            
    #if random.random() <mutation_rate: ### mutation on decay rate
        #decay=max(0,np.random.normal(decay,w_rate*B_max_w))
    #Parent_new=np.append(Parent_new,decay)   
    return Parent_new





        
"""
def Mutation(Parent,mutation_rate,w_rate,target_code): ## write the code for mutation on number of inputs
    NumNeurons=int(Parent[0])
    contextN=NumNeurons-1
    #decay=Parent[-1]
    
    Parent_new=np.array([NumNeurons])
    nodes=range(max_input)
    k_old=1
    for neuron in range(NumNeurons): # For each neuron in the network
        NumInputs=int(Parent[k_old])
        Inputnodes=[int(x) for x in Parent[k_old+1:k_old+1+NumInputs] ]
        InputWeights=Parent[k_old+1+NumInputs :k_old+1+ 2 *NumInputs  ]
        
        A=np.arange(max_input)
        A=np.delete(A,target_code)
        
        if neuron!=NumNeurons-1:   ### for neurons excluding the last one
            start=1
            for link in range(max_neurons-2,NumNeurons-2,-1):  #### No extra context nodes
                A=np.delete(A,link)
        else:                      ### last neuron
            start=0
            for link in range(max_neurons-2,-1,-1):          ### No context nodes
                A=np.delete(A,link)
                
        #############-----------------------------------------------
        for link in range(start,NumInputs):
            if random.random() < mutation_rate:     #mutate input nodes
                remain_A=[]
                remain_A=list(set(A)-set(Inputnodes))   #candidates for rewiring
                    
                if len(remain_A)>0:
                    index=random.randint(0,len(remain_A)-1)
                    node=remain_A[index]
                    Inputnodes[link]=node
                    if random.random() < mutation_rate: #mutate input wights
                        if node<=(contextN-1):       
                            InputWeights[link]=max(0,np.random.normal(InputWeights[link],w_rate*B_max_w))
                        else:
                            InputWeights[link]=np.random.normal(InputWeights[link],w_rate*B_max_w*2)
                            
        k_old=k_old+1 +2 *NumInputs 
        Parent_new=np.append(Parent_new,NumInputs ) 
        Parent_new=np.append(Parent_new,Inputnodes )
        Parent_new=np.append(Parent_new,InputWeights )
    
    
    #if random.random() <mutation_rate: ### mutation on decay rate
        #decay=max(0,np.random.normal(decay,w_rate*B_max_w))
    #Parent_new=np.append(Parent_new,decay)   
    return Parent_new  

"""


###########--------------------    Weight Mutation
def WMutation(Parent,mutation_rate,w_rate):
    NumNeurons=int(Parent[0])
    contextN=NumNeurons-1
    Parent_new=Parent.copy()

    k_old=1
    for neuron in range(0,NumNeurons): #
        NumInputs=int(Parent[k_old])
        Inputnodes=Parent[k_old+1:k_old+1+NumInputs]
        InputWeights=Parent[k_old+1+NumInputs :k_old+1+ 2 *NumInputs]

        for link in range(0,NumInputs):
            if random.random() < mutation_rate:       #mutate input wights
                if Inputnodes[link] <=(contextN-1): 
                    InputWeights[link]=max(0.01,np.random.normal(InputWeights[link],w_rate*B_max_w))
                else:
                    InputWeights[link]=np.random.normal(InputWeights[link],w_rate*B_max_w*2)
                
        Parent_new[k_old+1+NumInputs:k_old+1+ 2 *NumInputs]=  InputWeights 
        k_old=k_old+1 +2 *NumInputs
        
    #if random.random() <mutation_rate: ### mutation on decay rate
        #Parent_new[-1]=max(0,np.random.normal(Parent_new[-1],w_rate*B_max_w))
        
    return Parent_new  




###########--------------------   Evaluation 
"""
def Evaluate(Network,Data,target,n_exp,Sc):
    X=0.01
    T=int(len(Data)/n_exp) 
    NumNeuron=int(Network[0])
    target_g=target-max_neurons+1
    #########-----------------
    
    A=np.zeros([NumNeuron,max_input])
    k=1
    for i in range(0,NumNeuron):
        NumInputs=int(Network[k])   
        Inputnodes=[int(x) for x in Network[k+1:k+1+NumInputs] ]      
        InputWeights=Network[k+1+NumInputs :k+1+ 2 *NumInputs  ]
        k=k+1 +2 *NumInputs   
        A[i,Inputnodes]=InputWeights.copy()
    
    #landa=Network[-1]
    #########---------------------
    
    x_hat=np.zeros(T*n_exp)#np.reshape(,[T,1])
    
    x_hat[0]=Data[0,target_g].copy()
    
    XNext=X*np.ones(max_input)#copy()
    XNext[max_neurons-1:]=Data[0,:].copy()
       
    for exp in range(0,n_exp):
        
        for t in range(1,T):
            update=np.tanh(np.dot(A,XNext))
            #update[0]=update[0]*(1-landa)
            x_hat[t]=update[0].copy()
        
            XNext[max_neurons-1:]=Data[t,:].copy()
            XNext[0:NumNeuron-1]=update[1:]
            
    return x_hat

"""


###########--------------------  Graph
def Gragh_pannet(Network,target_code):   

    dot = Digraph('unix', format='png')#filename='unix.gv',
    dot.body.append('size="6,6"')
    dot.node_attr.update(color='black')#, style='filled' lightblue2
    
    context=['c'+str(i+1) for i in range(max_neurons-1)]   #context nodes
    Obs=['g'+str(i+1) for i in range(obsN)]   #genes
    
    value=context+Obs
    key=range(obsN+max_neurons)
    assign = dict(zip(key, value))
      
    k=1
    NumNeuron=int(Network[0])
    for neuron in range(0,NumNeuron):
        NumInputs=int(Network[k])   
        Inputnodes=[int(x) for x in Network[k+1:k+1+NumInputs] ]
        InputWeights=Network[k+1+NumInputs :k+1+ 2 *NumInputs]
        k=k+1 +2 *NumInputs
                
        if neuron==0:
            for link in range(0,NumInputs):
                if InputWeights[link] <0: 
                    dot.edge(assign[Inputnodes[link]],assign[target_code],color='red')
                else:
                    dot.edge(assign[Inputnodes[link]],assign[target_code])
                
                
            #dot.node_attr.update(color='lightblue2')#, style='filled'
            #dot.edge('neuron'+str(neuron+1),assign[target_code]) 
        else:
            for link in range(0,NumInputs):
                if InputWeights[link] <0: 
                    dot.edge(assign[Inputnodes[link]],assign[neuron-1],color='red')
                else:
                    dot.edge(assign[Inputnodes[link]],assign[neuron-1])
            
            #dot.node_attr.update(color='black')#, style='filled'
            #dot.edge('neuron'+str(neuron+1),assign[neuron-1])
                
    dot.render('Regulation_g'+ str(target_code-max_neurons+2))  #+'.gv'
 
    ####################################################################
def origGragh(matrix_o): 

    dot = Digraph('unix', format='png')#filename='unix.gv',
    dot.body.append('size="6,6"')
    dot.node_attr.update(color='black')#, style='filled' lightblue2
    
    value=['g'+str(i+1) for i in range(obsN)]   #genes
    key=range(obsN)
    assign = dict(zip(key, value))
    
    for lag in range(max_neurons):
        for i in range(obsN):
            for j in range(obsN):
                if matrix_o[lag][i][j]==1:
                    dot.edge(assign[j],assign[i],label="lag="+str(lag))
                elif matrix_o[lag][i][j]==-1:
                    dot.edge(assign[j],assign[i],color='red',label="lag="+str(lag)) 
                elif matrix_o[lag][i][j]==2:   ## up and down regulation with same time delay
                    dot.edge(assign[j],assign[i],color='red',label="lag="+str(lag))
                    dot.edge(assign[j],assign[i],label="lag="+str(lag))     
    return dot                

##############################      
def pre_matrix(networks,max_neurons,obsN):  ### row i shows the relations of gene i 
    matrix_p=np.zeros((max_neurons,obsN, obsN))

    for gene in range(obsN):
        G = nx.DiGraph()
        net=networks[gene]
        #matrix_p[0][gene][gene]=-1
        all_input=[]
        k=1
        NumNeuron=int(net[0])
        for neuron in range(0,NumNeuron):
            NumInputs=int(net[k])   
            Inputnodes=[int(x) for x in net[k+1:k+1+NumInputs] ]
            InputWeights=net[k+1+NumInputs:k+1+ 2 *NumInputs]
            all_input.extend(Inputnodes)
            k=k+1 +2 *NumInputs

            if neuron==0:
                for link in range(0,NumInputs):
                    G.add_edges_from([(Inputnodes[link],gene+max_neurons-1)],weight=InputWeights[link])
            else:
                for link in range(0,NumInputs):
                    G.add_edges_from([(Inputnodes[link],neuron-1)],weight=InputWeights[link])
          
        all_leafs=[x for x in all_input if x>=max_neurons-1]  
        leafs=list(set(all_leafs)) 
        for leaf in leafs:
            paths=nx.all_simple_paths(G, source=leaf, target=gene+max_neurons-1)
            
            for path in paths:
                lag=len(path)-2
                #print('g=',gene, 'path=',path)
                W=G[path[0]][path[1]]['weight'] ## all context nodes have positive weights, only weight of leaf node is effective
                index=leaf-max_neurons+1
                if matrix_p[lag][gene][index]==0: 
                    if W>0:
                        matrix_p[lag][gene][index]=1
                    else:
                        matrix_p[lag][gene][index]=-1
                elif matrix_p[lag][gene][index]==1: 
                    if W>0:
                        matrix_p[lag][gene][index]=1
                    else:
                        matrix_p[lag][gene][index]=2 ### up and down
                elif matrix_p[lag][gene][index]==-1:
                    if W>0:
                        matrix_p[lag][gene][index]=2 ### up and down
                    else:
                        matrix_p[lag][gene][index]=-1 
                elif matrix_p[lag][gene][index]==2:  
                    matrix_p[lag][gene][index]=2 ### up and down

    return matrix_p
"""
def pre_matrix(networks,max_neurons,obsN):  ### row i shows the relations of gene i 
    matrix_p=np.zeros((max_neurons,obsN, obsN))
    for gene in range(obsN):
        net=networks[gene]
        #matrix_p[0][gene][gene]=-1
        
        k=1
        NumNeuron=int(net[0])
        for neuron in range(0,NumNeuron):
            NumInputs=int(net[k])   
            Inputnodes=[int(x) for x in net[k+1:k+1+NumInputs] ]
            InputWeights=net[k+1+NumInputs:k+1+ 2 *NumInputs]
            k=k+1 +2 *NumInputs

            for link in range(0,NumInputs):
                ### c1,c2,...,c_{max_neurons-1},g1,g2,...
                ### 0, 1, ..., max_neurons-2   ,max_neurons-1,max_neurons, ...
                
                if Inputnodes[link]>=max_neurons-1: 
                    index1=Inputnodes[link]-max_neurons+1
                    if InputWeights[link]>=0:
                        matrix_p[neuron][gene][index1]=1
                    else:  
                        matrix_p[neuron][gene][index1]=-1 
                   
    return matrix_p

""" 
#########################   
def compact(matrix,max_neurons,obsN):
    new_matrix=np.zeros((obsN,obsN))
    
    for dim in range(max_neurons):        
        new_matrix=np.add(new_matrix,matrix[dim]) 
        
    Link_matrix=np.array(new_matrix)
    Link_matrix[Link_matrix>1]=1 
    #Link_matrix[Link_matrix<-1]=-1 
    return Link_matrix
    
######################### Link: gene pair and direction
def Link_TP_FP_FN(matrix_o,matrix_p,obsN ):
    TP=0
    FP=0
    FN=0
    
    A_matrix_o=np.absolute(matrix_o)
    A_matrix_p=np.absolute(matrix_p)
    
    C_A_matrix_o=compact(A_matrix_o,max_neurons,obsN)
    C_A_matrix_p=compact(A_matrix_p,max_neurons,obsN)
    
    for i in range(obsN):
        for j in range(obsN):
            if C_A_matrix_p[i][j]==1 and C_A_matrix_o[i][j]==1:
                TP+=1
            elif C_A_matrix_p[i][j]==1 and C_A_matrix_o[i][j]==0:
                FP+=1
            elif C_A_matrix_p[i][j]==0 and C_A_matrix_o[i][j]==1: 
                FN+=1
    return TP,FP,FN 
#########################     
def precision_recall(TP,FP,FN):                    
    if TP+FP==0:
        precision=0
    else:    
        precision=TP/float(TP+FP)
        
    if TP+FN==0:
        precision=0
    else:         
        recall=TP/float(TP+FN) 
        
    if precision==0 and recall==0:
        print('precision and recall are zero')
        F_score=0
    else:    
        F_score=2*precision* recall/float(precision+recall)  
        
    return  precision, recall, F_score 
    
######################### Delay: link and delay
def Delay_TP_FP_FN(matrix_o,matrix_p,max_neurons,obsN): 
    
    TP=0
    FP=0
    FN=0
    
    A_matrix_o=np.absolute(matrix_o)
    A_matrix_p=np.absolute(matrix_p)
    
    Delay_matrix_o=np.array(A_matrix_o)
    Delay_matrix_o[A_matrix_o>1]=1
    
    Delay_matrix_p=np.array(A_matrix_p)
    Delay_matrix_p[A_matrix_p>1]=1    
        

    
    for dim in range(max_neurons):
        for i in range(obsN):
            for j in range(obsN):
                if Delay_matrix_p[dim][i][j]==1 and Delay_matrix_o[dim][i][j]==1:
                    TP+=1
                elif Delay_matrix_p[dim][i][j]==1 and Delay_matrix_o[dim][i][j]==0:
                    FP+=1
                elif Delay_matrix_p[dim][i][j]==0 and Delay_matrix_o[dim][i][j]==1: 
                    FN+=1
    return TP,FP,FN 

######################### Effect: Link and sign
def Effect_TP_FP_FN(matrix_o,matrix_p,max_neurons,obsN):
    
    ####  array in C_matrix_o and C_matrix_p can be {0, 1, -1, 2}. 
    ####  2 means that there there are both up and down regulations.
    
    TP=0
    FP=0
    FN=0
    
    C_matrix_o=np.zeros((obsN,obsN))
    C_matrix_p=np.zeros((obsN,obsN))
    
    for i in range(obsN):
        for j in range(obsN):
            ary_o=[]
            ary_p=[]
            for k in range(max_neurons):                
                if matrix_o[k][i][j]!=0:               ## find all real time-delayed regulations of gene i by gene j: j->i
                    ary_o.append(matrix_o[k][i][j])
                if matrix_p[k][i][j]!=0:               ## find all predicted time-delayed regulations of gene i by gene j: j->i
                    ary_p.append(matrix_p[k][i][j])
     
            if len(ary_o)>0 and  abs(sum(ary_o))==len(ary_o) and sum(ary_o)>0:     ### All regulations are positive 
                C_matrix_o[i][j]=1
            elif len(ary_o)>0 and  abs(sum(ary_o))==len(ary_o) and sum(ary_o)<0:   ### All regulations are negative
                C_matrix_o[i][j]=-1
            elif len(ary_o)>0 and  abs(sum(ary_o))!=len(ary_o):                    ### both positive and negative regulationns
                C_matrix_o[i][j]=2    

                
            if len(ary_p)>0 and  abs(sum(ary_p))==len(ary_p) and sum(ary_p)>0:     ### All regulations are positive 
                C_matrix_p[i][j]=1
            elif len(ary_p)>0 and  abs(sum(ary_p))==len(ary_p) and sum(ary_p)<0:   ### All regulations are negative
                C_matrix_p[i][j]=-1
            elif len(ary_p)>0 and  abs(sum(ary_p))!=len(ary_p):                    ### both positive and negative regulationns
                C_matrix_p[i][j]=2                 
         
    
    for i in range(obsN):
        for j in range(obsN):
            if C_matrix_p[i][j]==1 and C_matrix_o[i][j]==1:
                TP+=1
            elif C_matrix_p[i][j]==-1 and C_matrix_o[i][j]==-1:   
                TP+=1
            elif C_matrix_p[i][j]==1 and C_matrix_o[i][j]==0:
                FP+=1
            elif C_matrix_p[i][j]==-1 and C_matrix_o[i][j]==0:
                FP+=1                  
            elif C_matrix_p[i][j]==0 and C_matrix_o[i][j]==1: 
                FN+=1
            elif C_matrix_p[i][j]==0 and C_matrix_o[i][j]==-1: 
                FN+=1      
                
            elif C_matrix_p[i][j]==1 and C_matrix_o[i][j]==-1: 
                FP+=1  
                FN+=1
            elif C_matrix_p[i][j]==-1 and C_matrix_o[i][j]==1: 
                FP+=1
                FN+=1   
            elif C_matrix_p[i][j]==2 and C_matrix_o[i][j]==1: 
                TP+=1
                FN+=1   
            elif C_matrix_p[i][j]==2 and C_matrix_o[i][j]==-1: 
                TP+=1
                FN+=1  
            elif C_matrix_p[i][j]==2 and C_matrix_o[i][j]==0: 
                FP+=2                 
            elif C_matrix_p[i][j]==1 and C_matrix_o[i][j]==2: 
                TP+=1
                FN+=1   
            elif C_matrix_p[i][j]==-1 and C_matrix_o[i][j]==2: 
                TP+=1
                FN+=1 
            elif C_matrix_p[i][j]==0 and C_matrix_o[i][j]==2: 
                FN+=2
        
    return TP,FP,FN 
   

### -----   generate matrix of regulations and weights of syntethic data under different time lags
def orig_matrix():
    initialization() 
    ## X(t+1)=AX(t)+BX(t-1)+CX(t-2)+...
    
    regulation_o=np.zeros((max_neurons,obsN, obsN))
    weight_o=np.zeros((max_neurons,obsN, obsN))
    

    ta_max=max_neurons     ###   maximm number of delay
    r_max=3  #int(np.ceil(obsN/4))      ###   maximum number of regulators
    
    a_min=0.5    ###   minimum cofficient
    a_max=1    ###   maximum cofficient
    
    ## simultanius direct and delayed regulations, no self regulation
    
    for i in range(obsN):
        n_regulators=random.randint(1,r_max)
        n_lags=np.random.randint(0, high=ta_max, size=n_regulators)

        A=np.arange(0,obsN)
        A=np.delete(A,i)      # no self regulation
        
        #s=np.random.randint(0,len(A), n_regulators)  # gene can be both direct and delayed regulator
        #indices=A[s]
        indices=random.sample(list(A), n_regulators)##  gene can be either direct or delayed regulator

        
        for j in range(n_regulators):
            index=indices[j]
            n_lag=n_lags[j]
            if random.random()>=0.5:
                weight_o[n_lag][i][index]=np.random.uniform(a_min, a_max)
                regulation_o[n_lag][i][index]=1
            else:
                weight_o[n_lag][i][index]=-np.random.uniform(a_min, a_max)
                regulation_o[n_lag][i][index]=-1    
    
    return  regulation_o,weight_o   
    
    
    

    