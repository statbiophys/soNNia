import numpy as np 
from sonia.sonia_leftpos_rightpos import SoniaLeftposRightpos
from tqdm import tqdm
import tensorflow.keras as keras

class Linear(object):
    '''
    Logistic Regression over sonia features for epitope prediction
    
    '''
    def __init__(self,sonia_model=None,include_indep_genes=False,include_joint_genes=True):
        if not sonia_model is None:
            self.sonia_model=sonia_model
        else: self.sonia_model=SoniaLeftposRightpos(include_indep_genes=include_indep_genes,
                                                    include_joint_genes=include_joint_genes)
        self.input_size=len(self.sonia_model.features)
        self.update_model_structure(initialize=True)

    def update_model_structure(self,input_layer=None,output_layer=None,initialize=False):
        if initialize:
            input_layer=keras.layers.Input(shape=(self.input_size,))
            output_layer=keras.layers.Dense(2, activation='softmax')(input_layer)
        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=["accuracy"])
    
    def encode(self,seq_features):
        length_input=len(self.sonia_model.features)
        data=np.array(seq_features)
        data_enc = np.zeros((len(data), length_input), dtype=np.int8)
        for i in range(len(data_enc)): data_enc[i][data[i]] = 1
        return data_enc
    
    def fit(self,x,y,val_split=0):
        x_enc=[self.sonia_model.find_seq_features(d) for d in tqdm(x)]
        X=self.encode(x_enc)
        shuffle=np.arange(len(y))
        np.random.shuffle(shuffle)
        self.history=self.model.fit(X[shuffle], y[shuffle],batch_size=300, epochs=100, verbose=0, validation_split=val_split)
    
    def predict(self,x):
        x_enc=[self.sonia_model.find_seq_features(d) for d in tqdm(x)]
        X=self.encode(x_enc)
        return self.model.predict(X)

class SoniaRatio(object):
    '''
    Sonia classifier as log likelihood ratio.
    '''
    def __init__(self,include_indep_genes=False,include_joint_genes=True):
        
        self.sonia_model_positive=SoniaLeftposRightpos(include_indep_genes=include_indep_genes,
                                                    include_joint_genes=include_joint_genes)
        self.sonia_model_negative=SoniaLeftposRightpos(include_indep_genes=include_indep_genes,
                                                    include_joint_genes=include_joint_genes)
    
    def fit(self,x,y,val_split=0,epochs=30,gen_seqs=None):
        sel=y[:,0].astype(np.bool)
        class1=x[sel]
        class2=x[np.logical_not(sel)]
        if gen_seqs is None:
            self.sonia_model_positive.add_generated_seqs(np.min([len(class1)*100,int(5e5)]))
            self.sonia_model_negative.add_generated_seqs(np.min([len(class2)*100,int(5e5)]))                                   
            self.sonia_model_positive.update_model(add_data_seqs=list(class1))
            self.sonia_model_negative.update_model(add_data_seqs=list(class2))
        else:
            self.sonia_model_positive.update_model(add_data_seqs=list(class1),add_gen_seqs=list(gen_seqs[:len(class1)*100]))
            self.sonia_model_negative.update_model(add_data_seqs=list(class2),add_gen_seqs=list(gen_seqs[:len(class2)*100]))
        self.sonia_model_negative.infer_selection(epochs=epochs,batch_size=int(5e3))
        self.sonia_model_positive.infer_selection(epochs=epochs,batch_size=int(5e3))
                                                     
    def predict(self,x):
        x_enc=[self.sonia_model_positive.find_seq_features(d) for d in tqdm(x)]
        q1=np.exp(-self.sonia_model_positive.compute_energy(x_enc))/self.sonia_model_positive.Z
        q2=np.exp(-self.sonia_model_negative.compute_energy(x_enc))/self.sonia_model_negative.Z
        return np.log(q2)-np.log(q1)