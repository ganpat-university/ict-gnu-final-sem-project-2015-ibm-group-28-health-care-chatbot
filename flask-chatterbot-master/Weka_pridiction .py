#!/usr/bin/env python
# coding: utf-8

# # file prediction
# 

# In[1]:



import weka.core.jvm as jvm
jvm.start()
from weka.core.converters import Loader
from weka.classifiers import Classifier
import weka.core.serialization as serialization

from weka.classifiers import PredictionOutput
from weka.classifiers import Evaluation
from weka.core.classes import Random
import pandas as pd


# In[2]:


loader = Loader(classname="weka.core.converters.ArffLoader")

data_dir = "D:\\Rahul\\HealthCare ChatBot\\healthcare-dataset-stroke-data\\"



objects = serialization.read_all("J48.model")
csr = Classifier(jobject=objects[0])
#print(classifier)


# In[3]:


f= open("instances.arff","a")

#enter attributes

st= "\nFemale,54,1,0,Yes,Private,Urban,72.93,35.7,'never smoked',?"
f.write(st)
f.close()


# In[4]:


f= open("instances.arff","r")
print(f.read())
f.close()


# In[10]:


from io import StringIO
output_results = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.CSV")
data1 = loader.load_file("instances.arff")
data1.class_is_last()
ev2 = Evaluation(data1)
ev2.test_model(csr,data1,output_results)
print("Class prediction: ",output_results.buffer_content()[-13:-10])
print("\n\n     Instance","     Actual","    Predicted")
print(output_results.buffer_content())
TESTDATA = StringIO("Instance,Actual,Predicted,"+output_results.buffer_content())
# jvm.stop()
x = pd.read_csv(TESTDATA)


# In[14]:


list(x.Predicted).pop().split(":")[1]

