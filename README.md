# Operationalizing Machine Learning

## Summary

This is the second project of the Udacity Azure ML Nanodegree. The goal of this project to predict if a client will subscribe to a term deposit of a Portuguese banking institution. [Bankmarketing Datset](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) was used to train the models. The models were trained using AutoML and among all the model, best Machine Learning model was selected on the basis of **Accuracy** deployed as an webservice to Azure Container Instance. Finally all the process was automated by creating and deploying a pipeline.



## Architectural Diagram
<img src = './img/arch.jpg' />

## Key Steps
#### Step 1:

Firstly, we have to register the dataset into the Azure ML Studio to run the Auto ML. I named the datasets as **Bank-marketing**

<img src = './img/datasets.png' />



#### Step 2:

Next step, now we can run auto ML experiments. During this experiments we need to do following

1. Create a computer cluster 
2. Run the experiment using classification and select the Explaining Best Model



When the run complete, we can get all the details including the best model.

<img src ='./img/train1.png' />



Like, this picture we can see the auto ml experiment is completed. We all so get the best Model. Here the best model is **VotingEnsemble** which works by combining the predictions from multiple models. And the accuracy 92%. 

If we move to models tab, we can see more information about differents for example the second best model **XGBoostClassifier** which accuracy is 91.56 %.

<img src = './img/train2.png' />



#### Step 3:

Now time for deployment, we have to deploy our best model. In azure one click deployment available. During deployment we need to enable **authentication** and we have to deploy in **Azure Container Instance**. When deployment is successful, we get additional details from deployment options.

<img src = './img/deploy.png' />

We can see in this image, the rest api endpoint is available. Azure also provide Swagger Option. To see the Swagger UI, we need to download the swagger.json and put in the same folder of these two file

1. swagger.sh
2. serve.py

After that we have to run this two command

```
bash swagger.sh
```

```
python serve.py
```

Now if we visit the http://127.0.0.1:85 we will a nice UI like this. But we have to replace the swagger json file.



<img src = './img/swagger.png'/>

**Note: Here I change the default port, I serve into port 85.**

```
docker run -p 85:8080 swaggerapi/swagger-ui
```



Using this swagger UI, we can easily get the information that we need to send with the http call, and also sample output from the Rest Server.

<img src='./img/swagger2.png'/>



#### Step 4:

Enable logging: To enable logging we will use python ML Azure SDK and enable logging. To do this we need to down the config.json file from the workspace. Then we can run the below python script.

```python
from azureml.core import Workspace
from azureml.core.webservice import Webservice

# Requires the config to be downloaded first to the current working directory
ws = Workspace.from_config()

# Set with the deployment name
name = "project2"

# load existing web service
service = Webservice(name=name, workspace=ws)

# enable application insight
service.update(enable_app_insights=True)

logs = service.get_logs()

for line in logs.split('\n'):
    print(line)
```

To run this, in the terminal we can use this command (here script name logs.py)

```
python logs.py
```

Output:

<img src = './img/logs.png' />

Now If we go back to the azure website and in the deployment, we will see the the Application Insights enabled and we have an option to see the insights by clicking the URL.

<img src='./img/deploy.png' /> 



#### Step 5:

Consume the API model: Now time for testing our deployed model. To do the testing we need the rest api endpoint and the api key. We can get the from deploy details (STEP 3) and to get the api key, click on the consume tab and there we will find a primary key.

<img src = './img/api.png' />



To do the testing, we will below script

```
import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://0340c046-6a84-4dfa-96b3-dc77a8c9e1fe.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'EfIa5jPEWhYmP7yS2iQEAbzA7Hh5NNuk'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "age": 17,
            "campaign": 1,
            "cons.conf.idx": -46.2,
            "cons.price.idx": 92.893,
            "contact": "cellular",
            "day_of_week": "mon",
            "default": "no",
            "duration": 971,
            "education": "university.degree",
            "emp.var.rate": -1.8,
            "euribor3m": 1.299,
            "housing": "yes",
            "job": "blue-collar",
            "loan": "yes",
            "marital": "married",
            "month": "may",
            "nr.employed": 5099.1,
            "pdays": 999,
            "poutcome": "failure",
            "previous": 1
          },
          {
            "age": 87,
            "campaign": 1,
            "cons.conf.idx": -46.2,
            "cons.price.idx": 92.893,
            "contact": "cellular",
            "day_of_week": "mon",
            "default": "no",
            "duration": 471,
            "education": "university.degree",
            "emp.var.rate": -1.8,
            "euribor3m": 1.299,
            "housing": "yes",
            "job": "blue-collar",
            "loan": "yes",
            "marital": "married",
            "month": "may",
            "nr.employed": 5099.1,
            "pdays": 999,
            "poutcome": "failure",
            "previous": 1
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
```

In this script, we are sending the request to our REST service with some data that required for prediction. To run this script, we can use bellow command in the terminal (script saved as endpoints.py)

```
python endpoints.py
```

<img src = './img/output.png' />



I also run the apache benchmark, to the server status And I found, the server working well, I get 0 failed response.

<img src = './img/brenchmark.png' />



#### Step 6:







## Screen Recording

*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate: