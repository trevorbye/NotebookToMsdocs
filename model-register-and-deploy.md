---
title: Deploy a model with Azure Machine Learning
titleSuffix: Azure Machine Learning
description: 
services: machine-learning
ms.service: machine-learning
ms.subservice: core
ms.topic: conceptual
ms.author: aashishb
author: 
ms.reviewer: aashishb
ms.date: 02/13/2020 
---

# Register model and deploy as webservice in ACI

Following this notebook, you will:

 - Learn how to register a model in your Azure Machine Learning Workspace.
 - Deploy your model as a web service in an Azure Container Instance.

## Prerequisites

If you are using an Azure Machine Learning Notebook VM, you are all set. Otherwise, make sure you go through the [configuration notebook](../../../configuration.ipynb) to install the Azure Machine Learning Python SDK and create a workspace.


```python
import azureml.core


# Check core SDK version number.
print('SDK version:', azureml.core.VERSION)
```

## Initialize workspace

Create a [Workspace](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.workspace%28class%29?view=azure-ml-py) object from your persisted configuration.


```python
from azureml.core import Workspace


ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')
```

## Register input and output datasets

For this example, we have provided a small model (`sklearn_regression_model.pkl` in the notebook's directory) that was trained on scikit-learn's [diabetes dataset](https://scikit-learn.org/stable/datasets/index.html#diabetes-dataset). Here, you will register the data used to create this model in your workspace.


```python
from azureml.core import Dataset


datastore = ws.get_default_datastore()
datastore.upload_files(files=['./features.csv', './labels.csv'],
                       target_path='sklearn_regression/',
                       overwrite=True)

input_dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, 'sklearn_regression/features.csv')])
output_dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, 'sklearn_regression/labels.csv')])
```

## Register model

Register a file or folder as a model by calling [Model.register()](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model.model?view=azure-ml-py#register-workspace--model-path--model-name--tags-none--properties-none--description-none--datasets-none--model-framework-none--model-framework-version-none--child-paths-none-).

In addition to the content of the model file itself, your registered model will also store model metadata -- model description, tags, and framework information -- that will be useful when managing and deploying models in your workspace. Using tags, for instance, you can categorize your models and apply filters when listing models in your workspace. Also, marking this model with the scikit-learn framework will simplify deploying it as a web service, as we'll see later.


```python
from azureml.core import Model
from azureml.core.resource_configuration import ResourceConfiguration


model = Model.register(workspace=ws,
                       model_name='my-sklearn-model',                # Name of the registered model in your workspace.
                       model_path='./sklearn_regression_model.pkl',  # Local file to upload and register as a model.
                       model_framework=Model.Framework.SCIKITLEARN,  # Framework used to create the model.
                       model_framework_version='0.19.1',             # Version of scikit-learn used to create the model.
                       sample_input_dataset=input_dataset,
                       sample_output_dataset=output_dataset,
                       resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
                       description='Ridge regression model to predict diabetes progression.',
                       tags={'area': 'diabetes', 'type': 'regression'})

print('Name:', model.name)
print('Version:', model.version)
```

## Deploy model

Deploy your model as a web service using [Model.deploy()](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model.model?view=azure-ml-py#deploy-workspace--name--models--inference-config--deployment-config-none--deployment-target-none-). Web services take one or more models, load them in an environment, and run them on one of several supported deployment targets. For more information on all your options when deploying models, see the [next steps](#Next-steps) section at the end of this notebook.

For this example, we will deploy your scikit-learn model to an Azure Container Instance (ACI).

### Use a default environment (for supported models)

The Azure Machine Learning service provides a default environment for supported model frameworks, including scikit-learn, based on the metadata you provided when registering your model. This is the easiest way to deploy your model.

Even when you deploy your model to ACI with a default environment you can still customize the deploy configuration (i.e. the number of cores and amount of memory made available for the deployment) using the [AciWebservice.deploy_configuration()](https://docs.microsoft.com/python/api/azureml-core/azureml.core.webservice.aci.aciwebservice#deploy-configuration-cpu-cores-none--memory-gb-none--tags-none--properties-none--description-none--location-none--auth-enabled-none--ssl-enabled-none--enable-app-insights-none--ssl-cert-pem-file-none--ssl-key-pem-file-none--ssl-cname-none--dns-name-label-none--). Look at the "Use a custom environment" section of this notebook for more information on deploy configuration.

**Note**: This step can take several minutes.


```python
from azureml.core import Webservice
from azureml.exceptions import WebserviceException


service_name = 'my-sklearn-service'

# Remove any existing service under the same name.
try:
    Webservice(ws, service_name).delete()
except WebserviceException:
    pass

service = Model.deploy(ws, service_name, [model])
service.wait_for_deployment(show_output=True)
```

After your model is deployed, perform a call to the web service using [service.run()](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice%28class%29?view=azure-ml-py#run-input-).


```python
import json


input_payload = json.dumps({
    'data': [
        [ 0.03807591,  0.05068012,  0.06169621, 0.02187235, -0.0442235,
         -0.03482076, -0.04340085, -0.00259226, 0.01990842, -0.01764613]
    ],
    'method': 'predict'  # If you have a classification model, you can get probabilities by changing this to 'predict_proba'.
})

output = service.run(input_payload)

print(output)
```

When you are finished testing your service, clean up the deployment with [service.delete()](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice%28class%29?view=azure-ml-py#delete--).


```python
service.delete()
```

### Use a custom environment

If you want more control over how your model is run, if it uses another framework, or if it has special runtime requirements, you can instead specify your own environment and scoring method. Custom environments can be used for any model you want to deploy.

Specify the model's runtime environment by creating an [Environment](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.environment%28class%29?view=azure-ml-py) object and providing the [CondaDependencies](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.conda_dependencies.condadependencies?view=azure-ml-py) needed by your model.


```python
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies


environment = Environment('my-sklearn-environment')
environment.python.conda_dependencies = CondaDependencies.create(pip_packages=[
    'azureml-defaults',
    'inference-schema[numpy-support]',
    'joblib',
    'numpy',
    'scikit-learn'
])
```

When using a custom environment, you must also provide Python code for initializing and running your model. An example script is included with this notebook.


```python
with open('score.py') as f:
    print(f.read())
```

Deploy your model in the custom environment by providing an [InferenceConfig](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model.inferenceconfig?view=azure-ml-py) object to [Model.deploy()](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.model.model?view=azure-ml-py#deploy-workspace--name--models--inference-config--deployment-config-none--deployment-target-none-). In this case we are also using the [AciWebservice.deploy_configuration()](https://docs.microsoft.com/python/api/azureml-core/azureml.core.webservice.aci.aciwebservice#deploy-configuration-cpu-cores-none--memory-gb-none--tags-none--properties-none--description-none--location-none--auth-enabled-none--ssl-enabled-none--enable-app-insights-none--ssl-cert-pem-file-none--ssl-key-pem-file-none--ssl-cname-none--dns-name-label-none--) method to generate a custom deploy configuration.

**Note**: This step can take several minutes.


```python
from azureml.core import Webservice
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.exceptions import WebserviceException


service_name = 'my-custom-env-service'

# Remove any existing service under the same name.
try:
    Webservice(ws, service_name).delete()
except WebserviceException:
    pass

inference_config = InferenceConfig(entry_script='score.py', environment=environment)
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(workspace=ws,
                       name=service_name,
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=aci_config)
service.wait_for_deployment(show_output=True)
```

After your model is deployed, make a call to the web service using [service.run()](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice%28class%29?view=azure-ml-py#run-input-).


```python
import json


input_payload = json.dumps({
    'data': [
        [ 0.03807591,  0.05068012,  0.06169621, 0.02187235, -0.0442235,
         -0.03482076, -0.04340085, -0.00259226, 0.01990842, -0.01764613]
    ]
})

output = service.run(input_payload)

print(output)
```

When you are finished testing your service, clean up the deployment with [service.delete()](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.webservice%28class%29?view=azure-ml-py#delete--).


```python
service.delete()
```

### Model profiling

You can also take advantage of the profiling feature to estimate CPU and memory requirements for models.

```python
profile = Model.profile(ws, "profilename", [model], inference_config, test_sample)
profile.wait_for_profiling(True)
profiling_results = profile.get_results()
print(profiling_results)
```

### Model packaging

If you want to build a Docker image that encapsulates your model and its dependencies, you can use the model packaging option. The output image will be pushed to your workspace's ACR.

You must include an Environment object in your inference configuration to use `Model.package()`.

```python
package = Model.package(ws, [model], inference_config)
package.wait_for_creation(show_output=True)  # Or show_output=False to hide the Docker build logs.
package.pull()
```

Instead of a fully-built image, you can also generate a Dockerfile and download all the assets needed to build an image on top of your Environment.

```python
package = Model.package(ws, [model], inference_config, generate_dockerfile=True)
package.wait_for_creation(show_output=True)
package.save("./local_context_dir")
```

## Next steps

 - To run a production-ready web service, see the [notebook on deployment to Azure Kubernetes Service](../production-deploy-to-aks/production-deploy-to-aks.ipynb).
 - To run a local web service, see the [notebook on deployment to a local Docker container](../deploy-to-local/register-model-deploy-local.ipynb).
 - For more information on datasets, see the [notebook on training with datasets](../../work-with-data/datasets-tutorial/train-with-datasets/train-with-datasets.ipynb).
 - For more information on environments, see the [notebook on using environments](../../training/using-environments/using-environments.ipynb).
 - For information on all the available deployment targets, see [&ldquo;How and where to deploy models&rdquo;](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-deploy-and-where#choose-a-compute-target).
