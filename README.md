# LLMAdmin - Study stochastic parrots in the wild


## Deploy LLMAdmin 

The guide below walks you through a minimal installation of LLMAdmin

### Install the requirements
```
pip install "ray[air]"
pip install .[backend]
pip install .[frontend] 
```

### Start a Ray Cluster locally

```
pip install -U "ray>=2.4.0"
ray start --head --port=6379
```

### Enable the LLMAdmin
Start by cloning this repo to your local machine.

```shell
git clone https://github.com/llm-admin/llm-admin.git
cd llm-admin
pip install .

```

### Deploy a hugging face model
You can deploy any model in the `models` directory of this repo, 
or define your own model YAML file and run that instead. for example:
```
 llm-executor run-experimental --model=models/text-generation--gpt2.yaml
```
will deploy `gpt2` in the ray cluster
check `llm-executor --help` to get more details


# LLMAdmin Reference


## Using the LLMAdmin CLI

LLMAdmin comes with a CLI that allows you to interact with the backend directly, without
using the Gradio frontend.
Installing LLMAdmin as described earlier will install the `llm-executor` CLI as well.
You can get a list of all available commands by running `llm-executor --help`.

Currently, `llm-executor` supports a few basic commands, all of which can be used with the
`--help` flag to get more information:

```shell
# Get a list of all available models in LLMAdmin
llm-executor models

# Query a model with a list of prompts
llm-executor query --model <model-name> --prompt <prompt_1> --prompt <prompt_2>

# Run a query on a text file of prompts
llm-executor query  --model <model-name> --prompt-file <prompt-file>

# Start a new model in LLMAdmin from provided configuration
llm-executor run <model>
```

## LLMAdmin Model Registry

LLMAdmin allows you to easily add new models by adding a single configuration file.
To learn more about how to customize or add new models, 
see the [LLMAdmin Model Registry](models/README.md).



## Launch job by API:
```
import requests
import json
import time

resp = requests.post(
    "http://127.0.0.1:8265/api/jobs/",
    json={
        "entrypoint": "/home/ray/anaconda3/bin/python /home/ray/anaconda3/bin/llm-executor run-experimental --model models/amazon--LightGPT.yaml",
        "runtime_env": {},
        "job_id": None,
        "metadata": {"job_submission_id": "123"}
    }
)
rst = json.loads(resp.text)
job_id = rst["job_id"]
print(job_id)
```

Document for API:
https://docs.ray.io/en/latest/cluster/running-applications/job-submission/api.html#/paths/~1api~1jobs/post


```
ray up --no-config-cache deploy/ray/llmadmin-cluster.yaml
```

```
ray down deploy/ray/llmadmin-cluster.yaml
```
