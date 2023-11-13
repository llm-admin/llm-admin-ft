import sys
from typing import List, Union, Optional

import ray._private.usage.usage_lib
from ray import serve

from llmadmin.backend.server.app import LLMDeployment, RouterDeployment, ExperimentalDeployment, ApplicationDeployment, ApiServer
from llmadmin.backend.server.models import LLMApp, ServeArgs, FTApp
from llmadmin.backend.server.utils import parse_args, parse_args_ft
import uuid
import os
from llmadmin.backend.llm.ft import TransformersFT


# ray.init(address="auto")


def llm_server(args: Union[str, LLMApp, List[Union[LLMApp, str]]]):
    """Serve LLM Models

    This function returns a Ray Serve Application.

    Accepted inputs:
    1. The path to a yaml file defining your LLMApp
    2. The path to a folder containing yaml files, which define your LLMApps
    2. A list of yaml files defining multiple LLMApps
    3. A dict or LLMApp object
    4. A list of dicts or LLMApp objects

    You can use `serve.run` to run this application on the local Ray Cluster.

    `serve.run(llm_backend(args))`.

    You can also remove
    """
    models = parse_args(args)
    if not models:
        raise RuntimeError("No enabled models were found.")

    # For each model, create a deployment
    deployments = {}
    model_configs = {}
    for model in models:
        if model.model_config.model_id in model_configs:
            raise ValueError(
                f"Duplicate model_id {model.model_config.model_id} specified. "
                "Please ensure that each model has a unique model_id. "
                "If you want two models to share the same Hugging Face Hub ID, "
                "specify initialization.hf_model_id in the model config."
            )
        print("Initializing LLM app", model.json(indent=2))
        user_config = model.dict()
        deployment_config = model.deployment_config.dict()
        model_configs[model.model_config.model_id] = model

        deployment_config = deployment_config.copy()
        max_concurrent_queries = deployment_config.pop(
            "max_concurrent_queries", None
        ) or user_config["model_config"]["generation"].get("max_batch_size", 1)
        deployments[model.model_config.model_id] = LLMDeployment.options(
            name=model.model_config.model_id.replace("/", "--").replace(".", "_"),
            max_concurrent_queries=max_concurrent_queries,
            user_config=user_config,
            **deployment_config,
        ).bind()
    # test = []
    return RouterDeployment.bind(deployments, model_configs)

def llm_experimental(args: Union[str, LLMApp, List[Union[LLMApp, str]]]):
    """Serve LLM Models

    This function returns a Ray Serve Application.

    Accepted inputs:
    1. The path to a yaml file defining your LLMApp
    2. The path to a folder containing yaml files, which define your LLMApps
    2. A list of yaml files defining multiple LLMApps
    3. A dict or LLMApp object
    4. A list of dicts or LLMApp objects

    You can use `serve.run` to run this application on the local Ray Cluster.

    `serve.run(llm_backend(args))`.

    You can also remove
    """
    models = parse_args(args)
    if not models:
        raise RuntimeError("No enabled models were found.")

    # For model, create a deployment
    model = models[0]

    print("Initializing LLM app", model.json(indent=2))
    user_config = model.dict()
    deployment_config = model.deployment_config.dict()
    deployment_config = deployment_config.copy()
    max_concurrent_queries = deployment_config.pop(
        "max_concurrent_queries", None
    ) or (user_config["model_config"]["generation"].get("max_batch_size", 1) if user_config["model_config"]["generation"] else 1)
    
    deployment = LLMDeployment.options(
        name=model.model_config.model_id.replace("/", "--").replace(".", "_"),
        max_concurrent_queries=max_concurrent_queries,
        user_config=user_config,
        **deployment_config,
    ).bind()

    serve_conf = {
        "name": model.model_config.model_id.replace("/", "--").replace(".", "_"),
    }

    return (ExperimentalDeployment.bind(deployment, model), serve_conf)


def llm_flow(input: dict, tweaks: Optional[dict] = None, build=True):
    """Serve LLM Models

    This function returns a Ray Serve Application.

    Accepted inputs:
    the dict, actually it's a json object

    You can use `serve.run` to run this application on the local Ray Cluster.

    `serve.run(llm_backend(args))`.

    You can also remove
    """
    # If input is a dictionary, assume it's a JSON object
    if isinstance(input, dict):
        flow_graph = input
    else:
        raise TypeError(
            "Input must be a JSON object (dict)"
        )

    return ApplicationDeployment.bind(flow_graph, tweaks)


def llm_application(args):
    """This is a simple wrapper for LLM Server
    That is compatible with the yaml config file format

    """
    serve_args = ServeArgs.parse_obj(args)
    return llm_server(serve_args.models)


def run(*models: Union[LLMApp, str]):
    """Run the LLM Server on the local Ray Cluster

    Args:
        *models: A list of LLMApp objects or paths to yaml files defining LLMApps

    Example:
       run("models/")           # run all models in the models directory
       run("models/model.yaml") # run one model in the model directory
       run({...LLMApp})         # run a single LLMApp
       run("models/model1.yaml", "models/model2.yaml", {...LLMApp}) # mix and match
    """
    app = llm_server(list(models))
    ray._private.usage.usage_lib.record_library_usage("llmadmin")
    serve.run(app, name = "cmp_models_default", route_prefix = "/cmp_models_default", host="0.0.0.0")

def run_experimental(*models: Union[LLMApp, str]):
    """Run the LLM Server on the local Ray Cluster

    Args:
        model: A LLMApp objects or paths to yaml files defining LLMApps

    Example:
       run("models/")           # run all models in the models directory
       run("models/model.yaml") # run one model in the model directory
       run({...LLMApp})         # run a single LLMApp
       run("models/model1.yaml", "models/model2.yaml", {...LLMApp}) # mix and match
    """
    app = llm_experimental(list(models))
    serve_conf = app[1]
    ray._private.usage.usage_lib.record_library_usage("llmadmin")
    serve.run(app[0], host="0.0.0.0", name = serve_conf["name"], route_prefix = "/" + serve_conf["name"])


def del_experimental(app_name: str):
    serve.delete(app_name, _blocking = True)


def run_application(flow: dict, tweaks: Optional[dict] = None):
    """Run the LLM Server on the local Ray Cluster

    Args:
        *models: A list of LLMApp objects or paths to yaml files defining LLMApps

    Example:
       run("models/")           # run all models in the models directory
       run("models/model.yaml") # run one model in the model directory
       run({...LLMApp})         # run a single LLMApp
       run("models/model1.yaml", "models/model2.yaml", {...LLMApp}) # mix and match
    """
    app = llm_flow(flow, tweaks)

    ray._private.usage.usage_lib.record_library_usage("llmadmin")
    serve.run(app, host="0.0.0.0")

def start_apiserver():
    """Run the API Server on the local Ray Cluster

    Args:
        *host: The host ip to run.
        *port: The port to run.     
    
    """
    app = ApiServer.bind()
   
    ray._private.usage.usage_lib.record_library_usage("llmadmin")
    #ray.init(address="auto")
    serve.run(app, host="0.0.0.0",name="apiserver")

def run_ft(ft: Union[FTApp, str]):
    """Run the LLM Server on the local Ray Cluster

    Args:
        model: A LLMApp objects or paths to yaml files defining LLMApps

    Example:
       run("models/model.yaml") # run one model in the model directory
       run(FTApp)         # run a single LLMApp
    """

    ft = parse_args_ft(ft)
    if not ft:
        raise RuntimeError("No valiabled fine tune defination were found.")
    
    print(ft)

    ray._private.usage.usage_lib.record_library_usage("llmadmin")

    runner = TransformersFT(ft)
    runner.train()
    
def run_comparation():
    """Run the LLM Server on the local Ray Cluster

    Args:
        *models: A list of LLMApp objects or paths to yaml files defining LLMApps

    Example:
       run("models/")           # run all models in the models directory
       run("models/model.yaml") # run one model in the model directory
       run({...LLMApp})         # run a single LLMApp
       run("models/model1.yaml", "models/model2.yaml", {...LLMApp}) # mix and match
    """
    from llmadmin.frontend.app import LLMAdminFrontend
    app = LLMAdminFrontend.options(ray_actor_options={"num_cpus": 1}, name="LLMAdminFrontend").bind("http://127.0.0.1:8000/cmp_models_default")
    ray._private.usage.usage_lib.record_library_usage("llmadmin")
    serve.run(app, host="0.0.0.0", name = "cmp_default", route_prefix="/cmp_default")


if __name__ == "__main__":
    run(*sys.argv[1:])


