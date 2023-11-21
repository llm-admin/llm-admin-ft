from setuptools import find_packages, setup

setup(
    name="llm-executor",
    version="0.0.2",
    description="A tool to deploy and query LLMs",
    packages=find_packages(include="llmadmin*"),
    include_package_data=True,
    package_data={"llmadmin": ["models/*"]},
    entry_points={
        "console_scripts": [
            "llm-executor=llmadmin.api.cli:app",
        ]
    },
    install_requires=["typer>=0.9", "rich"],
    extras_require={
        # TODO(tchordia): test whether this works, and determine how we can keep requirements
        # in sync
        "backend": [
            "async_timeout",
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "torchvision>=0.15.2",
            "accelerate",
            "transformers>=4.25.1",
            "datasets",
            "ftfy",
            "tensorboard",
            "sentencepiece",
            "Jinja2",
            "numexpr>=2.7.3",
            "hf_transfer",
            "evaluate",
            "bitsandbytes",
            "deepspeed @ git+https://github.com/Yard1/DeepSpeed.git@aviary",
            "numpy<1.24",
            "ninja",
            "protobuf<3.21.0",
            "optimum @ git+https://github.com/huggingface/optimum.git",
            "torchmetrics",
            "safetensors",
            "pydantic==1.10.7",
            "einops",
            "markdown-it-py[plugins]",
            "scipy==1.11.1",
            "jieba==0.42.1",
            "rouge_chinese==1.0.3",
            "nltk==3.8.1",
            "sqlalchemy==1.4.41",
            "typing-extensions==4.5.0",
            "linkify-it-py==2.0.2",
            "markdown-it-py==2.2.0",
            "gradio",
            "httpx[socks]==0.23.3"
        ],
        "frontend": [
            "gradio",
            "aiorwlock",
            "ray",
            "pymongo",
            "pandas",
            "boto3",
        ],
        "dev": [
            "pre-commit",
            "ruff==0.0.270",
            "black==23.3.0",
        ],
        "test": [
            "pytest",
        ],
        "docs": [
            "mkdocs-material",
        ],
    },
    dependency_links=["https://download.pytorch.org/whl/cu118"],
    python_requires=">=3.8",
)
