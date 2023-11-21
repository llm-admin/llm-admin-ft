# build from Dockerfile-base
#FROM docker.io/vincentpli/aviary:base-0.0.1
FROM wanghaihui/llmray:base-0.0.1

COPY "./dist" "/home/ray/dist"
RUN cd /home/ray/dist && pip install "$(ls *.whl | head -n1)[backend, frontend]"

# The build context should be the root of the repo
# So this gives the model definitions
COPY "./models" "/home/ray/models"

ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN echo "Testing llmadmin install" && python -c "import llmadmin.backend"

RUN pip cache purge && conda clean -a && rm -rf ~/.cache
