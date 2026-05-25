FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime

WORKDIR /srv/home/users/kalinchukd23cs/InfEstimation_benchmark

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    traker==0.3.2 \
    peft==0.18.1 \
    kronfluence==1.0.1 \
    rank-bm25==0.2.2 \
    matplotlib==3.10.7 \
    scikit-learn==1.7.2 \
    pandas==2.2.3 \
    transformers==5.0.0 \
    datasets==4.3.0
    
CMD ["python3"]

LABEL org.opencontainers.image.source="https://github.com/DarynaKalinchuk/InfEstimation_benchmark"
