FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NLTK_DATA=/usr/local/share/nltk_data

# Install OS deps early (spacy can need build tooling on some arches)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel

# Install your NLP libs
RUN pip install "numpy==1.26.4"

RUN pip install "spacy==3.5.0" nltk
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl

# RUN python -m spacy download en_core_web_sm
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')"

RUN pip install coreferee \
 && python -m coreferee install en

COPY AgOCQs/requirements.txt /app/requirements.txt

# KEY: install these WITHOUT deps so pip does NOT try to install tokenizers
RUN pip install --no-deps \
    torch==1.13.1 \
    transformers==4.27.0 \
    sentence-transformers==2.2.0 \
    fsspec==2023.12.2 \
    pytorch-lightning==1.2.2 \
    fastt5==0.0.5

# Now install everything else (no tokenizers inside requirements.txt)
RUN pip install -r /app/requirements.txt

COPY AgOCQs/model/t5  /app/model/t5
COPY AgOCQs/CLaROv2.csv /app/CLaROv2.csv
COPY AgOCQs/inputText /app/inputText
COPY AgOCQs/agocqs.ipynb /app/agocqs.ipynb

RUN pip install jupyter

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
