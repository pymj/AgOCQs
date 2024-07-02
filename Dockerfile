FROM python:3.8.5
# Set the working directory in the container
RUN pip install spacy
RUN pip install nltk
RUN python -m spacy download en_core_web_sm
RUN [ "python3", "-c", "import nltk; nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')" ]
RUN pip install --upgrade pip
# ENV PATH="/app:${PATH}"
COPY AgOCQs/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY AgOCQs/model/t5  /app/model/t5
COPY AgOCQs/CLaROv2.csv /app/CLaROv2.csv
COPY AgOCQs/inputText /app/inputText
COPY AgOCQs/agocqs.ipynb /app/agocqs.ipynb
# COPY AgOCQs/inputText /app/inputText/request
WORKDIR /app
#COPY . /app
RUN pip install jupyter
RUN ls
# ENV TINI_VERSION v0.6.0
# ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
# RUN chmod +x /usr/bin/tini
# ENTRYPOINT ["/usr/bin/tini", "--"]
#expose container
EXPOSE 8888

# ENV NAME AgEnv
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]


