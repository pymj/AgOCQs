FROM python:3.8.5
# Set the working directory in the container
RUN pip install spacy
RUN pip install nltk
RUN python -m spacy download en_core_web_sm
RUN [ "python3", "-c", "import nltk; nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')" ]
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install jupyter
#expose container
EXPOSE 8888
ENV NAME AgEnv
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]


