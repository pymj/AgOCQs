# AgOCQs
AgOCQs is a methodology developed for automated authoring of Competency Questions (CQs) for ontology engineering purposes. Leveraging the power on Large Language models, domain corpus, linguistic abstraction and NLP technigues; AgOCQs paves the way for automatic development of ontological CQs and their re-usability within a given domain or sub-domain.

This repo contains initial codes and supporting files. Initial results of human evaluation of the output of AgOCQs can be found in the survey_resultd_by_grp.csv 
# Important Note: 
Installation should be carried out with the versions provided to avoid dependency issues. On your terminal window, do the following:
## install docker
```
pip install docker
``` 
## Build docker container
```
docker build -t <name-of-image> .
```
## Run container
```
docker run -p 8888:8888 <name-of-image>

```
Navigate to the Url where the code base will be opened. Double click on agocqs.ipynb to run the notebook with the current dataset. 
Data used can be found in inputText/request folder and can be replaced with own data. Preferrably, used pdf files for easy adaptation.