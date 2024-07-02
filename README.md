# AgOCQs
AgOCQs is a methodology developed for automated authoring of Competency Questions (CQs) for ontology engineering purposes. Leveraging the power on Large Language models, domain corpus, linguistic abstraction and NLP technigues; AgOCQs paves the way for automatic development of ontological CQs and their re-usability within a given domain or sub-domain.

This repo contains initial codes and supporting files. Initial results of human evaluation of the output of AgOCQs can be found in the survey_resultd_by_grp.csv 
# Important Note: 
Installation should be carried out with the versions provided to avoid dependency issues.
## install docker
```
pip install docker
```
## Build docker container
```
docker build -t <name-of-container> .
```
## Run container
```
docker run -p 8888:8888 <name-of-container>

```


