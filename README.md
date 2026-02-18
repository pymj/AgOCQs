# AgOCQs
AgOCQs is a methodology developed for automated authoring of Competency Questions (CQs) for ontology engineering purposes. Leveraging the power on Large Language models, domain corpus, linguistic abstraction and NLP technigues; AgOCQs paves the way for automatic development of ontological CQs and their re-usability within a given domain or sub-domain.

This repo contains initial codes and supporting files
# Recent update Notes:
- Switch from full finetunning to parameter based finetunning with PEFT LORA
- This is to allow for reproducibility of the finetunning process, critical for the solution.
- Memory and compute was carried out with AWS EC2 instance m7iLarge with 2vCPUs and 8 Gigs of Ram
- Finetunning time differes depending on the number of epochs used
- Already finetuned model with SQuAd dataset available
- Intrinsic metrics developed for assessment of CQs added (Answerability, Relevance, Coverage and Combined score metrics)
- Quality assessment carried out as a collective rather than individual CQs
- CQ fitness scale used in combine-score metric to determine suitability of CQs for ontology or would be ontology
- process allows for flexibility to incerase domain data for improved CQs fitness
- Result of template study can be found in the folder: template_flies
# Important Note: 
- Use the instructions below to recreate the environment.
-Installation should be carried out with the versions provided to avoid dependency issues. On your terminal window, do the following:

## install docker 
```
pip install docker
``` 
## Build docker container, and  installs requirements.txt at the same time
```
docker build -t myapp .
```
## Run container to load Jupeter lab environment which creates an enviroment
```
<!-- docker run -p 8888:8888 <name-of-image> -->

docker run -d --name myapp --restart unless-stopped -p 8888:8888 myapp:latest

```
## Start AgOCQs for automatic Competency Questions (CQs) generation
- Navigate to the Url where the code base will be opened.
- Double click on agocqs.ipynb to run the notebook with the current dataset.
- Select Run all cells from the dropdown to automatically generate CQs.
- Change data in inputText if you want to use your own data
- Your data must either be in PDF format or text files (.txt)
- Wastewater Data can be found in inputText/request folder and can be replaced with own data. Preferrably
# Next: 
## Start quality assessments with the CQ-Metrics
- Double click on CQ_metric_wastewater.ipynb to run the notebook to assess quality with the current wastewater dataset.
-  Select Run all cells from the dropdown to automatically run the intrinsic metrics and assess the  wastewater CQs
### Or for Covid-19 data
- Double click on CQ_metric_covid.ipynb to run the notebook to assess quality with the current Covid dataset.
- Select Run all cells from the dropdown to automatically run the intrinsic metrics and assess the Covid-19 CQs
## Outputs 
- outputs folder hold all results
- Inner folders distinguish the results of the domains 