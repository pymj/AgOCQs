import os
import pandas as pd
from procee_data import TextPreprocessor, readTextFile, format
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import T5ForConditionalGeneration, T5Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
# Generate questions with our data using trained model
def generate(df2, model, tokenizer):
    
    context = df2['sentences_new']
    text = "context: "+context
    input_ids =[]
    attention_mask = []
    for x in (text):
      encoding = tokenizer.encode_plus(x,max_length =512, padding=True, return_tensors="pt")
      print (encoding.keys())
      input_ids.append(encoding["input_ids"].to(device))
      attention_mask.append(encoding["attention_mask"].to(device))
      input_ids, attention_mask = encoding["input_ids"].to(device),encoding["attention_mask"].to(device)
    data = dict(zip(input_ids, attention_mask))

    model.eval()
    cqs = []

    for x in (text):
      encoding = tokenizer.encode_plus(x,max_length =512, padding=True, return_tensors="pt")
      print (encoding.keys())
      input_ids, attention_mask = encoding["input_ids"].to(device),encoding["attention_mask"].to(device)
      beam_outputs = model.generate(
          input_ids=input_ids,attention_mask=attention_mask,
          max_length=72,
          early_stopping=True,
          num_beams=5,
          num_return_sequences=3
      )

      for beam_output in beam_outputs:
          sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
          cqs.append(sent)
    return cqs

# Post processing of generated questions
def postprocess(cqs):
    questions =[]
    with open(os.getcwd() + "/output/CQs.txt", "w") as file:
        file.write(str(cqs))
    with open(os.getcwd() + "/output/CQs.txt","r",encoding="utf-8") as f:
        for line in f.readlines():
            questions.append(line)
    new_questions = []
    for i in (questions):
        x = str(i).split(":")
        x = str(i).replace("question", '')
        new_questions.append(x)
    new_questions2 = str(new_questions).split(",")
    dcqs =pd.DataFrame(new_questions2)
    dcqs.columns = ['CQs']
    dcqs['CQs'] = dcqs['CQs'].map(lambda x: x.replace(":", '').replace("'", '').replace('[', '').replace(']', '').replace('"', ''))
    
    for i in range(0, len(dcqs)):
        dcqs.at[i, 'newCQs'] =re.sub(r"\s\([A-Z][a-z]+,\s[A-Z][a-z]?\.[^\)]*,\s\d{4}\)","",str(dcqs.loc[i,'CQs']))
    with open(os.getcwd() + "/output/cq_output.txt", "w") as f:
        f.write("\n".join(dcqs["newCQs"]))
    return dcqs
    # # dcqs.head()

# Abstract pattern extraction
def mark_chunk(cq, spans, chunktype, offset, counter):
    for (start, end) in spans:  # for each span of EC/PC candidate
        cq = cq[:start - offset] + chunktype + str(counter) +\
            cq[end - offset:]  # substitute that candidate with EC/PC marker
        offset += (end - start) - len(chunktype) - 1  # check by how much the total length of CQ changed
    return cq, offset


def extract_EC_chunks(cq):
    """
        Find EC chunks and replace their occurences with EC tags
    """

    def _get_EC_span_reject_wh_starters(chunk):
        """
            By default, SpaCy treats question words (wh- pronouns starting questions: where, what,...)
            as nouns, so often if questions starts with wh- pronoun + noun (like: What software is the best?)
            the whole "what software" is interpreted as EC chunk - this function tries to fix that issue by
            omitting wh- word if EC candidate consists of multiple tokens, and first token in question is wh- word.
            The same issue occurs for "How" starter.
            Moreover, chunks extracted with SpaCy enclose words like 'any', 'some' which are important for us, so
            they shouldn't be substituted with 'EC' marker. Thus we remove such words if they prepend the EC.
            The result is returned as the span - the position of beginning of the fixed EC and position of end.
        """
        if (len(chunk) > 1 and
                (chunk[0].text.lower().startswith("wh")) or
                (chunk[0].text.lower() == 'how')):
            chunk = chunk[1:]

        if chunk[0].text.lower() in ['any', 'some', 'many', 'well', 'its']:
            chunk = chunk[1:]

        return (chunk.start_char, chunk.end_char)

    doc = nlp(cq)

    #  thins classified as ECs which shouldn't be interpreted that way
    rejecting_ec = ["what", "which", "when", "where", "who", "type", "types",
                    "kinds", "kind", "category", "categories", "difference",
                    "differences", "extent", "i", "we", "respect", "there",
                    "not", "the main types", "the possible types", "the types",
                    "the difference", "the differences", "the main categories"]

    counter = 1  # counter indicating current chunk id (EC1, EC2, ...)
    offset = 0   # if we replace for example "Weka" with 'EC1' then the new CQ will be shorter by one char the offset remembers by how much we have shortened the current CQ with EC markers, so the new substitutions can be applied in correct places.


    # we decided to treat qualities defined as adjectives in: How + Quality(adjective) + Verb as EC
    if (doc[0].text.lower() == 'how' and
            doc[1].pos_ == 'ADJ' and
            doc[2].pos_ == 'VERB'):

        start = doc[1].idx  # mark where quality starts
        end = start + len(doc[1])  # mark where quality ends

        cq, offset = mark_chunk(cq, [(start, end)], "EC", offset, counter)  # substitute quality with EC identifier
        counter += 1  # the next EC chunk should have a new, bigger identifier, for example EC2

    for chunk in doc.noun_chunks:  # for each EC chunk candidate detected
        (start, end) = _get_EC_span_reject_wh_starters(chunk)  # check where chunk begins and ends

        ec = cq[start - offset:end - offset]  # extract text of potential EC, apply offsets

        if ec.lower() in rejecting_ec:  # if it should be rejected - do nothing
            continue

        if "the thing" in ec and end - start > len("the thing"):
            cq = cq[:start - offset] + "EC" + str(counter) +\
                " EC" + str(counter + 1) + cq[end - offset:]
            counter += 2
            offset += (end - start) - 7
        else:
            cq, offset = mark_chunk(cq, [(start, end)], "EC", offset, counter)
            counter += 1

    if (doc[-2].pos_ == 'VERB' and doc[-3].text in ['are', 'is', 'were', 'was'] and doc[-1].text == '?') or (doc[-2].pos_ in ['ADJ', 'ADV'] and doc[-1].text == '?'):
        # if CQ ends with are/is/were/was + VERB + ? or the last token is ADJective or ADVerb, treat the
        # verb / adverb / adjective as EC
        # Which animals are endangered -> endangered is EC
        # Which animals are quick -> quick is EC
        if doc[-2].text.lower() not in rejecting_ec:
            start = doc[-2].idx
            end = start + len(doc[-2])

            cq, offset = mark_chunk(
                cq, [(start, end)], "EC", offset, counter)
            counter += 1

    return cq


def get_PCs_as_spans(cq):
    def _is_auxilary(token, chunk_token_ids):
        """
            Check if given token is an auxilary verb of detected PC.
            The auxilary verb can be in a different place than the main part
            of the PC, so pos-tag-sequence based rules don't work here.
            For example in "What system does Weka require?" - the main part
            of PC is the word 'required'. The auxilary verb 'does' is separeted
            from the main part by 'Weka' noun. Thus dependency tree is used
            to identify auxilaries.
        """
        if (token.head.i in chunk_token_ids and  # if dep-tree current token's parent (head) is somewhere inside the main part of PC represented as chunk_token_ids (sequence of numeric token identifiers)
                token.dep_ == 'aux' and  # if the dep-tree label on the edge between some word from main part of PC and current token is AUX (auxilary)
                token.i not in chunk_token_ids):  # if token is outside of detected main part of PC
            return True  # yep, it's auxilary
        else:
            return False

    def _get_span(group, doc):
        id_tags = group.split(",")
        ids = [int(id_tag.split("::")[0]) for id_tag in id_tags]
        aux = None
        for token in doc:
            if _is_auxilary(token, ids):
                aux = token

        return (doc[ids[0]].idx, doc[ids[-1]].idx + len(doc[ids[-1]]),
                aux)

    def _reject_subspans(spans):
        """
            Given list of (chunk begin index, chunk end index) spans,
            return only those spans that aren't subspans of any other span.
            For instance form list [(1,10), (2,5)], the second span
            will be rejected because it is a subspan of the first one.
        """
        filtered = []
        for i, span in enumerate(spans):
            subspan = False
            for j, other in enumerate(spans):
                if i == j:
                    continue

                if span[0] >= other[0] and span[1] <= other[1]:
                    subspan = True
                    break
            if subspan is False:
                filtered.append(span)
        return filtered

    doc = nlp(cq)

    """
        Transform CQ into a form of POS-tags with token sequence identifier.
        Each token is described with "{ID}::{POS_TAG}".
        Tokens are separated with ","
        Having that form, we can extract longest sequences of expected pos-tags
        using regexes. The extracted parts can be explored to collect identifiers
        of tokens, so we know where they are located in text.
        Ex: "Kate owns a cat" should be translated into: "1::NOUN,2::VERB,3::DET,4::NOUN"
    """
    pos_text = ",".join(
        ["{id}::{pos}".format(id=id, pos=t.pos_) for id, t in enumerate(doc)])

    regexes = [   # rules describing PCs
        r"([0-9]+::(PART|VERB),?)*([0-9]+::VERB)",
        r"([0-9]+::(PART|VERB),?)+([0-9]+::AD(J|V),)+([0-9]+::ADP)",
        r"([0-9]+::(PART|VERB),?)+([0-9]+::ADP)",
    ]

    spans = []  # list of beginnings and endings of each chunk
    for regex in regexes:  # try to extract chunks with regexes
        for m in re.finditer(regex, pos_text):
            spans.append(_get_span(m.group(), doc))  # get chunk begin and end if matched
    spans = _reject_subspans(spans)  # reject subspans
    return spans


def extract_PC_chunks(cq):
    rejecting_pc = ['is', '痴', 'are', 'was', 'do', 'does', 'did', 'were',
                    'have', 'had', 'can', 'could', 'categorise', 'regarding',
                    'is of', 'are of', 'are in', 'given', 'is there']

    offset = 0
    counter = 1

    for begin, end, aux in get_PCs_as_spans(cq):
        if cq[begin - offset:end - offset].lower() in rejecting_pc:
            continue
        spans = [(begin, end)]
        if aux:
            spans.insert(0, (aux.idx, aux.idx + len(aux)))

        cq, offset = mark_chunk(cq, spans, "PC", offset, counter)
        counter += 1
    return cq

# post abstraction processing
path_to_output = os.getcwd() + "/output/output.txt"
def post_abstract(path_to_output):
    genQuestion=[] 
    splitQuestions =[]
    patterns = []
    pattern_candidates = {}
    T5Cqs_patterns = []
    with open(path_to_output, 'r') as f:
      for line in f.readlines():
        line= line.replace('?', '? ')
        line2 = line.split("?")
        line3= [p.join(' ?') for p in line2]
        line3 = [p.lstrip() for p in line3]
        genQuestion.append(line3)
    genQuestionDF= pd.DataFrame(genQuestion).T
    genQuestionDF2 = genQuestionDF.drop_duplicates(subset='questions', keep="first")
    genQuestionDF.columns=['questions']

    allList=[]
    pat_candidates={}
    for sublist in  genQuestion: 
        final = []
        for i in sublist:
            if(i==''):
                continue
            for item in i.split(','):
                pat = extract_EC_chunks(item)
                pat = extract_PC_chunks(pat)
                final.append(pat)
                pat_candidates[item]=(pat)
        allList.append(final)
    return allList, pat_candidates, genQuestionDF2

def processCQs():
    absWithCQs= pd.DataFrame(pat_candidates.items(), columns=['questions', 'patterns'])
    absWithCQs = absWithCQs.iloc[:-1]
    absWithCQs.to_csv(os.getcwd() + "/output/geneCQsNov.csv", sep='|')
    #import CLaRO templates
    templates= pd.read_csv(os.getcwd() + "/CLaROv2.csv", sep = ";")
    templates.columns = 'ID', 'patterns'
    return absWithCQs, templates
    #check if present in CLaRO

def compare_patterns(absWithCQs, templates):
    check = pd.merge(absWithCQs, templates, on=['patterns'], how='left', indicator='Exist')
    generated_CQs_with_patterns = check[check['Exist'] == 'both']
    check_with_cqPatterns_only = check[check['Exist'] == 'left_only']
    check_in__templates_only = check[check['Exist'] == 'right_only']
    #extract distict ptterns found in templates 
    distinct_patterns = generated_CQs_with_patterns.drop_duplicates(subset='patterns', keep="first")
    return distinct_patterns, check_in__templates_only, check_with_cqPatterns_only, generated_CQs_with_patterns

#semantic similarity to check CQs that are the same and should be dropped and also find questions that not similar
def final_output(genQuestionDF2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    CovidCorpus = list(set(list(genQuestionDF2['questions'])))
    query_Corpus =list(set(list(genQuestionDF2['questions'])))
    #Compute embedding for both lists
    corpus_embed = model.encode(query_Corpus, convert_to_tensor=True)
    covidSemanticCQs= {}
    top_k = min(5, len(CovidCorpus))
    for qry in query_Corpus:
        query_embed = model.encode(qry, convert_to_tensor=True, device=device)
        cos_scores = util.cos_sim(query_embed, corpus_embed)[0]
        top_results = torch.topk(cos_scores, k=top_k)
    for score, idx in zip(top_results[0], top_results[1]):
        new_score= "{:.4f}".format(score)
        new_score= float(new_score)
        if  new_score <= 0.7500:
            covidSemanticCQs[qry]=[]
            covidSemanticCQs[qry].append({query_Corpus[idx]: (new_score)})
        QuesSimilar= pd.json_normalize([ { "firstCQs": key_, "secondCQs" : i, "score" : child[i] }  for key_ in covidSemanticCQs for child in covidSemanticCQs[key_] for i in child ])

    QuesSimilar.to_csv(os.getcwd() +  "/output/semanticCQs.csv", sep='|')
    return QuesSimilar

def generateCQs(textFile):   
    outText= readTextFile(textFile)
    text = format(outText)
    preprocessor= TextPreprocessor()
    processed_text = preprocessor.process_file(text=text, output_path=None,
                                               remove_numbers=True,
                                               remove_headers=True,
                                               normalize_spaces=True,
                                               fix_breaks=True,
                                               clean_chars=True,
                                               merge_sentences=True)
    df= pd.DataFrame([processed_text], columns=['sentences'])
    cqs = generate(df)
    dcqs = postprocess(cqs)
    allList, pat_candidates, genQuestionDF2 = post_abstract(path_to_output)
    Similar_CQs_filter= final_output(genQuestionDF2)
    absWithCQs,templates= processCQs(pat_candidates)
    distinct_patterns, check_in_templates_only, check_with_cqPatterns_only, generated_CQs_with_patterns = compare_patterns(absWithCQs, templates)
        
    return Similar_CQs_filter, distinct_patterns, generated_CQs_with_patterns