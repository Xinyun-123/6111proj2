import sys
import time
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
import urllib.request
import re
import spacy
from spanbert import SpanBERT
from urllib.error import HTTPError, URLError
import logging
import openai
import socket
import requests

# from spacy_help_functions import extract_relations
import spacy_help_functions
from collections import defaultdict


spacy2bert = {
    "ORG": "ORGANIZATION",
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE"
}

bert2spacy = {
    "ORGANIZATION": "ORG",
    "PERSON": "PERSON",
    "LOCATION": "LOC",
    "CITY": "GPE",
    "COUNTRY": "GPE",
    "STATE_OR_PROVINCE": "GPE",
    "DATE": "DATE"
}

bertInternalName = {
    1: "per:schools_attended",
    2: "per:employee_of",
    3: "per:cities_of_residence",
    4: "org:top_members/employees"
}

relation_entity_dict = {1: {"subject": ["PERSON"], "object": ["ORGANIZATION"]},
                        2: {"subject": ["PERSON"], "object": ["ORGANIZATION"]},
                        3: {"subject": ["PERSON"], "object": ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]},
                        4: {"subject": ["ORGANIZATION"], "object": ["PERSON"]}}

relation_name_dict = {1: "Schools_Attended",
                      2: "Work_For",
                        3: "Live_In",
                        4: "Top_Member_Employees"}

# One shot example in GTP prompt
gpt_example = {1: "[relationship: Schools_Attended; subject: Jeff Bezos; object: Princeton University]",
               2: "[relationship: Work_For; subject: Alec Radford; object: OpenAI]",
                3: "[relationship: Live_In; subject: Mariah Carey; object: New York City]",
                4: "[relationship: Top_Member_Employees; subject: Jensen Huang; object: Nvidia]"}

# Return the top 10 results of the query on Google
def web_search(query, API_key, engine_id):
    # Build a service object for interacting with the API. Visit
    # the Google APIs Console <http://code.google.com/apis/console>
    # to get an API key for your own application.

    service = build(
        "customsearch", "v1", developerKey=API_key
    )

    res = (
        service.cse()
        .list(
            q=query,
            cx=engine_id,
        )
        .execute()
    )
    return res

# Extract relation tuples using SpanBert
def get_tuples_spanbert(spanbert, sent, entities_of_interest,r,t,X, tuple_added_count, tuple_total_count):

    extracted_entity_pairs = spacy_help_functions.create_entity_pairs(sent, entities_of_interest)

    # Check if the sentence contains entities pairs of interests
    annotated = 0
    if len(extracted_entity_pairs) > 0:
        examples = []
        for ep in extracted_entity_pairs:   #processing each entity pair
            entity1 = ep[1][1]
            entity2 = ep[2][1]
            for ep in extracted_entity_pairs:
                if entity1 in relation_entity_dict[r]['subject'] and entity2 in relation_entity_dict[r]['object']:
                    examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
                elif entity2 in relation_entity_dict[r]['subject'] and entity1 in relation_entity_dict[r]['object']:
                    examples.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
        if len(examples) > 0: #move to get relations only if the candidate pair (examples) is not empty
            annotated += 1
            X, tuple_added_count, tuple_total_count = get_relations(spanbert, examples, r, t, X, tuple_added_count, tuple_total_count)
    return X, tuple_added_count, tuple_total_count, annotated


# Extract relation tuples using SpanBert
def get_relations(spanbert, examples, r, t, X, tuple_added_count, tuple_total_count):
    preds = spanbert.predict(examples)
    for ex, pred in list(zip(examples, preds)):
        relation = pred[0]
        confidence = pred[1]
        subj = ex["subj"][0]
        obj = ex["obj"][0]

        # Check if the relations match the relation of interest
        if relation != bertInternalName[r] or confidence < t:
            continue

        print("                === Extracted Relation ===")
        print("                Input tokens: ", ex["tokens"])
        print("                Output Confidence: {} ; Subject: {} ; Object: {} ;".format(confidence, subj, obj))

        # Add tuples into X
        new_tuple = (subj, obj)
        tuple_total_count += 1
        if new_tuple in X and confidence > X[new_tuple]: #check if the new tuple is in X already
            X[(subj, obj)] = confidence
            print("                Duplicate with higher confidence than existing record. Replace the old with this one.")
        elif new_tuple not in X:
            X[(subj, obj)] = confidence
            tuple_added_count += 1
            print("                Adding to set of extracted relations")
        else:
            print("                Duplicate with lower confidence than existing record. Ignoring this.")
    return X, tuple_added_count, tuple_total_count


# Fetch the raw text from URL using BeautifulSoup and preprocess the raw texts
def get_texts(page):
    soup = BeautifulSoup(page, "html.parser")
    text = soup.get_text()
    preprocessed_text = re.sub(u'\xa0', ' ', text)
    preprocessed_text = re.sub('\t+', ' ', preprocessed_text)
    preprocessed_text = re.sub('\n+', ' ', preprocessed_text)
    preprocessed_text = re.sub(' +', ' ', preprocessed_text)
    preprocessed_text = preprocessed_text.replace('\u200b', '')

    # truncate the text
    if len(preprocessed_text) > 10000:
        print("        Trimming webpage content from {} to 10000 characters".format(len(preprocessed_text)))
        # truncate text to 20,000 characters
        preprocessed_text = preprocessed_text[:10000]
    else:
        print("        Webpage length (num characters): ", len(preprocessed_text))
    return preprocessed_text

# Call OpenAI
def get_openai_completion(prompt, model, max_tokens, temperature=0.2, top_p=1, frequency_penalty=0,
                              presence_penalty=0):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    #get the response text from openai api
    response_text = response['choices'][0]['text']
    return response_text


# Extract relation tuples using OpenAI Gpt-3
def get_tuples_gpt(r, sent, X, entities_of_interest, tuple_added_count, tuple_total_count):
    # Check if the sentences contain the entity pair of interest
    annotated = 0
    extracted_entity_pairs = spacy_help_functions.create_entity_pairs(sent, entities_of_interest)
    if len(extracted_entity_pairs) > 0:
        examples = []
        for ep in extracted_entity_pairs:
            entity1 = ep[1][1]
            entity2 = ep[2][1]
            for ep in extracted_entity_pairs:
                if entity1 in relation_entity_dict[r]['subject'] and entity2 in relation_entity_dict[r]['object']:
                    examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
                elif entity2 in relation_entity_dict[r]['subject'] and entity1 in relation_entity_dict[r]['object']:
                    examples.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
        # if there is no desired entity pairs, return directly rather than using gpt3
        if len(examples) <= 0:
            return X, tuple_added_count, tuple_total_count, annotated
    else:
        return X, tuple_added_count, tuple_total_count, annotated

    # query gpt3
    annotated += 1
    prompt_text = """ Given a sentence, extract all instances of the following relationship types as possible:
    relationship type: {}

    Output: [relationship: RELATIONSHIP; subject: {}; object: {}]
    
    An example for output is: {}

    Sentence: {} 

    Output: """.format(relation_name_dict[r], ", ".join(relation_entity_dict[r]['subject']), " or ".join(relation_entity_dict[r]['object']), gpt_example[r], sent)

    model = 'text-davinci-003'
    max_tokens = 100
    temperature = 0.3
    top_p = 1
    frequency_penalty = 0
    presence_penalty = 0

    response_text = get_openai_completion(prompt_text, model, max_tokens, temperature, top_p, frequency_penalty,
                                          presence_penalty)

    # parse the response from gpt3 and add the tuple into X
    relations = response_text.split("]")
    for relation in relations:
        temp = relation.split('; ')  # "[relationship: Live_In", "subject: John ", "object: Beijing, China]"

        if len(temp) < 3:
            continue
        subj = temp[1].split(': ')[1]
        subj = subj.lower().strip()
        obj = temp[2].split(': ')[1]
        obj = obj.lower().strip()

        print('')
        print("                === Extracted Relation ===")
        print("                Sentence: {}".format(sent.text.replace('\n', ' ')))
        print("                Subject: {} ; Object: {} ;".format(subj, obj))
        tuple_total_count += 1
        if (subj, obj) in X:
            print("                Duplicate. Ignoring this.")
        else:
            X[(subj, obj)] = 1
            tuple_added_count += 1
            print("                Adding to set of extracted relations")
        print("                ==========")
        print("")
    return X, tuple_added_count, tuple_total_count, annotated


# Iterative Set Expansion
def main():
    # Initialize command line parameters
    model = sys.argv[1]
    API_key = sys.argv[2]
    engine_id = sys.argv[3]
    openai_key = sys.argv[4]
    r = int(sys.argv[5])
    t = float(sys.argv[6])  # Confidence Threshold
    q = sys.argv[7]
    k = int(sys.argv[8])
    X = defaultdict(float) #dictionary in form of {(subject,object): confidence score}

    ## print headers
    print("Parameters:")
    print("Client key      = ", API_key)
    print("Engine key      = ", engine_id)
    print("OpenAI key      = ", openai_key)
    print("Method  = ", model[1:])
    print("Relation        = ", relation_name_dict[r])
    print("Threshold       = ", t)
    print("Query           = ", q)
    print("# of Tuples     = ", k)
    print("Loading necessary libraries; This should take a minute or so ...)")

    q_history = [q.lower()]

    # Load models
    nlp = spacy.load("en_core_web_lg")
    spanbert = SpanBERT("./pretrained_spanbert")
    openai.api_key = openai_key

    #iteration
    count_iteration = 1
    while True:
        print("=========== Iteration: ", count_iteration, " - Query: ", q, " ===========")
        res = web_search(q, API_key, engine_id)
        items = res['items'][:10]
        seenUrls = []

        count = 0
        # Process each webpage
        for item in items:
            count += 1
            url = item['formattedUrl']
            print('\n')
            print("URL ( {} / {}): ".format(count, len(items)), url)

            # Continue only for unseen Urls
            if url not in seenUrls:
                seenUrls.append(url)
                print("        Fetching text from url ...")
                try:
                    page = requests.get(url)
                except:
                    print('        Unable to fetch URL ...')
                    continue
            else:
                print("        URL already seen ...")
                continue

            # use beautifulsoup to get text
            text = get_texts(page.text)
            print("        Annotating the webpage using spacy...")
            doc = nlp(text)
            sentences = list(doc.sents)
            print("        Extracted {} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...".format(len(sentences)))

            entities_of_interest = relation_entity_dict[r]['subject'] + relation_entity_dict[r]['object']

            sent_count = 0
            tuple_added_count = 0
            tuple_total_count = 0
            num_ann = 0
            # Process each sentences for a webpage
            for sent in sentences:
                sent_count += 1
                if sent_count % 5 == 0:
                    print("        Processed {} / {} sentences".format(sent_count, len(sentences)))
                    print('')
                if model == "-gpt3":
                    X, tuple_added_count, tuple_total_count, annotated = get_tuples_gpt(r, sent, X, entities_of_interest, tuple_added_count, tuple_total_count)
                    time.sleep(1.5)
                else:
                    X, tuple_added_count, tuple_total_count, annotated = get_tuples_spanbert(spanbert, sent, entities_of_interest, r, t, X, tuple_added_count, tuple_total_count)
                    num_ann += annotated

            print("        Extracted annotations for  {}  out of total  {}  sentences".format(num_ann, len(sentences)))
            print("        Relations extracted from this website: {} (Overall: {})".format(tuple_added_count, tuple_total_count))

        # sort X in descending order of confidence score
        X = {k: v for k, v in sorted(X.items(), key=lambda item: item[1], reverse=True)}

        # if get k relations already, end the iteration
        if len(X) >= k:
            # print("### The top k tuples are: ", list(X.keys())[:k])
            break

        # Update query using the unseen tuple with the highest confidence
        i = 0
        y = list(X.keys())[i]
        while " ".join(y).lower() in q_history:
            i += 1
            y = list(X.keys())[i]
        q = " ".join(y).lower()
        q_history.append(q)
        count_iteration += 1
    print("================== ALL RELATIONS for {} ( {} ) =================".format(relation_name_dict[r], len(X)))
    if model == "-gpt3":
        for subj, obj in X.keys():
            print("Subject: {}                | Object: {}".format(subj, obj))
    else:
        for subj, obj in X.keys():
            print("Confidence: {}           | Subject: {}           | Object: {}".format(X[(subj, obj)], subj, obj))
    print("Total # of iterations = ", count_iteration)


if __name__ == "__main__":
    main()
