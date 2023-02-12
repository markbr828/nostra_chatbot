import pandas as pd
import re
from typing import Set
from transformers import GPT2TokenizerFast
import numpy as np
from nltk.tokenize import sent_tokenize
import openai

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
openai.api_key = "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"
# export OPENAI_API_KEY="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"
def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

def reduce_long(
    long_text: str, long_text_tokens: bool = False, max_len: int = 590
) -> str:
    """
    Reduce a long text to a maximum of `max_len` tokens by potentially cutting at a sentence end
    """
    if not long_text_tokens:
        long_text_tokens = count_tokens(long_text)
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + count_tokens(sentence)
            if ntokens > max_len:
                return ". ".join(sentences[:i][:-1]) + "."

    return long_text
def get_questions(context):
    try:
        response = openai.Completion.create(
            engine="davinci-instruct-beta-v3",
            prompt=f"Write questions based on the text below\n\nText: {context}\n\nQuestions:\n1.",
            temperature=0,
            max_tokens=257,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n\n"]
        )
        return response['choices'][0]['text']
    except:
        return ""
def process_page(text):
    output=[]
    tokens = count_tokens(text)
    text = reduce_long(text, max_len=1500)
    return [(text, tokens)]

def get_answers(row):
    try:
        response = openai.Completion.create(
            engine="davinci-instruct-beta-v3",
            prompt=f"Write answer based on the text below\n\nText: {row.context}\n\nQuestions:\n{row.questions}\n\nAnswers:\n1.",
            temperature=0,
            max_tokens=257,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response['choices'][0]['text']
    except Exception as e:
        print (e)
        return ""

def prepare_metadata():
    res = []
    pages = []
    data = "dataset.txt"
    with open(data,'r',encoding='utf-8') as f:
        page = ""
        for line in f.readlines():
            if line.strip() == "":
                pages.append(page)
                page =""

            else:
                page += line

    for page in pages:
        res += process_page(page)
    df = pd.DataFrame(res, columns=["context", "tokens"])
    df = df[df.tokens>40]

    # df = df.drop_duplicates(['title','heading'])
    df = df.reset_index().drop('index',axis=1) # reset index
    print (df.head())
    df['questions']= df.context.apply(get_questions)
    df['questions'] = "1." + df.questions
    print(df[['questions']].values[0][0])

    df['answers']= df.apply(get_answers, axis=1)
    df['answers'] = "1." + df.answers
    df = df.dropna().reset_index().drop('index',axis=1)
    print(df[['answers']].values[0][0])

    df = df[df.tokens<2000]
    df.to_csv('dataset.csv', index=False)

def prepare_jsonl():
    df = pd.read_csv('dataset.csv')
    values = df[['questions', 'answers']].values
    res = []
    for value in values:
        # print(value)
        prompts = value[0].split("\n")
        comps = value[1].split("\n")
        assert len(prompts) == len(comps)
        for i in range(len(prompts)):
            prompt = prompts[i].split(str(i+1)+". ")[1]+"\n\n###\n\n"
            comp = " "+comps[i].split(str(i+1)+". ")[1]+"###"
            res.append((prompt, comp)) 

    print(len(res))
    df = pd.DataFrame(res, columns=["prompt", "completion"])
    df[['prompt', 'completion']].to_json('dataset.jsonl', orient='records', lines=True)


# prepare_metadata()

prepare_jsonl()

