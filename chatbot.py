#####################################################
# Fine-Tuning Chatbot, Refer to Readme.md for start #
#####################################################
import openai
# from flask import Flask, redirect, render_template, request, url_for
import prepare
# app = Flask(__name__)
openai.api_key = "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"

def generate_prompt(question):
    return """{}""".format(
        question.strip()
    )

def answer(question):
    response = openai.Completion.create(
        model="curie:ft-blockx-2023-02-12-13-06-40",
        prompt=generate_prompt(question),
        max_tokens=40,
        temperature=0.1,
    )
    # print("response: ", response.choices[0].text)
    return response.choices[0].text.strip().split("\n")[0]

# @app.route("/", methods=("GET", "POST"))
# def index():
#     if request.method == "GET":
#         print(request)
def main():
    print("Hello, I am Nosta sales manager! Ask me anything about Nosta.\n----------------------------------------\n")
    while True:
        question = input("Me: ")
        if question.strip():
            print("Nosta: "+answer(question))
        print("\n")
main()
