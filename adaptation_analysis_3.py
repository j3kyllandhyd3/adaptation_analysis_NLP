from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

import numpy as np
import pandas as pd
import openai
from llama_index.llms.openai import OpenAI
from openai import AsyncOpenAI

import json
import asyncio
import re
import os
import sys

PROMPT_TEMPLATE_GENERAL = ("""
You are an analyst of corporate reports, tasked with reviewing a company's report for information on whether they provide goods or services related to climate adaptation and resilience. Based on the following extracts
from the report, answer the given QUESTIONS. If you don't know the answer, just say that you don't know by answering "NA". Don't try to make up an answer.

These are the report extracts:
--------------------- [START OF EXTRACTS]\n
{sources}\n
--------------------- [END OF EXTRACTS]\n

QUESTIONS:
1. What company issued the report?
2. What sector does the company belong to?
3. Where is the company located?

Format your answers in JSON format with the following keys: COMPANY_NAME, COMPANY_SECTOR, COMPANY_LOCATION.
Your FINAL_ANSWER in JSON (ensure there's no format error):
""")

PROMPT_TEMPLATE_YEAR = ("""
You are an analyst of corporate reports, tasked with reviewing a company's report for information on whether they provide goods or services related to climate adaptation and resilience. Based on the following extracts
from the report, answer the given QUESTION. If you don't know the answer, just say that you don't know by answering "NA". Don't try to make up an answer.

These are the report extracts:
--------------------- [START OF EXTRACTS]\n
{sources}\n
--------------------- [END OF EXTRACTS]\n

QUESTION:
In which year was the report published?

Format your answers in JSON format with the following key: YEAR
Your FINAL_ANSWER in JSON (ensure there's no format error):
""")

PROMPT_TEMPLATE_QA = ("""
You are a veteran analyst of corporate reports, tasked with reviewing a company's report for information on whether they provide goods or services related to climate adaptation and resilience.

This is basic information on the company:
{basic_info}

You are presented with the following extracts from the company's report:
--------------------- [START OF EXTRACTS]\n
{sources}\n
--------------------- [END OF EXTRACTS]\n

Using the extracts and no prior knowledge, your task is to respond to the question encapsulated in "||".
Question: ||{question}||

Please consider the following additional explanation to the question encapsulated in "+++++" as crucial for answering the question:
+++++ [BEGIN OF EXPLANATION]
{explanation}
+++++ [END OF EXPLANATION]

Please adhere to the following guidelines when providing your answer:
1. Your response must be precise, thorough, and grounded on specific extracts from the report to verify its authenticity.
2. If you are unsure, simply acknowledge the lack of knowledge, rather than fabricating an answer.
3. Keep your ANSWER within {answer_length} words.
4. Be skeptical to the information disclosed in the extracts as there might be greenwashing (exaggerations of the company's adaptation and resilience credentials). Always answer in a critical tone.
5. Always acknowledge that the information provided is representing the company's view based on its report.
6. Scrutinize whether the report is grounded in quantifiable, concrete data or vague, unverifiable statements, and communicate your findings.
7. Start your answer with a "[[YES]]" or "[[NO]]" depending on whether you would answer the question with a yes or no. Always complement your judgment on yes or no with a short explanation that summarizes the sources in an informative way, i.e. provide details.

Format your answer in JSON format with the two keys: ANSWER (this should contain your answer string without sources), and SOURCES (this should be a list of the SOURCE numbers that were referenced in your answer).
Your FINAL_ANSWER in JSON (ensure there's no format error):
""")

def createRetriever(REPORT, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K):
    # Load in document
    documents = SimpleDirectoryReader(input_files=[REPORT]).load_data()
    parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)  # Tries to keep sentences together
    nodes = parser.get_nodes_from_documents(documents)

    # Build indexes
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    index = VectorStoreIndex(
        nodes,
        embed_model=embed_model
    )

    # Configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K,
    )
    return retriever

def basicInformation(retriever, PROMPT_TEMPLATE_GENERAL, MODEL):
    # Query content
    retrieved_nodes = retriever.retrieve(
        "What is the name of the company, the sector it operates in, and location of its headquarters?")
    # Create the "sources" block
    sources = []
    for i in retrieved_nodes:
        page_num = i.metadata.get('page_label', 'Unknown')
        # Remove "\n" from the sources
        source = i.get_content().replace("\n", "")
        sources.append(f"PAGE {page_num}: {source}")
    sources_block = "\n\n\n".join(sources)

    qa_template = PromptTemplate(PROMPT_TEMPLATE_GENERAL)
    # Create text prompt (for completion API)
    prompt = qa_template.format(sources=sources_block)

    # Get response
    response = OpenAI(temperature=0, model=MODEL).complete(prompt)
    # Replace front or back ```json {} ```
    response_text_json = response.text.replace("```json", "").replace("```", "")
    response_text = json.loads(response_text_json)

    # Create a dictionary to store the information
    basic_info_dict = {
        "COMPANY_NAME": response_text.get('COMPANY_NAME', 'Unknown'),
        "COMPANY_SECTOR": response_text.get('COMPANY_SECTOR', 'Unknown'),
        "COMPANY_LOCATION": response_text.get('COMPANY_LOCATION', 'Unknown')
    }

    return basic_info_dict

def yearInformation(retriever, PROMPT_TEMPLATE_YEAR, MODEL):
    # Query content
    retrieved_nodes = retriever.retrieve(
        "In which year was the report published?")
    # Create the "sources" block
    sources = []
    for i in retrieved_nodes:
        page_num = i.metadata.get('page_label', 'Unknown')
        # Remove "\n" from the sources
        source = i.get_content().replace("\n", "")
        sources.append(f"PAGE {page_num}: {source}")
    sources_block = "\n\n\n".join(sources)

    qa_template = PromptTemplate(PROMPT_TEMPLATE_YEAR)
    # Create text prompt (for completion API)
    prompt = qa_template.format(sources=sources_block)

    # Get response
    response = OpenAI(temperature=0, model=MODEL).complete(prompt)
    # Replace front or back ```json {} ```
    response_text_json = response.text.replace("```json", "").replace("```", "")
    response_text = json.loads(response_text_json)

    return response_text

def createPromptTemplate(retriever, BASIC_INFO, QUERY_STR, PROMPT_TEMPLATE_QA, EXPLANATION, ANSWER_LENGTH):
    # Query content
    retrieved_nodes = retriever.retrieve(QUERY_STR)
    # Create the "sources" block
    sources = []
    for i in retrieved_nodes:
        page_num = i.metadata.get('page_label', 'Unknown')
        # Remove "\n" from the sources
        source = i.get_content().replace("\n", "")
        sources.append(f"PAGE {page_num}: {source}")
    sources_block = "\n\n\n".join(sources)

    qa_template = PromptTemplate(PROMPT_TEMPLATE_QA)
    # Create text prompt (for completion API)
    prompt = qa_template.format(basic_info=BASIC_INFO, sources=sources_block, question=QUERY_STR,
                                explanation=EXPLANATION, answer_length=ANSWER_LENGTH)

    return prompt

def createPrompts(retriever, PROMPT_TEMPLATE_QA, BASIC_INFO, ANSWER_LENGTH, MASTERFILE):
    prompts = []
    questions = []
    for i in np.arange(0, MASTERFILE.shape[0]):
        QUERY_STR = MASTERFILE.iloc[i]["question"]
        questions.append(QUERY_STR)
        EXPLANATION = MASTERFILE.iloc[i]["question definitions"]
        prompts.append(
            createPromptTemplate(retriever, BASIC_INFO, QUERY_STR, PROMPT_TEMPLATE_QA, EXPLANATION, ANSWER_LENGTH))
    print("Prompts Created")
    return prompts, questions

# Asynced creation of answers
async def answer_async(prompts, MODEL):
    coroutines = []
    llm = OpenAI(temperature=0, model=MODEL)
    for p in prompts:
        co = llm.acomplete(p)
        coroutines.append(co)
    # Schedule all calls *concurrently*:
    out = await asyncio.gather(*coroutines)
    return out

async def createAnswersAsync(prompts, MODEL):
    # Async answering
    answers = await answer_async(prompts, MODEL)
    return answers

def createAnswers(prompts, MODEL):
    # Sync answering
    answers = []
    llm = OpenAI(temperature=0, model=MODEL)
    for p in prompts:
        response = llm.complete(p)
        answers.append(response)

    print("Answers Given")
    return answers

def outputExcel(answers, questions, prompts, REPORT, MASTERFILE, MODEL, basic_info_dict, option="", excels_path="Excels_SustReps"):
    # Create the columns
    categories, ans, ans_verdicts, source_pages, source_texts = [], [], [], [], []
    subcategories = MASTERFILE['identifier'].tolist()
    themes = MASTERFILE['theme'].tolist()
    dimensions = MASTERFILE['dimension'].tolist()

    for i, a in enumerate(answers):
        try:
            # Replace front or back ```json {} ```
            a_text = a.text if hasattr(a, 'text') else str(a)
            a_text = a_text.replace("```json", "").replace("```", "")
            answer_dict = json.loads(a_text)
        except:
            print(f"{i} with formatting error")
            try:
                answer_dict = {"ANSWER": "CAUTION: Formatting error occurred, this is the raw answer:\n" + a_text,
                               "SOURCES": "See In Answer"}
            except:
                answer_dict = {"ANSWER": "Failure in answering this question.", "SOURCES": "NA"}

        # Final verdict
        verdict = re.search(r"\[\[([^]]+)\]\]", answer_dict.get("ANSWER", ""))
        if verdict:
            ans_verdicts.append(verdict.group(1))
        else:
            ans_verdicts.append("NA")

        # Other values
        ans.append(answer_dict.get("ANSWER", ""))
        source_pages.append(", ".join(map(str, answer_dict.get("SOURCES", []))))
        source_texts.append(prompts[i].split("---------------------")[1])

        # Assign theme to each question based on the order of answers
        category = themes[i] if i < len(themes) else "Unknown"
        categories.append(category)

    # Create DataFrame and export as Excel
    df_out = pd.DataFrame(
        {
            "theme": categories,
            "subcategory": subcategories,
            "dimension": dimensions,
            "question": questions,
            "decision": ans_verdicts,
            "answer": ans,
            "source_pages": source_pages,
            "source_texts": source_texts,
            "Company name": basic_info_dict['COMPANY_NAME'],
            "Industry": basic_info_dict['COMPANY_SECTOR'],
            "Headquarters": basic_info_dict['COMPANY_LOCATION']
        }
    )
    excel_path_qa = f"./{excels_path}/" + REPORT.split("/")[-1].split(".")[0] + f"_{MODEL}" + f"{option}" + ".xlsx"
    df_out.to_excel(excel_path_qa, index=False)
    return excel_path_qa

async def main():
    print(sys.argv)
    if len(sys.argv) < 4:
        print("WRONG USAGE PATTERN!\nPlease use: 'python api_key report model dimension [num indicators]'")
        return
    args = sys.argv[1:]
    os.environ["OPENAI_API_KEY"] = args[0]
    openai.api_key = args[0]
    # Global parameters
    MASTERFILE = pd.read_excel("questions_masterfile_020824.xlsx")
    CHUNK_SIZE = 350
    CHUNK_OVERLAP = 50
    TOP_K = 8
    ANSWER_LENGTH = 200

    REPORT = args[1]
    MODEL = args[2]
    DIMENSION = args[3]
    
    if DIMENSION.lower() != "all":
        # If the option of less is given
        try:
            less = int(args[4])
            MASTERFILE = MASTERFILE[MASTERFILE['dimension'] == DIMENSION][:less].copy()
            print(f"Execution with subset of {less} indicators in dimension '{DIMENSION}'.")
        except:
            less = "all"
            MASTERFILE = MASTERFILE[MASTERFILE['dimension'] == DIMENSION].copy()
            print(f"Execution with all parameters in dimension '{DIMENSION}'.")
    else:
        # Process all dimensions
        try:
            less = int(args[4])
            MASTERFILE = MASTERFILE[:less].copy()
            print(f"Execution with subset of {less} indicators across all dimensions.")
        except:
            less = "all"
            print("Execution with all parameters across all dimensions.")
    
    retriever = createRetriever(REPORT, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K)
    basic_info_dict = basicInformation(retriever, PROMPT_TEMPLATE_GENERAL, MODEL)
    year_info = yearInformation(retriever, PROMPT_TEMPLATE_YEAR, MODEL)
    basic_info_dict["YEAR"] = year_info["YEAR"]
    basic_info_dict["REPORT_NAME"] = REPORT
    print(basic_info_dict)


    prompts, questions = createPrompts(retriever, PROMPT_TEMPLATE_QA, basic_info_dict, ANSWER_LENGTH, MASTERFILE)

    # Avoid hitting rate limits
    answers = []
    step_size = 5
    print(f"In order to avoid Rate Limit Errors, we answer {step_size} questions at a time, not everything in parallel.\nThis increases the execution time significantly but decreases the error rate. Another potential way to overcome this is to upgrade your OpenAI API key.\nFollow the tutorials on Medium to learn more.")
    for i in np.arange(0, len(prompts), step_size):
        p_loc = prompts[i:i + step_size]
        a_loc = await createAnswersAsync(p_loc, MODEL)
        answers.extend(a_loc)
        num = i + step_size
        if num > len(prompts):
            num = len(prompts)
        print(f"{num} Answers Given")
    
    excels_path = "Excel_Output"
    option = f"_topk{TOP_K}_params{less}_dimension_{DIMENSION}"
    path_excel = outputExcel(answers, questions, prompts, REPORT, MASTERFILE, MODEL, basic_info_dict, option, excels_path)

# For usage on windows:
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
asyncio.run(main())
