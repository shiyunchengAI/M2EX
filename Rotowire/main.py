import getpass
import os
import pathlib
import ast
import csv
import itertools
from langchain_openai import ChatOpenAI

# Imported from the https://github.com/langchain-ai/langgraph/tree/main/examples/plan-and-execute repo

# from langchain.sql_database import SQLDatabase
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import END, MessageGraph, START


from typing import Sequence, Dict

from langchain import hub
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

import sys
# sys.path.append(os.path.dirname(os.getcwd()) + '/ceasura_langgraph')
sys.path.append(os.path.dirname(os.getcwd()) )

from Rotowire.src.joiner import *
from Rotowire.src.planner import *
from Rotowire.src.task_fetching_unit import *
from Rotowire.src.build_graph import graph_construction
from Rotowire.src.utils import *


import json

from tqdm import tqdm
from pathlib import Path


import logging



import os
import sys
# sys.path.append(os.path.dirname(os.getcwd()) + '/src')
# sys.path.append(os.path.dirname(os.getcwd()) + '/ceasura_langgraph/tools')

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")
        

def pretty_print_stream_chunk(chunk):
    for node, updates in chunk.items():
        print(f"Update from node: {node}")
        if "messages" in updates:
            updates["messages"][-1].pretty_print()
        else:
            print(updates)

        print("\n")
        
def load_json(file_path, data):
    fp = Path(file_path)
    if not fp.exists():
        fp.touch()
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        with open(file_path, 'r') as f:
            data = json.load(f)
    return data

def append_json(data, file_path):
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, 'r+') as f:
        _data = json.load(f)
        if type(data) == dict:
            _data.append(data)
        elif type(data) == list:
            _data.extend(data)
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
        f.seek(0)
        json.dump(_data, f, ensure_ascii=False, indent=4)
    return _data

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("LANGCHAIN_API_KEY")
    # _set_if_undefined("TAVILY_API_KEY")
    # # Optional, add tracing in LangSmith

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "XMODE-Rotowire"
    
    model="gpt-4o" #gpt-4-turbo-preview
    db_path="/home/ubuntu/workspace/XMODE/Rotowire/rotowire.db"
    temperature=0
    language='en'
    
    ceasura_rotowire=[]
    output_path = f'/home/ubuntu/workspace/XMODE/Rotowire/experiments/{language}'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    output_file= f'{output_path}/ceasura_rotowire-{language}-test.json'
   
    load_json(output_file,ceasura_rotowire)
    
    results=[]
    
    LOG_PATH="/home/ubuntu/workspace/XMODE/Rotowire/experiments/log"
    
    # Load the questions from csv
    rotowire_questions_file = "/home/ubuntu/workspace/XMODE/Rotowire/rotowire_questions.csv"
    with open(rotowire_questions_file, 'r') as f:
        csv_reader = csv.reader(f)
        rotowire_questions = [row[0] for row in csv_reader]
    
    # if no csv file, use the default questions
    if not any(rotowire_questions):
        rotowire_questions=[
            "Who is the smallest power forward in the database"
            "What is the youngest team in the Southeast Division in terms of the founding date",
            "Who is the oldest player per nationality",
            "What is the oldest team per conference in terms of the founding date",
            "Plot the age of the youngest player per position",
            "Plot the age of the oldest team per conference in terms of the founding date",
            "Who made the lowest number of assists in any game",
            "Which team made the highest percentage of field goals in any game",
            "For each player, what is the highest number of assists they made in a game",
            "How many games did each team loose",
            "Plot the highest number of three pointers made by players from each nationality",
            "Plot the  lowest percentage of field goals made by teams from each division",
            ##new
        ]
        
    for id, question in enumerate(rotowire_questions): 
        iddx = 1
        result={}
        result_str=''
        #logging the current use-case
        use_case_log_path = f'{LOG_PATH}/{id}'
        pathlib.Path(use_case_log_path).mkdir(parents=True, exist_ok=True) 
        for handler in logging.root.handlers:
                handler.level = logging.root.level
        file_handler = logging.FileHandler(Path(use_case_log_path) / 'out.log')
        logging.root.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
        logger.debug(f"Question: {question}")
       
        result["question"]=question
        result["id"]=id
        chain = graph_construction(model,temperature=temperature, db_path=db_path,log_path=use_case_log_path)
        # steps=[]
        
        
        config = {"configurable": {"thread_id": "2"}}
       
        
       
        for step in chain.stream(question, config, stream_mode="values"):
            print(step)
            # for k,v in step.items():
            #     print(k)
            #     print("---------------------")
            #     for ctx in v:
            #         print (ctx)
            result_str += f"Step {iddx:}\n {step}\n\n"
            iddx+=1
            print("---------------------")
        
        to_json=[]
        try:
            for msg in step:
                value= msg.to_json()['kwargs']
                to_json.append(value)
                # needs code or prompt imporvements
            prediction=[ast.literal_eval(step[-1].content)]
        except Exception as e:
            print(str(e)) # comes basicly from ast.literal_eval becuase the output sometimes not in JSON structure
            prediction= step[-1].content
        
        result["xmode"] = to_json
        result["prediction"]= prediction
        
        
        results.append(result)
        file_result_path = Path(use_case_log_path) / "xmode.json"
        with open(file_result_path, 'w') as f:
            json.dump([result], f, ensure_ascii=False, indent=4)
       
        path = Path(use_case_log_path) / "steps-values.log"
        with open(path, "w") as f: 
            print(result_str, file=f)
    
    append_json(results,output_file)
   
    # all_states = []
    # for state in chain.get_state(config):
    #    print(state) 
    # it is better to creat graph for each question
if __name__ == '__main__':
    main()