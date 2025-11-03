import getpass
import os
import pathlib
import ast
import argparse
from pydantic import BaseModel, Field, computed_field
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

from src.joiner import *
from src.planner import *
from src.task_fetching_unit import *
from ArtWork.src.build_graph import graph_construction
from src.utils import *


import json

from tqdm import tqdm
from pathlib import Path


import logging

from dotenv import load_dotenv
load_dotenv()

import os
import sys
# sys.path.append(os.path.dirname(os.getcwd()) + '/src')
# sys.path.append(os.path.dirname(os.getcwd()) + '/ceasura_langgraph/tools')

class LLMServingPlatformParserModel(BaseModel):
    """args
    llm_service: str 
    model: str
    base_url: str (computed field)
    api_key: str (computed field)
    """
    llm_service: str = Field(
        ...,
        default="swissai",
        description="The LLM serving platform to use. Must be one of the following: openai, swissai",
    )

    model: str = Field(
        ...,
        default="meta-llama/Llama-3.3-70B-Instruct",
        description="The model to use for the LLM serving platform",
        # model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    )
    
    @computed_field
    @property
    def base_url(self):
        if self.llm_service == "openai":
            return None
        elif self.llm_service == "swissai":
            assert os.environ.get("RC_API_BASE"), "RC_API_BASE environment variable must be set in .env"
            return os.environ.get("RC_API_BASE")
        raise ValueError("Invalid LLM serving platform")
    
    @computed_field
    @property
    def api_key(self):
        if self.llm_service == "openai":
            _set_if_undefined("OPENAI_API_KEY")
            return os.environ.get("OPENAI_API_KEY")
        elif self.llm_service == "swissai":
            _set_if_undefined("RC_API_KEY")
            return os.environ.get("RC_API_KEY")
        raise ValueError("Invalid LLM serving platform")


def add_model(parser: argparse.ArgumentParser, parser_model):
    fields = parser_model.__fields__
    for name, field in fields.items():
        parser.add_argument(
            f"--{name}",
            dest=name,
            type=field.type_,
            default=field.default,
            help=field.field_info.description,  
            
        )

parser = argparse.ArgumentParser()
add_model(parser, LLMServingPlatformParserModel)
args = parser.parse_args()
llm_config = LLMServingPlatformParserModel(**vars(args))

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
    
    base_url = llm_config.base_url
    model = llm_config.model
    
    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("LANGCHAIN_API_KEY")
    # _set_if_undefined("TAVILY_API_KEY")
    # # Optional, add tracing in LangSmith

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = f"XMODE-ArtWork-{llm_config.name}-{llm_config.model}"
    
    os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_YI_KEY") # experiment by Yi
    # model="gpt-4o" #gpt-4-turbo-preview

    model_str = model.split("/")[-1].replace(".", "_")
    

    db_path="/home/ubuntu/workspace/XMODE/ArtWork/art.db"
    temperature=0
    language='en'
    
    ceasura_artWork=[]
    output_path = f'/home/ubuntu/workspace/XMODE/ArtWork/experiments/{model_str}/{language}'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True) 
    output_file= f'{output_path}/ceasura_artWork-{language}-test.json'
   
    load_json(output_file,ceasura_artWork)
    
    results=[]
    
    
    LOG_PATH=f"/home/ubuntu/workspace/XMODE/ArtWork/experiments/{model_str}/log"
    pathlib.Path(LOG_PATH).mkdir(parents=True, exist_ok=True) 
    ArtWork_Questions=[
        "What is the oldest impressionist artwork in the database?",
        "What is the newest painting in the database?",
        "What is the movement of the painting that depicts the highest number of swords?",
        "What is the movement of the painting that depicts the highest number of babies?",
        "What is the genre of the oldest painting in the database?",
        "What is the genre of the newest painting in the database?",
        "What is depicted on the oldest Renaissance painting in the database?",
        "What is depicted on the oldest religious artwork in the database?",
        "Plot the year of the oldest painting per movement",
        "Plot the year of the oldest painting per genre",
        "Plot the number of paintings that depict War for each year",
        "Plot the number of paintings that depict War for each century",
        "Plot the number of paintings for each year",
        "Plot the number of paintings for each century",
        "Plot the lowest number of swords depicted in each year",
        "Plot the lowest number of swords depicted in each genre",
        "Get the number of paintings that depict Fruit for each century",
        "Get the number of paintings that depict Animals for each movement",
        "Get the number of paintings for each year",
        "Get the number of paintings for each century",
        "Get the highest number of swords depicted in paintings of each movement",
        "Get the highest number of swords depicted in paintings of each genre",
        "Get the century of the newest painting per movement",
        "Get the century of the newest painting per genre",
        ##new
        "Find the total number of paitings depicting war and number of painting depecting sword in Renaissance",
        "Retrieve the average number of paintings depicting a person and the total number of artworks depicting swords for each century.",
        "Create a plot displaying the count of paintings in each genre, as well as a plot showing the count of paintings in each movement",
        "Visualize the number of paintings corresponding to each century and Show the century of the most recent painting for each genre.",
        "Plot the total number of paintings depicting war and the number of paintings depicting swords in religious arts.",
        "Create a plot to display the number of paintings for each year and get the total number of paintings for each genre."
    ]
    for id, question in enumerate(ArtWork_Questions): 
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
        chain = graph_construction(model, temperature=temperature, db_path=db_path, log_path=use_case_log_path, base_url=base_url)
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