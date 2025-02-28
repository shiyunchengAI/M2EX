import re
from typing import List, Optional, Union
import json
import ast
import re, sys,os
sys.path.append(os.path.dirname(os.getcwd()) + '/src')

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pathlib import Path
from src.utils import correct_malformed_json

from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)

_DESCRIPTION = (
    " text_analysis(question:str, context: Union[str, List[str]])-> str\n"
    " This tools is a text analysis task. For given text and a question, it analysis the text corpora and provide answer to the question. \n"
    " Comparision should be done after each analysis.\n"
    "- You cannot analyse multiple games texts in one call. For instance, `text_analysis('Which game has greater score differnece?','[{{'game_id':xxx,'game_id':yyy}}, {{'game_id':zzz,'game_id':www}})` does not work. "
    "If you need to analyse reports of multiple games, you need to call them separately like `text_analysis('What's the score difference in this gam?','{{'game_id':yyy}}')` and then `What's the score difference in this game?','{{'game_id':wwww}}')`\n"
    "These are the samples and you should consider the give question and act accordingly. "
    " - Minimize the number of `text_analysis` actions as much as possible."
    # Context specific rules below
    " - You can optionally provide either list of strings or string as `context` to help the agent solve the problem. "
    "If there are multiple contexts you need to answer the question, you can provide them as a list of strings.\n"
    " - `text_analysis` action will not see the output of the previous actions unless you provide it as `context`. "
    "You MUST provide the output of the previous actions as `context` if you need to do text_analysis on it.\n"
    " - You MUST NEVER provide `text2SQL` type action's outputs as a variable in the `question` argument. "
    "This is because `text2SQL` returns a text blob that contains the information about the database record, and needs to be process and extract game_id which `text_analysis` requires "
    "Therefore, when you need to provide an output of `text2SQL` action, you MUST provide it as a `context` argument to `text_analysis` action. "
)


_SYSTEM_PROMPT = """You are a text analysis assistant. Analyze the the provided question and report to answer the question.
"""

_ADDITIONAL_CONTEXT_PROMPT = """The following additional context is provided from other functions.\
    Use it to substitute into any ${{#}} variables or other words in the problem.\
    \n\n${context}\n\nNote that context variables are not defined in code yet.\
You must extract the relevant study_id and directly put them in code.
"""


def _get_report(_d):
     
    print("_d",_d)
    if 'game_id' not in _d and 'report' not in _d:
        return ValueError(f"The report analysis task requires game_id and report\nstate:\n{_d}")
    return _d

def _load_report(_d):
    return _d['report']
  
class ExecuteCode(BaseModel):
    reasoning: str = Field(
        ...,
        description="The reasoning behind the answer, including how context is included, if applicable.",
    )

    answer: str = Field(
        ...,
        description="an answer to the question about the report",
    )
   


def get_text_analysis_tools(llm: ChatOpenAI,db_path:str):
    """
   
    Args:
        question (str): The question about the report.
        context Union[str, List[str]]
    Returns:
        str: the answer to the question about the  report.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{question}"),
            MessagesPlaceholder(variable_name="report_info"),
        ]
    )
    extractor = create_structured_output_runnable(ExecuteCode, llm, prompt)


    def text_analysis(
        question: str,
        context: Union[str, List[str]],
        config: Optional[RunnableConfig] = None,
    ):
        chain_input = {"question": question}
        
        print("context-first:", context, type(context))
        if context :
            # if context_str.strip():
            #     context_str = _ADDITIONAL_CONTEXT_PROMPT.format(
            #         context=context_str.strip()
            #     )
            # If context is a string, parse it as JSON
            # print("context-before:", context)
            if isinstance (context, List) and isinstance(context[0], int):
                context=[str(ctx) for ctx in context]
                
            if isinstance (context, int):
                context=str(context)
                
            if isinstance(context, str) :
                context=correct_malformed_json(context)
                context = [ast.literal_eval(context)]
                if 'status' in context[0]:
                    context = context[0]
                # If the context contains 'data' key, use its value
            else:
                #     print("context-2", context)
                context = ast.literal_eval(context[0])
            
            print("context-2", context)
            
            if 'data' in context:
                #["{'status': 'success', 'data': [{'studydatetime': '2105-09-06 18:18:18'}]}"]
                context = context['data']

            print("context-after:", context)
            
            report_urls = [_get_report(ctx) for ctx in context]
            
            if isinstance(report_urls, ValueError):
                chain_input["context"] = [SystemMessage(content=str(report_urls))]
                print("Error on report_urls",report_urls)
            else:
                print("report_urls",report_urls)
                try:    
                    reports = [_load_report(url['report_url']) for url in report_urls[0]]
                    _humMessage=[{"type": "text", "text": x} for x in reports]
                    print("_humMessage",_humMessage)
                    chain_input["report_info"] = [
                            HumanMessage(
                                content=_humMessage
                            )
                        ]
                except ValueError as e:
                     chain_input["context"] = [SystemMessage(content=str(e))]

        model = extractor.invoke(chain_input, config)
        try:
            return model.answer
        except Exception as e:
            return repr(e)

    return StructuredTool.from_function(
        name="text_analysis",
        func=text_analysis,
        description=_DESCRIPTION,
    )

