import re
from typing import List, Optional, Union
import json
import ast
import re, sys,os
sys.path.append(os.path.dirname(os.getcwd()) + '/src')

from langchain.chains.openai_functions import create_structured_output_runnable
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
    " text_analysis(question:str, context: Union[str, List[str]])-> Union[str, List[str]]\n"
    " This tools is a text analysis task. For given text and a question, it analysis the text report and provide answer to the question. \n"
    " Each report contains statistics about all the teams and players that participated in a single game, e.g. number of points scored by each player / team, number of assists by each player / team, etc."
    " The question should target only one report. For example: What's the score difference in the game <x> ? or Who made the highest number of goals in the game <Y>?"
    " The question can be anything that can be answered by looking at a game report: For example. 'How many points did team <x> score? ...\n"
    " - Minimize the number of `text_analysis` actions as much as possible."
    # Context specific rules below
   " - You should provide either list of strings or string as `context` from previous agent to help the `text analysis` agent solve the problem."
   "The format of the context for text_analysis should be `[{'game_id': 'xxxx', 'report':'xxxx'}, {'game_id': 'xxxx', 'report':'xxxx'}, ...]`. For example for one game: `[{'game_id': '0', 'report':'....'}]`"
    "If there are multiple contexts you need to answer the question, you can provide them as a list of strings.\n"
    " - `text_analysis` action will not see the output of the previous actions unless you provide it as `context`. "
    "You MUST provide the output of the previous actions as `context` if you need to do text_analysis on it.\n"
    " - You MUST NEVER provide `text2SQL` type action's outputs as a variable in the `question` argument. "
    "This is because `text2SQL` returns a text blob that contains the information about the database record, and needs to be process and extract game_id which `text_analysis` requires "
    "Therefore, when you need to provide an output of `text2SQL` action, you MUST provide it as a `context` argument to `text_analysis` action. "
)


_SYSTEM_PROMPT = """You are a text analysis assistant. Analyze the the provided question and report to answer the question. Only answer the question and don't provide extra information in your answer. In your answer be concrete and use None if you can't find the answer in the report.
"""



def _get_report(_d):
    #{'game_id': '524', 'report_url': 'reports/524.txt'}
    if 'game_id' not in _d and ('report_url' not in _d or 'report' not in _d or 'report_path' not in _d ):
        return ValueError(f"The report analysis task requires game_id and report_url or report \nstate:\n{_d}")
    files_path = os.path.join(os.path.dirname(os.getcwd()))
    if 'report_url' in _d:
        _d['report_url'] = f"{files_path}/Rotowire/{_d['report_url']}"
    elif 'report' in _d:
        _d['report_url'] = f"{files_path}/Rotowire/{_d['report']}"
    elif 'report_path' in _d:
        _d['report_url'] = f"{files_path}/Rotowire/{_d['report_path']}"
    return _d


def _load_report(report_url):
    try:
        with open(report_url, "r") as f:
            text = f.read()
            return text
    except FileNotFoundError:
        raise FileNotFoundError(f"report_path <{report_url}> not found")
    except Exception as e:
        raise e

class ExecuteCode(BaseModel):
    reasoning: str = Field(
        ...,
        description="The reasoning behind the answer, including how context is included, if applicable.",
    )

    answer: str = Field(
        ...,
        description="an answer to the question about the report",
    )


def clean_json_string_and_parse(string_dict: str):
    """Cleans a malformed JSON-like string and parses it into a valid Python dictionary.
    
    The function applies the following heuristic fixes:
      1. Replaces Python literals with JSON literals.
      2. Removes spaces around hyphens between digits (e.g. "56 - 26" becomes "56-26").
      3. Inserts spaces between a lowercase and an uppercase letter.
      4. Fixes broken contractions.
      5. Converts keys and values enclosed in single quotes to double quotes.
      6. Removes internal quotes within string values.
      7. Decodes Unicode escapes and fixes common encoding issues.
      8. Removes extraneous whitespace around braces.
      9. Checks for truncation.
    """
    # Step 1: Replace Python-style literals.
    string_dict = string_dict.replace("None", "null")\
                             .replace("True", "true")\
                             .replace("False", "false")
    
    # Step 2: Remove spaces around hyphens between digits.
    string_dict = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1-\2', string_dict)
    
    # Step 3: Insert a space between a lowercase letter and an uppercase letter.
    string_dict = re.sub(r'([a-z])([A-Z])', r'\1 \2', string_dict)
    
    # Step 4: Fix broken contractions like "does nt" -> "doesn't".
    string_dict = re.sub(r'(\w)\s*nt\b', r"\1n't", string_dict)
    
    # Step 5: Convert keys (and values) with single quotes to double quotes.
    # Fix keys: {'key': becomes {"key":
    string_dict = re.sub(r"([{,]\s*)'([a-zA-Z0-9_]+)'\s*:", r'\1"\2":', string_dict)
    # Fix values: : 'value' becomes : "value"
    string_dict = re.sub(r':\s*\'([^\']+)\'([,}])', r': "\1"\2', string_dict)
    
    # Step 6: Remove internal quotes inside string values.
    # For each double-quoted string, remove any internal ' or " characters.
    def remove_internal_quotes(match):
        content = match.group(1)
        cleaned = re.sub(r'[\'"]', '', content)
        return '"' + cleaned + '"'
    string_dict = re.sub(r'"((?:\\.|[^"\\])*)"', remove_internal_quotes, string_dict)
    
    # Step 7: Decode Unicode escapes.
    string_dict = string_dict.encode("utf-8").decode("unicode_escape")
    
    # Step 8: Fix known encoding issues.
    string_dict = string_dict.replace("â", "'").replace("\u2019", "'")
    
    # Step 9: Remove extra spaces around braces.
    string_dict = re.sub(r'\s*([{}])\s*', r'\1', string_dict)
    
    # Step 10: Check for truncation.
    if not re.search(r'[\]}]$', string_dict.strip()):
        print("⚠️ Warning: JSON string might be truncated or incomplete!")
    
    
    # Step 12: Attempt to parse the cleaned string.
    try:
        parsed_dict = json.loads(string_dict)
        print("Successfully parsed JSON!")
        return parsed_dict
    except json.JSONDecodeError as e:
        error_pos = e.pos
        print(f"Error parsing JSON at position {error_pos}: {e.msg}")
        start = max(0, error_pos - 50)
        end = min(len(string_dict), error_pos + 50)
        print("Problematic JSON section:\n", string_dict[start:end])
        return None



def get_text_analysis_tools(llm: ChatOpenAI, llm_local :Optional[ChatOpenAI] = None):
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
    
    def build_message_for_llm_local(context):
        """context
        {"question": question,
         "report_info": [HumanMessage(
             content="The host Toronto Raptors defeated the Philadelphia 76ers, ..."
             )]}
        """
        if "question" in context and "report_info" in context:
            return [
                SystemMessage(content="You are a text analysis assistant. Analyze the the provided question based on the report_info to answer the question.\n Only answer the question and don't provide extra inforamtion in your answer.\nThe output should be in the format: {{'reasoning': '...', 'answer': '...'}}"),
                HumanMessage(content=context["question"]),
                context["report_info"][0],
            ]

    
    def text_analysis(
        question: str,
        context: Union[str, List[str]],
        config: Optional[RunnableConfig] = None,
    ):
        chain_input = {"question": question}
        
        print("context-first:", context, type(context))

        if isinstance(context, str):
            try:
                context = clean_json_string_and_parse(context)
                # print("context-updated:", context, type(context))
                if isinstance(context['data'], str):
                    try:
                        context['data'] = clean_json_string_and_parse(context['data'])

                    except json.JSONDecodeError as e:
                        print(f'JSON inside decode error: {str(e)}')
            except json.JSONDecodeError as e:
                print(f'JSON decode error: {str(e)}')
                        
            # print(type(context['data']),context['data'])
            
            # If the context contains 'data' key, use its value
        elif isinstance(context, list):
            context=context
        else:
            raise ValueError(f"Invalid context type: {type(context)}")

        if 'data' in context:
            #["{'status': 'success', 'data': [{'studydatetime': '2105-09-06 18:18:18'}]}"]
            context = context['data']

        # print("context-after:", context)
            
        reports = [_get_report(ctx) for ctx in context]
        print(reports)
       
        try:    
            if len(reports)>1:
                _answers=[]
                print("# of reports",len(reports))
                for report in reports:
                    _humMessage=_load_report(report['report_url'])
                    # print("_humMessage",_humMessage)
                    chain_input["report_info"] = [
                            HumanMessage(
                                content=_humMessage
                            )
                        ]
                    # if llm_local is None:
                    _model = extractor.invoke(chain_input, config) # error happens here! must use another model call 
                    try:
                        print("model.answer",_model.answer)
                        _answers.append(f"{question}: Considering the information for : {', '.join([f'{k}:{v}' for k, v in report.items() if k!='report_url'])} : {_model.answer}")
                    
                    except Exception as e:
                        return repr(e)
                    # else:
                    #     _model = llm_local.invoke(build_message_for_llm_local(chain_input), config)
                    #     try:
                    #         _parsed_content = clean_json_string_and_parse(_model.content)
                    #         _answer = _parsed_content['answer']
                    #         _reasoning = _parsed_content['reasoning']
                    #         print("model.answer",_answer)
                    #         print("model.reasoning",_reasoning)
                    #         _answers.append(f"Considering the information : {', '.join([f'{k}:{v}' for k, v in report.items() if k!='report_url'])} : {_answer}")
                    #     except Exception as e:
                    #         return repr(e)
                return _answers
            else:
                _humMessage=_load_report(reports[0]['report_url'])
                # print("_humMessage",_humMessage)
                chain_input["report_info"] = [
                        HumanMessage(
                            content=_humMessage
                        )
                    ]
                # if llm_local is None:
                _model = extractor.invoke(chain_input, config) # error happens here! must use another model call 
                try:
                    return f"{question}: Considering the information : {', '.join([f'{k}:{v}' for k, v in reports[0].items() if k!='report_url'])} : {_model.answer}"
                except Exception as e:
                    return repr(e)
                # else:
                #     _model = llm_local.invoke(build_message_for_llm_local(chain_input), config)
                #     try:
                #         _parsed_content = clean_json_string_and_parse(_model.content)
                #         _answer = _parsed_content['answer']
                #         _reasoning = _parsed_content['reasoning']
                #         print("model.answer",_answer)
                #         print("model.reasoning",_reasoning)
                #         return f"Considering the information : {', '.join([f'{k}:{v}' for k, v in reports[0].items() if k!='report_url'])} : {_answer}"
                #     except Exception as e:
                #         return repr(e)
        except ValueError as e:
            return SystemMessage(content=str(e) +'the output of the previous task should be in this format: ["{\'status\': \'success\', \'data\': [{\'game_id\': \'...\',\'report\': \'...\'`}]}"]')               



    return StructuredTool.from_function(
        name="text_analysis",
        func=text_analysis,
        description=_DESCRIPTION,
    )

