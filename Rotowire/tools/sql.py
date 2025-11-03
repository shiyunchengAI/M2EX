from typing import List, Optional
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from langchain_community.utilities import SQLDatabase
# from langchain import SQLDatabase
from sqlalchemy import create_engine
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Type, Union,List
import re, sys,os
sys.path.append(os.path.dirname(os.getcwd()) + '/src')

# The Table 'players' contains general information about basketball players such as their name, position they play (e.g., Power forward Small forward, Power forward center,Right fielder,...), birth_date, nationality.
# The Table 'teams' contains general information about basketball teams.
# The Table 'players_to_games' maps players to games. This table only links player to the game.
# The Table 'teams_to_games' maps teams to games. This table only links team to the game.
# The Table 'game_reports' contains file path to the text game reports about basketball games. The report includes the majority of the game's statistics and results, which should be extracted using text analysis tools.

from src.utils import _get_db_schema

_meta_data="""


"""

_DESCRIPTION = (
    "text2SQL(problem: str, context: Optional[Union[str,list[str]]])-->str\n"
    "The input for this tools should be `problem` as a textual question\n"
    # Context specific rules below
    "You can optionally provide a list of strings as `context` to help the agent solve the problem. "
    "If there are multiple contexts you need to answer the question, you can provide them as a list of strings.\n"
    "In the 'context' you could add any other information that you think is required to generate te SQL code. It can be the information from previous tasks.\n" 
    "This tools is able to translate the question to the SQL code considering the database information.\n"
    "The SQL code can be executed using sqlite3 library.\n"
    "Use the output of running generated SQL code to answer the question.\n"
)


_SYSTEM_PROMPT = """  
You are a database expert. Generate a SQL query given the following user question, database information and other context that you receive.
You should analyse the question, context and the database schema and come with the executabel sqlite3 query. 
Provide all the required information in the SQL code to answer the original user question that may required in other tasks utilizing the relevant database schema.
Ensure you include all necessary information, including columns used for filtering, especially when the task involves plotting or data exploration.
This must be taken into account when performing any time-based data queries or analyses.
Translate a text question into a SQL query that can be executed on the SQLite database.
You should stick to the available schema including tables and columns in the database and should not bring any new tables or columns.
In SQL query, don't create any alias for the tables or columns. User their original names.
....
"""

_ADDITIONAL_CONTEXT_PROMPT = """
"""


class ExecuteCode(BaseModel):

    reasoning: str = Field(
        ...,
        description="The reasoning behind the SQL expression, including how context is included, if applicable.",
    )

    SQL: str = Field(
        ...,
        description="The SQL Code that can be runnable on the corresponding database ",
    )
    

    
def _execute_sql_query(query: str, db_path: str, as_dict=True) -> Dict[str, Any]:
    try:
        if as_dict:
            import sqlite3
            import json
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            # print("SQL:",change_current_time(query))
            cur.execute(query)
            results = [dict(row) for row in cur.fetchall()]
            # # Ensure results can be serialized to JSON
            # json_str = json.dumps(results)
            # json.loads(json_str)  # Validate JSON loadable
            # results = json_str
            conn.close()
        else:
            engine = create_engine(f'sqlite:///{db_path}')
            database = SQLDatabase(engine, sample_rows_in_table_info=0)
            results = database.run(query)
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}



def get_text2SQL_tools(llm: ChatOpenAI, db_path:str):
    """
    Provide the SQL code from a given question.

    Args:
        raw_question (str): The raw user question.
        schema (str): The database information such as the Tables and Columns.

    Returns:
        results (SQL QUERY str)
    """

    _db_schema = _get_db_schema(db_path,sample_rows_in_table_info=3)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{problem}"),
            ("user", f"{_meta_data} {_db_schema}"),
            MessagesPlaceholder(variable_name="info", optional=True),
        ]
    )
    extractor= prompt | llm.with_structured_output(ExecuteCode)
    
    def text2SQL(
        problem: str,
        context: Optional[Union[str,List[str]]] = None,
    ):
        #tables_columns=_parse_input(tables_columns)
        chain_input = {"problem": problem}
        if context:
            if isinstance(context,list):
                context_str = "\n".join(context)
            else:
                context_str = context
            chain_input["info"] = [HumanMessage(content=context_str)]
        code_model = extractor.invoke(chain_input)
        try:
            return _execute_sql_query(code_model.SQL, db_path)
        except Exception as e:
            # self_debugging 
            err = repr(e)
            _error_handiling_prompt=f"Something went wrong on executing SQL: `{code_model.SQL}`. This is the error I got: `{err}`. \\ Can you fixed the problem and write the fixed SQL code?"
            chain_input["info"] =[HumanMessage(content= [context_str, _error_handiling_prompt])]
            code_model = extractor.invoke(chain_input)
            try:
                return _execute_sql_query(code_model.SQL, db_path)
            except Exception as e:
                return repr(e)

    return StructuredTool.from_function(
        name="text2SQL",
        func=text2SQL,
        description=_DESCRIPTION,
    )
