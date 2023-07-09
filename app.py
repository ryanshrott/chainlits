import openai
import json, ast
import os
import chainlit as cl
from dotenv import load_dotenv
import openai
import os
from funkagent import agents, parser
load_dotenv(".env")
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
from helper_utils import *
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from chainlit import AskUserMessage, Message, on_chat_start
from function_utils import *
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
openai.api_key = os.environ.get("OPENAI_API_KEY")
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.prompts import ChatPromptTemplate
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
from airtable import Airtable

MAX_ITER = 5

llm = ChatOpenAI(temperature=0)
airtable = Airtable(os.environ['AIRTABLE_BASE_ID'], os.environ['AIRTABLE_TABLE_NAME'], api_key=os.environ['AIRTABLE_API_KEY'])

async def process_new_delta(new_delta, openai_message, content_ui_message, function_ui_message):
    if "role" in new_delta:
        openai_message["role"] = new_delta["role"]
    if "content" in new_delta:
        new_content = new_delta.get("content") or ""
        openai_message["content"] += new_content
        await content_ui_message.stream_token(new_content)
    if "function_call" in new_delta:
        if "name" in new_delta["function_call"]:
            openai_message["function_call"] = {
                "name": new_delta["function_call"]["name"]}
            await content_ui_message.send()
            function_ui_message = cl.Message(
                author=new_delta["function_call"]["name"],
                content="", indent=1, language="json")
            await function_ui_message.stream_token(new_delta["function_call"]["name"])

        if "arguments" in new_delta["function_call"]:
            if "arguments" not in openai_message["function_call"]:
                openai_message["function_call"]["arguments"] = ""
            openai_message["function_call"]["arguments"] += new_delta["function_call"]["arguments"]
            await function_ui_message.stream_token(new_delta["function_call"]["arguments"])
    return openai_message, content_ui_message, function_ui_message


class PersonalDetails(BaseModel):
    email: Optional[str] = Field(
        None,
        description="The email address that client associates as theirs i.e. johndoe@gmail.com"
    )
    name: Optional[str] = Field(
        None,
        description="The name of the client. Include surname if possible, i.e. John Smith."
    )
    city: Optional[str] = Field(
        None,
        description="The name of the city where the client is looking to make a real estate transaction. Ensure it's a real city in Canada and spelled correctly. i.e. Toronto"
    )
    preferred_language: Optional[str] = Field(
        None,
        description="The language that the person prefers to communicate in, i.e. English, French, Spanish, etc."
    )



def ask_for_info(ask_for = ['name','city', 'preferred_language']):
    # prompt template 1
    first_prompt = ChatPromptTemplate.from_template(
        "Below are some things to ask the user for in a coversation way. You should only ask one question at a time even if you don't get all the info. Don't ask as a list! Don't greet the user! Don't say Hi. Explain you need to get some info. Note that you are a realtor extracting info from a client. If you need to gather email and phone, ask for either one of them. Prioritize collecting email address. \n\n \
        ### ask_for list: {ask_for}"
    )

    # info_gathering_chain
    info_gathering_chain = LLMChain(llm=llm_random, prompt=first_prompt)
    ai_chat = info_gathering_chain.run(ask_for=ask_for)
    return ai_chat

def check_what_is_empty(user_peronal_details):
    ask_for = []
    # Check if fields are empty
    for field, value in user_peronal_details.dict().items():
        if value in [None, "", 0]:  # You can add other 'empty' conditions as per your requirements
            ask_for.append(f'{field}')
    return ask_for


## checking the response and adding it
def add_non_empty_details(current_details: PersonalDetails, new_details: PersonalDetails):
    non_empty_details = {k: v for k, v in new_details.dict().items() if v not in [None, ""]}
    updated_details = current_details.copy(update=non_empty_details)
    return updated_details

def filter_response(text_input, user_details):
    chain = create_tagging_chain_pydantic(PersonalDetails, llm)
    print('text_input', text_input)
    res = chain.run(text_input)
    # add filtered info to the
    print('res', res)
    user_details = add_non_empty_details(user_details,res)
    print('user_details', user_details)
    ask_for = check_what_is_empty(user_details)
    return ask_for, user_details

def get_first_name_and_capitalize(full_name):
    first_name = full_name.split()[0]
    capitalized_first_name = first_name.capitalize()
    return capitalized_first_name

@cl.on_chat_start
async def start_chat():
    ask_msg = '''Hey there! I'm your friendly Canadian realtor, ready to help with your housing adventures! Whether you're hunting for a dream home, looking to sell, or need advice on market trends and mortgages, I've got your back. To get started, could I kindly get your email address in case we get disconnected?'''
    await cl.Message(content=ask_msg).send()
    cl.user_session.set(
        "message_history",
        [
            {"role": "system", "content": sys_msg},
            {"role": "assistant", "content": ask_msg},
        ],
    )
    cl.user_session.set('person_details', PersonalDetails(name="",
                                city="",
                                email="",
                                preferred_language=""))
    cl.user_session.set('lead_gathered', False)
    cl.user_session.set('new_client', False)


def ask_for_info(ask_for = ['name','city', 'preferred_language']):
    # prompt template 1
    first_prompt = ChatPromptTemplate.from_template(
        "Below are some things to ask the user for in a coversation way. You should only ask one question at a time even if you don't get all the info. Don't ask as a list! Don't greet the user! Don't say Hi. Explain you need to get some info. Note that you are a realtor extracting info from a client. Always prioritize gathering client email address first. \n\n \
        ### ask_for list: {ask_for}"
    )

    # info_gathering_chain
    info_gathering_chain = LLMChain(llm=llm, prompt=first_prompt)
    ai_chat = info_gathering_chain.run(ask_for=ask_for)
    return ai_chat


def find_client_in_airtable(email):
    print('finding client', email)
    client = airtable.search('email', email)
    print(client)
    return client
def add_lead_to_airtable(details, client_id=None):
    print('Adding/updating lead in Airtable', details)
    record = {
        'name': details.name,
        'email': details.email,
        'city': details.city,
        'preferred_language': details.preferred_language
    }
    if not client_id:
        airtable.insert(record)
    else:
        airtable.update(client_id, record)


N = 10  # You can adjust this value based on your requirements
@cl.on_message
async def run_conversation(user_message: str):
    if not cl.user_session.get('lead_gathered'):
        person_details = cl.user_session.get('person_details')
        ask_for, updated_details = filter_response(user_message, person_details)
        if updated_details.email and not cl.user_session.get('new_client') and not cl.user_session.get('client_id'):
            client = find_client_in_airtable(updated_details.email)
            if client:
                cl.user_session.set('client_id', client[0]['id'])
                if 'name' in client[0]['fields']:
                    updated_details.name = client[0]['fields']['name']
                    ask_for.remove('name')
                if 'city' in client[0]['fields']:
                    updated_details.city = client[0]['fields']['city']
                    ask_for.remove('city')
                if 'preferred_language' in client[0]['fields']:
                    updated_details.preferred_language = client[0]['fields']['preferred_language']
                    ask_for.remove('preferred_language')
            else: # it's a new client
                cl.user_session.set('new_client', True)
        cl.user_session.set('person_details', updated_details)
        print(cl.user_session.get('person_details'))
        print(ask_for)
        if ask_for:
            ai_chat = ask_for_info(ask_for)
            await cl.Message(content=ai_chat).send()
            return
        
        else:
            print('LEAD GATHERED!')
            add_lead_to_airtable(updated_details, cl.user_session.get('client_id'))  # Add the lead to Airtable if this is a new client only
            message_history = cl.user_session.get("message_history")
            person_details = cl.user_session.get('person_details')
            message_history.append({"role": "user", "content": f'My contact information is as follows: \n + {str(person_details)}. From now on, please only respond to me in my preferred language, which is {person_details.preferred_language}.'} )
            cl.user_session.set('person_details', updated_details)
            cl.user_session.set('lead_gathered', True)
            capitalized_first_name = get_first_name_and_capitalize(updated_details.name)
            ast_msg = f'Thanks for the info, {capitalized_first_name}. How can I help you today?'
            message_history.append({"role": "assistant", "content": ast_msg})
            await cl.Message(content=ast_msg).send()
            return
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": user_message})
    print(message_history[1:])
    cur_iter = 0

    while cur_iter < MAX_ITER:

        # OpenAI call
        openai_message = {"role": "", "content": ""}
        function_ui_message = None
        content_ui_message = cl.Message(content="")
        async for stream_resp in await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo-0613",
            messages=message_history[-N:],
            stream=True,
            function_call="auto",
            functions=functions,
            temperature=0
        ):

            new_delta = stream_resp.choices[0]["delta"]
            openai_message, content_ui_message, function_ui_message = await process_new_delta(
                new_delta, openai_message, content_ui_message, function_ui_message)

        message_history.append(openai_message)
        if function_ui_message is not None:
            await function_ui_message.send()

        if stream_resp.choices[0]["finish_reason"] == "stop":
            break

        elif stream_resp.choices[0]["finish_reason"] != "function_call":
            raise ValueError(stream_resp.choices[0]["finish_reason"])

        # if code arrives here, it means there is a function call
        function_name = openai_message.get("function_call").get("name")
        arguments = ast.literal_eval(
            openai_message.get("function_call").get("arguments"))

        function_response = handle_function_request(function_name, arguments)

        message_history.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        ) 

        await cl.Message(
            author=function_name,
            content=str(function_response),
            language="json",
            indent=1,
        ).send()
        cur_iter += 1