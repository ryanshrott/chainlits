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
import random
import string
from pymongo import MongoClient
import tiktoken
MAX_ITER = 5

llm = ChatOpenAI(temperature=0)
client = MongoClient(os.getenv("MONGO_AUTH"))  # Make sure to set the MONGO_CONNECTION_STRING in your .env
db = client.get_database('smartbids')  # replace 'your_database_name' with your actual database name
clients_collection = db['leads']
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

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
        description="The email address of the client. Ensure it's a real email address and spelled correctly. Never use a generic email address. Leave empty if you don't know it."
    )
    name: Optional[str] = Field(
        None,
        description="The name of the client. Include surname if possible, i.e. John Smith. Never use a generic name. Leave empty if you don't know it."
    )
    city: Optional[str] = Field(
        None,
        description="The name of the city where the client is looking to make a real estate transaction. Ensure it's a real city in Canada and spelled correctly. Never use a generic city. Leave empty if you don't know it."
    )
    preferred_language: Optional[str] = Field(
        None,
        description="The language that the person prefers to communicate in, i.e. English, French, Spanish, etc. Ensure it's a real language and spelled correctly. Never use a generic language. Leave empty if you don't know it."
    )

def send_verification_code(subject, verification_code, to_address):
    print('trying to send email')
    from_address = 'ryan@smartbids.ai'
    password = os.getenv("EMAIL_PASS")
    msg = MIMEMultipart()
    msg['From'] = "SmartBids.ai - Email verification <" + from_address + ">"
    msg['To'] = to_address
    msg['Subject'] = subject
    msg.attach(MIMEText(f"Your verification code is {verification_code}", 'html'))
    server = smtplib.SMTP_SSL('mail.privateemail.com', 465)
    server.login(from_address, password)
    text = msg.as_string()
    server.sendmail(from_address, to_address, text)
    print('email sent')
    server.quit()


def ask_for_info(ask_for = ['name','city', 'preferred_language']):
    # prompt template 1
    first_prompt = ChatPromptTemplate.from_template(
        "Below are some things to ask the user for in a coversation way. You should only ask one question at a time even if you don't get all the info. Don't ask as a list! Don't greet the user! Don't say Hi. Explain you need to get some info. Note that you are a realtor extracting info from a client. If you need to gather email and phone, ask for either one of them.\n\n \
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
    cl.user_session.set('client_verified', False)


def ask_for_info(ask_for = ['name','city', 'preferred_language']):
    # prompt template 1
    first_prompt = ChatPromptTemplate.from_template(
        "Below are some things to ask the user for in a friendly, coversation way. You should only ask one question at a time even if you don't get all the info. Don't ask as a list! Don't greet the user! Don't say Hi. Explain that you need to get some info in order to look up client account and conversation history. Always prioritize gathering client email address first. \n\n \
        ### ask_for list: {ask_for}"
    )
    print(first_prompt)

    # info_gathering_chain
    info_gathering_chain = LLMChain(llm=llm_random, prompt=first_prompt)
    ai_chat = info_gathering_chain.run(ask_for=ask_for)
    return ai_chat


def find_client_in_mongo(email):
    print('finding client', email)
    client = clients_collection.find_one({"email": email})
    print(client)
    return client

def add_lead_to_mongo(details, client_id=None):
    print('Adding/updating lead in MongoDB', details)
    record = {
        'name': details.name,
        'email': details.email,
        'city': details.city,
        'preferred_language': details.preferred_language
    }
    if not client_id:
        clients_collection.insert_one(record)
    else:
        clients_collection.update_one({"_id": client_id}, {"$set": record})

def add_lead_to_octupus(details):
    try:                    
        # Add user to Octupus list
        headers = {
            'Content-Type': 'application/json',
        }

        # Check if OCTOPUS_KEY exists in environment variables
        api_key = os.environ['OCTUPUS_KEY']
        # Assuming email and name are previously defined
        split_name = details.name.split(' ')
        data = {
            "api_key": api_key,
            "email_address": details.email,
            "fields": {"Name": details.name,
                        "FirstName": split_name[0],
                        "City": details.city,
                        "Language": details.preferred_language},
            "tags": ["chatbot"],
            "status": "SUBSCRIBED"
        }

        response = requests.post('https://emailoctopus.com/api/1.6/lists/a7f14044-54c0-11ee-bed9-57e59232c7ed/contacts', headers=headers, data=json.dumps(data))

        print(response.text)

    except Exception as e:
        print(e)

def update_lead_as_verified(client_id):
    print('Updating lead as verified in MongoDB', client_id)
    clients_collection.update_one({"_id": client_id}, {"$set": {"verified": True}})



N = 20  # You can adjust this value based on your requirements
@cl.on_message
async def run_conversation(user_message: str):
    if not cl.user_session.get('lead_gathered'):
        person_details = cl.user_session.get('person_details')
        ask_for, updated_details = filter_response(user_message, person_details)
        if updated_details.email and not cl.user_session.get('new_client') and not cl.user_session.get('client_id'):
            client = find_client_in_mongo(updated_details.email)
            if client:
                cl.user_session.set('client_id', client['_id'])
                if 'name' in client and 'name' in ask_for:
                    updated_details.name = client['name']
                    ask_for.remove('name')
                if 'city' in client and 'city' in ask_for:
                    updated_details.city = client['city']
                    ask_for.remove('city')
                if 'preferred_language' in client and 'preferred_language' in ask_for:
                    updated_details.preferred_language = client['preferred_language']
                    ask_for.remove('preferred_language')
                verified = client.get('verified')
                verified = True if verified else False
                print('verified', verified)
                cl.user_session.set('client_verified', client.get('verified'))
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
            add_lead_to_mongo(updated_details, cl.user_session.get('client_id'))  # Add the lead to Airtable if this is a new client only
            add_lead_to_octupus(updated_details)  # Add the lead to Octupus if this is a new client only
            message_history = cl.user_session.get("message_history")
            person_details = cl.user_session.get('person_details')
            message_history.append({"role": "user", "content": f'My contact information is as follows: \n + {str(person_details)}. From now on, please only respond to me in my preferred language, which is {person_details.preferred_language}.'} )
            cl.user_session.set('person_details', updated_details)
            cl.user_session.set('lead_gathered', True)
            capitalized_first_name = get_first_name_and_capitalize(updated_details.name)
            message_history.append({"role": "assistant", "content": f'Thanks for the info, {capitalized_first_name}. How can I help you today?'})
            if cl.user_session.get('client_verified'):
                ast_msg = f'Thanks for the info, {capitalized_first_name}. I see that your account has been fully verified. How can I help you today?'
            else:
                verification_code = ''.join(random.choice(string.digits) for _ in range(3))
                cl.user_session.set('verification_code', verification_code)
                send_verification_code('SmartBids Verification Code', verification_code, updated_details.email)
                cl.user_session.set('sent_code', True)
                ast_msg = f'Thanks for the info, {capitalized_first_name}. I see that your account has not been fully verified. Please check your email for a verification code. Please paste the code here.'
            await cl.Message(content=ast_msg).send()
            return
    if not cl.user_session.get('client_verified') and cl.user_session.get('sent_code'):
        if user_message == cl.user_session.get('verification_code'):
            cl.user_session.set('client_verified', True)
            if not cl.user_session.get('client_id'):
                cl.user_session.set('client_id', find_client_in_mongo(cl.user_session.get('person_details').email)['_id'])
            update_lead_as_verified(cl.user_session.get('client_id'))
            await cl.Message(content=f'Verification successful! How can I help you today?').send()
            return
        else:
            await cl.Message(content=f'Verification failed. Please try again.').send()
            return

    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": user_message})
    print(message_history[1:])
    #message_history = [message for message in message_history if message["role"] != "function"]
    cur_iter = 0

    
    while cur_iter < MAX_ITER:

        # OpenAI call
        openai_message = {"role": "", "content": ""}
        function_ui_message = None
        content_ui_message = cl.Message(content="")
        messages_model = message_history[:2] + message_history[max(2, len(message_history) - N):]
        token_count = num_tokens_from_messages(messages_model)
        print('token_count', token_count)
        async for stream_resp in await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo-16k-0613" if token_count > 3200 else "gpt-3.5-turbo-0613",
            messages=messages_model,
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
        
