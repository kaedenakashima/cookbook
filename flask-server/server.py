#  npm i serve -D
#  npx serve -s build
# conda deactivate
# rm -rf venv
# python3 -m venv venv
# source venv/bin/activate
# lsof -i :5000
# kill -9 <PIN>
# pip install -r requirements.txt
#  pip install datetime python-dateutil requests flask python-dotenv flask flask-cors twilio huggingface_hub torch glob2 langchain sentence-transformers faiss-cpu openai
# pip install -U langchain-community pybase64
import os
import sys
import json
import requests
import time
import smtplib
import smtplib, ssl
import glob
import base64
from io import BytesIO
from PIL import Image
from smtplib import SMTPException
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from openai import OpenAI
from huggingface_hub import InferenceClient
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import TavilySearchResults
from langchain import PromptTemplate
# from langchain.llms import OpenAI
from datetime import date 
from transformers import pipeline
load_dotenv()
app = Flask(__name__)
cors = CORS(app, support_credentials=True)
sys.path.insert(0, os.path.dirname(__file__))
@app.route('/0')
def mein1():
    return render_template('index.html', welcomeText='text from python file')
@app.route('/1')
@cross_origin(supports_credentials=True)
def hello_word():
    return '???'
@app.route('/2')
@cross_origin(supports_credentials=True)
def sample():
    return jsonify(
        {
            'sample':[
                'a',
                'b',
                'c'
            ]
        }
    )
@app.route('/3',methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def gettest():
    users = [{'id': 1, 'username': 'sweety'}, {'id': 2, 'username': 'pallavi'}]
    if request.method == 'GET':
        return jsonify({'users': users})
    if request.method == 'POST':
        received1 = request.json
        received2 = received1['data']
        return jsonify({'result': received2})
hf_token = os.environ['HF_TOKEN']
headers = {'Authorization': f'Bearer {hf_token}'}
hf = InferenceClient()
@app.route('/llm/abconversation',methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def getScript():
    if request.method == 'POST':
        received1 = request.json
        received2 = received1['first-msg']
        msgsA = ['start!']
        msgsB = [received2]
        msgsAB = []
        def sayA():
            msg = [{'role': 'system', 'content': 'Please answer with short sentence.'}]
            for x, y in zip(msgsA, msgsB):
                msg.append({'role': 'assistant', 'content': x})
                msg.append({'role': 'user', 'content': y})
            print('sayA:',msg)
            llm = hf.chat.completions.create(
                model= 'meta-llama/Meta-Llama-3-8B-Instruct',
                messages=msg
            )
            return llm.choices[0].message.content
        def sayB():
            msg = [{'role': 'user', 'content': 'Please answer with short sentence.'}]
            for x, y in zip(msgsA, msgsB):
                msg.append({'role': 'user', 'content': x})
                msg.append({'role': 'assistant', 'content': y})
            msg.append({'role': 'user', 'content': msgsA[-1]})
            print('sayA:',msg)
            llm = hf.chat.completions.create(
                model= 'meta-llama/Meta-Llama-3-8B-Instruct',
                messages=msg
            )
            return llm.choices[0].message.content
        for i in range(5):
            a_next = sayA()
            msgsA.append(a_next)
            msgsAB.append({'user':'A','msg':a_next})
            b_next = sayB()
            msgsB.append(b_next)
            msgsAB.append({'user':'B','msg':b_next})
        print(msgsAB)
        return jsonify(msgsAB)
@app.route('/llm/translator/en-es',methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def getTranslation():
    if request.method == 'POST':
        received1 = request.json
        received2 = received1['en']
        data = json.dumps({'inputs':received2})
        time.sleep(1)
        while True:
            try:
                response = requests.request('POST', 'https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-es', headers=headers, data=data)
                break
            except Exception:
                continue
        res = json.loads(response.content.decode('utf-8'))
        return jsonify({'es':res[0]['translation_text']})
@app.route('/llm/context',methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def getContext():
    if request.method == 'POST':
        received1 = request.json
        received2 = received1['word']
        qa = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', device='mps:0')
        context='''
        Kernel:The core part of an operating system responsible for managing hardware, memory, and processes.
        AI:Artificial Intelligence is a broad term used to describe engineered systems that have been taught to do a task that typically requires human intelligence.
        Data Lake:A storage repository where data is stored in its raw format. Data lakes allow for more flexibility than a more rigid data warehouse.
        '''
        return jsonify({'definition':(qa(question=received2, context=context))['answer']})
openai_api_key = os.environ['OPENAI_API_KEY']   
openai = OpenAI()
msgsUser3 = []
msgsSystem3= []
@app.route('/llm/tools',methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def getTools():
    received1 = request.json
    received2 = received1['question']
    print(received2)
    cpu_prices ={'intel core ultra 200s': 'cost range is $299 to $619','intel 14th gen':'$90 to $432','amd ryzen 9000-series': 'cost range is $229 to $869','amd ryzen 7000-series':'$178 to $739','amd ryzen 5000-series':'$107 to $337'}
    def get_cpu_price(x):
        print(f'Tool get_cpu_price called for {x}')
        product = x.lower()
        return cpu_prices.get(product, 'Unknown')
    price_function = {
        "name": "get_cpu_price",
        "description": "Get the price range of the CPU. Call this whenever you need to know the product price, for example when a customer asks 'How much is the CPU?'",
        "parameters": {
            "type": "object",
            "properties": {
                "cpu_name": {
                    "type": "string",
                    "description": "The cpu that the customer wants to buy",
                },
            },
            "required": ["cpu_name"],
            "additionalProperties": False
        }
    }
    tools = [{'type': 'function', 'function': price_function}]
    sys = 'You are a helpful assistant for an Electric Store called BestBuy. '
    sys += 'Give short, courteous answers, no more than 1 sentence. '
    sys += "Always be accurate. If you don't know the answer, say so."
    def chat(x): 
        print(f'A:{x}\n') 
        msgs = [{'role': 'system', 'content': sys}] 
        for x1, y1 in zip(msgsUser3, msgsSystem3):
            msgs.append({'role': 'user', 'content': x1})
            msgs.append({'role': 'assistant', 'content': y1})
        msgs += [{'role': 'user', 'content': x}]
        res = openai.chat.completions.create(model='gpt-4o-mini', messages=msgs, tools=tools)
        print('res',res.choices[0])
        msgsUser3.append(x)
        if res.choices[0].finish_reason=='tool_calls':
            msg = res.choices[0].message
            res, product = handle_tool_call(msg)
            msgs.append(msg)
            msgs.append(res)
            res = openai.chat.completions.create(model='gpt-4o-mini', messages=msgs)
            msgsSystem3.append(res.choices[0].message.content)
            print(f'B:{res.choices[0].message.content}\n') 
            return res.choices[0].message.content
        else:
            msgsSystem3.append(res.choices[0].message.content)
            print(f'B:{res.choices[0].message.content}\n') 
            return res.choices[0].message.content
    def handle_tool_call(x):
        tool_call = x.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        product = arguments.get('cpu_name')
        price = get_cpu_price(product)
        response = {
            'role': 'tool',
            'content': json.dumps({'cpu_name': product,'price': price}),
            'tool_call_id': tool_call.id
        }
        return response, product
    return jsonify({'answer':chat(received2)})
    return jsonify({'answer':chat(received2)})
SystemMsgs1= ['']
UserMsgs1  = []
@app.route('/llm/speech-to-text',methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def getAnswer():
    if request.method == 'POST':
        received1 = request.json
        received2 = received1['user']
        UserMsgs1.append(received2)
        msg = [{'role': 'system', 'content': 'Please answer with short sentence.'}]
        for x, y in zip(SystemMsgs1, UserMsgs1):
            msg.append({'role': 'assistant', 'content': x})
            msg.append({'role': 'user', 'content': y})
        llm = hf.chat.completions.create(
            model= 'meta-llama/Meta-Llama-3-8B-Instruct',
            messages=msg
        )
        res = llm.choices[0].message.content
        SystemMsgs1.append(res)
        return jsonify({'answer':res})
SystemMsgs2= ['']
UserMsgs2  = []
@app.route('/llm/rag1',methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def getRag1():
    if request.method == 'POST':
        received1 = request.json
        received2 = received1['question']
        rag_context = {}
        #1 create API
        business = glob.glob('coreweave/md/company/*')
        for x1 in business:
            name = x1.split(' ')[-1][:-3]
            x2 = ''
            with open(x1, 'r', encoding='utf-8') as f:
                x2 = f.read()
                rag_context[name]=x2
        products = glob.glob('coreweave/md/products/*')
        for x1 in products:
            name = x1.split(os.sep)[-1][:-3]
            x2 = ''
            with open(x1, 'r', encoding='utf-8') as f:
                x2 = f.read()
                rag_context[name]=x2
        employee = glob.glob('coreweave/md/employee/*')
        for x1 in employee:
            name = x1.split(os.sep)[-1][:-3]
            x2 = ''
            with open(x1, 'r', encoding='utf-8') as f:
                x2 = f.read()
                rag_context[name]=x2

        #2-1 get md if keywords*(*md titles) is included in question.
        def find_context(x1):
            x2 = []
            for x3, y in rag_context.items(): #x3-name,y-details
                print(x3.lower(), y)
                if x3.lower() in x1.lower():
                    x2.append(y)
            return x2            
        #2-2 use context to ask questions  
        def md_context(x1):
            x2 = find_context(x1)
            if x2:
                x1 += '\n\nThe following additional context might be relevant in answering this question:\n\n'
                for y in x2:
                    x1 += y + '\n\n'
            return x1
        msg = [{'role': 'system', 'content': "You are an expert in answering accurate questions about CoreWeave, the cloud-computing company. Give brief, accurate answers. If you don't know the answer, say so. Do not make anything up if you haven't been provided with relevant context."}]
        for x, y in zip(SystemMsgs2, UserMsgs2):
            msg.append({'role': 'assistant', 'content': x})
            msg.append({'role': 'user', 'content': y})
        msg.append({'role': 'user', 'content': md_context(received2)})   
        llm = hf.chat.completions.create(
            model= 'meta-llama/Meta-Llama-3-8B-Instruct',
            messages=msg
        )
        print('msg: ',msg)
        res = llm.choices[0].message.content
        SystemMsgs2.append(res)
        return jsonify({'answer':res})
@app.route('/llm/rag2',methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def getRag2():
    if request.method == 'POST':
        received1 = request.json
        received2 = received1['question']
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-l6-v2')
        folders = glob.glob('coreweave/md/*')
        documents = []
        for x1 in folders:
            doc_type = os.path.basename(x1)
            loader = DirectoryLoader(x1, glob='**/*.md', loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
            folder_docs = loader.load()
            for x2 in folder_docs:
                x2.metadata['doc_type'] = doc_type
                documents.append(x2)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        # FAISS gets related data from vector formatted data
        db = FAISS.from_documents(chunks, embeddings)
        searchDocs = db.similarity_search(received2)
        return jsonify({'answer':searchDocs[0].page_content})
@app.route('/llm/generate-img',methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def getAIImg():
    if request.method == 'POST':
        received1 = request.json
        received2 = received1['theme']
        # image_response = openai.images.generate(
        #     model = "dall-e-3",
        #     prompt=f'Create a zoomed luxury picture with road that car can cross the image left to right. Do not include car in image. And set background as {received2}.',
        #     n=1,
        #     size="1024x1024",
        #     response_format="url",
        # )
        # generated_image_filepath = os.path.join('./images', 'ai-generated-pic.png')
        # generated_image = requests.get(image_response.data[0].url).content
        # with open(generated_image_filepath, 'wb') as image_file:
        #     image_file.write(generated_image) 
        with open('./images/ai-generated-pic.png', 'rb') as x:
            img = base64.b64encode(x.read())
        img = img.decode('utf-8')
        return jsonify({'img':img})
@app.route('/llm/stock-news',methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def getNews():
    if request.method == 'POST':
        received1 = request.json
        received2 = received1['stockname']
        try:
            with open('data.json') as f:
                alldata = json.load(f)
        except FileNotFoundError:
            print('system: File not found.')
        else:
            if(alldata['stock_news'][received2][0]['updateddate'] == date.today()):
                return jsonify({'stock-info':alldata[received2][0]})
            else:
                news_params={
                    'apiKey':os.environ['NEWS_API'],
                    'qInTitle':received2,
                }
                res1=requests.get('https://newsapi.org/v2/everything', params=news_params)
                res2 =res1.json()['articles'][:1]
                for x1 in res2:
                    classifier = pipeline('sentiment-analysis')
                    res3 = classifier(x1['description'])
                updateItem = {'title':x1['title'], 'description':x1['description'], 'content':x1['content'], 'res':res3[0]['label'], 'score':res3[0]['score'],'updateddate':str(date.today())}
                try:
                    with open('data.json') as f:
                        alldata = json.load(f)
                except FileNotFoundError:
                    print('system: File not found.')
                else:
                    for x, y in alldata['stock_news'].items():
                        if x==received2:
                            alldata['stock_news'][received2][0]=updateItem
                    with open('data.json', 'w') as f:
                        json.dump(alldata, f)  
                return jsonify({'stock-info':updateItem})
@app.route('/llm/summarize',methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def getSummery():
    if request.method == 'POST':
        received1 = request.json
        received2 = received1['context']
        template: str = """
            Given the information about article {question} I want you to create:
            1. a short summary
            2. list up 5 important things
        """
        prompt = PromptTemplate.from_template(template=template)
        prompt_formatted_str: str = prompt.format(
            question= {received2}
        )
        return jsonify({'answer':openai.predict(prompt_formatted_str)})
@app.route('/llm/googlesearch',methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def getGoogleSearch():
    if request.method == 'POST':
        received1 = request.json
        received2 = received1['question']
        ctx = TavilySearchResults().run(f'{received2}')
        print('ctx',ctx)
        msgs = [{'role': 'system', 'content': f"You are an assistant. Please answer by 1 short sentence less than 10 words. Based on this data: data {ctx[0]}"}]
        llm = InferenceClient('meta-llama/Meta-Llama-3-8B-Instruct')
        res = llm.chat_completion(messages=msgs, max_tokens=100)
        return jsonify({'answer':res.choices[0].message.content,'url':ctx[0]['url']})
@app.route('/mail-notification',methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def mail1():
    if request.method == 'GET':
        return jsonify({'info': 'input emailadress to send a mail'})
    if request.method == 'POST':
        received1 = request.json
        received2 = received1['data']
        with open('./mail_templates/verify_account.txt') as mailscript_file:
            contents = mailscript_file.read()
            contents = contents.replace('[EMAIL]', received2)
        msg = MIMEMultipart('alterative')
        msg['Subject']='Katie Portfolio One Time Passcode'
        msg['From']=os.environ['EMAIL_FROM']
        msg['To']=received2
        msg.attach(MIMEText(contents, 'plain'))
        context=ssl.create_default_context()
        with smtplib.SMTP_SSL("kaedenakashima.com",465, context=context) as connection:
            connection.login(os.environ['EMAIL_FROM'],password=os.environ['SMTP_PWD'])
            connection.sendmail(
                os.environ['EMAIL_FROM'],
                received2,
                msg.as_string()
            )
            response_code, response_msg = connection.noop()
            connection.quit()
        return jsonify({'result':'email sent'})
    return jsonify({'result':'email could not sent'})
client = Client(os.environ['TWILIO_ACCOUNT_SID'], os.environ['TWILIO_AUTH_TOKEN'])
@app.route('/sms-notification',methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def sms():
    if request.method == 'GET':
        return jsonify({'info': 'input phone number to send a messege'})
    if request.method == 'POST':
        received1 = request.json
        received2 = received1['data']
        client.messages.create(
            from_=os.environ['TWILIO_PHONE_NUMBER'],
            to=received2,
            body = 'You have logged in'
        )
        return jsonify({'result':'sms sent'})
    return jsonify({'result':'sms could not sent'})

if __name__== '__main__':
    app.run()