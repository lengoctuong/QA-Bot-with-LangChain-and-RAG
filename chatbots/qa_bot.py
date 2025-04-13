import os
import tiktoken
from fastapi import HTTPException
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

class LLMService:
    '''
    provider in ['AzureOpenAI', 'Google', 'HuggingFace']
    model_name: e.g 'gpt-35-turbo-16k', 'DeepSeek-R1-houab', gemini-2.0-flash, 'HuggingFaceH4/zephyr-7b-beta',...
    '''

    def __init__(self, provider='AzureOpenAI', model_name='DeepSeek-R1-houab', max_gen_tokens=512, task='text2text-generation', local=False):
        self.provider = provider
        self.model_name = model_name
        self.max_gen_tokens = max_gen_tokens

        try:
            if provider == 'AzureOpenAI':
                if model_name == "gpt-35-turbo-16k":
                    self.model = AzureChatOpenAI(api_version="2024-08-01-preview", azure_deployment=model_name, max_tokens=max_gen_tokens)
                elif model_name == "DeepSeek-R1-houab":
                    self.model = AzureAIChatCompletionsModel(model_name=model_name, max_tokens=max_gen_tokens)
                else:
                    raise HTTPException(status_code=500, detail='Model name is not supported.')
            elif provider == 'Google':
                self.model = ChatGoogleGenerativeAI(model=model_name, max_tokens=max_gen_tokens)
            elif provider == 'HuggingFace':
                if local == False:
                    self.model = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id=model_name, task=task, max_new_tokens=max_gen_tokens), verbose=True)
                # else:
                #     tokenizer = AutoTokenizer.from_pretrained(model_name)
                #     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                #     text2text_pipeline = pipeline(task, model=model, tokenizer=tokenizer)
                #     self.model = HuggingFacePipeline(pipeline=text2text_pipeline)
            else:
                raise HTTPException(status_code=500, detail='Provider is not supported.')
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

class EmbeddingService:
    def __init__(self, model_name='text-embedding-3-large'):
        self.model_name = model_name

        try:
            self.tokenizer = tiktoken.get_encoding(tiktoken.encoding_for_model(model_name).name)
            self.model = AzureOpenAIEmbeddings(
                model=model_name,
                azure_endpoint=os.getenv('AZURE_OPENAI_EMB_ENDPOINT'),
                api_key=os.getenv('AZURE_OPENAI_EMB_API_KEY'),
                openai_api_version="2023-05-15"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))