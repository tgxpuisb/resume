import os
import PyPDF2
import re
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o", temperature = 0.0)
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm = llm,
    memory = memory
)


def extract_text_from_pdf(pdf_path):
  """从PDF文件中提取文本"""
  with open(pdf_path, 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page_num in range(len(reader.pages)):
      page = reader.pages[page_num]
      text += page.extract_text()
  return text

def remove_invalid_content(text):
  pattern = r'(?=.{16,})[0-9A-Za-z]{16,}'
  cleaned_text = re.sub(pattern, '', text)
  return cleaned_text


pdf_path = 'test.pdf'
pdf_text = remove_invalid_content(extract_text_from_pdf(pdf_path))

print(conversation.predict(input=f"你是一个资深的HR助理，你的任务是从简历内容中准确的整理出有效信息并以json的格式给出，你的回答只需要json形式，以下是简历内容：\n {pdf_text} \n\n 这条信息不需要任何回复，请等待后续指令"))

print(conversation.predict(input="请告诉我用户的手机号和国籍"))