import openai
import PyPDF2
import re
import json


client = openai.OpenAI(
)

messages = [
  {"role": "system", "content": "你是一个资深的HR助理，你的任务是从简历内容中准确整理出有效信息并以json的格式给出，你的回答只需要json形式"}
]
merged_json = {}
def chat_with_gpt4(prompt):
    """调用GPT-4 API进行对话"""
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4",  # 使用GPT-4模型
        messages=messages
    )
    answer = response.choices[0].message.content
    messages.append({"role": "system", "content": answer})
    return answer

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


def main():
  global merged_json
  # 提取PDF文本
  pdf_path = 'test.pdf'
  pdf_text = remove_invalid_content(extract_text_from_pdf(pdf_path))
  # print(pdf_text)

  # # 与GPT-4进行对话
  messages.append({
    "role": "system", "content": f"请根据以下简历内容回答问题：\n{pdf_text}"
  })
  # response = chat_with_gpt4('请告诉我候选人的手机号和用户的国籍，如果用户没写国籍则根据手机号或者文字内容猜测')
  response = chat_with_gpt4(
"""
To summarize the user's English speaking level, the following is the scoring criteria 1=Can recognize and understand very basic words and simple, familiar phrases. Able to write simple isolated phrases and sentences. 2=Can read and understand short, simple texts. Capable of producing basic sentences, writing brief notes and messages related to immediate needs. 3=Can read straightforward information within a known area and write connected text on topics that are familiar or of personal interest. 4=Able to read and write texts about various topics. Can understand the main ideas of complex text and produce clear, detailed writing on a wide range of subjects. 5=Proficient in reading and writing complex texts. Capable of understanding implicit meanings and producing sophisticated and well-structured writing on complex subjects.
Note: in the format {"en_speaking_level": number or null}
Note: If the candidate does not specify this ability, null is returned.
"""
  )

  print("GPT-4的回复：")
  print(response)
  r = json.loads(response)
  print(r)
  # merged_json = {**merged_json, **r}
  # response = chat_with_gpt4('请告诉我用户名，分别以first_name，last_name，full_name的形式给到，如果用户名是中文，请翻译成对应汉语拼音，中文的姓氏对应 last_name 姓氏后的名字对应 first_name. full_name 是 first_name加上last_name')
  # r = json.loads(response)
  # merged_json = {**merged_json, **r}
  # print(response)


if __name__ == "__main__":
  main()