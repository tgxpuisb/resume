import os
import re
import json
import traceback
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader

# LangChain 相关
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# 初始化 LangChain 模型（若无 GPT-4 权限可改为 "gpt-3.5-turbo"）
chat_model = ChatOpenAI(model="gpt-4", temperature=0.0)

"""
PROMPTS 定义了多个解析需求。注意：
1. 所有需要作为文本出现的花括号都使用双花括号 `{{` 和 `}}` 转义。
2. 仅使用 {pdf_text} 作为占位符。
3. 在找不到信息时，Prompt 让模型返回空字段，避免 KeyError。
"""
PROMPTS = {
    "phone_number": (
        "你是一个资深的HR助理，你的任务是从简历内容中准确的整理出有效信息并以json的格式给出，你的回答只需要json形式，以下是简历内容：\n{{\"phone\": \"+86 123456789\"}}。\n\n"
        "请根据以下简历内容回答问题：\n{pdf_text}\n\n"
        "请告诉我候选人的手机号 phone 手机号前需要有候选人的国籍信息如果候选人没填写国籍信息则根据简历内容中的工作地点进行猜测，example: 如果候选人是中国则数据是 +86，如果候选人是美国则数据是 +1，地址和手机号中间不需要有空格。"
        "如果无法找到手机号，请返回 {{\"phone\": \"\"}}。"
    ),
    "name_extraction": (
        "请告诉我候选人用户名，分别以first_name，last_name，full_name的形式给到，如果用户名是中文，请翻译成对应汉语拼音，中文的姓氏对应 last_name 姓氏后的名字对应 first_name. full_name 是 first_name加上last_name。"
        "返回格式仅需JSON形式，示例：{{\"first_name\": \"Wei\", \"last_name\": \"Wang\", \"full_name\": \"Wei Wang\"}}。"
        "如果无法找到姓名，请返回 {{\"first_name\": \"\", \"last_name\": \"\", \"full_name\": \"\"}}。"
        "\n\n简历内容：\n{pdf_text}"
    ),
    "work_years": (
        "请告诉我候选人的工作年限，返回格式为{{\"work_year\": number}}，如果候选人在简历内直接写明工作年限则直接使用，如果候选人没有写明，则根据候选人的第一份工作和最后一份的时间间隔进行计算。"
        "如果无法找到工作年限信息，请返回 {{\"work_year\": 0}}。"
        "\n\n简历内容：\n{pdf_text}"
    ),
    "job_titles": (
        "这里是一份job title 的 json 数据：[{\"id\":13,\"name\":\"Admin\"}, ... {\"id\":135,\"name\":\"CircleCI\"}]."
        "你的任务是根据候选人的简历总结用户的job title, 你总结的候选人job title需要是在候选人在项目中主要从事的内容，请给出最多不超过5个job title，以 {{\"titles\": []}} 的数据结构返回。"
        "如果无法找到job title，请返回 {{\"titles\": []}}。"
        "\n\n简历内容：\n{pdf_text}"
    ),
    "tagline": (
        "为候选人总结一个英文的 tagline，格式为 {{\"tagline\": \"tagline detail\"}}。Although all kinds of content can be employed as a tagline, here are some general types of taglines you might want to use: \n"
        "1. A philosophy mini-statement or personal description \n"
        "2. One or more achievements each described separately \n"
        "3. A mini achievement summarizing career-long accomplishments\n\n"
        "如果无法生成tagline，请返回 {{\"tagline\": \"\"}}。"
        "\n\n简历内容：\n{pdf_text}"
    ),
    "summary": (
        "为候选人的简历写一个英文的 summary, A resume summary statement is a two- to three-sentence professional introduction that you can add at the top of your resume to highlight your most valuable skills and experiences. The resume summary can help employers quickly learn whether you have the skills and background they require."
        "summary 格式为 {{\"self_introduce\": \"summary content\"}} 其中 summary content 可以是一个由<ol><li></li></ol>组成的有序列表的html字符串。"
        "如果无法生成summary，请返回 {{\"self_introduce\": \"\"}}。"
        "\n\n简历内容：\n{pdf_text}"
    ),
    "work_experiences": (
        "Please summarize the candidate's work experiences and projects in each work experience, I will use Typescript type declarations and annotations to describe the data structure:\n"
        "type workExperiences = {\n"
        "  experiences: Array<{\n"
        "    where: string // The company the candidate works for  translate to english\n"
        "    from: string // Candidate's employment start date\n"
        "    to: string // The end time of the candidate's employment\n"
        "    mainly_as: string // The candidate's job level or position  translate to english\n"
        "    description: string // Description and summary of the candidate's work in the company. If the content involves multiple items and is independent of each other, you can use an html string description consisting of an ordered list of <ol><li></li></ol>\n"
        "  }>\n"
        "}\n"
        "Note: The resume may be written in Chinese, and the summary needs to be translated into English."
        "\n\n简历内容：\n{pdf_text}"
    ),
    "company_skills_rating": (
        "这是一份公司信息技术的技能列表: [{\"i\":8,\"n\":\"React\",\"t\":\"Front End\"}, ... {\"i\":135,\"n\":\"CircleCI\",\"t\":\"DevOps\"}]."
        "你的任务是结合候选人简历对候选人的信息技术技能进行评级,评级级别为1-5的正整数熟练程度随着数字增大而增大,1代表候选人只是使用过该技术3代表候选人对技术比较熟练5代表候选人精通该项技术并熟悉这项技术的运行原理。"
        "注意: 如果候选人简历中没有提到公司技术列表中的技术则不需要评级。"
        "请以以下格式返回：{{\"skills\": [{\"id\": \"number\", \"name\": \"name\", \"type\": \"大类别\", \"level\": \"评级\"}]}}。"
        "如果没有需要评级的技能，请返回 {{\"skills\": []}}。"
        "\n\n简历内容：\n{pdf_text}"
    ),
    "educations": (
        "Please summarize the educations in the candidate's resume., I will use Typescript type declarations and annotations to describe the data structure:\n"
        "type educations = {\n"
        "  educations: Array<{\n"
        "    where: string // Where the candidate received his or her education translate to english\n"
        "    degree: string // degree  translate to english\n"
        "    from: string // The time when the candidate's education started  translate to english\n"
        "    to: string // The time when the candidate's education ended translate to english\n"
        "    mainly_as: string // college major translate to english\n"
        "  }>\n"
        "}\n"
        "Note: The resume may be written in Chinese, and the summary needs to be translated into English."
        "\n\n简历内容：\n{pdf_text}"
    ),
    "patents": (
        "Summary of candidate patents, in the format {{\"patents\": \"string\"}}, If the content involves multiple items, you can use an HTML string description consisting of an ordered list of <ol><li></li></ol>."
        "Note: if candidate has no Patents give the empty string."
        "Note: The resume may be written in Chinese, and the summary needs to be translated into English."
        "\n\n简历内容：\n{pdf_text}"
    ),
    "en_speaking_level": (
        "To summarize the user's English speaking level, the following is the scoring criteria:\n"
        "1=Can recognize and understand very basic words and simple, familiar phrases. Able to write simple isolated phrases and sentences.\n"
        "2=Can read and understand short, simple texts. Capable of producing basic sentences, writing brief notes and messages related to immediate needs.\n"
        "3=Can read straightforward information within a known area and write connected text on topics that are familiar or of personal interest.\n"
        "4=Able to read and write texts about various topics. Can understand the main ideas of complex text and produce clear, detailed writing on a wide range of subjects.\n"
        "5=Proficient in reading and writing complex texts. Capable of understanding implicit meanings and producing sophisticated and well-structured writing on complex subjects.\n"
        "Note: in the format {{\"en_speaking_level\": number or null}}.\n"
        "Note: If the candidate does not specify this ability, null is returned."
        "\n\n简历内容：\n{pdf_text}"
    ),
    "en_read_write_level": (
        "To summarize the user's English read and write level, the following is the scoring criteria:\n"
        "1=Can use simple phrases and sentences to communicate basic needs in familiar contexts. Interaction is limited to slow speech and repetition.\n"
        "2=Able to engage in simple conversation on familiar topics with some assistance. Can ask and answer basic questions and use common expressions.\n"
        "3=Can handle short, routine exchanges without disruption. Able to communicate in simple and direct exchanges of information on familiar tasks and topics.\n"
        "4=Can communicate with some confidence on familiar routine and non-routine matters. Able to express personal opinions but sometimes requires assistance for more complex conversation.\n"
        "5=Fully fluent and comfortable in any situation. Can engage in nuanced, complex conversations, understanding slang, idioms, and cultural references.\n"
        "Note: in the format {{\"en_read_write_level\": number or null}}.\n"
        "Note: If the candidate does not specify this ability, null is returned."
        "\n\n简历内容：\n{pdf_text}"
    )
}

def extract_text_from_pdf(file_path):
    """
    使用 PyPDF2 提取 PDF 文本。若遇到异常或无法提取，则返回空字符串。
    """
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        app.logger.error("Error extracting PDF text: %s", e, exc_info=True)
        return ""

def clean_pdf_text(pdf_text):
    """
    简单地删除 16 位以上的随机字符串，可视情况定制。
    """
    pattern = r'(?=.{16,})[0-9A-Za-z]{16,}'
    return re.sub(pattern, '', pdf_text)

def process_prompt(task_name, pdf_text):
    """
    调用 LangChain，发送 HumanMessage，并返回其 content。
    若出错，返回一个可 JSON 化的错误提示字符串，供后续解析。
    """
    try:
        prompt_template = PROMPTS[task_name]
    except KeyError:
        # 若字典里没有对应 key，返回错误信息
        return json.dumps({"error": f"No prompt for task '{task_name}'"})

    # 只对 pdf_text 做插值，不插值其他
    formatted_prompt = prompt_template.format(pdf_text=pdf_text)

    try:
        response = chat_model([HumanMessage(content=formatted_prompt)])
        return response.content  # 通常为 JSON 格式字符串
    except Exception as e:
        app.logger.error("Error calling LangChain for task %s: %s", task_name, e, exc_info=True)
        return json.dumps({"error": "LangChain call failed", "details": str(e)})

@app.route("/upload", methods=["POST"])
def upload_pdf():
    """
    1. 上传 PDF 文件
    2. 提取并清理文本
    3. 依次调用各个 prompt
    4. 若无法解析成 JSON 或缺少字段，就返回空或默认值，而不是报错
    """
    # region: 检查上传
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed"}), 400
    # endregion

    # region: 保存PDF到本地
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    try:
        file.save(file_path)
    except Exception as e:
        app.logger.error("Failed to save file: %s", e, exc_info=True)
        return jsonify({"error": "Failed to save file"}), 500
    # endregion

    # region: 提取并清理文本
    pdf_text_raw = extract_text_from_pdf(file_path)
    pdf_text_cleaned = clean_pdf_text(pdf_text_raw)
    # endregion

    results = {}
    for task_name in PROMPTS:
        response_str = process_prompt(task_name, pdf_text_cleaned)

        # 解析 JSON
        try:
            parsed_result = json.loads(response_str)
        except json.JSONDecodeError:
            parsed_result = {
                "error": "Failed to parse JSON",
                "raw_response": response_str
            }

        # 字段检查：若缺失则补默认值
        if task_name == "phone_number":
            # 应当包含 { "phone": "" }
            if isinstance(parsed_result, dict) and "phone" not in parsed_result:
                parsed_result["phone"] = ""
        elif task_name == "name_extraction":
            if isinstance(parsed_result, dict):
                parsed_result.setdefault("first_name", "")
                parsed_result.setdefault("last_name", "")
                parsed_result.setdefault("full_name", "")
        elif task_name == "work_years":
            if isinstance(parsed_result, dict) and "work_year" not in parsed_result:
                parsed_result["work_year"] = 0
        elif task_name == "job_titles":
            if isinstance(parsed_result, dict):
                if "titles" not in parsed_result or not isinstance(parsed_result["titles"], list):
                    parsed_result["titles"] = []
        elif task_name == "tagline":
            if isinstance(parsed_result, dict) and "tagline" not in parsed_result:
                parsed_result["tagline"] = ""
        elif task_name == "summary":
            if isinstance(parsed_result, dict) and "self_introduce" not in parsed_result:
                parsed_result["self_introduce"] = ""
        elif task_name == "work_experiences":
            if isinstance(parsed_result, dict) and "experiences" not in parsed_result:
                parsed_result["experiences"] = []
        elif task_name == "company_skills_rating":
            if isinstance(parsed_result, dict) and "skills" not in parsed_result:
                parsed_result["skills"] = []
        elif task_name == "educations":
            if isinstance(parsed_result, dict) and "educations" not in parsed_result:
                parsed_result["educations"] = []
        elif task_name == "patents":
            if isinstance(parsed_result, dict) and "patents" not in parsed_result:
                parsed_result["patents"] = ""
        elif task_name == "en_speaking_level":
            if isinstance(parsed_result, dict) and "en_speaking_level" not in parsed_result:
                parsed_result["en_speaking_level"] = None
        elif task_name == "en_read_write_level":
            if isinstance(parsed_result, dict) and "en_read_write_level" not in parsed_result:
                parsed_result["en_read_write_level"] = None

        results[task_name] = parsed_result

    return jsonify(results), 200

# 全局异常处理，隐藏 Python 堆栈给客户端
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error("Unhandled Exception: %s", e, exc_info=True)
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    # 生产中请保持 debug=False
    app.run(debug=False)
