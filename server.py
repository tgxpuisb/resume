import os
import re
import json
import traceback
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from openai import OpenAI


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

client = OpenAI()

# 定义各个任务的 Prompt，确保所有花括号都被正确转义
PROMPTS = {
    "phone_number": (
        "你是一个资深的HR助理，你的任务是从简历内容中准确地整理出有效信息并以JSON的格式给出。"
        "请仅以JSON形式回答，以下是简历内容：\n{{\"phone\": \"+86 123456789\"}}。\n\n"
        "请根据以下简历内容回答问题：\n{pdf_text}\n\n"
        "请告诉我候选人的手机号phone，手机号前需要有候选人的国籍信息。"
        "如果候选人没填写国籍信息，则根据简历内容中的工作地点进行猜测。例如：如果候选人是中国则数据是+86，如果候选人是美国则数据是+1。"
        "地址和手机号中间不需要有空格。"
        "如果无法找到手机号，请返回{{\"phone\": \"\"}}。"
    ),
    "name_extraction": (
        "请告诉我候选人的用户名，分别以first_name，last_name，full_name的形式给出。"
        "如果用户名是中文，请翻译成对应汉语拼音，中文的姓氏对应last_name，名字对应first_name。"
        "full_name是first_name加上last_name。"
        "返回格式仅需JSON形式，示例：{{\"first_name\": \"Wei\", \"last_name\": \"Wang\", \"full_name\": \"Wei Wang\"}}。"
        "如果无法找到姓名，请返回{{\"first_name\": \"\", \"last_name\": \"\", \"full_name\": \"\"}}。"
    ),
    "work_years": (
        "请告诉我候选人的工作年限，返回格式为{{\"work_year\": number}}。"
        "如果候选人在简历内直接写明工作年限，则直接使用；如果候选人没有写明，则根据候选人的第一份工作和最后一份工作的时间间隔进行计算。"
        "如果无法找到工作年限信息，请返回{{\"work_year\": 0}}。"
    ),
    "job_titles": (
        "这里是一份job title的JSON数据："
        "[{{\"id\":13,\"name\":\"Admin\"}}, {{\"id\":12,\"name\":\"Android Engineer\"}}, {{\"id\":3,\"name\":\"Back-end Engineer\"}}, "
        "{{\"id\":19,\"name\":\"Business Dev Rep\"}}, {{\"id\":23,\"name\":\"CV\"}}, {{\"id\":28,\"name\":\"Data Scientist\"}}, "
        "{{\"id\":18,\"name\":\"DevOps Engineer\"}}, {{\"id\":2,\"name\":\"Front-end Engineer\"}}, {{\"id\":17,\"name\":\"Full-stack Engineer\"}}, "
        "{{\"id\":15,\"name\":\"HR\"}}, {{\"id\":16,\"name\":\"HRBP\"}}, {{\"id\":11,\"name\":\"IOS Engineer\"}}, {{\"id\":14,\"name\":\"IT\"}}, "
        "{{\"id\":22,\"name\":\"ML Engineer\"}}, {{\"id\":26,\"name\":\"Mobile Engineer\"}}, {{\"id\":21,\"name\":\"NLP\"}}, {{\"id\":20,\"name\":\"OPS\"}}, "
        "{{\"id\":4,\"name\":\"Product Manager\"}}, {{\"id\":7,\"name\":\"Project Manager\"}}, {{\"id\":8,\"name\":\"QA\"}}, "
        "{{\"id\":9,\"name\":\"QA Lead\"}}, {{\"id\":5,\"name\":\"Software Architecture\"}}, {{\"id\":24,\"name\":\"SRE Engr\"}}, "
        "{{\"id\":6,\"name\":\"Team Lead\"}}, {{\"id\":10,\"name\":\"UI Design Lead\"}}, {{\"id\":1,\"name\":\"UI Designer\"}}, "
        "{{\"id\":25,\"name\":\"UX\"}}, {{\"id\":27,\"name\":\"Web3 Researcher\"}}]"
        ".\n"
        "你的任务是根据候选人的简历总结用户的job title。"
        "你总结的候选人job title需要是在候选人在项目中主要从事的内容，请给出最多不超过5个job title，"
        "以{{\"titles\": []}}的数据结构返回。"
        "如果无法找到job title，请返回{{\"titles\": []}}。"
    ),
    "tagline": (
        "为候选人总结一个英文的tagline，格式为{{\"tagline\": \"tagline detail\"}}。"
        "Although all kinds of content can be employed as a tagline, here are some general types of taglines you might want to use: \n"
        "1. A philosophy mini-statement or personal description \n"
        "2. One or more achievements each described separately \n"
        "3. A mini achievement summarizing career-long accomplishments\n\n"
        "请仅以JSON形式回答。"
        "如果无法生成tagline，请返回{{\"tagline\": \"\"}}。"
    ),
    "summary": (
        "为候选人的简历写一个英文的summary。"
        "Resume summary statement是一个两到三句的专业介绍，您可以将其添加在简历顶部，以突出您的最有价值的技能和经验。"
        "Resume summary可以帮助雇主快速了解您是否具备他们所需的技能和背景。"
        "summary格式为{{\"self_introduce\": \"summary content\"}}，其中summary content可以是一个由<ol><li></li></ol>组成的有序列表的HTML字符串。"
        "请仅以JSON形式回答。"
        "如果无法生成summary，请返回{{\"self_introduce\": \"\"}}。"
    ),
    "work_experiences": (
        "请总结候选人的工作经历和项目。以下是数据结构的描述：\n"
        "type workExperiences = {{\n"
        "  experiences: Array<{{\n"
        "    where: string // The company the candidate works for, translate to English\n"
        "    from: string // Candidate's employment start date\n"
        "    to: string // The end time of the candidate's employment\n"
        "    mainly_as: string // The candidate's job level or position, translate to English\n"
        "    description: string // Description and summary of the candidate's work in the company. If the content involves multiple items and is independent of each other, you can use an ordered list of <ol><li></li></ol> for the description.\n"
        "  }}>;\n"
        "}}\n"
        "Note: The resume may be written in Chinese, and the summary needs to be translated into English。"
        "\n\n简历内容：\n{pdf_text}\n\n"
        "请确保返回的数据是有效的JSON格式，符合上述数据结构描述。"
    ),
    "company_skills_rating": (
        "这是一份公司信息技术的技能列表: "
        "[{{\"i\":8,\"n\":\"React\",\"t\":\"Front End\"}}, {{\"i\":9,\"n\":\"Visual Basic\",\"t\":\"Front End\"}}, {{\"i\":10,\"n\":\"JavaScript\",\"t\":\"Front End\"}}, "
        "{{\"i\":11,\"n\":\"Swift\",\"t\":\"Front End\"}}, {{\"i\":12,\"n\":\"HTML5\",\"t\":\"Front End\"}}, {{\"i\":13,\"n\":\"Angular\",\"t\":\"Front End\"}}, "
        "{{\"i\":14,\"n\":\"Flutter\",\"t\":\"Front End\"}}, {{\"i\":15,\"n\":\"Bootstrap\",\"t\":\"Front End\"}}, {{\"i\":16,\"n\":\"Layui\",\"t\":\"Front End\"}}, "
        "{{\"i\":17,\"n\":\"Vue.js\",\"t\":\"Front End\"}}, {{\"i\":28,\"n\":\"Objective-C\",\"t\":\"Front End\"}}, {{\"i\":57,\"n\":\"After Effects\",\"t\":\"Front End\"}}, "
        "{{\"i\":58,\"n\":\"Illustrator\",\"t\":\"Front End\"}}, {{\"i\":59,\"n\":\"Photoshop\",\"t\":\"Front End\"}}, {{\"i\":60,\"n\":\"XD\",\"t\":\"Front End\"}}, "
        "{{\"i\":61,\"n\":\"Sketch\",\"t\":\"Front End\"}}, {{\"i\":62,\"n\":\"Figma\",\"t\":\"Front End\"}}, {{\"i\":63,\"n\":\"Principle\",\"t\":\"Front End\"}}, "
        "{{\"i\":64,\"n\":\"Firebase\",\"t\":\"Front End\"}}, {{\"i\":95,\"n\":\"React Native\",\"t\":\"Front End\"}}, {{\"i\":96,\"n\":\"RxSwift\",\"t\":\"Front End\"}}, "
        "{{\"i\":109,\"n\":\"Swiftui\",\"t\":\"Front End\"}}, {{\"i\":110,\"n\":\"Combine\",\"t\":\"Front End\"}}, {{\"i\":111,\"n\":\"Mobile media\",\"t\":\"Front End\"}}, "
        "{{\"i\":112,\"n\":\"Android Framework\",\"t\":\"Front End\"}}, {{\"i\":113,\"n\":\"Java\",\"t\":\"Front End\"}}, {{\"i\":114,\"n\":\"Kotlin\",\"t\":\"Front End\"}}, "
        "{{\"i\":115,\"n\":\"Cocoa Touch\",\"t\":\"Front End\"}}, {{\"i\":137,\"n\":\"Elixir\",\"t\":\"Front End\"}}, {{\"i\":7,\"n\":\"Java\",\"t\":\"Back End\"}}, "
        "{{\"i\":18,\"n\":\"C\",\"t\":\"Back End\"}}, {{\"i\":19,\"n\":\"Python\",\"t\":\"Back End\"}}, {{\"i\":20,\"n\":\"C++\",\"t\":\"Back End\"}}, "
        "{{\"i\":21,\"n\":\"C#\",\"t\":\"Back End\"}}, {{\"i\":22,\"n\":\"R\",\"t\":\"Back End\"}}, {{\"i\":23,\"n\":\"PHP\",\"t\":\"Back End\"}}, "
        "{{\"i\":24,\"n\":\"SQL\",\"t\":\"Back End\"}}, {{\"i\":25,\"n\":\"Go\",\"t\":\"Back End\"}}, {{\"i\":26,\"n\":\"Perl\",\"t\":\"Back End\"}}, "
        "{{\"i\":27,\"n\":\"TypeScript\",\"t\":\"Back End\"}}, {{\"i\":29,\"n\":\"Kotlin\",\"t\":\"Back End\"}}, {{\"i\":30,\"n\":\"Node js\",\"t\":\"Back End\"}}, "
        "{{\"i\":31,\"n\":\"Mysql\",\"t\":\"Back End\"}}, {{\"i\":32,\"n\":\"Oracle\",\"t\":\"Back End\"}}, {{\"i\":33,\"n\":\"SQL Server\",\"t\":\"Back End\"}}, "
        "{{\"i\":34,\"n\":\"Sybase\",\"t\":\"Back End\"}}, {{\"i\":35,\"n\":\"DB2\",\"t\":\"Back End\"}}, {{\"i\":36,\"n\":\"Mongodb\",\"t\":\"Back End\"}}, "
        "{{\"i\":37,\"n\":\"Cassandra\",\"t\":\"Back End\"}}, {{\"i\":38,\"n\":\"CouchDB\",\"t\":\"Back End\"}}, {{\"i\":39,\"n\":\"Redis\",\"t\":\"Back End\"}}, "
        "{{\"i\":40,\"n\":\"Membase\",\"t\":\"Back End\"}}, {{\"i\":41,\"n\":\"HBase\",\"t\":\"Back End\"}}, {{\"i\":42,\"n\":\"Express\",\"t\":\"Back End\"}}, "
        "{{\"i\":43,\"n\":\"GraphQL\",\"t\":\"Back End\"}}, {{\"i\":44,\"n\":\"Laravel\",\"t\":\"Back End\"}}, {{\"i\":45,\"n\":\"CakePHP\",\"t\":\"Back End\"}}, "
        "{{\"i\":46,\"n\":\"Django\",\"t\":\"Back End\"}}, {{\"i\":47,\"n\":\"Ruby on Rails\",\"t\":\"Back End\"}}, {{\"i\":48,\"n\":\"Flask\",\"t\":\"Back End\"}}, "
        "{{\"i\":49,\"n\":\"Phoenix\",\"t\":\"Back End\"}}, {{\"i\":50,\"n\":\"Spring Boot\",\"t\":\"Back End\"}}, {{\"i\":51,\"n\":\"Robot Frame\",\"t\":\"Back End\"}}, "
        "{{\"i\":52,\"n\":\"Web Driver\",\"t\":\"Back End\"}}, {{\"i\":53,\"n\":\"TestNG\",\"t\":\"Back End\"}}, {{\"i\":54,\"n\":\"Appium\",\"t\":\"Back End\"}}, "
        "{{\"i\":55,\"n\":\"Jenkins\",\"t\":\"Back End\"}}, {{\"i\":56,\"n\":\"Docker\",\"t\":\"Back End\"}}, {{\"i\":65,\"n\":\"Hadoop\",\"t\":\"Back End\"}}, "
        "{{\"i\":66,\"n\":\"Hive\",\"t\":\"Back End\"}}, {{\"i\":67,\"n\":\"Spark\",\"t\":\"Back End\"}}, {{\"i\":68,\"n\":\"Storm\",\"t\":\"Back End\"}}, "
        "{{\"i\":69,\"n\":\"Flink\",\"t\":\"Back End\"}}, {{\"i\":70,\"n\":\"Flume\",\"t\":\"Back End\"}}, {{\"i\":71,\"n\":\"Elasticsearch\",\"t\":\"Back End\"}}, "
        "{{\"i\":77,\"n\":\"GitOps\",\"t\":\"Back End\"}}, {{\"i\":97,\"n\":\"Kafka\",\"t\":\"Back End\"}}, {{\"i\":98,\"n\":\"Zookeeper\",\"t\":\"Back End\"}}, "
        "{{\"i\":99,\"n\":\"Vertica\",\"t\":\"Back End\"}}, {{\"i\":116,\"n\":\"Computer Visoon(CV)\",\"t\":\"Back End\"}}, "
        "{{\"i\":117,\"n\":\"Natural language processing（NLP)\",\"t\":\"Back End\"}}, {{\"i\":118,\"n\":\"Reinforcement Learning(RL)\",\"t\":\"Back End\"}}, "
        "{{\"i\":119,\"n\":\"Tensorflow\",\"t\":\"Back End\"}}, {{\"i\":120,\"n\":\"Pytorch\",\"t\":\"Back End\"}}, {{\"i\":121,\"n\":\"Opencv\",\"t\":\"Back End\"}}, "
        "{{\"i\":122,\"n\":\"Langchain\",\"t\":\"Back End\"}}, {{\"i\":123,\"n\":\"TensorRT\",\"t\":\"Back End\"}}, {{\"i\":124,\"n\":\"LLM\",\"t\":\"Back End\"}}, "
        "{{\"i\":133,\"n\":\"Rust\",\"t\":\"Back End\"}}, {{\"i\":136,\"n\":\"Elixir\",\"t\":\"Back End\"}}, {{\"i\":1,\"n\":\"Communication\",\"t\":\"Soft Skills\"}}, "
        "{{\"i\":3,\"n\":\"Leadership\",\"t\":\"Soft Skills\"}}, {{\"i\":4,\"n\":\"Management\",\"t\":\"Soft Skills\"}}, {{\"i\":5,\"n\":\"Creative\",\"t\":\"Soft Skills\"}}, "
        "{{\"i\":6,\"n\":\"Execution\",\"t\":\"Soft Skills\"}}, {{\"i\":72,\"n\":\"Ethereum\",\"t\":\"Web3\"}}, {{\"i\":73,\"n\":\"Solana\",\"t\":\"Web3\"}}, "
        "{{\"i\":74,\"n\":\"Solidity\",\"t\":\"Web3\"}}, {{\"i\":75,\"n\":\"Vyper\",\"t\":\"Web3\"}}, {{\"i\":76,\"n\":\"The Graph\",\"t\":\"Web3\"}}, "
        "{{\"i\":100,\"n\":\"Hardhat\",\"t\":\"Web3\"}}, {{\"i\":101,\"n\":\"Truffle\",\"t\":\"Web3\"}}, {{\"i\":102,\"n\":\"Mocha\",\"t\":\"Web3\"}}, "
        "{{\"i\":103,\"n\":\"web3.js\",\"t\":\"Web3\"}}, {{\"i\":104,\"n\":\"ethers.js\",\"t\":\"Web3\"}}, {{\"i\":105,\"n\":\"Alchemy\",\"t\":\"Web3\"}}, "
        "{{\"i\":106,\"n\":\"Infura\",\"t\":\"Web3\"}}, {{\"i\":107,\"n\":\"Tenderly\",\"t\":\"Web3\"}}, {{\"i\":108,\"n\":\"Defender\",\"t\":\"Web3\"}}, "
        "{{\"i\":78,\"n\":\"Catch Requirement\",\"t\":\"QA Skills\"}}, {{\"i\":79,\"n\":\"Design Test Case\",\"t\":\"QA Skills\"}}, {{\"i\":80,\"n\":\"Execute Test Case\",\"t\":\"QA Skills\"}}, "
        "{{\"i\":81,\"n\":\"Defect Tracking\",\"t\":\"QA Skills\"}}, {{\"i\":82,\"n\":\"Design Test Plan\",\"t\":\"QA Skills\"}}, {{\"i\":83,\"n\":\"Report Test Result\",\"t\":\"QA Skills\"}}, "
        "{{\"i\":84,\"n\":\"Write Manual\",\"t\":\"QA Skills\"}}, {{\"i\":85,\"n\":\"Automation Test\",\"t\":\"QA Skills\"}}, {{\"i\":86,\"n\":\"Risk Management\",\"t\":\"QA Skills\"}}, "
        "{{\"i\":87,\"n\":\"Reproduce Live Issues\",\"t\":\"QA Skills\"}}, {{\"i\":88,\"n\":\"UE (UX) Design\",\"t\":\"PM Skills\"}}, {{\"i\":89,\"n\":\"Data Analysis\",\"t\":\"PM Skills\"}}, "
        "{{\"i\":90,\"n\":\"Market Research\",\"t\":\"PM Skills\"}}, {{\"i\":91,\"n\":\"Business Strategy\",\"t\":\"PM Skills\"}}, {{\"i\":92,\"n\":\"Customer Development\",\"t\":\"PM Skills\"}}, "
        "{{\"i\":93,\"n\":\"Agile Methodologies\",\"t\":\"PM Skills\"}}, {{\"i\":94,\"n\":\"Project Management\",\"t\":\"PM Skills\"}}, {{\"i\":125,\"n\":\"AWS\",\"t\":\"DevOps\"}}, "
        "{{\"i\":126,\"n\":\"Azure\",\"t\":\"DevOps\"}}, {{\"i\":127,\"n\":\"GCP\",\"t\":\"DevOps\"}}, {{\"i\":128,\"n\":\"Ali Cloud\",\"t\":\"DevOps\"}}, {{\"i\":129,\"n\":\"Kubernetes\",\"t\":\"DevOps\"}}, "
        "{{\"i\":130,\"n\":\"Helm\",\"t\":\"DevOps\"}}, {{\"i\":131,\"n\":\"Docker\",\"t\":\"DevOps\"}}, {{\"i\":132,\"n\":\"Terraform\",\"t\":\"DevOps\"}}, "
        "{{\"i\":134,\"n\":\"Jenkins\",\"t\":\"DevOps\"}}, {{\"i\":135,\"n\":\"CircleCI\",\"t\":\"DevOps\"}}]"
        ".\n"
        "你的任务是结合候选人简历对候选人的信息技术技能进行评级，评级级别为1-5的正整数，熟练程度随着数字增大而增大，"
        "1代表候选人只是使用过该技术，3代表候选人对技术比较熟练，5代表候选人精通该项技术并熟悉这项技术的运行原理。"
        "注意：如果候选人简历中没有提到公司技术列表中的技术则不需要评级。"
        "请以以下格式返回：{{\"skills\": [{{\"id\": \"number\", \"name\": \"name\", \"type\": \"大类别\", \"level\": \"评级\"}}]}}。"
        "如果没有需要评级的技能，请返回{{\"skills\": []}}。"
    ),
    "educations": (
        "请总结候选人的教育经历。以下是数据结构的描述：\n"
        "type educations = {{\n"
        "  educations: Array<{{\n"
        "    where: string // Where the candidate received his or her education, translate to English\n"
        "    degree: string // degree, translate to English\n"
        "    from: string // The time when the candidate's education started, translate to English\n"
        "    to: string // The time when the candidate's education ended, translate to English\n"
        "    mainly_as: string // college major, translate to English\n"
        "  }}>;\n"
        "}}\n"
        "Note: The resume may be written in Chinese, and the summary needs to be translated into English。"
        "\n\n简历内容：\n{pdf_text}\n\n"
        "请确保返回的数据是有效的JSON格式，符合上述数据结构描述。"
    ),
    "patents": (
        "请总结候选人的专利信息，格式为{{\"patents\": \"string\"}}。"
        "如果内容涉及多个项目，可以使用一个由<ol><li></li></ol>组成的HTML字符串描述。"
        "注意：如果候选人没有专利，请返回空字符串。"
        "Note: The resume may be written in Chinese, and the summary needs to be translated into English。"
    ),
    "en_speaking_level": (
        "请总结候选人的英语口语水平，评分标准如下：\n"
        "1=Can recognize and understand very basic words and simple, familiar phrases. Able to write simple isolated phrases and sentences.\n"
        "2=Can read and understand short, simple texts. Capable of producing basic sentences, writing brief notes and messages related to immediate needs.\n"
        "3=Can read straightforward information within a known area and write connected text on topics that are familiar or of personal interest.\n"
        "4=Able to read and write texts about various topics. Can understand the main ideas of complex text and produce clear, detailed writing on a wide range of subjects.\n"
        "5=Proficient in reading and writing complex texts. Capable of understanding implicit meanings and producing sophisticated and well-structured writing on complex subjects.\n"
        "请仅以JSON形式回答。"
        "返回格式为{{\"en_speaking_level\": number or null}}。"
        "Note: If the candidate does not specify this ability, null is returned。"
    ),
    "en_read_write_level": (
        "请总结候选人的英语读写水平，评分标准如下：\n"
        "1=Can use simple phrases and sentences to communicate basic needs in familiar contexts. Interaction is limited to slow speech and repetition.\n"
        "2=Able to engage in simple conversation on familiar topics with some assistance. Can ask and answer basic questions and use common expressions.\n"
        "3=Can handle short, routine exchanges without disruption. Able to communicate in simple and direct exchanges of information on familiar tasks and topics.\n"
        "4=Can communicate with some confidence on familiar routine and non-routine matters. Able to express personal opinions but sometimes requires assistance for more complex conversation.\n"
        "5=Fully fluent and comfortable in any situation. Can engage in nuanced, complex conversations, understanding slang, idioms, and cultural references.\n"
        "请仅以JSON形式回答。"
        "返回格式为{{\"en_read_write_level\": number or null}}。"
        "Note: If the candidate does not specify this ability, null is returned。"
    )
}

def extract_text_from_pdf(file_path):
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

def call_openai(prompt):
    """
    使用 OpenAI API 进行聊天完成，返回模型的回复内容。
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # 若没有 GPT-4 权限可改为 "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content":  "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=1024
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        app.logger.error("OpenAI API call failed: %s", e, exc_info=True)
        return json.dumps({"error": "OpenAI API call failed", "details": str(e)})


@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

    try:
        file.save(file_path)
    except Exception as e:
        app.logger.error("Failed to save file: %s", e, exc_info=True)
        return jsonify({"error": "Failed to save file"}), 500

    pdf_text_raw = extract_text_from_pdf(file_path)
    pdf_text_cleaned = clean_pdf_text(pdf_text_raw)
    print(pdf_text_cleaned)
    results = {}

    # 处理所有任务，跳过 "company_skills_rating"
    for task_name, prompt_content in PROMPTS.items():
        if task_name == "company_skills_rating":
            continue  # 后面单独处理

        app.logger.debug(f"Processing task: {task_name}")

        # 填充 Prompt
        try:
            filled_prompt = prompt_content.format(pdf_text=pdf_text_cleaned)
        except KeyError as e:
            app.logger.error("Formatting prompt failed for task '%s': %s", task_name, e, exc_info=True)
            results[task_name] = DEFAULT_EMPTY_RESPONSES.get(task_name, {})
            continue

        app.logger.debug(f"Filled prompt for {task_name}: {filled_prompt}")

        # 调用 OpenAI API
        response_str = call_openai(filled_prompt)

        # 解析 JSON 响应
        if isinstance(response_str, str) and response_str.startswith("{") and response_str.endswith("}"):
            try:
                parsed_result = json.loads(response_str)
            except json.JSONDecodeError:
                app.logger.error(f"JSON decoding failed for task '{task_name}'. Response: {response_str}")
                parsed_result = DEFAULT_EMPTY_RESPONSES.get(task_name, {})
        else:
            app.logger.error(f"Invalid response format for task '{task_name}'. Response: {response_str}")
            parsed_result = DEFAULT_EMPTY_RESPONSES.get(task_name, {})

        results[task_name] = parsed_result

    # 处理 work_experiences
    work_experiences = results.get("work_experiences", {}).get("experiences", [])
    if isinstance(work_experiences, list):
        for idx, experience in enumerate(work_experiences):
            # 提取工作描述
            experience_description = experience.get("description", "")
            if not experience_description:
                app.logger.warning(f"No description found for experience {idx}, setting company_skills_rating to empty.")
                work_experiences[idx]["company_skills_rating"] = DEFAULT_EMPTY_RESPONSES.get("company_skills_rating", {})
                continue

            # 填充 company_skills_rating Prompt
            company_prompt_template = PROMPTS["company_skills_rating"]
            try:
                company_prompt = company_prompt_template.format(pdf_text=experience_description)
            except KeyError as e:
                app.logger.error("Formatting company_skills_rating prompt failed for experience %d: %s", idx, e, exc_info=True)
                work_experiences[idx]["company_skills_rating"] = DEFAULT_EMPTY_RESPONSES.get("company_skills_rating", {})
                continue

            app.logger.debug(f"Filled prompt for company_skills_rating for experience {idx}: {company_prompt}")

            # 调用 OpenAI API
            company_response_str = call_openai(company_prompt)

            # 解析 JSON 响应
            if isinstance(company_response_str, str) and company_response_str.startswith("{") and company_response_str.endswith("}"):
                try:
                    company_parsed_result = json.loads(company_response_str)
                except json.JSONDecodeError:
                    app.logger.error(f"JSON decoding failed for company_skills_rating on experience {idx}. Response: {company_response_str}")
                    company_parsed_result = DEFAULT_EMPTY_RESPONSES.get("company_skills_rating", {})
            else:
                app.logger.error(f"Invalid response format for company_skills_rating on experience {idx}. Response: {company_response_str}")
                company_parsed_result = DEFAULT_EMPTY_RESPONSES.get("company_skills_rating", {})

            work_experiences[idx]["company_skills_rating"] = company_parsed_result

        # 更新 results 中的 work_experiences
        results["work_experiences"]["experiences"] = work_experiences
        for idx, experience in enumerate(work_experiences):
            # 提取工作描述
            experience_description = experience.get("description", "")
            if not experience_description:
                app.logger.warning(f"No description found for experience {idx}, skipping company_skills_rating.")
                work_experiences[idx]["company_skills_rating"] = {"error": "No description provided."}
                continue

            # 填充 company_skills_rating Prompt
            company_prompt_template = PROMPTS["company_skills_rating"]
            try:
                company_prompt = company_prompt_template.format(pdf_text=experience_description)
            except KeyError as e:
                app.logger.error("Formatting company_skills_rating prompt failed for experience %d: %s", idx, e, exc_info=True)
                work_experiences[idx]["company_skills_rating"] = {"error": f"Formatting prompt failed: {str(e)}"}
                continue

            app.logger.debug(f"Filled prompt for company_skills_rating for experience {idx}: {company_prompt}")

            # 调用 OpenAI API
            company_response_str = call_openai(company_prompt)

            # 解析 JSON 响应
            if isinstance(company_response_str, str) and company_response_str.startswith("{") and company_response_str.endswith("}"):
                try:
                    company_parsed_result = json.loads(company_response_str)
                except json.JSONDecodeError:
                    app.logger.error("JSON decoding failed for company_skills_rating on experience %d", idx)
                    company_parsed_result = {"error": "Invalid JSON response"}
            else:
                company_parsed_result = {"error": "Invalid JSON response"}

            work_experiences[idx]["company_skills_rating"] = company_parsed_result

        # 更新 results 中的 work_experiences
        results["work_experiences"]["experiences"] = work_experiences

        for idx, experience in enumerate(work_experiences):
            # Extract the description of the experience
            experience_description = experience.get("description", "")
            if not experience_description:
                app.logger.warning(f"No description found for experience {idx}, skipping company_skills_rating.")
                work_experiences[idx]["company_skills_rating"] = {"error": "No description provided."}
                continue

            # Prepare the prompt for company_skills_rating by filling in the description
            company_prompt = PROMPTS["company_skills_rating"].format(pdf_text=experience_description)
            app.logger.debug(f"Filled prompt for company_skills_rating for experience {idx}: {company_prompt}")
            company_response_str = call_openai(company_prompt)
            
            if company_response_str.startswith("{") and company_response_str.endswith("}"):
                try:
                    company_parsed_result = json.loads(company_response_str)
                except json.JSONDecodeError:
                    app.logger.error("JSON decoding failed for company_skills_rating on experience %d", idx)
                    company_parsed_result = {"error": "Invalid JSON response"}
            else:
                company_parsed_result = {"error": "Invalid JSON response"}

            work_experiences[idx]["company_skills_rating"] = company_parsed_result

        # Update the results with the modified work_experiences
        results["work_experiences"]["experiences"] = work_experiences
    return jsonify(results), 200

# 全局异常处理，隐藏 Python 堆栈给客户端
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error("Unhandled Exception: %s", e, exc_info=True)
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    # 生产中请保持 debug=False
    app.run(host='0.0.0.0', port=8081, debug=False)

