import os
import re
import json
import traceback
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from openai import OpenAI
from flask_cors import CORS

DEFAULT_EMPTY_RESPONSES = {
    "phone_number": {"phone": ""},
    "name_extraction": {"first_name": "", "last_name": "", "full_name": ""},
    "work_years": {"work_year": 0},
    "job_titles": {"titles": []},
    "tagline": {"tagline": ""},
    "summary": {"self_introduce": ""},
    "work_experiences": {"experiences": []},
    "educations": {"educations": []},
    "patents": {"patents": ""},
    "en_speaking_level": {"en_speaking_level": None},
    "en_read_write_level": {"en_read_write_level": None},
    "projects": {"projects": []}
}
        
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = './uploads'

client = OpenAI()

# 定义各个任务的 Prompt，确保所有花括号都被正确转义
PROMPTS = {
    "phone_number": (
        "Your task is to find out the candidate's mobile phone number and email, which must be preceded by the candidate's nationality information."
        "If the candidate does not fill in the nationality information, the data is guessed based on the work location in the resume. For example, if the candidate is from China, the data is +86, and if the candidate is from the United States, the data is +1."
        "There does not need to be a space between the address and the phone number."
        "If you cannot find your phone number, please return {{\"phone\": \"unknown\", \"email\": \"unknown\"}}。"
    ),
    "name_extraction": (
        "Your task is to find the candidate's username, given in the fields first_name, last_name, full_name."
        "Note: If the username is in Chinese, please translate it into the corresponding Chinese pinyin. The Chinese last name corresponds to last_name and the first name corresponds to first_name."
        "full_name is first_name concat last_name."
        "The return format only needs to be in JSON format, example: {{\"first_name\": \"Wei\", \"last_name\": \"Wang\", \"full_name\": \"Wei Wang\"}}。"
        "If the name cannot be found, please return {{\"first_name\": \"unknown\", \"last_name\": \"unknown\", \"full_name\": \"unknown\"}}。"
    ),
    "work_years": (
        "Your task is to calculate the candidate's years of work experience and return the format: {{\"work_year\": number}}。"
        "If the candidate directly states his/her years of work in the resume, it will be used directly; if the candidate does not state it, it will be calculated based on the time interval between the candidate's first job and the last job."
        "If you cannot find the years of service information, please return {{\"work_year\": 0}}。"
    ),
    "job_titles": (
        "Here is a JSON data of job title: "
        "[{{\"id\":13,\"name\":\"Admin\"}}, {{\"id\":12,\"name\":\"Android Engineer\"}}, {{\"id\":3,\"name\":\"Back-end Engineer\"}}, "
        "{{\"id\":19,\"name\":\"Business Dev Rep\"}}, {{\"id\":23,\"name\":\"CV\"}}, {{\"id\":28,\"name\":\"Data Scientist\"}}, "
        "{{\"id\":18,\"name\":\"DevOps Engineer\"}}, {{\"id\":2,\"name\":\"Front-end Engineer\"}}, {{\"id\":17,\"name\":\"Full-stack Engineer\"}}, "
        "{{\"id\":15,\"name\":\"HR\"}}, {{\"id\":16,\"name\":\"HRBP\"}}, {{\"id\":11,\"name\":\"IOS Engineer\"}}, {{\"id\":14,\"name\":\"IT\"}}, "
        "{{\"id\":22,\"name\":\"ML Engineer\"}}, {{\"id\":26,\"name\":\"Mobile Engineer\"}}, {{\"id\":21,\"name\":\"NLP\"}}, {{\"id\":20,\"name\":\"OPS\"}}, "
        "{{\"id\":4,\"name\":\"Product Manager\"}}, {{\"id\":7,\"name\":\"Project Manager\"}}, {{\"id\":8,\"name\":\"QA\"}}, "
        "{{\"id\":9,\"name\":\"QA Lead\"}}, {{\"id\":5,\"name\":\"Software Architecture\"}}, {{\"id\":24,\"name\":\"SRE Engr\"}}, "
        "{{\"id\":6,\"name\":\"Team Lead\"}}, {{\"id\":10,\"name\":\"UI Design Lead\"}}, {{\"id\":1,\"name\":\"UI Designer\"}}, "
        "{{\"id\":25,\"name\":\"UX\"}}, {{\"id\":27,\"name\":\"Web3 Researcher\"}}] \n"
        "Your task is to summarize the user's job title based on the candidate's resume."
        "The job title of the candidate you summarize needs to be the main content of the candidate in the project. Please give no more than 5 job titles."
        "The return format only needs to be in JSON format, example: {{\"titles\": [{{\"id\": 13, \"name\": \"Admin\"}}]}}"
        "If the job_titles cannot be found, please return {{\"titles\": []}}。"
    ),
    "tagline": (
        "your task is Summarize an English tagline for the candidate"
        "Although all kinds of content can be employed as a tagline, here are some general types of taglines you might want to use: \n"
        "1. A philosophy mini-statement or personal description \n"
        "2. One or more achievements each described separately \n"
        "3. A mini achievement summarizing career-long accomplishments\n\n"
        "The return format only needs to be in JSON format, example: {{\"tagline\": \"tagline detail\"}}。\n"
        "If the tagline cannot be found, please return {{\"tagline\": \"\"}}。"
    ),
    "summary": (
        "your task Write an English summary for the candidate's resume."
        "A resume summary statement is a two- to three-sentence professional introduction that you add to the top of your resume to highlight your most valuable skills and experiences."
        "A resume summary can help employers quickly understand whether you have the skills and background they need."
        "The return format only needs to be in JSON format, example: {{\"self_introduce\": \"summary content\"}}，The summary content can be an HTML string consisting of an ordered list of <ol><li></li></ol>."
        "If the summary cannot be found, please return {{\"self_introduce\": \"\"}}。"
    ),
    "work_experiences": (
        "your task is summarize the candidate's work experience and projects. The following is a description of the data structure:\n"
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
        "The return format only needs to be in JSON format, example: {{\"experiences\": [{{\"where\": \"\", \"from\": \"\", \"to\": \"\", \"mainly_as\": \"\", \"description\": \"\"}}]}}"
    ),
    "skills": (
        "Here is a list of skills for company information technology:"
        "[{{\"i\":8,\"n\":\"React\",\"t\":\"Front End\"}}, {{\"i\":9,\"n\":\"Visual Basic\",\"t\":\"Front End\"}}, {{\"i\":10,\"n\":\"JavaScript\",\"t\":\"Front End\"}}, "
        "{{\"i\":11,\"n\":\"Swift\",\"t\":\"Front End\"}}, {{\"i\":12,\"n\":\"HTML5\",\"t\":\"Front End\"}}, {{\"i\":13,\"n\":\"Angular\",\"t\":\"Front End\"}}, "
        "{{\"i\":14,\"n\":\"Flutter\",\"t\":\"Front End\"}}, {{\"i\":15,\"n\":\"Bootstrap\",\"t\":\"Front End\"}}, {{\"i\":16,\"n\":\"Layui\",\"t\":\"Front End\"}}, "
        "{{\"i\":17,\"n\":\"Vue.js\",\"t\":\"Front End\"}}, {{\"i\":28,\"n\":\"Objective-C\",\"t\":\"Front End\"}}, {{\"i\":57,\"n\":\"After Effects\",\"t\":\"Front End\"}}, "
        "{{\"i\":58,\"n\":\"Illustrator\",\"t\":\"Front End\"}}, {{\"i\":59,\"n\":\"Photoshop\",\"t\":\"Front End\"}}, {{\"i\":60,\"n\":\"XD\",\"t\":\"Front End\"}}, "
        "{{\"i\":61,\"n\":\"Sketch\",\"t\":\"Front End\"}}, {{\"i\":62,\"n\":\"Figma\",\"t\":\"Front End\"}}, {{\"i\":63,\"n\":\"Principle\",\"t\":\"Front End\"}}, "
        "{{\"i\":64,\"n\":\"Firebase\",\"t\":\"Front End\"}}, {{\"i\":95,\"n\":\"React Native\",\"t\":\"Front End\"}}, {{\"i\":96,\"n\":\"RxSwift\",\"t\":\"Front End\"}}, "
        "{{\"i\":109,\"n\":\"Swiftui\",\"t\":\"Front End\"}}, {{\"i\":110,\"n\":\"Combine\",\"t\":\"Front End\"}}, {{\"i\":111,\"n\":\"Mobile media\",\"t\":\"Front End\"}}, "
        "{{\"i\":112,\"n\":\"Android Framework\",\"t\":\"Front End\"}}, {{\"i\":114,\"n\":\"Kotlin\",\"t\":\"Front End\"}}, "
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
        "Your task is to rate the candidate's information technology skills based on his/her resume. The rating level is a positive integer from 1 to 5, and the proficiency increases as the number increases."
        "1 means the candidate has only used the technology, 3 means the candidate is relatively proficient in the technology, and 5 means the candidate is proficient in the technology and familiar with how it works."
        "Note: If the candidate's resume does not mention any technology from the company's technology list, no rating is required."
        "The return format only needs to be in JSON format, example: {{\"skills\": [{{\"id\": \"number\", \"name\": \"name\", \"type\": \"\", \"level\": 3}}]}}。"
    ),
    "educations": (
        "your task is summarize the candidate's educational experience. The following is a description of the data structure:\n"
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
        "The return format only needs to be in JSON format, example: {{\"experiences\": [{{\"where\": \"\", \"from\": \"\", \"to\": \"\", \"mainly_as\": \"}}]}}"
    ),
    "patents": (
        "Your task is to summarize the candidate's patent information:"
        "The return format only needs to be in JSON format, example:  {{\"patents\": \"patents string\"}} If the content involves multiple items, The patents string can be can use an HTML string consisting of <ol><li></li></ol> to describe it."
        "Note：If the patents cannot be found, please return {{\"patents\": \"\"}}。"
        "Note: The resume may be written in Chinese, and the summary needs to be translated into English。"
    ),
    "en_speaking_level": (
        "your task is summarize the candidate's English speaking proficiency, the scoring criteria are as follows：\n"
        "1=Can recognize and understand very basic words and simple, familiar phrases. Able to write simple isolated phrases and sentences.\n"
        "2=Can read and understand short, simple texts. Capable of producing basic sentences, writing brief notes and messages related to immediate needs.\n"
        "3=Can read straightforward information within a known area and write connected text on topics that are familiar or of personal interest.\n"
        "4=Able to read and write texts about various topics. Can understand the main ideas of complex text and produce clear, detailed writing on a wide range of subjects.\n"
        "5=Proficient in reading and writing complex texts. Capable of understanding implicit meanings and producing sophisticated and well-structured writing on complex subjects.\n"
        "The return format only needs to be in JSON format, example: {{\"en_speaking_level\": number or null}}。"
        "Note: If the candidate does not specify this ability, null is returned。"
    ),
    "en_read_write_level": (
        "your task is summarize the candidate's English reading and writing proficiency. The scoring criteria are as follows: \n"
        "1=Can use simple phrases and sentences to communicate basic needs in familiar contexts. Interaction is limited to slow speech and repetition.\n"
        "2=Able to engage in simple conversation on familiar topics with some assistance. Can ask and answer basic questions and use common expressions.\n"
        "3=Can handle short, routine exchanges without disruption. Able to communicate in simple and direct exchanges of information on familiar tasks and topics.\n"
        "4=Can communicate with some confidence on familiar routine and non-routine matters. Able to express personal opinions but sometimes requires assistance for more complex conversation.\n"
        "5=Fully fluent and comfortable in any situation. Can engage in nuanced, complex conversations, understanding slang, idioms, and cultural references.\n"
        "The return format only needs to be in JSON format, example: {{\"en_read_write_level\": number or null}}。"
        "Note: If the candidate does not specify this ability, null is returned。"
    ),
    "projects": (
        "Here is a list of skills for company information technology:"
        "[{{\"i\":8,\"n\":\"React\",\"t\":\"Front End\"}}, {{\"i\":9,\"n\":\"Visual Basic\",\"t\":\"Front End\"}}, {{\"i\":10,\"n\":\"JavaScript\",\"t\":\"Front End\"}}, "
        "{{\"i\":11,\"n\":\"Swift\",\"t\":\"Front End\"}}, {{\"i\":12,\"n\":\"HTML5\",\"t\":\"Front End\"}}, {{\"i\":13,\"n\":\"Angular\",\"t\":\"Front End\"}}, "
        "{{\"i\":14,\"n\":\"Flutter\",\"t\":\"Front End\"}}, {{\"i\":15,\"n\":\"Bootstrap\",\"t\":\"Front End\"}}, {{\"i\":16,\"n\":\"Layui\",\"t\":\"Front End\"}}, "
        "{{\"i\":17,\"n\":\"Vue.js\",\"t\":\"Front End\"}}, {{\"i\":28,\"n\":\"Objective-C\",\"t\":\"Front End\"}}, {{\"i\":57,\"n\":\"After Effects\",\"t\":\"Front End\"}}, "
        "{{\"i\":58,\"n\":\"Illustrator\",\"t\":\"Front End\"}}, {{\"i\":59,\"n\":\"Photoshop\",\"t\":\"Front End\"}}, {{\"i\":60,\"n\":\"XD\",\"t\":\"Front End\"}}, "
        "{{\"i\":61,\"n\":\"Sketch\",\"t\":\"Front End\"}}, {{\"i\":62,\"n\":\"Figma\",\"t\":\"Front End\"}}, {{\"i\":63,\"n\":\"Principle\",\"t\":\"Front End\"}}, "
        "{{\"i\":64,\"n\":\"Firebase\",\"t\":\"Front End\"}}, {{\"i\":95,\"n\":\"React Native\",\"t\":\"Front End\"}}, {{\"i\":96,\"n\":\"RxSwift\",\"t\":\"Front End\"}}, "
        "{{\"i\":109,\"n\":\"Swiftui\",\"t\":\"Front End\"}}, {{\"i\":110,\"n\":\"Combine\",\"t\":\"Front End\"}}, {{\"i\":111,\"n\":\"Mobile media\",\"t\":\"Front End\"}}, "
        "{{\"i\":112,\"n\":\"Android Framework\",\"t\":\"Front End\"}}, {{\"i\":114,\"n\":\"Kotlin\",\"t\":\"Front End\"}}, "
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
        ".\n",
        "your task is Summarize the projects that users have done at {company_name},The following is a description of the data structure:\n"
        "type projects = {{\n"
        "  projects: Array<{{\n"
        "    name: string // the project name translate to English\n"
        "    title: string // The position or rank held by the candidate in this project, translate to English\n"
        "    description: string // The candidate's detailed description of the project, translate to English\n"
        "    tech: Array<{{ \"id\": number, \"skill\": \"string\" }}> // The skills the candidate used in the project from list of skills for company information technology" 
        "  }}>;\n"
        "}}\n"
        "The return format only needs to be in JSON format, example: {{\"projects\": [{{\"name\": \"\", \"title\": \"\", \"description\": \"\", \"tech\": \"}}]}}"
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

def call_openai(prompt, pdf_text):
    """
    使用 OpenAI API 进行聊天完成，返回模型的回复内容。
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 若没有 GPT-4 权限可改为 "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are a senior HR assistant. Your task is to help your boss who only knows English to accurately filter out valid information from resumes and present it in JSON format."},
                {"role": "system", "content": "You work as an API, your input and output will be strictly formatted JSON  which can be directly parsed by the application. Do not enclose JSON string in markdown quotes."},
                {"role": "user", "content": f"answer the questions based on the following resume: \n{pdf_text}\n\n"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1024
        )
        print('======')
        print(response.choices[0].message.content)
        print('======')
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

    # 处理所有任务，跳过 "projects"
    for task_name, prompt_content in PROMPTS.items():
        if task_name == "projects":
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
        response_str = call_openai(filled_prompt, pdf_text_cleaned)

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

        results.update(parsed_result)

    # 处理 work_experiences
    work_experiences = results.get("experiences", [])
    if isinstance(work_experiences, list):
        for idx, experience in enumerate(work_experiences):
            projects_prompt_template = PROMPTS["projects"]
            try:
                projects_prompt = '\n'.join(projects_prompt_template).format(company_name=experience.get("name", ""))
            except KeyError as e:
                app.logger.error("Formatting projects prompt failed for experience %d: %s", idx, e, exc_info=True)
                work_experiences[idx].update(DEFAULT_EMPTY_RESPONSES.get("projects", {}))
                continue
            # 调用 OpenAI API
            project_response_str = call_openai(projects_prompt, pdf_text_cleaned)
            print("=====")
            print(project_response_str)
            print('======')
            # 解析 JSON 响应
            if isinstance(project_response_str, str) and project_response_str.startswith("{") and project_response_str.endswith("}"):
                try:
                    company_parsed_result = json.loads(project_response_str)
                except json.JSONDecodeError:
                    app.logger.error(f"JSON decoding failed for projects on experience {idx}. Response: {project_response_str}")
                    company_parsed_result = DEFAULT_EMPTY_RESPONSES.get("projects", {})
            else:
                app.logger.error(f"Invalid response format for projects on experience {idx}. Response: {project_response_str}")
                company_parsed_result = DEFAULT_EMPTY_RESPONSES.get("projects", {})
            
            work_experiences[idx].update(company_parsed_result)
        
        results["experiences"] = work_experiences
    return jsonify(results), 200

# 全局异常处理，隐藏 Python 堆栈给客户端
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error("Unhandled Exception: %s", e, exc_info=True)
    return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    # 生产中请保持 debug=False
    app.run(host='0.0.0.0', port=8081, debug=False)

