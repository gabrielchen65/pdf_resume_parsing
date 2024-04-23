import os
from utility import parse_json_markdown
from sentence_transformers import SentenceTransformer
from read_files import Read_files
from llm import LLM
import json
from evaluation import evaluate

def split_text_into_sections(resume_text, section_line_number):
    
    index_list = list(section_line_number.keys())
    sections = {
    "heading_info": resume_text.split('\n')[:index_list[0]],
    }

    for idx, (start_line, section_name) in enumerate(section_line_number.items()):
        if idx < len(index_list) - 1:
            sections[section_name] = resume_text.split('\n')[index_list[idx]:index_list[idx+1]]
        else:
            sections[section_name] = resume_text.split('\n')[index_list[idx]:],    
    return sections


def parse_all_sections(read_files, sections, llm_instance):
    print("Parsing heading information...")
    prompt = f'<s> [INST] {read_files.system_prompt_heading_info} {sections["heading_info"]} [/INST]'
    response = llm_instance.invoke(prompt)
    resume_json_heading_info = parse_json_markdown(response)

    print("Parsing education...")
    prompt = f'<s> [INST] {read_files.system_prompt_education} {sections["Education"]} [/INST]'
    response = llm_instance.invoke(prompt)
    resume_json_education = parse_json_markdown(response)

    print("Parsing skills...")
    prompt = f'<s> [INST] {read_files.system_prompt_skills} {sections["Skills"]} [/INST]'
    response = llm_instance.invoke(prompt)
    resume_json_skills = parse_json_markdown(response)

    print("Parsing work experience...")
    prompt = f'<s> [INST] {read_files.system_prompt_work_experience} {sections["Work Experience"]} [/INST]'
    response = llm_instance.invoke(prompt)
    resume_json_work_experience = parse_json_markdown(response)

    print("Parsing projects...")
    prompt = f'<s> [INST] {read_files.system_prompt_projects} {sections["Projects"]} [/INST]'
    response = llm_instance.invoke(prompt)
    resume_json_projects = parse_json_markdown(response) 

    resume_all = assemble_sections(resume_json_heading_info, 
                                   resume_json_education, 
                                   resume_json_skills, 
                                   resume_json_work_experience, 
                                   resume_json_projects)

    return resume_all


def assemble_sections(resume_json_heading_info, resume_json_education, resume_json_skills, resume_json_work_experience, resume_json_projects):
    resume_all = {
        "name": resume_json_heading_info["name"],
        "summary": resume_json_heading_info["summary"],
        "phone": resume_json_heading_info["phone"],
        "email": resume_json_heading_info["email"],
        "media": {
            "linkedin": resume_json_heading_info["linkedin"] if "linkedin" in resume_json_heading_info else "",
            "github": resume_json_heading_info["github"] if "github" in resume_json_heading_info else "",
            "devpost": resume_json_heading_info["devpost"] if "devpost" in resume_json_heading_info else "",
            "medium": resume_json_heading_info["medium"] if "medium" in resume_json_heading_info else "",
            "leetcode": resume_json_heading_info["leetcode"] if "leetcode" in resume_json_heading_info else "",
            "dagshub": resume_json_heading_info["dagshub"] if "dagshub" in resume_json_heading_info else "",
            "kaggle": resume_json_heading_info["kaggle"] if "kaggle" in resume_json_heading_info else "",
            "instagram": resume_json_heading_info["instagram"] if "instagram" in resume_json_heading_info else "",
        },
        "education": resume_json_education["education"],
        "skills": resume_json_skills["skills"],
        "work_experience": resume_json_work_experience["work_experience"],
        "projects": resume_json_projects["projects"],
        "certifications": [],
        "achievements": []
    }

    json.dump(resume_all, open("resume_generated.json", "w"), indent=2)
    return resume_all

def main():
    # Read the resume file
    read_files = Read_files()
    resume_text = read_files.resume_text

    # find the titles of the sections
    titles_finder = SentenceTransformer()
    section_line_number = titles_finder.locate_section_line_numbers(resume_text)

    # split the resume into sections
    sections = split_text_into_sections(resume_text, section_line_number)
    
    # call LLM for parsing the sections
    llm = LLM()
    resume_generated = parse_all_sections(read_files, sections, llm.llm_instance)
    
    # evaluate the results, default with cosine similarity
    evaluate(resume_generated)


if __name__ == "__main__":
    main()