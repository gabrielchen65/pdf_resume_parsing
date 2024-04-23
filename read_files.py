from utility import get_prompt, extract_text

class Read_files():
    def __init__(self):
        # Education, Skills, Work Experience, Projects, Certifications, Achievements
        folder = "./prompts/"
        system_prompt_path_heading_info = folder    + "resume-extractor_heading_info.txt"
        system_prompt_path_education = folder       + "resume-extractor_education.txt" 
        system_prompt_path_skills = folder          + "resume-extractor_skills.txt" 
        system_prompt_path_work_experience = folder + "resume-extractor_work_experience.txt" 
        system_prompt_path_projects = folder        + "resume-extractor_projects.txt"
        system_prompt_path_certifications = folder  + "resume-extractor_certifications.txt"
        system_prompt_path_achievements = folder    + "resume-extractor_achievements.txt"

        self.system_prompt_heading_info = get_prompt(system_prompt_path_heading_info)
        self.system_prompt_education = get_prompt(system_prompt_path_education)
        self.system_prompt_skills = get_prompt(system_prompt_path_skills)
        self.system_prompt_work_experience = get_prompt(system_prompt_path_work_experience)
        self.system_prompt_projects = get_prompt(system_prompt_path_projects)
        self.system_prompt_certifications = get_prompt(system_prompt_path_certifications)
        self.system_prompt_achievements = get_prompt(system_prompt_path_achievements)


        pdf_path = "data/my_resume.pdf"
        self.resume_text = extract_text(pdf_path)
