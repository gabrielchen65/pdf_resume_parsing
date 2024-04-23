import pdfplumber
from langchain_core.output_parsers import JsonOutputParser


def parse_json_markdown(json_string: str) -> dict:
    try:
        parser = JsonOutputParser()
        parsed = parser.parse(json_string)

        return parsed
    except Exception as e:
        print(e)
        return None

def get_prompt(system_prompt_path: str) -> str:
        """
        Reads the content of the file at the given system_prompt_path and returns it as a string.

        Args:
            system_prompt_path (str): The path to the system prompt file.

        Returns:
            str: The content of the file as a string.
        """
        with open(system_prompt_path, encoding="utf-8") as file:
            return file.read().strip() + "\n"
        
def extract_text(pdf_path: str):
    resume_text = "" 
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in range(len(pdf.pages)):
            resume_text  += pdf.pages[page_num].extract_text()
        return resume_text