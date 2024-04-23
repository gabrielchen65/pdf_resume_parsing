from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity_sklearn


class SentenceTransformer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def extract_sentences_embeddings(self, sentences):
        # Sentences we want sentence embeddings for
        # sentences = ['This is an example sentence', 'Each sentence is converted']
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings
    
    def locate_section_line_numbers(self, resume):
        # using BERT embeddings
        lines = resume.split("\n")
        targets_dict = {
            "Education": ["Education", "Educations", "educational background", "education and training'"],
            "Skills": ["Skills", "Skill"],
            "Work Experience": ["Work", "Work Experience", "Experience"],
            "Projects": ["Projects", "Project"],
            "Certifications": ["Certification", "Certifications"],
            "Achievements": ["Achievements"]
        }

        lines_embeddings = self.extract_sentences_embeddings(lines)

        section_line_number = {}
        for target_name, target_set in targets_dict.items():
                name = ""
                targets_embeddings = self.extract_sentences_embeddings(target_set)

                # Calculate cosine similarity between each line and the target
                cosine_similarities = cosine_similarity_sklearn(targets_embeddings, lines_embeddings)
                # convert to numpy array, shape of (num_targets, num_lines), (4,42 in this case)
                cosine_similarities = np.array(cosine_similarities)
                # Find the max similarity overall
                (target_idx, line_idx) = np.unravel_index(np.argmax(cosine_similarities, axis=None), cosine_similarities.shape)
                # skip if there's no match
                print(target_name, target_set)
                print(target_name, cosine_similarities[target_idx][line_idx])
                print()
                if cosine_similarities[target_idx][line_idx] < 0.7:
                    continue
                # save the line number
                section_line_number[line_idx] = target_name

        section_line_number = dict(sorted(section_line_number.items()))
        print(section_line_number.keys())
        print([lines[i] for i in section_line_number.keys()])

        return section_line_number

