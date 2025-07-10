import torch
import transformers
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
sys_prompt = "[INST] <<SYS>> You are a helpful recipe assistant. You **only answer using the provided recipe documents**. Do not use any outside knowledge. If the question can't be answered from the documents, say you do not have that information. Follow the user’s query carefully, and ensure your answer is correct based on the documents.<</SYS>>"
end_prompt = " \n Using **only** the recipes above, answer the user's query in detail.  If information is missing, respond with a brief apology and that you cannot find it. [/INST]"
 
 
def build_content_prompt(i, name, tags, ingredients, description, steps, score = None):
    if not score:
        return f" **\n Retrieved Recipes:**\n Recipe {i}: {name}\n  - **Tags:** {tags} \n - **Ingredients:** {ingredients} \n - **Desription:** {description} \n- **steps:** {steps}"
    else:
        return f" **Retrieved Recipes:** \nRecipe {i}: {name} \n - **Tags:** {tags} \n - **Ingredients:** {ingredients} \n - **Description:** {description} \n- **steps:** {steps} \n- **score:** {score}"

def LLM(query, res):

    prompt = sys_prompt + f"**\n User Query:** {query}" +" \n ".join([build_content_prompt(row['official_id'], row['name'], row['tags'], row['ingredients'], row['description'], row['steps']) for row in res.to_dict(orient="records")])  + end_prompt


    print("Loading Model")
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"


    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map='auto'
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Done")
    print("Awaiting LLM Response")
    encoded_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded_prompt = encoded_prompt.to("cuda")

    generated_ids = model.generate(**encoded_prompt, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    print("####################################################")
    print("Response:")
    print(decoded[0])