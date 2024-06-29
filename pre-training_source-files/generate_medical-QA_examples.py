import os
import json
import argparse

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--corpus_dir', type=str, default='medical_datasets/source/')
    parser.add_argument('--output_p', type=str, default='medical_datasets/')
    args = parser.parse_args()

    prompt = json.load(open("task_prompts/pre-train_macro-prompts.json", encoding="utf8"))["answer_MedLFQA_prompt"]

    res = {}
    for file in [f'{args.corpus_dir}{"MedCalc.json"}']:
    #for file in [f'{args.corpus_dir}{f}' for f in os.listdir(args.corpus_dir) if f.endswith('.jsonl')]:
        source_data = [json.loads(line) for line in open(file, 'r', encoding='utf8') if line != None] if file.endswith('.jsonl') else json.load(open(file, 'r', encoding='utf8'))

        if "MedInstruct" in file:
            for i, data in enumerate(source_data):
                if "instruction" in data and (data['instruction'] == None or data['instruction'] == ""):
                    continue

                if data["input"] == "<noinput>":
                    prompt = json.load(open("task_prompts/pre-train_macro-prompts.json", encoding="utf8"))["MedInstruct-52k_no-input_prompt"]

                    res[f'{file.split("/")[-1][:-5]}_{i}'] = {
                        "id" : f'{file.split("/")[-1][:-5]}_{i}',
                        "text" : f'{prompt.replace("$instruction", data["instruction"])}{data["output"]}</s>'
                    }

                else:
                    prompt = json.load(open("task_prompts/pre-train_macro-prompts.json", encoding="utf8"))["MedInstruct-52k_input_prompt"]

                    res[f'{file.split("/")[-1][:-5]}_{i}'] = {
                        "id" : f'{file.split("/")[-1][:-5]}_{i}',
                        "text" : f'{prompt.replace("$instruction", data["instruction"]).replace("$input", data["input"])}{data["output"]}</s>'
                    }

        elif "MedCalc" in file:
            prompt = json.load(open("task_prompts/pre-train_macro-prompts.json", encoding="utf8"))["MedCalc_prompt"]

            for i, data in enumerate(source_data):
                if "Question" in data and (data['Question'] == None or data['Question'] == ""):
                    continue

                res[f'{file.split("/")[-1][:-5]}_{i}'] = {
                    "id" : f'{file.split("/")[-1][:-5]}_{i}',
                    "text" : f'{prompt.replace("$question", data["Question"]).replace("$patient_note", data["Patient Note"])}{data["Ground Truth Explanation"]}</s>'
                }

        else:
            for i, data in enumerate(source_data):
                if "question" in data and (data['Question'] == None or data['Question'] == ""):
                    continue

                res[f'{file.split("/")[-1][:-6]}_{i}'] = {
                    "id" : f'{file.split("/")[-1][:-6]}_{i}',
                    "text" : f'{prompt.replace("$question", data["Question"])}{data["Free_form_answer"]}</s>'
                }
        

    print(f'Generated {len(res)} examples')

    #with safe_open_w(f'{args.output_p}pre-train_MedInstruct-52k.json') as out_f:
    #    json.dump(res, out_f, indent=4)

    with safe_open_w(f'{args.output_p}pre-train_MedCalc_train.json') as out_f:
        json.dump({key: value for i, (key, value) in enumerate(res.items()) if i % 10 != 0}, out_f, indent=4)

    with safe_open_w(f'{args.output_p}pre-train_MedCalc_dev.json') as out_f:
        json.dump({key: value for i, (key, value) in enumerate(res.items()) if i % 10 == 0}, out_f, indent=4)


if __name__ == '__main__':
    main()