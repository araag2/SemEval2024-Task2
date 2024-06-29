import os
import json
import argparse

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--corpus_dir', type=str, default='medical_datasets/')
    parser.add_argument('--output_p', type=str, default='medical_datasets/')
    args = parser.parse_args()

    devs = [f'{args.corpus_dir}{f}' for f in os.listdir(args.corpus_dir) if f.endswith('dev.json')]
    trains = [f'{args.corpus_dir}{f}' for f in os.listdir(args.corpus_dir) if f.endswith('train.json')]

    n_devs = {}
    n_trains = {}

    for dev in devs:
        source_data = json.load(open(dev, 'r', encoding='utf8'))
        for i in source_data:
            n_devs[i] = {
                "id" : i,
                "text" : source_data[i]["text"]
            }

    for train in trains:
        source_data = json.load(open(train, 'r', encoding='utf8'))
        for i in source_data:
            n_trains[i] = {
                "id" : i,
                "text" : source_data[i]["text"]
            }

    with safe_open_w(f'{args.output_p}pre-train_MedMix_dev.json') as out_f:
        json.dump(n_devs, out_f, indent=4)

    with safe_open_w(f'{args.output_p}pre-train_MedMix_train.json') as out_f:
        json.dump(n_trains, out_f, indent=4)


if __name__ == '__main__':
    main()