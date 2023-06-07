from pathlib import Path
import argparse
import shutil

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset-path', '-dp', type=Path, default='./', help='path to dataset')
    args = argparser.parse_args()
    dataset_path = args.dataset_path
    data_path = dataset_path / 'data'
    if data_path.exists():
        with open(data_path / 'train.txt', 'r') as f:
            train_files = f.readlines()
            breakpoint()
    else:
        files = [file for file in dataset_path.glob('*')]
        train_percent = 0.8
        train_num = int(len(files) * train_percent)
        train_files = files[:train_num]
        test_files = files[train_num:]
        with open(dataset_path / 'train.txt', 'w') as f:
            for file in train_files:
                f.write(str(file) + '\n')
        with open(dataset_path / 'val.txt', 'w') as f:
            for file in test_files:
                f.write(str(file) + '\n')
        with open(dataset_path / 'test.txt', 'w') as f:
            for file in test_files:
                f.write(str(file) + '\n')    
        pass