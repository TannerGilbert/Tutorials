import os
import pandas as pd
import argparse


def folder_structure_to_csv(path: str, save_path: str):
    paths = []
    labels = []
    for folder in os.listdir(path):
        for image in os.listdir(path+'/'+folder):
            paths.append(folder+'/'+image)
            labels.append(folder)
    df = pd.DataFrame({'image_name': paths,'label': labels})
    df.to_csv(save_path, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create CSV from folder structure of a data-set')
    parser.add_argument('-p', '--path', type=str, required=True, help='Path to the data-set directory')
    parser.add_argument('-s', '--save_path', type=str, required=True, help='Path where the csv will be saved')
    args = parser.parse_args()
    folder_structure_to_csv(args.path, args.save_path)
