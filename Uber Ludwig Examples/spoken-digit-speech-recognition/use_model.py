from ludwig.api import LudwigModel
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use Ludwig model')
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    args = parser.parse_args()

    model = LudwigModel.load(args.model)

    # make prediction on other example
    print(model.predict(data_dict={'audio_path': [args.input]}))