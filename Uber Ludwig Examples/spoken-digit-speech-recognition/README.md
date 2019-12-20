# Audio Classification with Uber Ludwig
from the [official documentation](https://uber.github.io/ludwig/examples/#spoken-digit-speech-recognition)

## Download data-set and create csv

```bash
git clone https://github.com/Jakobovski/free-spoken-digit-dataset.git
mkdir speech_recog_digit_data
cp -r free-spoken-digit-dataset/recordings speech_recog_digit_data
cd speech_recog_digit_data
```

```bash
echo "audio_path","label" >> "spoken_digit.csv"
cd "recordings"
ls | while read -r file_name; do
   audio_path=$(readlink -m "${file_name}")
   label=$(echo ${file_name} | cut -c1)
   echo "${audio_path},${label}" >> "../spoken_digit.csv"
done
cd "../"
```

## Creating the model_definition file
To classify audio with ludwig you will have one input and one output feature. The input feature is of type audio. The output is a category.

model_definition.yaml
```yaml
input_features:
    -
        name: audio_path
        type: audio
        encoder: stacked_cnn
        preprocessing:
            audio_feature:
                type: fbank
                window_length_in_s: 0.025
                window_shift_in_s: 0.01
                num_filter_bands: 80
            audio_file_length_limit_in_s: 1.0
            norm: per_file
        reduce_output: concat
        conv_layers:
            -
                num_filters: 16
                filter_size: 6
                pool_size: 4
                pool_stride: 4
                dropout: true
            -
                num_filters: 32
                filter_size: 3
                pool_size: 2
                pool_stride: 2
                dropout: true
        fc_layers:
            -
                fc_size: 64
                dropout: true

output_features:
    -
        name: label
        type: category

training:
    dropout_rate: 0.4
    early_stop: 10
```

Now that you have the model_definition.yaml file created you can train the model by executing the following command.

```bash
ludwig experiment \
  --data_csv <PATH_TO_SPOKEN_DIGIT_CSV> \
  --model_definition_file model_definition_file.yaml
```

## Making predictions using Python

To use the trained model in Python you need to import ```LudwigModel``` from ```ludwig.api```, then you can load the model using the ```load``` method. After that you can make predictions using the ```predict``` method. 

```python
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
```

## Serving the model as a RestAPI

Since Uber Ludwig v0.2 you can deploy your Ludwig model as a RestAPI by executing a single command.

```bash
ludwig serve --model_path results/experiment_run/model/
```

Now you can make a prediction with curl:
```bash
curl http://0.0.0.0:8000/predict -X POST -F 'audio_path=@<path to audio file>'
```