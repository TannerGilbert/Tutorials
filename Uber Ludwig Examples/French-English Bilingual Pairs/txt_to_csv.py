import os
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('/media/gilbert/948A92E98A92C760/Local_Programming/Datasets/Other/French-English Bilingual Pairs/fra.txt', sep='\t', header=None, names=['english', 'french'])
    df.head()
    df.to_csv('/media/gilbert/948A92E98A92C760/Local_Programming/Datasets/Other/French-English Bilingual Pairs/fra.csv')