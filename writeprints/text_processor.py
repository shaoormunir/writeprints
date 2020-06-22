from writeprints.feature_extractor import FeatureExtractor
from tqdm.auto import tqdm
import pandas as pd


class Processor(object):
    def __init__(self, flatten=True):
        self.flatten = flatten
        self.extractor = FeatureExtractor(self.flatten)

    def extract(self, text):
        return self.extractor.process(text)

    def extract_df(self, df):
        assert('text' in df)

        df_out = pd.DataFrame()

        for i, row in tqdm(df.iterrows(), total=len(df)):
            output = self.extractor.process(row['text'])
            output['text'] = row['text']
            df_out = df_out.append(output, ignore_index=True)
          
        return df_out
