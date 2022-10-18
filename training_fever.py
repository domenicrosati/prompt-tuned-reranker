"""
This examples trains a CrossEncoder for the STSbenchmark task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continious labels 0...1 to indicate the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python training_stsbenchmark.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator, CERerankingEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
from datasets import load_dataset
from softprompt_crossencoder import PromptTunedCrossEncoder

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout


EVIDENCE_LABEL = 'CONTRADICT'

# #Define our Cross-Encoder
train_batch_size = 16
num_epochs = 4
warmup_steps = 0
model_save_path = f'output/ce-{EVIDENCE_LABEL}'

# #We use cross-encoder/ms-marco-MiniLM-L-12-v2 as base model and set num_labels=1, which predicts a continous score between 0 and 1
model = PromptTunedCrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', num_labels=1)

# Read scifact dataset
logger.info("Read FEVER train dataset")
wiki_pages = load_dataset('fever', 'wiki_pages')
wiki_df = wiki_pages['wikipedia_pages'].to_pandas()
fever = load_dataset('fever', 'v1.0')

EVIDENCE_LABEL = 'SUPPORTS'

fever_df = fever['train'].to_pandas()
fever_filtered = fever_df[fever_df['evidence_sentence_id'] != -1]
fever_filtered = fever_filtered[(fever_filtered['label'] == 'SUPPORTS') | (fever_filtered['label'] == 'REFUTES')]
wiki_df_filtered = wiki_df[wiki_df['id'].isin(fever_filtered['evidence_wiki_url'])]

train_samples = []
train_df = fever_filtered
for i, doc in train_df.iterrows():
    try:
        claim = doc['claim']
        evidence = wiki_df_filtered[wiki_df_filtered['id'] == doc['evidence_wiki_url']]['lines'].iloc[0].split('\n')[
            doc['evidence_sentence_id']
        ].split('\t')[1]
        type_ = doc['label']
        score = 1 if type_ == EVIDENCE_LABEL else 0
        train_samples.append(InputExample(texts=[claim, evidence], label=score))
    except:
        pass

fever_df = fever['validation'].to_pandas()
fever_filtered = fever_df[fever_df['evidence_sentence_id'] != -1]
fever_filtered = fever_filtered[(fever_filtered['label'] == 'SUPPORTS') | (fever_filtered['label'] == 'REFUTES')]
wiki_df_filtered = wiki_df[wiki_df['id'].isin(fever_filtered['evidence_wiki_url'])]

dev_samples = []
dev_samples_ranking = {}
dev_df = fever_filtered
for i, doc in dev_df.iterrows():
    try:
        claim = doc['claim']
        evidence = wiki_df_filtered[wiki_df_filtered['id'] == doc['evidence_wiki_url']]['lines'].iloc[0].split('\n')[
            doc['evidence_sentence_id']
        ].split('\t')[1]
        type_ = doc['label']
        score = 1 if type_ == EVIDENCE_LABEL else 0
        dev_samples.append(InputExample(texts=[claim, evidence], label=score))

        if claim not in dev_samples_ranking:
            dev_samples_ranking[claim] = {
                'query': claim, 'positive': [], 'negative': []
            }

        if score == 1:
            dev_samples_ranking[claim]['positive'].append(evidence)

        dev_samples_ranking[claim]['negative'].extend(
            sent.split('\t')[1] for sent in
            wiki_df_filtered[wiki_df_filtered['id'] == doc['evidence_wiki_url']]['lines'].iloc[0].split('\n')
            if sent.split('\t')[1] != evidence and score == 1
        )

    except:
        pass

dev_ranking_samples_filtered = []
for sample in list(dev_samples_ranking.values()):
    if len(sample['positive']) == 0 or len(sample['negative']) == 0:
        continue
    dev_ranking_samples_filtered.append(sample)

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# # We add an evaluator, which evaluates the performance during training
binary_evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name='scifact-test')
ranking_evaluator = CERerankingEvaluator.from_input_examples(dev_ranking_samples_filtered, name='scifact-test')

# # Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=binary_evaluator,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path, show_progress_bar=True)
model.save_soft_prompt('./output', f'{EVIDENCE_LABEL}-prompt.torch')

# #### Load model and eval on test set


print("Performance on randomly intitialized prompt")
model = PromptTunedCrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', num_labels=1)
print("Ranking evaluation", ranking_evaluator(model))
print("Binary evaluation", binary_evaluator(model))

print("Performance on trained prompt")
model = PromptTunedCrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', num_labels=1)
model.set_soft_prompt_embeds(f'./output/{EVIDENCE_LABEL}-prompt.torch')
evaluator = CERerankingEvaluator.from_input_examples(dev_samples, name='scifact-test')
print("Ranking evaluation", ranking_evaluator(model))
print("Binary evaluation", binary_evaluator(model))

print("Performance on without prompt")
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', num_labels=1)
evaluator = CERerankingEvaluator.from_input_examples(dev_samples, name='scifact-test')
print("Ranking evaluation", ranking_evaluator(model))
print("Binary evaluation", binary_evaluator(model))
