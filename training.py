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
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
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
model_save_path = f'output/ce-{EVIDENCE_LABEL}'

# #We use cross-encoder/ms-marco-MiniLM-L-12-v2 as base model and set num_labels=1, which predicts a continous score between 0 and 1
model = PromptTunedCrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', num_labels=1)


# Read scifact dataset
logger.info("Read scifact train dataset")
corpus = load_dataset('scifact', 'corpus')
claims = load_dataset('scifact', 'claims')


corpus_df = corpus['train'].to_pandas()
corpus_df['doc_id_str'] = corpus_df['doc_id'].apply(lambda x: str(x))

train_samples = []
train_df = claims['train'].to_pandas()
train_df = train_df.loc[(train_df['evidence_label'] == 'CONTRADICT') | (train_df['evidence_label'] == 'SUPPORT')]
for i, doc in train_df.iterrows():
    claim = doc['claim']
    evidence = corpus_df[corpus_df['doc_id_str'].apply(lambda x: str(x)) == doc['evidence_doc_id']]['abstract'].iloc[0][doc['evidence_sentences'][0]]
    type_ = doc['evidence_label']
    score = 1 if type_ == EVIDENCE_LABEL else 0
    train_samples.append(InputExample(texts=[claim, evidence], label=score))

dev_samples = []
dev_df = claims['validation'].to_pandas()
dev_df = dev_df.loc[(dev_df['evidence_label'] == 'CONTRADICT') | (dev_df['evidence_label'] == 'SUPPORT')]
for i, doc in dev_df.iterrows():
    claim = doc['claim']
    evidence = corpus_df[corpus_df['doc_id_str'].apply(lambda x: str(x)) == doc['evidence_doc_id']]['abstract'].iloc[0][doc['evidence_sentences'][0]]
    type_ = doc['evidence_label']
    score = 1 if type_ == EVIDENCE_LABEL else 0
    dev_samples.append(InputExample(texts=[claim, evidence], label=score))

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)


# # We add an evaluator, which evaluates the performance during training
# evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='scifact-test')


# # Train the model
# model.fit(train_dataloader=train_dataloader,
#           evaluator=evaluator,
#           epochs=num_epochs,
#           warmup_steps=warmup_steps,
#           output_path=model_save_path, show_progress_bar=True)
# model.save_soft_prompt('./output', f'{EVIDENCE_LABEL}-prompt.torch')

# #### Load model and eval on test set
model = PromptTunedCrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', num_labels=1)
model.set_soft_prompt_embeds(f'./output/{EVIDENCE_LABEL}-prompt.torch')

evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='scifact-test')
evaluator(model)
