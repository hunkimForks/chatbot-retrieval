import csv
import sys
import gzip
import os
import csv
import numpy as np
import random

"""
Note:
Use this script to generate the necessary data sets to run udc_predict/test/train.py scripts.

Command:
$python [Command] input_file_path [output_file_path_if_required]

1. Process the raw FB export so that each line consists of a "context" and "utterance."
$python prepare_fb_data.py process_raw_export raw_export_file_path output_file_path
ex) $python ./scripts/prepare_fb_data.py process_raw_export ./data/tf.txt.gz ./data/tf_processed.csv

2. Generate training/test/valid sets using the processed data from the above step
$python prepare_fb_data.py generate_data_sets processed_file_path
ex) $python ./scripts/prepare_fb_data.py generate_data_sets ./data/tf_processed.csv
"""

PROCESS_RAW_EXPORT = 'process_raw_export'
GENERATE_DATA_SETS = 'generate_data_sets'

# NOTE: we don't make use of num_likes at all
def _parse_info_from_line(line):
  line = line.decode('utf-8')
  raw = line.split("\t")
  post = {'num_likes': raw[0], 'body': raw[1].rstrip()}
  raw_comments = raw[2:]
  comments = []
  for i in range(0, len(raw_comments), 2):
    comments.append({'num_likes': raw_comments[i], 'body': raw_comments[i+1].rstrip()})
  return {'post': post, 'comments': comments}

# Process the raw data so each row contains a post and its reply.
# Note that "post" == "context" and "reply" == "utterance".
def process_raw_export(filename, output):
  with gzip.open(filename) as f1:
    with open(output, 'w') as f2:
      fieldnames = ['Context', 'Utterance']
      writer = csv.DictWriter(f2, fieldnames=fieldnames, delimiter='\t')
      writer.writeheader()

      # NOTE: if a post has many comments, we add multiple rows with the same post body
      for line in f1:
        info = _parse_info_from_line(line)
        for comment in info['comments']:
          writer.writerow({
            'Context': info['post']['body'].encode('utf-8'),
            'Utterance': comment['body'].encode('utf-8')
          })

# With the processed_data from above, we have to make train.csv, test.csv, and valid.csv.
# Let's separate it so that the ratio is 2:1:1 respectively.
# columns for train.csv: [Context, Utterance, Label]
# columns for test/valid.csv: [Context, Utterance, Distractor_0, .., Distractor_n]
def generate_data_sets(path_to_processed_data):
  with open(path_to_processed_data) as f1:
    rows = [r.decode('utf-8') for r in f1.read().splitlines()[1:]]
  rows = np.array(rows)
  np.random.shuffle(rows)

  # NOTE: we don't care if they are not exactly separated four ways -- close enough
  four_way_separated = np.array_split(rows, 4)
  train_rows = four_way_separated[0].tolist() + four_way_separated[1].tolist()
  test_rows = four_way_separated[2].tolist()
  valid_rows = four_way_separated[3].tolist()
  make_train_set(train_rows, './data/train_set.csv')
  make_test_set(test_rows, './data/test_set.csv')
  make_valid_set(valid_rows, './data/valid_set.csv')

# Get all the rows that do not have the given context
# i.e. get all the other posts
def _get_rows_with_different_context(context, rows):
  rows_with_different_context = []
  for row in rows:
    row_info = row.split('\t')
    if row_info[0] != context:
      rows_with_different_context.append(row)
  return rows_with_different_context

# Input: [Context, Utterance]
# Output: [Context, Utterance, Label]
# Label == 1 if Utterance belongs to the right Context
# Label == 0 otherwise.
# Make one set correctly labeled and the other set incorrectly labeled
# The output's size will be twice the input size
def make_train_set(train_rows, output):
  correct_rows = ['\t'.join([r, '1']) for r in train_rows]

  # assign a wrong utterance for each context
  incorrect_rows = []
  for row in train_rows[:]:
    row_info = row.split('\t')
    differently_contexted_rows = _get_rows_with_different_context(row_info[0], train_rows)
    random_row = random.choice(differently_contexted_rows)
    wrong_utterance = random_row.split('\t')[1]
    row_info[1] = wrong_utterance
    row_info.append('0')
    incorrect_rows.append('\t'.join(row_info))

  combined = correct_rows + incorrect_rows
  np.random.shuffle(combined)

  with open(output, 'w') as f2:
    fieldnames = ['Context', 'Utterance', 'Label']
    writer = csv.DictWriter(f2, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()

    for row in combined:
      row_info = row.split('\t')
      writer.writerow({
        'Context': row_info[0].encode('utf-8'),
        'Utterance': row_info[1].encode('utf-8'),
        'Label': row_info[2].encode('utf-8'),
      })

# Input: [Context, Utterance]
# Output: [Context, Utterance, Distractor_0, Distractor_1, Distractor_2]
def make_test_set(test_rows, output, num_distractors=3):
  output_rows = []

  for row in test_rows:
    row_info = row.split('\t')
    differently_contexted_rows = _get_rows_with_different_context(row_info[0], test_rows)
    distractor_rows = random.sample(differently_contexted_rows, num_distractors)
    distractor_utterances = [r.split('\t')[1] for r in distractor_rows]
    test_row = row_info + distractor_utterances
    output_rows.append('\t'.join(test_row))

  with open(output, 'w') as f2:
    fieldnames = ['Context', 'Utterance'] + [('Distractor_%d' % i) for i in range(num_distractors)]
    writer = csv.DictWriter(f2, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()

    for row in output_rows:
      row_info = row.split('\t')
      writer.writerow({
        'Context': row_info[0].encode('utf-8'),
        'Utterance': row_info[1].encode('utf-8'),
        'Distractor_0': row_info[2].encode('utf-8'),
        'Distractor_1': row_info[3].encode('utf-8'),
        'Distractor_2': row_info[4].encode('utf-8')
      })

# Note: redundant wrapper to express that we make valid_set and test_set the same way
def make_valid_set(valid_rows, output, num_distractors=3):
  make_test_set(valid_rows, output, num_distractors)

if __name__ == "__main__":
  cmd = sys.argv[1]
  if (cmd == PROCESS_RAW_EXPORT):
    print('Processing the raw export: ' + sys.argv[2])
    process_raw_export(sys.argv[2], sys.argv[3])
  elif (cmd == GENERATE_DATA_SETS):
    print('Generating data sets with: ' + sys.argv[2])
    generate_data_sets(sys.argv[2])
