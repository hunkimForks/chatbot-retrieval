import pdb
import csv
import sys

"""
Process the historical data from FB groups to conform the
chatbot-retrieval data format.
"""

# NOTE: we don't make use of num_likes at all
def parse_info_from_line(line):
  raw = line.split("\t")
  post = {'num_likes': raw[0], 'body': raw[1].rstrip()}
  raw_comments = raw[2:]
  comments = []
  for i in range(0, len(raw_comments), 2):
    comments.append({'num_likes': raw_comments[i], 'body': raw_comments[i+1].rstrip()})
  return {'post': post, 'comments': comments}

def process_raw_data(filename, output):
  with open(filename) as f1:
    with open(output, 'w') as f2:
      fieldnames = ['Context', 'Utterance']
      writer = csv.DictWriter(f2, fieldnames=fieldnames)
      writer.writeheader()

      # NOTE: if a post has many comments, we add multiple rows with the same post body
      for line in f1:
        info = parse_info_from_line(line)
        for comment in info['comments']:
          writer.writerow({'Context': info['post']['body'], 'Utterance': comment['body']})

if __name__ == "__main__":
  print("Processing FB data: " + sys.argv[1])
  print("Output: " + sys.argv[2])
  process_raw_data(sys.argv[1], sys.argv[2])
