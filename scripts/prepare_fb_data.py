import pdb
import csv
import sys
import gzip

"""
Process the historical data from FB groups to conform the
chatbot-retrieval data format.
To run: $python prepare_fb_data.py input_file_path output_file_path
"""

# NOTE: we don't make use of num_likes at all
def parse_info_from_line(line):
  line = line.decode('utf-8')
  raw = line.split("\t")
  post = {'num_likes': raw[0], 'body': raw[1].rstrip()}
  raw_comments = raw[2:]
  comments = []
  for i in range(0, len(raw_comments), 2):
    comments.append({'num_likes': raw_comments[i], 'body': raw_comments[i+1].rstrip()})
  return {'post': post, 'comments': comments}

def process_raw_data(filename, output):
  with gzip.open(filename) as f1:
    with open(output, 'w') as f2:
      fieldnames = ['Context', 'Utterance']
      writer = csv.DictWriter(f2, fieldnames=fieldnames, delimiter='\t')
      writer.writeheader()

      # NOTE: if a post has many comments, we add multiple rows with the same post body
      for line in f1:
        info = parse_info_from_line(line)
        for comment in info['comments']:
          writer.writerow({
            'Context': info['post']['body'].encode('utf-8'),
            'Utterance': comment['body'].encode('utf-8')
          })

if __name__ == "__main__":
  print("Processing FB data: " + sys.argv[1])
  print("Output: " + sys.argv[2])
  process_raw_data(sys.argv[1], sys.argv[2])
