import os
from tqdm import tqdm
import csv
import codecs
from utils.normalizeString import normalizeString
from dataset.vocab import Vocab

class CornellData(object):

    def __init__(self,data_path,output_file,delimiter,max_len):
        self.data_path = data_path
        self.output_file = output_file
        self.delimiter = delimiter
        self.max_len = max_len
        
        self.lines = self.loadLines()
        self.conversations = self.loadConversations()
        self.write()

        self.voc,self.pair = self.load_data()

    # Splits each line of the file into a dictionary of fields
    def loadLines(self):
        fileName = os.path.join(self.data_path, "movie_lines.txt")
        MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
        fields = MOVIE_LINES_FIELDS
        print("\nProcessing corpus...")
        lines = {}
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in tqdm(f):
                values = line.split(" +++$+++ ")
                # Extract fields
                lineObj = {}
                for i, field in enumerate(fields):
                    lineObj[field] = values[i]
                lines[lineObj['lineID']] = lineObj
        return lines

    # Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
    def loadConversations(self):
        fileName = os.path.join(self.data_path, "movie_conversations.txt")
        MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
        fields = MOVIE_CONVERSATIONS_FIELDS
        print("\nLoading conversations...")
        conversations = []
        with open(fileName, 'r', encoding='iso-8859-1') as f:
            for line in tqdm(f):
                values = line.split(" +++$+++ ")
                # Extract fields
                convObj = {}
                for i, field in enumerate(fields):
                    convObj[field] = values[i]
                # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
                lineIds = eval(convObj["utteranceIDs"])
                # Reassemble lines
                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(self.lines[lineId])
                conversations.append(convObj)
        return conversations

    # Extracts pairs of sentences from conversations
    def extractSentencePairs(self,conversations):
        qa_pairs = []
        for conversation in conversations:
            # Iterate over all the lines of the conversation
            for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
                inputLine = conversation["lines"][i]["text"].strip()
                targetLine = conversation["lines"][i + 1]["text"].strip()
                # Filter wrong samples (if one of the lists is empty)
                if inputLine and targetLine:
                    qa_pairs.append([inputLine, targetLine])
        return qa_pairs

    def write(self):
        # Write new csv file
        print("\nWriting newly formatted file...")
        with open(self.output_file, 'w', encoding='utf-8', newline='') as outputfile:
            writer = csv.writer(outputfile, delimiter=delimiter)
            for pair in self.extractSentencePairs(self.conversations):
                writer.writerow(pair)

    # Read query/response pairs and return a voc object
    def readVocs(self,datafile, corpus_name):
        print("Reading lines...")
        # Read the file and split into lines
        lines = open(datafile, encoding='utf-8'). \
            read().strip().split('\n')
        # Split every line into pairs and normalize
        pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
        voc = Vocab(corpus_name)
        return voc, pairs

    # Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
    def filterPair(self,p):
        # Input sequences need to preserve the last word for EOS token
        return len(p[0].split(' ')) < self.max_len and len(p[1].split(' ')) < self.max_len

    # Filter pairs using filterPair condition
    def filterPairs(self,pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    # Using the functions defined above, return a populated voc object and pairs list
    def load_data(self):
        print("Start preparing training data ...")
        voc, pairs = self.readVocs(self.output_file, self.data_path)
        print("Read {!s} sentence pairs".format(len(pairs)))
        pairs = self.filterPairs(pairs)
        print("Trimmed to {!s} sentence pairs".format(len(pairs)))
        print("Counting words...")
        for pair in pairs:
            voc.addSentence(pair[0])
            voc.addSentence(pair[1])
        print("Counted words:", voc.num_words)
        return voc, pairs

if __name__ == '__main__':
    corpus = "cornell movie-dialogs corpus"
    save_dir = "save"
    # Define path to new file
    output_file = os.path.join(corpus, "formatted_movie_lines.txt")
    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))
    cornell_data = CornellData(data_path=corpus,output_file=output_file,delimiter=delimiter,max_len=10)
