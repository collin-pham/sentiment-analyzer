import math, os, pickle, re

class Bayes_Classifier:
   def __init__(self):
      """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text."""
      self.pos = dict()
      self.neg = dict()

      if os.path.isfile("Negative_Dict"):
         self.neg = self.load("Negative_Dict")
 
      if os.path.isfile("Positive_Dict"):
         self.pos = self.load("Positive_Dict")
 
      else:
         self.train()

   def train(self):   
      """Trains the Naive Bayes Sentiment Classifier."""
      
      lFileList = []
      #Extract the file names and put them in a list
      for fFileObj in os.walk("movies_reviews/"):
         lFileList = fFileObj[2]
         
      #For each file name, load the text into a string and then tokenize the string.
      for f in lFileList:
         review = self.loadFile("movies_reviews/" + f)
         words = self.tokenize(review)
         #If the file is a negative review,
         if f[7] == '1':
            #For the word in the tokenized string,
            for word in words:
               #If the word is in the dict already, add to its count.
               if word in self.neg:
                  self.neg[word] = self.neg[word] + 1
                  #Else, create it.
               else:
                  self.neg[word] = 1
                  
         #Same process for pos dict.
         else:
            for word in words:
               if word in self.pos:
                  self.pos[word] = self.pos[word] + 1
               else:
                  self.pos[word] = 1
      #Save the dict.
      self.save(self.neg, "Negative_Dict")
      self.save(self.pos, "Positive_Dict")
    
   def classify(self, sText):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """

      #tokenize the given text
      words = self.tokenize(sText)

      #initialize denominators and numerators
      pos_denom = sum(self.pos.values()) * 1.0
      neg_denom = sum(self.neg.values()) * 1.0
      
      denom = (sum(self.pos.values()) + sum(self.neg.values())) * 1.0
      
      #prior probability of a word being pos or neg
      pos_prob = pos_denom / denom
      neg_prob = neg_denom / denom
      
      #take log to minimize round off error of small numbers
      pos_prob = math.log(pos_prob)
      neg_prob = math.log(neg_prob)

      #for the word in the tokenized words
      for word in words:
         #if word in pos dict, add to pos prob
         if word in self.pos:
            pos_prob = pos_prob + math.log((self.pos[word]) / pos_denom)
         #add one smoothing
         else:
            pos_prob = pos_prob + math.log(1.0 / pos_denom)

         #if word in neg dict, add to neg prob
         if word in self.neg:
            neg_prob = neg_prob + math.log((self.neg[word]) / neg_denom)
         #add one smoothing
         else:
            neg_prob = neg_prob + math.log(1.0 / neg_denom)

      #if the difference is less than .1, call it neutral
      if abs(pos_prob - neg_prob) < .1:
         return "neutral"
      #if pos > neg then its pos
      elif pos_prob > neg_prob:
         return "positive"
      #its neg if it's not pos
      else:
         return "negative"
            

   def loadFile(self, sFilename):
      """Given a file name, return the contents of the file as a string."""

      f = open(sFilename, "r")
      sTxt = f.read()
      f.close()
      return sTxt
   
   def save(self, dObj, sFilename):
      """Given an object and a file name, write the object to the file using pickle."""

      f = open(sFilename, "w")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()
   
   def load(self, sFilename):
      """Given a file name, load and return the object stored in the file."""

      f = open(sFilename, "r")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

   def tokenize(self, sText): 
      """Given a string of text sText, returns a list of the individual tokens that 
      occur in that string (in order)."""

      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-":
            sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))
               
      if sToken != "":
         lTokens.append(sToken)

      return lTokens

