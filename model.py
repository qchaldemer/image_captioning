import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_dim = hidden_size
        
        # embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # the LSTM takes embed features from the images
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # the linear layer that maps the hidden state output dimension
        self.hidden2vocab = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        #print(features.shape, captions.shape)
        
        captions = captions[:,:-1]
        # create embedded word vectors for each word in a sentence
        embeds = self.word_embeddings(captions)
        
        features = features.view(features.shape[0], 1, features.shape[1])        
        
        inputs = torch.cat((features, embeds), dim=1)
        
        #print(embeds.shape)
        #print(inputs.shape)
            
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        
        lstm_out, _ = self.lstm(inputs)
        
        # get the scores for the most likely word 
        vocab_outputs = self.hidden2vocab(lstm_out)
        
        return vocab_outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        predicted_sentence = []
        
        for i in range(max_len):        
            sentence_lstm, states = self.lstm(inputs, states)
            x = self.hidden2vocab(sentence_lstm)
            prediction = x.argmax(dim=2)
            predicted_sentence.append(prediction[0].item())
            inputs = self.word_embeddings(prediction)
            #sentence_scores = nn.Softmax(sentence_outputs, dim=2)
            #_, max_word = torch.max(sentence_scores, 1)

        
        return predicted_sentence