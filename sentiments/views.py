from django.shortcuts import render

# Create your views here.

import torch
import pickle
import torch.nn as nn

class SentimentLSTM(nn.Module):
    """
    The LSTM model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentLSTM, self).__init__()

         # output size of the FC after the RNN
        self.output_size = output_size
        
        # number of hidden states for the RNN
        self.n_layers = n_layers
        
        # size of each hidden state of the RNN
        self.hidden_dim = hidden_dim
        
        # dimension for the embedding before feeding it into the RNN
        self.embedding_dim = embedding_dim
        
        # define embedding, LSTM, dropout and Linear layers here
        
        # embedding layer before feeding input to lstm
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, # The number of expected features in the input x
            hidden_dim, # The number of features in the hidden state h
            n_layers, # Number of recurrent layers (i.e. number of hidden states)
            dropout=drop_prob, # If non-zero, introduces a dropout layer on the outputs of each RNN layer except 
                               # the last layer
            batch_first=True # If True, then the input and output tensors are provided as (batch, seq, feature)
        )
        
        # dropout layer to add after the LSTM and before the FC
        self.dropout = nn.Dropout(0.4)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        
        # get the number of samples in the batch
        batch_size = x.size(0)

        # first we need to perform an embedding of our input
        embedded = self.embedding(x)
        
        # feed the embetted input (i.e. batch) to the LSTM
        # get the output which is feeded to FC
        rnn_out, hidden = self.lstm(embedded, hidden)
    
        # stack up LSTM outputs
        rnn_out = rnn_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer 
        # input is the output of the LSTM
        out = self.dropout(rnn_out)
        out = self.fc(out)
        
        # sigmoid function for the prediction
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        
        # get last batch of labels
        sig_out = sig_out[:, -1] 
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
        
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden


# device = torch.device('cuda')
with open('dict.pkl','rb') as f :
    vocab_to_int = pickle.load(f)

# Instantiate the model with these hyperparameters


vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 300
hidden_dim = 256
n_layers = 2

model = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)


device = torch.device('cpu')
checkpoint = torch.load('lstmmodelgpu2.tar', map_location=device)

state_dict =checkpoint['model_state_dict']
state_dict['embedding.weight'] = state_dict['encoder.weight']
state_dict.pop('encoder.weight')

model.load_state_dict(state_dict)

def predict_sentiment(sentence, seq_length=200):
    tokenized = [vocab_to_int[word] for word in sentence.split()]
    review = tokenized
    tensor = torch.LongTensor(review).to(device)
    tensor = tensor.unsqueeze(0)
    # n_layers, batch, hidden_dim
    test_hidden = ((torch.zeros(2, 1, 256)),(torch.zeros(2, 1, 256)))
    output, h = model(tensor, test_hidden)
    prediction = torch.sigmoid(output)
    return prediction.item()


def home(request):
    # if request.user.is_authenticated:
    #     if request.user.is_owner:
    #         return redirect('owners:quiz_change_list')
    #     else:
    #         return redirect('freelancers:quiz_list')
    sentiment = ''
    if request.method == "POST":
        # Do validation stuff here
        sentence = request.POST['text']
        score = predict_sentiment(sentence)
        if score > 0.5:
            sentiment = 'positive'
        else:
            sentiment = 'negative'


    return render(request, 'home.html', {'sentiment': sentiment})