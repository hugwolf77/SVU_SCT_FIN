import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, num_layers=1, batch_norm=False):
        super(Model, self).__init__()
        self.input_size = args.channels
        self.seq_len = args.seq_len
        # self.embeding_size = args.vae_hid_size
        # self.hidden_size_1 = (self.embeding_size*2)
        self.hidden_size_1 = (self.input_size//2)
        self.hidden_size_2 = (self.hidden_size_1//2)
        self.latent_size = args.vae_latent_size

        self.num_layers = num_layers
        self.batch_norm = batch_norm

        self.encoder_lstm_1 = nn.LSTM(
            self.input_size, self.input_size, num_layers, batch_first=True)
        self.encoder_lstm_2 = nn.LSTM(
            self.input_size, self.hidden_size_1, num_layers, batch_first=True)
        self.encoder_lstm_3 = nn.LSTM(
            self.hidden_size_1, self.hidden_size_2, num_layers, batch_first=True)
        self.encoder_lstm_4 = nn.LSTM(
            self.hidden_size_2, self.latent_size, num_layers, batch_first=True)
        
        
        self.encoder_fc_mu = nn.Linear(self.latent_size, self.latent_size)
        self.encoder_fc_logvar = nn.Linear(self.latent_size, self.latent_size)

        self.decoder_lstm_1 = nn.LSTM(
            self.latent_size*2, self.hidden_size_2, num_layers, batch_first=True)
        self.decoder_lstm_2 = nn.LSTM(
            self.hidden_size_2, self.hidden_size_1, num_layers, batch_first=True)
        self.decoder_lstm_3 = nn.LSTM(
            self.hidden_size_1, self.input_size, num_layers, batch_first=True)
        self.decoder_lstm_4 = nn.LSTM(
             self.input_size, self.input_size, num_layers, batch_first=True)
 
        # self.decoder_fc = nn.Linear(self.hidden_size, self.input_size)
        self.output_fc = nn.Linear(self.input_size, self.input_size)

        # self.relu = nn.ReLU()
        # self.batchnorm1d_encoder = nn.BatchNorm1d(self.hidden_size)
        # self.batchnorm1d_decoder = nn.BatchNorm1d(self.hidden_size)

    def encode(self, x):
        x, _ = self.encoder_lstm_1(x)
        x, _ = self.encoder_lstm_2(x)
        x, _ = self.encoder_lstm_3(x)
        x, _ = self.encoder_lstm_4(x)
        # x = x[:, -1, :]  # get only last hidden state ????
        # x = self.relu(x)
        # print("encoder lstm lyr out >>>> activation func :: ", x)
        # if self.batch_norm:
        #     x = self.batchnorm1d_encoder(x)  # apply batch normalization
        mu = self.encoder_fc_mu(x)
        logvar = self.encoder_fc_logvar(x)
        return x, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, x_z):
        # z = self.decoder_fc(z)
        # if self.batch_norm:
            # z = self.batchnorm1d_decoder(z)  # apply batch normalization
        # repeat along sequence length
        # z = z.unsqueeze(1).repeat(1, self.seq_len, 1)

        out, _ = self.decoder_lstm_1(x_z)
        out, _ = self.decoder_lstm_2(out)
        out, _ = self.decoder_lstm_3(out)
        out, _ = self.decoder_lstm_4(out)
        # output = self.relu(output)
        output = self.output_fc(out)
        return output

    def forward(self, x):
        x, mu, logvar = self.encode(x)
        mu = F.relu(mu)
        logvar = F.relu(logvar)
        z = self.reparameterize(mu, logvar)
        x_z = torch.cat([x, z], dim=2)
        output = self.decode(x_z)
        return output, mu, logvar, x, z
