import torch
import torch.nn as nn


class LSTMVAE(nn.Module):
    def __init__(self, args, num_layers=1, batch_norm=True):
        super(LSTMVAE, self).__init__()
        self.input_size = args.input_size
        self.seq_len = args.seq_len
        self.hidden_size = args.vae_hidden_size
        self.latent_size = args.vae_latent_size
        self.num_layers = num_layers
        self.batch_norm = batch_norm

        self.encoder_lstm = nn.LSTM(
            args.input_size, args.hidden_size, num_layers, batch_first=True)
        self.encoder_fc_mu = nn.Linear(args.hidden_size, args.latent_size)
        self.encoder_fc_logvar = nn.Linear(args.hidden_size, args.latent_size)

        self.decoder_lstm = nn.LSTM(
            args.hidden_size, args.hidden_size, num_layers, batch_first=True)
        self.decoder_fc = nn.Linear(args.latent_size, args.hidden_size)
        self.output_fc = nn.Linear(args.hidden_size, args.input_size)

        self.relu = nn.ReLU()
        self.batchnorm1d_encoder = nn.BatchNorm1d(args.hidden_size)
        self.batchnorm1d_decoder = nn.BatchNorm1d(args.hidden_size)

    def encode(self, x):
        x, _ = self.encoder_lstm(x)
        x = x[:, -1, :]  # get only last hidden state
        x = self.relu(x)
        # print("encoder lstm lyr out >>>> activation func :: ", x)
        if self.batch_norm:
            x = self.batchnorm1d_encoder(x)  # apply batch normalization
        mu = self.encoder_fc_mu(x)
        logvar = self.encoder_fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        z = self.decoder_fc(z)
        if self.batch_norm:
            z = self.batchnorm1d_decoder(z)  # apply batch normalization
        # repeat along sequence length
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)

        z, _ = self.decoder_lstm(z)
        z = self.relu(z)
        z = self.output_fc(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar, z
