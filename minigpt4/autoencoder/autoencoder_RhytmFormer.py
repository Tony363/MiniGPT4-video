import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class AE(torch.nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(160, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32)
        )

        self.middle_model = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 5*4096),
            torch.nn.Sigmoid()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(5*4096, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 160),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        middle_output = self.middle_model(encoded)
        decoded = self.decoder(middle_output)
        decoded = decoded.view(x.size(0), 160)
        return middle_output, decoded



def collect_tensors(tensor_dict):
    collected_tensors = []
    keys = []
    for key in tensor_dict:
        tensor = tensor_dict[key]
        collected_tensors.append(tensor)
        keys.append(key)
    return collected_tensors, keys


def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    tensor_dict = torch.load('/home/tup30353/autoencoder/RhytmFormer_tensors.pth')

    
    tensors, keys = collect_tensors(tensor_dict)
    autoencoder_input = torch.stack(tensors, dim=0).float().to(device)
    
    
    
    
    
    
    autoencoder = AE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    
    num_epochs = 200
    batch_size = 32
    train_loss = []

    for epoch in range(num_epochs):
        autoencoder.train()
        epoch_loss = 0
        for i in range(0, len(autoencoder_input), batch_size):
            batch = autoencoder_input[i:i+batch_size]
            
            optimizer.zero_grad()
            _, output = autoencoder(batch)
            
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_loss.append(epoch_loss / len(autoencoder_input))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(autoencoder_input):.4f}")
    
    plt.plot(train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('/home/tup30353/autoencoder/training_loss4.png')
    plt.show()
    torch.save(autoencoder.state_dict(), "model_weights_RhytmFormer.pth")

    autoencoder = AE()
    autoencoder.load_state_dict(torch.load("/home/tup30353/autoencoder/model_weights_RhytmFormer.pth"))

    autoencoder.eval()
    latent, output = autoencoder(autoencoder_input)
    print(f"Latent shape: {latent[0].shape}, output shape: {output[0].shape}")
    results = {}
    for i, key in enumerate(keys):
        results[key] = {
            'initial_tensor': tensors[i].cpu(),
            'latent_representation': latent[i].cpu(),
            'reconstructed_tensor': output[i].cpu()
        }

    torch.save(results, '/home/tup30353/autoencoder/rppg/results_RhytmFormer.pt')
    print("Results saved successfully.")

if __name__ == '__main__':
    main()
