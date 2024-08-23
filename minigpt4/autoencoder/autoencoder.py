import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class AE(torch.nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(296, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Tanh(),
            torch.nn.Linear(1024, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.Tanh(),
            torch.nn.Linear(2048, 5*4096)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(5*4096, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.Tanh(),
            torch.nn.Linear(2048, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Tanh(),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 296)
            # torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(x.size(0), 296)
        return encoded, decoded


    def encode(self,x:torch.tensor)->torch.tensor:
        encoded, decoded = self.forward(x)
        return encoded.reshape(
            encoded.shape[0],
            5,
            encoded.shape[1]//5
        )
    


# def collect_tensors(tensor_dict):
#     collected_tensors = []
#     keys = []
#     for key in tensor_dict:
#         tensor = tensor_dict[key]
#         for key2 in tensor:
#             print(key2)
#             collected_tensors.append(tensor[key2])
#             keys.append(f"{key}_{key2}")
#     return collected_tensors, keys

def collect_tensors(tensor_dict):
    collected_tensors = []
    keys = []
    for key in tensor_dict:
        tensor = tensor_dict[key]
        collected_tensors.append(tensor)
        keys.append(key)
    return collected_tensors, keys

def test()->None:
    tensor_dict = torch.load('RhytmFormer_tensors.pth')
    tensors, keys = collect_tensors(tensor_dict)
    input = tensors[0].float().unsqueeze(0).to("cpu")
    print(input.shape)
    # input = torch.stack(tensors, dim=0).float().to("cpu")
    # print(input.shape)
    autoencoder = AE()
    autoencoder.load_state_dict(torch.load("model_weights_RhytmFormer.pth"))
    autoencoder.eval()
    print(autoencoder.encode(input).shape)
    # latent, output = autoencoder(autoencoder_input)
    
    # print(f"Latent shape: {latent.shape}, output shape: {output.shape}")
    # print(torch.reshape(latent,(latent.shape[0],5,4096)).shape)
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tensor_dict = torch.load('rppg/signals_1.pt')
    tensor_dict2 = torch.load('rppg/signals_2.pt')
    
    collected_tensors1, keys1 = collect_tensors(tensor_dict)
    collected_tensors2, keys2 = collect_tensors(tensor_dict2)
    
    collected_tensors = collected_tensors1 + collected_tensors2
    keys = keys1 + keys2
    
    autoencoder_input = torch.stack(collected_tensors, dim=0).to(device)
    
    autoencoder = AE().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
    
    num_epochs = 200
    batch_size = 32
    train_loss = []
    # print(tensor_dict[list(tensor_dict.keys())[0]][0].T.shape)
    # middle_output,decoded = autoencoder(tensor_dict[list(tensor_dict.keys())[0]][0].T.unsqueeze(0))
    # print("ENCODED - ",decoded.shape,middle_output.shape)
    # return
    for epoch in range(num_epochs):
        autoencoder.train()
        epoch_loss = 0
        for i in range(0, len(autoencoder_input), batch_size):
            batch = autoencoder_input[i:i+batch_size]
            middle_output,decoded = autoencoder(batch)
            print("ENCODED - ",decoded.shape,middle_output.shape)
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
    plt.savefig('training_loss3.png')
    plt.show()
    torch.save(autoencoder.state_dict(), "model_weights.pth")

    

    autoencoder.eval()
    latent, output = autoencoder(autoencoder_input)
    print(f"Latent shape: {latent[0].shape}, output shape: {output[0].shape}")
    results = {}
    for i, key in enumerate(keys):
        results[key] = {
            'initial_tensor': collected_tensors[i].cpu(),
            'latent_representation': latent[i].cpu(),
            'reconstructed_tensor': output[i].cpu()
        }

    torch.save(results, 'rppg/results.pt')
    print("Results saved successfully.")

if __name__ == '__main__':
    # main()
    test()
