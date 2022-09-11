from unet_prax.src.datas import *
from unet_prax.src.models import *
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    writer = SummaryWriter()

    net = UNet(channel_list=(1,12,24,48,96,192,))

    if torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    elif torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')
    else:
        DEVICE = torch.device('cpu')

    net.to(DEVICE)

    loss_fun = nn.BCEWithLogitsLoss().to(DEVICE)
    
    optim = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    optim.zero_grad()

    train_data = MnistData(True)
    test_data = MnistData(False)

    train_dataloader = DataLoader(train_data, batch_size=25, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data))

    
    def train_model(n_epochs):
        min_loss = np.inf
        step = 0
        for epoch in range(n_epochs):
            print(f'Epoch # {epoch}')
            for samples, labels in tqdm(train_dataloader):
                samples, labels = samples.to(DEVICE), labels.to(DEVICE)
                optim.zero_grad()
                out = net(samples)
                loss = loss_fun(out, transforms.CenterCrop(out.shape[2:3])(samples))
                step += 1
                writer.add_scalar('Loss/train', loss.item(), step)
                loss.backward()
                optim.step()
            if loss.item() < min_loss:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': loss,
                }, 'model.pth')
                min_loss = loss.item()
        
            
    train_model(20)
    writer.flush()