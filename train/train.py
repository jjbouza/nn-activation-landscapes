import torch


def train(model, 
          dataloader, 
          optimizer=(torch.optim.Adam, {"lr": 0.002})
          maximum_epochs=1000,
          device=None,
          cuttoff_accuracy=1.0):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.train()
    optimizer = optimizer[0](**optimizer[1]).to(device)

    def accuracy(output, target):
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        accuracy = pred.eq(target.view_as(pred)).sum().item()/output.shape[0]
        return accuracy
    
    epochs = tqdm.tqdm(range(1, maximum_epochs))
    for epoch in epochs:
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            if (acc := accuracy(output, target)) >= cuttoff_accuracy:
                return acc

    acc = accuracy(output, target)
    return acc


