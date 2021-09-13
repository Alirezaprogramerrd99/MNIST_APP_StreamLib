from datetime import time
import streamlit as st
import torch
import numpy
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# ---------------------------- start of web app ------------------------------
st.title('MNIST data set Demo')
st.markdown('## Model hyperparameters')


n_epochs = st.text_input('enter n_epochs: ')
n_epochs = int(n_epochs) if n_epochs != "" else n_epochs

batch_size_train = 64
batch_size_test = 1000

learning_rate = st.text_input('enter learning rate: ')
learning_rate = float(learning_rate) if learning_rate != "" else learning_rate

momentum = 0.5
log_interval = 60000

if n_epochs != '' and learning_rate != '':
    st.text('n_epochs: {}\nbatch_size_train: {}\nbatch_size_test: {}\nlearning_rate: {}'.format(n_epochs,
                                                                                                batch_size_train, batch_size_test, learning_rate))

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# --------------- getting data set ------------
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

st.markdown('## Some visualizations')

if 'count_batch' not in st.session_state:
    st.session_state.count_batch = 0

# callback function


def increment_counter():
    st.session_state.count_batch += 1


def decrement_counter():
    st.session_state.count_batch -= 1


col1, col2 = st.columns(2)

with col1:
    st.button('Increment batch number', on_click=increment_counter)

if st.session_state.count_batch > 0:    # dont get into negetive values.
    with col2:
        st.button('Decrement batch number', on_click=decrement_counter)
st.write('batch number = ', st.session_state.count_batch)


@st.cache
def itr_bathces(times, loader):
    examples = enumerate(loader)
    next_batch = (0, (None, None))
    for _ in range(times + 1):
        next_batch = next(examples)
    return next_batch


def visualizeBatch(example_data):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    return fig


# examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = itr_bathces(
    st.session_state.count_batch, train_loader)

st.sidebar.markdown('## some visualization options')
show_batchId = st.sidebar.checkbox('show batch_id')
show_exampleData = st.sidebar.checkbox('show exampleData')
show_exampleTargets = st.sidebar.checkbox('show exampleLabels')
show_dataShape = st.sidebar.checkbox('show batch {} shape'.format(batch_idx))
show_DataNumbers = st.sidebar.checkbox(
    'show batch {} some data'.format(batch_idx))

if show_batchId:
    st.write('batch id: ', batch_idx)

if show_exampleData:
    st.write('batch {}: '.format(batch_idx),  example_data)

if show_exampleTargets:
    st.write('batch {}: '.format(batch_idx), example_targets)

if show_dataShape:
    st.write('batch {} data shape: '.format(batch_idx), example_data.shape)
    st.write('batch {} labels shape: '.format(
        batch_idx), example_targets.shape)

if show_DataNumbers:
    st.write(visualizeBatch(example_data))


# --------------------------------- creating model --------------------------------
st.sidebar.markdown('## training plot')

if learning_rate != '' and n_epochs != '':

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                st.markdown('`Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}`'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
                # torch.save(network.state_dict(), '/results/model.pth')
                # torch.save(optimizer.state_dict(), '/results/optimizer.pth')

    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target,
                                        size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        st.markdown('`\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n`'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    if st.button('train model'):

        latest_iteration = st.empty()
        bar = st.progress(0)

        test()
        for epoch in range(1, n_epochs + 1):
            latest_iteration.text(f'epoch {epoch}')
            train(epoch)
            test()
            bar.progress(epoch + (100 // n_epochs) + 1)
            time.sleep(0.1)

        # if 'losses' not in st.session_state:
        #     st.session_state.losses = (train_losses, test_losses)
        st.markdown('### train loss plot')
        st.line_chart(train_losses)
        st.markdown('### test loss plot')
        st.line_chart(test_losses)

       # show_learningCurve = st.sidebar.checkbox('show learningCurve')

        # if show_learningCurve:

        # st.line_chart(st.session_state.losses[0])
        # st.line_chart(st.session_state.losses[1])
        # x_train = list(range(len(train_losses)))
        # x_test = list(range(len(test_losses)))

        # fig = plt.figure()
        # plt.plot(train_losses, color='blue')
        # plt.plot(test_losses, color='red')
        # plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        # plt.xlabel('number of training examples seen')
        # plt.ylabel('negative log likelihood loss')
        # st.write(fig)

        with torch.no_grad():
            output = network(example_data)
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title("Prediction: {}".format(
                output.data.max(1, keepdim=True)[1][i].item()))
            plt.xticks([])
            plt.yticks([])

        st.markdown('## Model prediction:')
        st.write(fig)
