import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import urllib.request
import torch.optim as optim
from sklearn.metrics import average_precision_score, f1_score
import cv2
from collections import Counter



# Create the directory
os.makedirs("./cvpr_emotic", exist_ok=True)

# Downloading ResNet model, that we are going to use for feature extraction 
# url = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
# file_path = "./cvpr_emotic/resnet18_places365.pth.tar"
# urllib.request.urlretrieve(url, file_path)

arch = 'resnet18'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda integration

model = models.__dict__[arch](num_classes=365) 
model_path = "./cvpr_emotic/resnet18_places365.pth.tar"  
checkpoint = torch.load(model_path, map_location=device)
state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict, strict=False)  # collecting the weights of the ResNet model
model.to(device)
model.eval()  
torch.save(model.state_dict(), 'resnet18_places365_state_dict.pth') # saving the ResNet model.


# custom dataset class
class Emotic_PreDataset(Dataset):
    def __init__(self, x_context, x_body, y_cat, y_cont, transform, context_norm, body_norm):
        super(Emotic_PreDataset, self).__init__()
        self.x_context = x_context # context images
        self.x_body = x_body #body images
        self.y_cat = y_cat #categorical emotion labels
        self.y_cont = y_cont # cntinuous emotionlabels
        self.transform = transform # transformations
        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1]) #normalization
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1]) #normalization

    def __len__(self):
        return len(self.y_cat)
  
    def __getitem__(self, index):
        #context and body images and their labels are retrieved
        image_context = self.x_context[index]
        image_body = self.x_body[index]
        cat_label = self.y_cat[index]
        cont_label = self.y_cont[index]
        # transformations and normalization
        image_context = self.transform(image_context)
        image_context = self.context_norm(image_context)
        image_body = self.transform(image_body)
        image_body = self.body_norm(image_body)
        #convert categorical labels to tensors
        cat_label = torch.as_tensor(cat_label, dtype=torch.float32)
        #normalize continuous labels
        cont_label = torch.as_tensor(cont_label, dtype=torch.float32) / 10.0

        return image_context, image_body, cat_label, cont_label


# Define the Emotic model
class Emotic(nn.Module):
    def __init__(self, num_context_features, num_body_features):
        super(Emotic, self).__init__()
        self.context_dim = num_context_features
        self.body_dim = num_body_features
        self.fc1 = nn.Linear(num_context_features + num_body_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)

    def forward(self, context_input, body_input):
        context = context_input.view(-1, self.context_dim)
        body = body_input.view(-1, self.body_dim)
        combined_features = torch.cat((context, body), dim=1)
        x = self.fc1(combined_features)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        categorical_output = self.fc_cat(x)
        continuous_output = self.fc_cont(x)
        return categorical_output, continuous_output
    
#loss function
class DiscreteLoss(nn.Module):
    def __init__(self, weight_type='mean', device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.weight_type = weight_type
        self.weights = self._get_weights(weight_type) #already computed weights

    def _get_weights(self, weight_type):
        if weight_type == 'mean':
            return torch.full((26,), 1.0 / 26, device=self.device) #equal weights for all classes
        elif weight_type == 'static':
            return torch.tensor([    #predefined weights for each class
                0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
                0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
                0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520, 0.1537
            ], device=self.device)
        else:  
            return torch.ones((26,), device=self.device) #default weights

    def forward(self, predictions, targets):
        if self.weight_type == 'dynamic':
            weights = self._compute_dynamic_weights(targets) # calculate weights dynamically
        else:
            weights = self.weights # use precomputed weigthts
        loss = self._compute_weighted_loss(predictions, targets, weights) #loss calculation
        return loss

    def _compute_dynamic_weights(self, targets):
        class_totals = targets.sum(dim=0).float() #add occurances for each class
        dynamic_weights = torch.zeros_like(class_totals, device=self.device)
        non_zero_classes = class_totals > 0 #calculate weights for classes with non-zero occurances
        dynamic_weights[non_zero_classes] = 1.0 / torch.log(class_totals[non_zero_classes] + 1.2)
        dynamic_weights[~non_zero_classes] = 0.0001 #assign little weight to classes with zero occurences
        return dynamic_weights

    def _compute_weighted_loss(self, predictions, targets, weights):
        diff = predictions - targets #calculate difference between predictions and tagets
        squared_diff = diff ** 2
        weighted_diff = squared_diff * weights #apply weights to squared difference
        return weighted_diff.sum() #sum to get loss

class ContinuousLoss_SL1(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, predictions, targets):
        difference = torch.abs(predictions - targets) #calculate difference between predictions and tagets
        squared_loss = 0.5 * torch.pow(difference, 2)  
        linear_loss = difference - 0.5  #linear loss
        smooth_loss = torch.where(difference < self.threshold, squared_loss, linear_loss) #squared loss if difference < threshold, else linear loss
        return smooth_loss.sum() #summ to get loss
    
data_src = 'cvpr_emotic'
# Load preprocessed training data
train_context = np.load(os.path.join(data_src,'preprocessed_emotic','train_context_arr.npy'))
train_body = np.load(os.path.join(data_src,'preprocessed_emotic','train_body_arr.npy'))
train_cat = np.load(os.path.join(data_src,'preprocessed_emotic','train_cat_arr.npy'))
train_cont = np.load(os.path.join(data_src,'preprocessed_emotic','train_cont_arr.npy'))
# Load preprocessed validation data
val_context = np.load(os.path.join(data_src,'preprocessed_emotic','val_context_arr.npy'))
val_body = np.load(os.path.join(data_src,'preprocessed_emotic','val_body_arr.npy'))
val_cat = np.load(os.path.join(data_src,'preprocessed_emotic','val_cat_arr.npy'))
val_cont = np.load(os.path.join(data_src,'preprocessed_emotic','val_cont_arr.npy'))
# Load preprocessed test data
test_context = np.load(os.path.join(data_src,'preprocessed_emotic','test_context_arr.npy'))
test_body = np.load(os.path.join(data_src,'preprocessed_emotic','test_body_arr.npy'))
test_cat = np.load(os.path.join(data_src,'preprocessed_emotic','test_cat_arr.npy'))
test_cont = np.load(os.path.join(data_src,'preprocessed_emotic','test_cont_arr.npy'))

batch_size = 32 #batch size

context_mean = [0.4690646, 0.4407227, 0.40508908] #mean for context images
context_std = [0.2514227, 0.24312855, 0.24266963] #standard deviation for context images
body_mean = [0.43832874, 0.3964344, 0.3706214] #mean for body images
body_std = [0.24784276, 0.23621225, 0.2323653] #standard deviation for body images
context_norm = [context_mean, context_std] #normalization for context images
body_norm = [body_mean, body_std] #normalization for body images
#transformations for training data
train_transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.RandomHorizontalFlip(), 
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), 
    transforms.ToTensor()
])
#transformations for validation and test data
test_transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.ToTensor()
])
#creating datasets for training, validation and testing
train_dataset = Emotic_PreDataset(train_context, train_body, train_cat, train_cont, train_transform, context_norm, body_norm)
val_dataset = Emotic_PreDataset(val_context, val_body, val_cat, val_cont, test_transform, context_norm, body_norm)
test_dataset = Emotic_PreDataset(test_context, test_body, test_cat, test_cont, test_transform, context_norm, body_norm)
#creating data loaders for training, validation and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f'Train loader: {len(train_loader)}, Val loader: {len(val_loader)}, Test loader: {len(test_loader)}')

#initiate emotic model
emotic_model = Emotic(num_context_features=512, num_body_features=512) 

results_dir = "./cvpr_emotic_results"
os.makedirs(results_dir, exist_ok=True)

def train_emotic(epochs, model_path, opt, scheduler, models, disc_loss, cont_loss, cat_loss_param=0.5, cont_loss_param=0.5):
    if not os.path.exists(model_path):  # directory for saving models
        os.makedirs(model_path)
    
    # Initialize files for logging
    training_log_file = os.path.join(results_dir, "training_log.txt")
    
    min_loss = np.inf  # used to keep track of minimum validation loss
    train_loss, val_loss = [], []  # lists to store training and validation loss
    model_context, model_body, emotic_model = models  # models

    with open(training_log_file, "w") as log_file:  # Open log file for writing
        log_file.write("Epoch\tTraining Loss\tValidation Loss\n")  # Add headers to the file

        for e in range(epochs):
            running_train_loss = 0.0
            # Move models to device
            emotic_model.to(device)
            model_context.to(device)
            model_body.to(device)
            # Set models to training mode
            emotic_model.train()
            model_context.train()
            model_body.train()
            # Iterating over training dataset
            for images_context, images_body, labels_cat, labels_cont in iter(train_loader):
                images_context, images_body = images_context.to(device), images_body.to(device)
                labels_cat, labels_cont = labels_cat.to(device), labels_cont.to(device)
                
                opt.zero_grad()
                pred_context, pred_body = model_context(images_context), model_body(images_body)
                pred_cat, pred_cont = emotic_model(pred_context, pred_body)
                # Calculate loss
                cat_loss_batch = disc_loss(pred_cat, labels_cat)
                cont_loss_batch = cont_loss(pred_cont * 10, labels_cont * 10)
                loss = cat_loss_param * cat_loss_batch + cont_loss_param * cont_loss_batch
                running_train_loss += loss.item()
                
                loss.backward()
                opt.step()
            # Calculate average training loss for each epoch
            average_train_loss = running_train_loss / len(train_loader)
            train_loss.append(average_train_loss)
            # Set models to evaluation mode
            running_val_loss = 0.0
            emotic_model.eval()
            model_context.eval()
            model_body.eval()
            
            with torch.no_grad():
                for images_context, images_body, labels_cat, labels_cont in iter(val_loader):
                    images_context, images_body = images_context.to(device), images_body.to(device)
                    labels_cat, labels_cont = labels_cat.to(device), labels_cont.to(device)

                    pred_context, pred_body = model_context(images_context), model_body(images_body)
                    pred_cat, pred_cont = emotic_model(pred_context, pred_body)
                    # Calculate loss
                    cat_loss_batch = disc_loss(pred_cat, labels_cat)
                    cont_loss_batch = cont_loss(pred_cont * 10, labels_cont * 10)
                    loss = cat_loss_param * cat_loss_batch + cont_loss_param * cont_loss_batch
                    running_val_loss += loss.item()
                # Calculate average validation loss for each epoch
                average_val_loss = running_val_loss / len(val_loader)
                val_loss.append(average_val_loss)
            
            scheduler.step()  #adjust learning rate based on scheduler
            #save model when validation loss improves
            if average_val_loss < min_loss:
                min_loss = average_val_loss
                print(f"Saving model at epoch {e + 1}")
                torch.save(model_context.state_dict(), os.path.join(model_path, "model_context1.pth"))
                torch.save(model_body.state_dict(), os.path.join(model_path, "model_body1.pth"))
                torch.save(emotic_model.state_dict(), os.path.join(model_path, "model_emotic1.pth"))
            
            #print and log training and validation loss for each epoch
            log_message = f"{e + 1}\t{average_train_loss:.4f}\t{average_val_loss:.4f}\n"
            print(f"Epoch {e + 1}/{epochs} - Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}")
            log_file.write(log_message)  #write the log message to the file

    return train_loss, val_loss  #return training and validation loss


def test_emotic(models, test_loader, disc_loss, cont_loss, device, cat_loss_param=0.5, cont_loss_param=0.5):
    model_context, model_body, emotic_model = models
    test_loss = []
    #set models to evaluation mode
    emotic_model.eval()
    model_context.eval()
    model_body.eval()
    
    running_test_loss = 0.0
    cat_losses = 0.0
    cont_losses = 0.0

    test_log_file = os.path.join(results_dir, "test_results.txt")
    with torch.no_grad(): 
        for images_context, images_body, labels_cat, labels_cont in test_loader:
            images_context, images_body = images_context.to(device), images_body.to(device)
            labels_cat, labels_cont = labels_cat.to(device), labels_cont.to(device)
            #extracting features using context and body models
            pred_context = model_context(images_context)
            pred_body = model_body(images_body)
            #using emotic model to make predictions
            pred_cat, pred_cont = emotic_model(pred_context, pred_body)
            #calculating loss
            cat_loss_batch = disc_loss(pred_cat, labels_cat)
            cont_loss_batch = cont_loss(pred_cont * 10, labels_cont * 10)
            loss = cat_loss_param * cat_loss_batch + cont_loss_param * cont_loss_batch
            running_test_loss += loss.item()
            
            cat_losses += cat_loss_batch.item()
            cont_losses += cont_loss_batch.item()
    #calculating average loss for test dataset
    average_test_loss = running_test_loss / len(test_loader)
    test_loss.append(average_test_loss)
    average_cat_loss = cat_losses / len(test_loader)
    average_cont_loss = cont_losses / len(test_loader)
    
    print(f'Average Test Loss: {average_test_loss:.4f}')
    print(f'Average Categorical Loss: {average_cat_loss:.4f}')
    print(f'Average Continuous Loss: {average_cont_loss:.4f}')

    with open(test_log_file, "w") as f:
        f.write("Test Results:\n")
        f.write(f"Average Test Loss: {average_test_loss:.4f}\n")
        f.write(f"Average Categorical Loss: {average_cat_loss:.4f}\n")
        f.write(f"Average Continuous Loss: {average_cont_loss:.4f}\n")

    return test_loss #returning test loss

#path for saving and loading models
model_path_places = './'
#loading context model
model_context = models.__dict__['resnet18'](num_classes=365)
context_state_dict = torch.load(os.path.join(model_path_places, 'resnet18_places365_state_dict.pth'))
model_context.load_state_dict(context_state_dict)

# Loading body model architecture
model_body = models.resnet18(pretrained=True)

# Emotic model (initialize based on context and body features)
emotic_model = Emotic(list(model_context.children())[-1].in_features, list(model_body.children())[-1].in_features)

# Removing the last layer for both context and body
model_context = nn.Sequential(*(list(model_context.children())[:-1]))
model_body = nn.Sequential(*(list(model_body.children())[:-1]))
#freezing parameters to avoid updating weights
for param in emotic_model.parameters():
    param.requires_grad = True
for param in model_context.parameters():
    param.requires_grad = False
for param in model_body.parameters():
    param.requires_grad = False
#optimizer and scheduler
opt = optim.Adam((list(emotic_model.parameters()) + list(model_context.parameters()) + list(model_body.parameters())), lr=1e-3, weight_decay=5e-4)
scheduler = StepLR(opt, step_size=7, gamma=0.1)
#loss functions
disc_loss = DiscreteLoss('dynamic', device)
cont_loss_SL1 = ContinuousLoss_SL1()
#loading pre-trained models 
context_state_dict = torch.load('./cvpr_emotic/model_context1.pth', map_location=device)
model_context.load_state_dict(context_state_dict)

body_state_dict = torch.load('./cvpr_emotic/model_body1.pth', map_location=device)
model_body.load_state_dict(body_state_dict)

emotic_state_dict = torch.load('./cvpr_emotic/model_emotic1.pth', map_location=device)
emotic_model.load_state_dict(emotic_state_dict)
#set all models to evaluation mode
model_context.eval()
model_body.eval()
emotic_model.eval()


#loss graph between training, validation and test lossses
def save_loss_graph(train_loss, val_loss, test_loss, results_dir):
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Path to save the graph
    graph_path = os.path.join(results_dir, "loss_graph.png")

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot([0] + test_loss, label='Testing Loss', marker='o')
    plt.title('Training, Validation, and Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the graph as a .png file
    plt.savefig(graph_path)
    plt.close()

def compute_metrics_categorical(predictions, targets):
    #moving predictions to numpy
    predictions = torch.sigmoid(predictions).cpu().numpy()  
    targets = targets.cpu().numpy()
    map_score = average_precision_score(targets, predictions, average='macro') #calculate mean average precision
    predictions_binary = (predictions > 0.5).astype(int)
    f1 = f1_score(targets, predictions_binary, average='macro')#calculate f1 score

    return {'mAP': map_score, 'F1': f1}

def compute_metrics_continuous(predictions, targets):
    #moving predictions to numpy
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    mae = np.mean(np.abs(predictions - targets)) #calculate mean absolute error
    mse = np.mean((predictions - targets) ** 2) #calculate mean square error
    if predictions.shape[1] == targets.shape[1]: 
        pcc = np.mean([  #calculate PCC
            np.corrcoef(predictions[:, i], targets[:, i])[0, 1]
            for i in range(predictions.shape[1])
        ])
    else:  
        pcc = np.corrcoef(predictions.flatten(), targets.flatten())[0, 1]

    return {'MAE': mae, 'MSE': mse, 'PCC': pcc}

def evaluate_metrics(models, data_loader, device):
    
    model_context, model_body, emotic_model = models
    emotic_model.eval()
    model_context.eval()
    model_body.eval()

    all_preds_cat, all_preds_cont = [], []
    all_labels_cat, all_labels_cont = [], []

    with torch.no_grad():
        for images_context, images_body, labels_cat, labels_cont in data_loader:
            images_context, images_body = images_context.to(device), images_body.to(device)
            labels_cat, labels_cont = labels_cat.to(device), labels_cont.to(device)

            pred_context = model_context(images_context)
            pred_body = model_body(images_body)
            pred_cat, pred_cont = emotic_model(pred_context, pred_body)
            #append predictions and labels to respective lists
            all_preds_cat.append(pred_cat)
            all_preds_cont.append(pred_cont)
            all_labels_cat.append(labels_cat)
            all_labels_cont.append(labels_cont)
    #concatenate all predictions and labels across batches
    all_preds_cat = torch.cat(all_preds_cat)
    all_preds_cont = torch.cat(all_preds_cont)
    all_labels_cat = torch.cat(all_labels_cat)
    all_labels_cont = torch.cat(all_labels_cont)
    #compute metrics for categorical 
    cat_metrics = compute_metrics_categorical(all_preds_cat, all_labels_cat)
    #compute metrics for continuous 
    cont_metrics = compute_metrics_continuous(all_preds_cont, all_labels_cont)

    return cat_metrics, cont_metrics


def evaluate_and_save_metrics(models, data_loader, dataset_name, device, results_dir):
    # Set models to evaluation mode
    model_context, model_body, emotic_model = models
    emotic_model.eval()
    model_context.eval()
    model_body.eval()
    
    # Evaluate metrics
    print(f"Evaluating on {dataset_name.capitalize()} Set...")
    cat_metrics, cont_metrics = evaluate_metrics(
        models=[model_context, model_body, emotic_model],
        data_loader=data_loader,
        device=device
    )

    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_file = os.path.join(results_dir, f"{dataset_name}_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"{dataset_name.capitalize()} Metrics:\n")
        for key, value in {**cat_metrics, **cont_metrics}.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"{dataset_name.capitalize()} metrics saved to {metrics_file}")
    return cat_metrics, cont_metrics

# Directory for results
results_dir = "./cvpr_emotic_results"


#categorical labels
category_labels = [
    'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 
    'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 
    'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 
    'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning'
]
#transformations for input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4690646, 0.4407227, 0.40508908], std=[0.2514227, 0.24312855, 0.24266963])
])

def predict_image_emotions(image_path, model_context, model_body, emotic_model, device, threshold=0.61):
    image = Image.open(image_path).convert('RGB') #loading image and converting to RGB
    image_transformed = transform(image).unsqueeze(0).to(device) #transformations
    #set models to evaluation mode
    model_context.eval()
    model_body.eval()
    emotic_model.eval()

    with torch.no_grad():
        #extracting features
        context_features = model_context(image_transformed)
        body_features = model_body(image_transformed)
        #predicting categorical and continuous emotions using emoric model
        pred_cat, pred_cont = emotic_model(context_features, body_features)
        #sigmoid activation to predict categorical lebel to get probability
        pred_cat = torch.sigmoid(pred_cat)
        #if probabilty greaterthan threshold add them to predicted categories
        predicted_categories = [
            category_labels[i] for i, prob in enumerate(pred_cat.squeeze()) if prob >= threshold
        ]
        #convert continuous predictions to numpy
        pred_cont = pred_cont.squeeze().cpu().numpy() * 10  

    print(f"Predicted Categories: {predicted_categories}")
    print(f"Predicted Continuous Dimensions (Valence, Arousal, Dominance): {pred_cont}")
    
    return predicted_categories, pred_cont



#load yolo model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#save for future use
torch.save(yolo_model, 'yolov5s_full_model.pt')
#set model to evaluation mode
yolo_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4690646, 0.4407227, 0.40508908], std=[0.2514227, 0.24312855, 0.24266963])
])
#categorical labels
category_labels = [
    'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 
    'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 
    'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 
    'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning'
]


def process_video(video_path, yolo_model, emotic_models, device, frame_skip=30, threshold=0.3):
    cap = cv2.VideoCapture(video_path) #oprn video file
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #get total no.of frames
    
    aggregated_categories = Counter() #count occurences of emotion categories
    continuous_values = [] 
    
    print(f"Processing video: {video_path} with {frame_count} frames")
    frame_index = 0 #index to current frame
    model_context, model_body, emotic_model = emotic_models
    
    while cap.isOpened(): #loop through video
        ret, frame = cap.read() #read frame from video
        if not ret: 
            break

        if frame_index % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = yolo_model(frame_rgb) #perform object detection
            detections = results.pandas().xyxy[0] #extract detected objects

            for idx, row in detections.iterrows():
                label = row['name'] #label of detected object
                if label == 'person':  #process only when person is detected
                    x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    cropped_image = frame_rgb[y_min:y_max, x_min:x_max] #crop the detected person region
                    cropped_image_pil = Image.fromarray(cropped_image) #convert to PIL
                    image_transformed = transform(cropped_image_pil).unsqueeze(0).to(device) #transformation

                    context_features = model_context(image_transformed)
                    body_features = model_body(image_transformed)
                    pred_cat, pred_cont = emotic_model(context_features, body_features) #predict emotion

                    pred_cat = torch.sigmoid(pred_cat).squeeze()
                    detected_emotic_categories = [
                        category_labels[i] for i, prob in enumerate(pred_cat.cpu().detach().numpy()) if prob >= threshold
                    ]
                    aggregated_categories.update(detected_emotic_categories) #update category counts

                    pred_cont = pred_cont.squeeze().detach().cpu().numpy() * 10 
                    continuous_values.append(pred_cont) #store continuous predictions

        frame_index += 1 #increase frame index

    cap.release() #release video

    if continuous_values:
        continuous_avg = np.mean(continuous_values, axis=0)
    else:
        continuous_avg = [0, 0, 0]  

    return aggregated_categories, continuous_avg
       



if __name__ == "__main__":
    print("Training")
    #training
    train_loss, val_loss = train_emotic(epochs=25, model_path='./cvpr_emotic', opt=opt, scheduler=scheduler, models=[model_context, model_body, emotic_model], disc_loss=disc_loss, cont_loss=cont_loss_SL1)

    #testing
    test_loss = test_emotic([model_context, model_body, emotic_model], test_loader, disc_loss, cont_loss_SL1, device)

    # Loss functions
    disc_loss = DiscreteLoss('dynamic', device)
    cont_loss_SL1 = ContinuousLoss_SL1()

    save_loss_graph(train_loss, val_loss, test_loss, results_dir)

    # Evaluate on Training Set
    train_cat_metrics, train_cont_metrics = evaluate_and_save_metrics(
        models=[model_context, model_body, emotic_model],
        data_loader=train_loader,
        dataset_name="training",
        device=device,
        results_dir=results_dir
    )

    # Evaluate on Validation Set
    val_cat_metrics, val_cont_metrics = evaluate_and_save_metrics(
        models=[model_context, model_body, emotic_model],
        data_loader=val_loader,
        dataset_name="validation",
        device=device,
        results_dir=results_dir
    )

    # Evaluate on Test Set
    test_cat_metrics, test_cont_metrics = evaluate_and_save_metrics(
        models=[model_context, model_body, emotic_model],
        data_loader=test_loader,
        dataset_name="test",
        device=device,
        results_dir=results_dir
    )

    image_path = "cvpr_emotic/ade20k/images/sun_ardczdjeutfxllrc.jpg"
    predicted_categories, predicted_continuous = predict_image_emotions(
        image_path, model_context, model_body, emotic_model, device

    )
    print("Aggregated Categorical Emotions:", predicted_categories)
    print("Average Continuous Emotions (Valence, Arousal, Dominance):", predicted_continuous)

    video_path = "cvpr_emotic/singlePerson.mp4" 
    emotic_models = (model_context, model_body, emotic_model)
    aggregated_categories, continuous_avg = process_video(
        video_path, yolo_model, emotic_models, device, frame_skip=30, threshold=0.57
    )

    print("Aggregated Categorical Emotions:", dict(aggregated_categories))
    print("Average Continuous Emotions (Valence, Arousal, Dominance):", continuous_avg)







