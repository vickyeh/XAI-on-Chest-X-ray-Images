import os
import sys
import time
from easydict import EasyDict as edict
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import torchcam
from model.classifier import Classifier 
import csv
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from PIL import Image
import sklearn.metrics as metrics
import random
import torchvision.models as models
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve, auc

use_gpu = torch.cuda.is_available()
nnIsTrained = False                 #pre-trained using ImageNet
nnClassCount = 14                   #dimension of the output

# Training settings: batch size, maximum number of epochs
trBatchSize = 64
trMaxEpoch = 50

# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = (320, 320)
imgtransCrop = 224

##model structure
class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DenseNet121(14).to(device)


class CheXpertDataSet(Dataset):
    def __init__(self, image_list_file, transform=None, policy="ones"):
        
        image_names = []
        labels = []

        with open(image_list_file, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None)
            k=0
            for line in csvReader:
                k+=1
                image_name= line[0]
                label = line[5:]
                
                for i in range(14):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                        
                image_names.append('/gscratch/cardss/shuyiyeh/CheXpert/' + image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label),image_name

    def __len__(self):
        return len(self.image_names)


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
transformList.append(transforms.Resize((imgtransCrop,imgtransCrop)))
#transformList.append(transforms.RandomResizedCrop(imgtransCrop))
#transformList.append(transforms.CenterCrop(imgtransCrop))
#transformList.append(transforms.RandomHorizontalFlip())
transformList.append(transforms.ToTensor())
#transformList.append(normalize)      
transformSequence=transforms.Compose(transformList)

#load dataset
dataset = CheXpertDataSet("/gscratch/cardss/shuyiyeh/CheXpert/Chexpert/bin/new_test_labels.csv" ,transformSequence, policy="zeroes")
datasetTrain, datasetValid=random_split(dataset, [350, len(dataset) - 350])
datasetTest = CheXpertDataSet("/gscratch/cardss/shuyiyeh/CheXpert/Chexpert/bin/new_val_labels.csv", transformSequence)            

#data loader
dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
dataLoaderVal = DataLoader(dataset=datasetValid, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)
dataLoaderTest = DataLoader(dataset=datasetTest,num_workers=24, pin_memory=True)

class CheXpertTrainer():

    def train (model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, launchTimestamp, checkpoint):
        
        #SETTINGS: OPTIMIZER & SCHEDULER
        #optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
                
        loss = torch.nn.BCELoss(size_average = True)
        
        #load checkpoint
        #checkpoint = torch.load('/gscratch/cardss/shuyiyeh/CheXpert/Chexpert/model_zeroes_1epoch_densenet.pth.tar')
        #model.load_state_dict(checkpoint['state_dict'], strict=False)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        
        
        #training
        lossMIN = 100000
        
        for epochID in range(0, trMaxEpoch):
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
            
            batchs, losst, losse = CheXpertTrainer.epochTrain(model, dataLoaderTrain, optimizer, trMaxEpoch, nnClassCount, loss)
            lossVal = CheXpertTrainer.epochVal(model, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, loss)


            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-epoch'+str(epochID)+'-' + launchTimestamp + '.pth.tar')
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
        
        return batchs, losst, losse        
    
    def epochTrain(model, dataLoader, optimizer, epochMax, classCount, loss):
        
        batch = []
        losstrain = []
        losseval = []
        
        model.train()

        for batchID, (varInput, target, _) in enumerate(dataLoaderTrain):
            varInput=varInput.to(device)
            varTarget=target.to(device)

            varOutput = model(varInput).to(device)
            lossvalue = loss(varOutput, varTarget)
                       
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            
            l = lossvalue.item()
            losstrain.append(l)
            
            if batchID%35==0:
                print(batchID//35, "% batches computed")
                #Fill three arrays to see the evolution of the loss


                batch.append(batchID)
                
                le = CheXpertTrainer.epochVal(model, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, loss).item()
                losseval.append(le)
                
                print(batchID)
                print(l)
                print(le)
                
        return batch, losstrain, losseval
    
    #-------------------------------------------------------------------------------- 
    
    def epochVal(model, dataLoader, optimizer, epochMax, classCount, loss):
        
        model.eval()
        
        lossVal = 0
        lossValNorm = 0

        with torch.no_grad():
            for i, (varInput, target,_) in enumerate(dataLoaderVal):
                varInput=varInput.to(device)
                target = target.to(device)
                varOutput = model(varInput)
                
                losstensor = loss(varOutput, target)
                lossVal += losstensor
                lossValNorm += 1
                
        outLoss = lossVal / lossValNorm
        return outLoss
      
     
    #---- Computes AUROC
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes
    
    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                pass
        class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
        all_fpr = np.linspace(0, 1, 100)  
        mean_tpr = np.zeros_like(all_fpr) 

        plt.figure(figsize=(10, 8))

        
        outAccuracy = []
        outBestThreshold = []
        for i in range(classCount):
            try:
                # Compute AUROC 
                auroc = roc_auc_score(datanpGT[:, i], datanpPRED[:, i])
                
                # Automatically determine the best threshold for each class based on F1 score
                best_f1 = 0
                best_threshold = 0
                thresholds = np.linspace(0, 1, 101)  # Test thresholds from 0 to 1
                
                for threshold in thresholds:
                    # Apply the threshold to get binary predictions
                    pred_binary = (datanpPRED[:, i] >= threshold).astype(int)
                    true_binary = datanpGT[:, i].astype(int)
                    
                    # Compute F1 score for the current threshold
                    f1 = f1_score(true_binary, pred_binary)
                    
                    # Update the best threshold if we find a better F1 score
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                
                # Use the best threshold to calculate accuracy
                pred_binary = (datanpPRED[:, i] >= best_threshold).astype(int)
                true_binary = datanpGT[:, i].astype(int)
                accuracy = accuracy_score(true_binary, pred_binary)
                conf_matrix = confusion_matrix(true_binary, pred_binary)
                print(f"Confusion Matrix for {class_names[i]}:")
                print(conf_matrix)
                
                # Store the results
                outAccuracy.append(accuracy)
                outBestThreshold.append(best_threshold)
                
            except ValueError:
                # Skip if there's an error in calculation (e.g., due to lack of variation in data)
                outAccuracy.append(None)
                outBestThreshold.append(None)
            #chosen class
            if i == 2 or i == 5 or i == 6 or i ==  7 or i == 8 or i ==  9:
                try:
                    fpr, tpr, _ = roc_curve(datanpGT[:, i], datanpPRED[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.2f})')
                
                    mean_tpr += np.interp(all_fpr, fpr, tpr)
                 
                except ValueError:
                    pass

        #average tpr
        mean_tpr /= 6 

        # average ROC
        plt.plot(all_fpr, mean_tpr, color='black', linestyle='-', linewidth=2,label=f'Mean ROC (AUC = {auc(all_fpr, mean_tpr):.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve for Common Classes (Mean ROC)')
        plt.legend(loc="lower right")
        plt.grid(True)

        
        plt.savefig("roc_curve_all_classes_trainNIH_testCheXpert.png")
        plt.close()  
            
        for i in range(classCount):
            print(f"Class: {class_names[i]}")
            print(f"  Accuracy: {outAccuracy[i]}")
            print(f"  Best Threshold: {outBestThreshold[i]}")
            
        return outAUROC
        
        
    #-------------------------------------------------------------------------------- 
    
    
    def test(model, dataLoaderTest, nnClassCount, class_names):   
        
        cudnn.benchmark = True
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        outGT = torch.FloatTensor().to(device)
        outPRED = torch.FloatTensor().to(device)
        model.eval()
        # no Grad-CAM
        with torch.no_grad():
            for i, (input, target,_) in enumerate(dataLoaderTest):
                input=input.to(device)
                target = target.to(device)
                outGT = torch.cat((outGT, target), 0)

                bs, c, h, w = input.size()
                varInput = input.view(-1, c, h, w)
                out = model(varInput).to(device)
                outPRED = torch.cat((outPRED, out), 0)
        '''
        #gradcam: open grad
        for i, (input, target) in enumerate(dataLoaderTest):
            input=input.to(device)
            target = target.to(device)
            outGT = torch.cat((outGT, target), 0)

            bs, c, h, w = input.size()
            varInput = input.view(-1, c, h, w)
            out = model(varInput).to(device)
            outPRED = torch.cat((outPRED, out), 0)
        '''
        aurocIndividual = CheXpertTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
           print (class_names[i], ' ', aurocIndividual[i])
        
        return outGT, outPRED
class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
nnClassCount=14
timestampTime = time.strftime("%H%M%S")
timestampDate = time.strftime("%d%m%Y")
timestampLaunch = timestampDate + '-' + timestampTime


## Training ------------------------------------------
#batch, losst, losse = CheXpertTrainer.train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, timestampLaunch, checkpoint = None)
#print("Model trained")

## Testing ------------------------------------------

model = DenseNet121(14).to(device)
checkpoint = torch.load('/gscratch/cardss/shuyiyeh/CheXpert/Chexpert/bin/m-epoch30-25112024-234923.pth.tar')
#checkpoint = torch.load('/gscratch/cardss/shuyiyeh/CheXpert/Chexpert/bin/m-epoch10-04122024-033918.pth.tar')
model.load_state_dict(checkpoint['state_dict'], strict=False)
optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
optimizer.load_state_dict(checkpoint['optimizer'])
outGT1, outPRED1 = CheXpertTrainer.test(model, dataLoaderTest, nnClassCount, class_names)



# GradCam --------------------------------------------

img = DataLoader(dataset=datasetTest, batch_size=1, num_workers=24, pin_memory=True)
model = DenseNet121(14).to(device)


checkpoint = torch.load('/gscratch/cardss/shuyiyeh/CheXpert/Chexpert/bin/m-epoch30-25112024-234923.pth.tar')
#checkpoint = torch.load('/gscratch/cardss/shuyiyeh/CheXpert/Chexpert/bin/m-epoch10-04122024-033918.pth.tar')
model.load_state_dict(checkpoint['state_dict'], strict=False)
optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
optimizer.load_state_dict(checkpoint['optimizer'])

#model=model.eval().to(device) 

cam_extractor=GradCAM(model,target_layer=model.densenet121.features.denseblock4.denselayer16.conv2)
varInputs=[]
outs=[]
targets=[]
image_names=[]
for i, (input, target, image_name) in enumerate(img):
    input=input.to(device)
    bs, c, h, w = input.size()
    varInput = input.view(-1, c, h, w)
    out = model(varInput).to(device)
    outPRED = torch.FloatTensor().to(device)
    outPRED = torch.cat((outPRED, out), 0)
    out=outPRED
    outs.append(out)
    varInput.requires_grad=True
    varInputs.append(varInput)
    targets.append(target)
    image_names.append(image_name)


def overlay_mask2(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.5) -> Image.Image:
    """Overlay a colormapped mask on a background image"""
    
    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")
    
    # Ensure alpha is between 0 and 1
    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")
    
    # Convert mask to numpy array and scale to [0, 1]
    mask_array = np.array(mask) 
    min_val = np.min(mask_array)
    max_val = np.max(mask_array)

    mask_array = (mask_array - min_val) / (max_val - min_val)

    # Get colormap
    cmap = cm.get_cmap(colormap)

    # Apply colormap to the mask
    colored_mask = cmap(mask_array)  # Apply colormap

    # Convert the colormap to RGB (remove alpha channel)
    colored_mask = (255 * colored_mask[:, :, :3]).astype(np.uint8)  # Remove alpha and scale to [0, 255] 
    #print(colored_mask.shape)
    
    # Convert the image to numpy array
    img_array = np.array(img)
    
    # Overlay the image with the mask using alpha blending
    overlay_img = np.clip(alpha * img_array + (1 - alpha) * colored_mask, 0, 255)
    
    # Convert back to PIL image
    overlay_pil = Image.fromarray(overlay_img.astype(np.uint8))
    
    return overlay_pil, colored_mask

# Plot the image of Grad-CAM
for i in range(len(varInputs)):
    
    # Chosen class for Grad-CAM
    target_class=2
    grads = cam_extractor(target_class,scores=outs[i])
    heatmap = grads[0]

    if isinstance(heatmap, list):
        heatmap = torch.tensor(heatmap)
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(varInputs[i][0]) 

    heatmap_pil = to_pil(heatmap)  
    heatmap_pil = heatmap_pil.resize((224, 224), Image.NEAREST)
    overlay_img,colored_mask = overlay_mask2(img_pil, heatmap_pil, colormap="viridis")
    heatmap_pil = to_pil(colored_mask)  
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_pil)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(heatmap_pil)
    axes[1].set_title("Heatmap")
    axes[1].axis('off')

    axes[2].imshow(overlay_img)
    axes[2].set_title("Overlay Image")
    axes[2].axis('off')
    predicted = outs[i][0][1].item()
    actual = targets[i][0][1].item()

    predicted_text = f"Predicted: {predicted:.4f}"
    actual_text = f"Actual: {actual:.0f}"

    fig.text(0.85, 0.6, predicted_text, ha='left', va='center', fontsize=12, color='black', weight='bold')
    fig.text(0.85, 0.5, actual_text, ha='left', va='center', fontsize=12, color='black', weight='bold')

    plt.subplots_adjust(right=0.85)
    plt.show()
    
    path = image_names[i][0]
    result = path.split('patient', 1)[1] 
    path = 'patient' + result[:5]

    fig.savefig(f'/gscratch/cardss/shuyiyeh/CheXpert/Chexpert/bin/grad_cam_image_CheXpertmodel_on_Cardiomegaly/GradCAM_image_{i}_{path}.png', bbox_inches='tight')
