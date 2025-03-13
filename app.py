from flask import Flask, request, render_template
import os
import torch
import torch.nn as nn
import cv2
from collections import Counter
import numpy as np
from torchvision import transforms, models
from PIL import Image
from emotic_models import Emotic
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Flask app setup
app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load yolo model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False)
yolo_model.to(device).eval()
print("YOLOv5 model loaded successfully.")

# load context and body features
model_context = models.resnet18(pretrained=False)
model_context = nn.Sequential(*list(model_context.children())[:-1])  

model_body = models.resnet18(pretrained=False)
model_body = nn.Sequential(*list(model_body.children())[:-1])

#features of context and body
context_output_features = 512  
body_output_features = 512

# emotic model
emotic_model = Emotic(
    num_context_features=context_output_features,
    num_body_features=body_output_features
)

# load pre-trained weights
model_context.load_state_dict(torch.load('./cvpr_emotic/model_context1.pth', map_location=device), strict=False)
model_body.load_state_dict(torch.load('./cvpr_emotic/model_body1.pth', map_location=device), strict=False)
emotic_model.load_state_dict(torch.load('./cvpr_emotic/model_emotic1.pth', map_location=device), strict=False)

# Set all models to evaluation mode
model_context.to(device).eval()
model_body.to(device).eval()
emotic_model.to(device).eval()

#models combined
emotic_models = (model_context, model_body, emotic_model)

# transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4690646, 0.4407227, 0.40508908], std=[0.2514227, 0.24312855, 0.24266963])
])

category_labels = [
    'Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval',
    'Disconnection', 'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem',
    'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness',
    'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning'
]


def process_video(video_path, yolo_model, emotic_models, device, frame_skip=30, threshold=0.3):
    cap = cv2.VideoCapture(video_path) #open video file
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


#rroutes to index.html template
@app.route('/', methods=['GET', 'POST'])  #get and post routes
def index():
    if request.method == 'POST':   
        if 'video' not in request.files or request.files['video'].filename == '':
            return render_template('index.html', results="Error: No video file provided.")
        video = request.files['video']
        video_filename = os.path.splitext(video.filename)[0] #video file name extraction
        video_path = os.path.join('results', video_filename)   
        os.makedirs('results', exist_ok=True)
        video.save(video_path)
        #processing video
        aggregated_categories, continuous_avg = process_video(    
            video_path, yolo_model, emotic_models, device, frame_skip=15, threshold=0.57
        )
        #results
        formatted_results = (                                      
            f"Aggregated Categorical Emotions: {dict(aggregated_categories)}\n"
            f"Average Continuous Emotions (Valence, Arousal, Dominance): "
            f"{[round(val, 4) for val in continuous_avg]}"
        )
        #saving results
        results_file_name = f"{video_filename}.txt"
        results_file_path = os.path.join('results', results_file_name)
        with open(results_file_path, 'w') as results_file:
            results_file.write(formatted_results)
        #rendering to template
        return render_template('index.html', results=formatted_results)
    return render_template('index.html', results=None)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    app.run(debug=True)

