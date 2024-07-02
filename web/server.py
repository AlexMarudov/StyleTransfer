from flask import Flask, request, jsonify, render_template
import io
from io import BytesIO
from PIL import Image
import numpy as np
import torch.nn as nn
import torch
import base64
# Импортируйте необходимые библиотеки для вашей модели переноса стиля

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Training on GPU!")
else:
    device = torch.device('cpu')
    print("Training on CPU :(")

app = Flask(__name__)
IMG_DIMENSIONS = (256, 256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "C:/PythonProjects/2year/PROJECT-4/model/models/matrix_2.pt"

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = nn.Conv2d(in_channels = self.in_channels,
                               out_channels = self.out_channels,
                               kernel_size = 3)
        self.batch_norm = nn.InstanceNorm2d(self.out_channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        # First convolution
        orig_x = x.clone()
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        
        # Second convolution
        x = self.conv(x)
        x = self.batch_norm(x)
        
        # Now add the original to the new one (and use center cropping)
        # Calulate the different between the size of each feature (in terms 
        # of height/width) to get the center of the original feature
        height_diff = orig_x.size()[2] - x.size()[2]
        width_diff = orig_x.size()[3] - x.size()[3]
        
        # Add the original to the new (complete the residual block)
        x = x + orig_x[:, :,
                                 height_diff//2:(orig_x.size()[2] - height_diff//2), 
                                 width_diff//2:(orig_x.size()[3] - width_diff//2)]
        
        return x
    
class ImageTransformationNetwork(nn.Module):
    def __init__(self):
        super(ImageTransformationNetwork, self).__init__()
        # Use reflection padding to keep the end shape
        self.ref_pad = nn.ReflectionPad2d(40)
        
        # Initial convolutions
        self.conv1 = nn.Conv2d(in_channels = 3,
                               out_channels = 32,
                               kernel_size = 9,
                               padding = 6,
                               padding_mode = 'reflect')
        
        self.conv2 = nn.Conv2d(in_channels = 32,
                               out_channels = 64,
                               kernel_size = 3,
                               stride = 2)
        
        self.conv3 = nn.Conv2d(in_channels = 64,
                               out_channels = 128,
                               kernel_size = 3,
                               stride = 2)
        
        # Residual Blocks
        self.resblock1 = ResidualBlock(in_channels = 128,
                                       out_channels = 128)
        
        self.resblock2 = ResidualBlock(in_channels = 128,
                                       out_channels = 128)
        
        self.resblock3 = ResidualBlock(in_channels = 128,
                                       out_channels = 128)
        
        self.resblock4 = ResidualBlock(in_channels = 128,
                                       out_channels = 128)
        
        self.resblock5 = ResidualBlock(in_channels = 128,
                                       out_channels = 128)
        
        # Transpose convoltutions
        self.trans_conv1 = nn.ConvTranspose2d(in_channels=128,
                                             out_channels=64,
                                             kernel_size=2,
                                             stride=2)
        
        self.trans_conv2 = nn.ConvTranspose2d(in_channels=64,
                                              out_channels=32,
                                              kernel_size=2,
                                              stride=2)
        
        # End with one last convolution
        self.conv4 = nn.Conv2d(in_channels = 32,
                               out_channels = 3,
                               kernel_size = 9,
                               padding = 4,
                               padding_mode = 'reflect')
        
    def forward(self, x):
        # Apply reflection padding
        x = self.ref_pad(x)
        
        # Apply the initial convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Apply the residual blocks
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)        
        
        #  Apply the transpose convolutions
        x = self.trans_conv1(x)
        x = self.trans_conv2(x)
        
        # Apply the final convolution
        x = self.conv4(x)
        
        return x

transformation_net = ImageTransformationNetwork()
transformation_net.load_state_dict(torch.load(model_path))
transformation_net.to(device).eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()  # Получение JSON-данных из запроса
        if 'image' not in data:
            return jsonify({'error': 'Изображение не найдено'}), 400

        # Декодирование base64 строки в изображение
        img_data = base64.b64decode(data['image'])
        img = Image.open(BytesIO(img_data))

        # Преобразуйте изображение в формат, необходимый вашей модели
        img = np.asarray(img.resize(IMG_DIMENSIONS)).transpose(2, 0, 1)[0:3]
        img = torch.from_numpy(img.reshape(1, 3, 256, 256)).float().to(device)

        # Вызовите модель для переноса стиля
        output_image = transformation_net(img)
        output_image = output_image.detach().cpu().numpy()
    
        # Clip the floats
        output_image = np.clip(output_image, 0, 255)
        
        # Преобразуйте результат обратно в изображение и закодируйте его в base64
        output_image = np.transpose(output_image, (0, 2, 3, 1))  # Переставляем оси
        #output_image = (output_image * 0.5 + 0.5) * 255  # Денормализация
        output_image = output_image.clip(0, 255).astype('uint8')
        output_image = Image.fromarray(output_image[0])

        buffered = io.BytesIO()
        output_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({'image': img_str})
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)