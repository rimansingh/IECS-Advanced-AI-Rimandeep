import gin
from models.cnn_model import CNNModel, DeeperCNN
from train import train

# Load Gin configs
gin.parse_config_file('configs/model_config.gin')
gin.parse_config_file('configs/train_config.gin')

# Fetch the configured model
# ModelClass = gin.get_configurable('CNNModel')
ModelClass = gin.get_configurable('DeeperCNN')

model = ModelClass()

print(f"Running model: {model.__class__.__name__}")

train(model)
