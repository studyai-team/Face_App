# FaR-GAN for One-Shot Face Reenactment

このコードはFaR-GANの非公式実装です。ある人が写った写真と、その人の表情を変えたいようなターゲット画像を入力すると、元の画像の人が写ったまま表情を変えることができます。これを利用して証明写真用に撮影した写真の表情が悪かった時、理想的な表情に変換することが出来ます。

# Training
1. git clone
2. pip install -r requirements.txt
3. Download Voxceleb1 Dataset   
```wget http://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/data/zippedFaces.tar.gz```    
This dataset is used in Seeing Voices and Hearing Faces: Cross-modal
biometric matching(http://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/)
4. Download VGGFace pretrained model    
```wget https://drive.google.com/open?id=1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU```  
This model was introduced in this cite(https://github.com/cydonia999/VGGFace2-pytorch)
5. Train the model
```python train.py```

# Inference
1. git clone
2. pip install -r requirements.txt
3. Download model
Generator(https://drive.google.com/file/d/1eOzUqdvnjzybrX1hougGUgCYabNgT7mk/view?usp=sharing) and  
Embedder(https://drive.google.com/file/d/1MNid2VoMZpPKw8lYbqqMoI2F_2aqjg1j/view?usp=sharing)  
in ```saved_models/```
4. Select source image path and model path using argparse command  
ex.) ```python inference.py --source_image_path examples/ひげ.png```
