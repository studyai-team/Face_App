from PIL import Image

img = Image.open("target_landmarks/typeA.jpg")
img = img.crop((150, 20, 750, 620)).resize((256, 256))
img.save("a.png")

img = Image.open("target_landmarks/typeB.png")
img = img.crop((40, 10, 520, 490)).resize((256, 256))
img.save("b.png")
