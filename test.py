from PIL import Image

img = Image.open("target_landmarks/typeA.png")
img = img.crop((150, 20, 750, 620)).resize((256, 256))
img.save("a.png")

img = Image.open("target_landmarks/typeB.png")
img = img.crop((40, 10, 520, 490)).resize((256, 256))
img.save("b.png")

img = Image.open("examples/ひげ.jpg")
img = img.crop((30, 0, 350, 320)).resize((256, 256))
img.save("ひげ.png")

img = Image.open("examples/口ぽっかりな表情.JPG")
img = img.crop((30, 0, 350, 320)).resize((256, 256))
img.save("口ぽっかりな表情.png")

img = Image.open("examples/変顔.JPG")
img = img.crop((0, 0, 400, 400)).resize((256, 256))
img.save("変顔.png")

img = Image.open("examples/怖い表情.JPG")
img = img.crop((0, 0, 470, 470)).resize((256, 256))
img.save("怖い表情.png")

img = Image.open("examples/斜め.JPG")
img = img.crop((0, 0, 400, 400)).resize((256, 256))
img.save("斜め.png")

img = Image.open("examples/斜め (2).JPG")
img = img.crop((0, 0, 440, 440)).resize((256, 256))
img.save("斜め(2).png")

img = Image.open("examples/暗い表情.JPG")
img = img.crop((0, 0, 400, 400)).resize((256, 256))
img.save("暗い表情.png")

img = Image.open("examples/歯がみえる表情.JPG")
img = img.crop((30, 0, 370, 340)).resize((256, 256))
img.save("歯が見える表情.png")

img = Image.open("examples/目が不自然に開いた表情.JPG")
img = img.crop((30, 0, 400, 370)).resize((256, 256))
img.save("目が不自然に開いた表情.png")

img = Image.open("examples/目が閉じている.JPG")
img = img.crop((30, 30, 420, 420)).resize((256, 256))
img.save("目が閉じている.png")

img = Image.open("examples/自信なさげな表情.JPG")
img = img.crop((00, 0, 410, 410)).resize((256, 256))
img.save("自信なさげな表情.png")

img = Image.open("examples/頬が膨らんだ表情.JPG")
img = img.crop((30, 0, 370, 340)).resize((256, 256))
img.save("頬が膨らんだ表情.png")
