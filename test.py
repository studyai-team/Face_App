from networks import select_frames
a = select_frames("test_video.mp4", 8)
print(len(a), a[0].shape)

hr_shape = opt.hr_shape

image = cv2.imread("after.jpg")
# print(image.shape)
cv2.imwrite("aa.jpg", image)
RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
flame_list = [RGB]
l = generate_landmarks(flame_list)
# print(l[0])
cv2.imwrite("1.jpg", l[0][0])
# exit()