from PIL import ImageGrab


count = 0
while(True):
	input("Apretar Enter para sacar una imagen...")
	im=ImageGrab.grab(bbox=(400,100,1200,680))
	im.save(f"images/Img_{count}.png")
	count += 1