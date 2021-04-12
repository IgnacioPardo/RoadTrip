import pyautogui as auto
from pyperclip import paste
from time import sleep
import os

default_region = (472, 193, 512, 512) # el formato es (x, y, ancho, alto)
coord_text_start = (561, 714) # depende de la resolucion, yo lo hice con 1600x900

image_index = 0

def take_screenshot(**kwargs):
	global image_index
	region = kwargs.get("region", default_region)
	auto.screenshot(os.path.join("inputs", f"image-{image_index}.jpg"), region=region)
	with open("georeferences.csv", 'a') as coord_file:
		coord_file.write(', '.join([str(image_index),
									read_coords((region[0], region[1])),
									read_coords((region[0] + region[2], region[1] + region[3]))]
									) + '\n')
	image_index += 1

def read_coords(point):
	auto.moveTo(*point)
	auto.click(button="left")
	sleep(1)
	auto.moveTo(*coord_text_start)
	auto.dragRel(150, 0, 1)
	auto.hotkey("ctrl", 'c')
	auto.moveTo(0, 300)
	auto.click()
	sleep(1)
	return paste()

def take_line_screenshot(amount, offset_vector, **kwargs):
	region = kwargs.get("region", default_region)
	drag_point = (region[0] if offset_vector[0] > 0 else region[0] + region[2],
				  region[1] if offset_vector[1] < 0 else region[1] + region[3]) # esto es para que draggee desde la esquina que le da mas lugar para moverse
	for i in range(amount):
		take_screenshot(region=region)
		auto.moveTo(*drag_point)
		auto.dragRel(offset_vector[0], -offset_vector[1], (offset_vector[0] ** 2 + offset_vector[1] ** 2) ** 0.5 / 100) # El pitágoras es para que el tiempo que tarde en moverlo sea proporcional al largo del vector del movimiento
		auto.moveTo(0, 300) # saca el mouse de la region para sacar la foto
		sleep(0.1)



def take_area_screenshot(width, height, **kwargs):
	region = kwargs.get("region", default_region)
	drag_point_1 = (region[0], region[1])
	drag_point_2 = (region[0] + region[2], region[1] + region[3])
	going_right = True
	current_width = 0
	for i in range(width * height):
		take_screenshot(region=region)
		current_width += 1
		if current_width == width:
			auto.moveTo(drag_point_2)
			auto.dragRel(0, -region[3], region[3] / 100)
			going_right = not going_right
			current_width = 0
		else:
			auto.moveTo(drag_point_2 if going_right else drag_point_1)
			auto.dragRel(-region[2] if going_right else region[2], 0, region[2] / 100)




if __name__ == '__main__':
	take_area_screenshot(3, 3)