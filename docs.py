import os
import argparse

#Optional arguments for CLI
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--doctitle', help='Docs Title', required=False)
parser.add_argument('-g', '--githuburl', help='GitHub url', required=False)
args = parser.parse_args()
title = args.doctitle if args.doctitle else 'Documentation'
url = args.githuburl if args.githuburl else 'hola'

#[comment, class, comment, func] => [class, comment, func, comment]
def order(ls):
	first = []
	second = []
	for i, elem in enumerate(ls):
		if i % 2 == 0:
			second.append(elem)
		else:
			first.append(elem)
	
	newls = []

	for i, elem in enumerate(first):
		newls.append(elem)
		if 'object' in elem:
			newls.append(second[i].replace('description', 'object_description'))
		newls.append(second[i])

	return newls

#All files in folder exept self.
path = os.getcwd()
files = [os.path.join(path, "main.py")]
"""
for r, d, f in os.walk(path):
	for file in f:
		if '.py' in file and __file__ not in file and '__init__' not in file:
			files.append(os.path.join(r, file))
"""
#Output Lines
outlines = []

#HTML headers and styling
template = open('template.html', 'r')
for line in template:
	outlines.append(line.replace('TITLE', title).replace('https://github.com', url))
template.close()

outlines.append('<div class="list">')

for f in files:

	print(f.split('/')[-1])
	print('')

	file = open(f ,"r+")
	
	outlines.append('<div class="class">')

	output = open('docs/index.html' ,"w+")

	single = []
	func = 0
	for i, line in enumerate(file):
		if "coding: utf-8" not in line:
			if not func: 
				if '#' in line and "{" not in line and "# Dependency: " not in line:
					single.append('<div class="description">'+line.replace('	', '').replace('\n', "</div>"))
					func = True
			else:
				if 'def' in line:
					single.append(line.replace('	', '').replace('def ', '<div class="function">').replace(':', '').replace('self, ', '').replace('self', '').replace('\n', "</div>").replace('(', '<p class="white">(</p><p class="param">').replace(')', '</p><p class="white">)</p>').replace(', ', '<p class="white">, </p><p class="param">'))
					func = False
				else:
					single.append(line.replace('class ', '<div class="object">').replace(':', '').replace('\n', "</div>"))				
					func = False

	singe = order(single)
	for line in singe:
		outlines.append(line)

	outlines.append('</div>')
	
	file.close()

output = open('docs/index.html' ,"w+")
for line in outlines:
	output.write(line)

#HTML footer
output.write('</div>\n<div class="footer">\n<a href="'+url+'">\n<svg class="" style="fill: white;" height="50" viewBox="0 0 16 16" version="1.1" width="50" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>\n</a>\n</div></body>\n</html>')
output.close()

os.system('open docs/index.html')