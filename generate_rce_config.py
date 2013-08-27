import os
import string

filein = open('robot.cfg.template')
fileout = open('robot.cfg', 'w')

template = ''

for line in filein:
    li=line.strip()
    if not li.startswith("#"):
        template += line

src = string.Template(template)
result = src.substitute({ 'hostname':os.environ['HOSTNAME']})
fileout.write(result)

