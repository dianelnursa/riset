import importlib.util
import os
import sys


sys.path.insert(0, os.path.dirname(__file__))

wsgi = importlib.util.load_source('wsgi', 'apps.py')
application = wsgi.app
