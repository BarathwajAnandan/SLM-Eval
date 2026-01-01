import shutil
import os

src = '/Users/barathwajanandan/Documents/FV-eval/temporary_disco/data'
dst = '/Users/barathwajanandan/Documents/FV-eval/tests/disco'

if os.path.exists(src):
    shutil.move(src, dst)
    shutil.rmtree('/Users/barathwajanandan/Documents/FV-eval/temporary_disco')
    print("Success")
else:
    print("Source not found")
