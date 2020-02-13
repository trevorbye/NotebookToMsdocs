from nbconvert.preprocessors import Preprocessor
import re

class CustomPreprocessor(Preprocessor):
    def preprocess(self, nb, resources):
        # remove ![]() image tags
        for cell in nb.cells:
            if cell.get('cell_type') == 'markdown':
                edited_source = re.sub('!\[[^\]]*\]\((.*?)\s*("(?:.*[^"])")?\s*\)', '', cell.get('source'))
                cell['source'] = edited_source

        # remove first cell that contains copyright
        nb.cells.pop(0)
        return nb, resources
