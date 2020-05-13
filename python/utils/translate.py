from googletrans import Translator


def translate(text, src='en', tgt='ja'):
    translator = Translator()
    return translator.translate(text, dest=tgt)
   

if __name__ == '__main__':
    print(translate('Hello!'))
