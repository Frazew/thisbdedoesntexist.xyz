from flask import Flask, render_template, redirect
import base64
from io import BytesIO
import json
import numpy as np
from gensim.models import FastText
import nltk
from bs4 import BeautifulSoup
import requests
import re
import os
import hashlib
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageColor
from glob import glob
import colorsys


app = Flask(__name__)
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')

# This is a FastText english word2vec model with reduced dimensionality (down to 100 dimensions from 300)
# See here: https://fasttext.cc/docs/en/crawl-vectors.html
ft_en = FastText.load('cc.en.100.word2vec')


color_value_threshold = 200
color_categories = {
	"red": "rouge",
	"blue": "bleue",
	"purple": "violette",
	"yellow": "jaune",
	"beige": "beige",
	"orange": "orange",
	"green": "verte"
}
colors = {c: [] for c in list(color_categories.keys())}
with open("colornames.bestof.json") as f: # This list of colors comes from here https://github.com/meodai/color-names, MIT license
    colornames = json.load(f)
    for color in colornames:
        colorname = color["name"]
        color = color["hex"]
        for c in color_categories:
            if c in colorname.lower():
                colors[c].append(color)

animals = {'Alligator': 'Alligator', 'Anteater': 'Fourmilier', 'Armadillo': 'Tatou', 'Axolotl': 'Axolotl', 'Badger': 'Blaireau', 'Bat': 'Chauve souris', 'Beaver': 'Castor', 'Buffalo': 'Buffle', 'Camel': 'chameau', 'Capybara': 'Capybara', 'Chameleon': 'Caméléon', 'Cheetah': 'guépard', 'Chinchilla': 'Chinchilla', 'Chipmunk': 'Tamia', 'Chupacabra': 'Chupacabra', 'Cormorant': 'Cormoran', 'Coyote': 'Coyote', 'Crow': 'corbeau', 'Dinosaur': 'Dinosaure', 'Dolphin': 'Dauphin', 'Duck': 'Canard', 'Elephant': 'Éléphant', 'Ferret': 'Furet', 'Fox': 'Renard', 'Frog': 'Grenouille', 'Giraffe': 'Girafe', 'Grizzly': 'Grizzly', 'Hedgehog': 'Hérisson', 'Hippo': 'Hippopotame', 'Hyena': 'Hyène', 'Iguana': 'Iguane', 'Jackal': 'Chacal', 'Kangaroo': 'Kangourou', 'Koala': 'Koala', 'Kraken': 'Kraken', 'Lemur': 'Lémurien', 'Leopard': 'Léopard', 'Liger': 'ligre', 'Llama': 'Lama', 'Manatee': 'Lamantin', 'Mink': 'Vison', 'Monkey': 'Singe', 'Moose': 'élan', 'Narwhal': 'Narval', 'Orangutan': 'Orang-outan', 'Otter': 'loutre', 'Panda': 'Panda', 'Penguin': 'manchot', 'Platypus': 'Ornithorynque', 'Python': 'Python', 'Quagga': 'Quagga', 'Rabbit': 'Lapin', 'Raccoon': 'Raton laveur', 'Rhino': 'Rhinocéros', 'Sheep': 'Mouton', 'Shrew': 'Musaraigne', 'Skunk': 'Moufette', 'Squirrel': 'Écureuil', 'Tiger': 'tigre', 'Turtle': 'Tortue', 'Walrus': 'Morse', 'Wolf': 'Loup', 'Wolverine': 'Carcajou', 'Wombat': 'Wombat'}
lists = ["whispers", "kraken", "spectre", "blinders", "eclipse"] # This is just a sample
with open("listes_bde.json") as f: # Load all from a file that's not in version control, it's a secret
	lists = json.load(f)
in_vocab = []
not_in_vocab = []
for l in lists:
    if ft_en.wv.has_index_for(l):
        in_vocab.append(l)
    else:
        not_in_vocab.append(l)


def is_invalid(word, enforce=False):
    enforced_check = False
    if enforce:
        enforced_check = enforced_check or "-" in word
        enforced_check = enforced_check or "." in word
        enforced_check = enforced_check or len(nltk.tokenize.SyllableTokenizer().tokenize(word)) > 4

    check = enforced_check or len(word.strip()) == 0
    check = check or len(nltk.tokenize.SyllableTokenizer().tokenize(word)) < 2
    check = check or nltk.pos_tag(nltk.word_tokenize(word))[0][1] not in ("NN", "NNS")
    return check


def get_word():
    word = ""
    while word == "" or is_invalid(word):
        r = np.random.normal(0.15, 0.1) * ft_en.cum_table[-1]
        word_key = np.searchsorted(ft_en.cum_table, r)
        word = ft_en.wv.index_to_key[word_key]
    return word


def generate():
	count = 6 - np.random.randint(3)
	choices_in_vocab = list(np.random.choice(in_vocab, count))
	choices_not_in_vocab = list(np.random.choice(not_in_vocab, 6 - count))
	choices = choices_in_vocab + choices_not_in_vocab
	choices.append(get_word())
	choices.append(get_word())

	THE_NAME = ""
	similar = ft_en.wv.most_similar(positive=choices)
	np.random.shuffle(similar)

	for word, weight in similar:
	    if is_invalid(word, True):
	        continue
	    THE_NAME = word.capitalize()
	    break

	THE_COLOR = np.random.choice(list(color_categories.keys()))
	THE_ANIMAL = np.random.choice(list(animals.keys()))
	THE_COLOR_HEX = np.random.choice(colors[THE_COLOR])

	font = np.random.choice(glob("fonts/*.ttf"))
	W, H = (1600, 416)

	img = Image.new(mode="RGB", size=(W, H), color=THE_COLOR_HEX)
	draw = ImageDraw.Draw(img)
	font = ImageFont.truetype(font, 256)

	w, h = draw.textsize(THE_NAME, font=font)

	color_value = colorsys.rgb_to_hsv(*ImageColor.getcolor(THE_COLOR_HEX, "RGB"))[2]
	stroke = "#000000" if color_value > color_value_threshold else "#FFFFFF"
	fill = "#FFFFFF" if color_value > color_value_threshold else "#000000"

	draw.text(((W -w ) / 2, (H - h) / 2), THE_NAME, fill=fill, stroke_width=4, stroke_fill=stroke, font=font)

	buffered = BytesIO()
	img.save(buffered, format="JPEG", quality=100)

	return {
		"name": THE_NAME,
		"color": color_categories[THE_COLOR],
		"color_hex": THE_COLOR_HEX,
		"animal": THE_ANIMAL,
		"animal_french": animals[THE_ANIMAL].capitalize(),
		"logo": base64.b64encode(buffered.getvalue()).decode("utf-8"),
		"stroke": stroke
	}


@app.route('/')
def root():
	try:
		bde = generate()
		return render_template("index.html", bde=generate())
	except:
		return redirect("/", code=302)


if __name__ == "__main__":
    app.run(host='0.0.0.0')