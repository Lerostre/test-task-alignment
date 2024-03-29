good_tokens = {
    "good": 9.0,
    "acceptable": 9.0,
    "excellent": 9.0,
    "exceptional": 9.0,
    "favorable": 9.0,
    "great": 9.0,
    "marvelous": 9.0,
    "positive": 9.0,
    "satisfactory": 9.0,
    "satisfying": 9.0,
    "superb": 9.0,
    "valuable": 9.0,
    "wonderful": 9.0,
    "ace": 6.0,
    "boss": 6.0,
    "bully": 6.0,
    "capital": 6.0,
    "choice": 6.0,
    "crack": 6.0,
    "nice": 6.0,
    "pleasing": 6.0,
    "prime": 6.0,
    "quality": 6.0,
    "rad": 6.0,
    "sound": 6.0,
    "spanking": 6.0,
    "sterling": 6.0,
    "super": 6.0,
    "superior": 6.0,
    "welcome": 6.0,
    "worthy": 6.0,
    "admirable": 3.0,
    "agreeable": 3.0,
    "commendable": 3.0,
    "congenial": 3.0,
    "deluxe": 3.0,
    "first-class": 3.0,
    "first-rate": 3.0,
    "gnarly": 3.0,
    "gratifying": 3.0,
    "honorable": 3.0,
    "jake": 3.0,
    "neat": 3.0,
    "precious": 3.0,
    "recherché": 3.0,
    "reputable": 3.0,
    "select": 3.0,
    "shipshape": 3.0,
    "splendid": 3.0,
    "stupendous": 3.0,
    "super-eminent": 3.0,
    "super-excellent": 3.0,
    "tiptop": 3.0,
    "up to snuff": 3.0,
}

bad_tokens = {
    "bad": -9.0,
    "atrocious": -9.0,
    "awful": -9.0,
    "cheap": -9.0,
    "crummy": -9.0,
    "dreadful": -9.0,
    "lousy": -9.0,
    "poor": -9.0,
    "rough": -9.0,
    "sad": -9.0,
    "unacceptable": -9.0,
    "blah": -6.0,
    "bummer": -6.0,
    "diddly": -6.0,
    "downer": -6.0,
    "garbage": -6.0,
    "gross": -6.0,
    "imperfect": -6.0,
    "inferior": -6.0,
    "junky": -6.0,
    "synthetic": -6.0,
    "abominable": -3.0,
    "amiss": -3.0,
    "bad news": -3.0,
    "beastly": -3.0,
    "careless": -3.0,
    "cheesy": -3.0,
    "crappy": -3.0,
    "cruddy": -3.0,
    "defective": -3.0,
    "deficient": -3.0,
    "dissatisfactory": -3.0,
    "erroneous": -3.0,
    "fallacious": -3.0,
    "faulty": -3.0,
    "godawful": -3.0,
    "grody": -3.0,
    "grungy": -3.0,
    "icky": -3.0,
    "inadequate": -3.0,
    "incorrect": -3.0,
    "not good": -3.0,
    "off": -3.0,
    "raunchy": -3.0,
    "slipshod": -3.0,
    "stinking": -3.0,
    "substandard": -3.0,
    "the pits": -3.0,
    "unsatisfactory": -3.0
}

omit_tokens = [
    "<br", "<br /", "<br />", "< br />",
    "<br />< br />", "<br /><br />",
    "<lf><lf>", "<lf>",
]