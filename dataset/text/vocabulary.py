import re


class Vocabulary:
    def __init__(self):
        self.consonant_2_index = {
            "": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "đ": 4,
            "g": 5,
            "h": 6,
            "k": 7,
            "l": 8,
            "m": 9,
            "n": 10,
            "r": 11,
            "s": 12,
            "t": 13,
            "v": 14,
            "x": 15,
            "ch": 16,
            "gh": 17,
            "ng": 18,
            "tr": 19,
            "qu": 20,
            "ph": 21,
            "th": 22,
            "nh": 23,
            "kh": 24,
            "gi": 25,
            "ngh": 26
        }

        self.vowel_2_index = {
            # a
            "a": 0,
            "ac": 1,
            "ach": 2,
            "ai": 3,
            "am": 4,
            "an": 5,
            "ang": 6,
            "anh": 7,
            "ao": 8,
            "ap": 9,
            "at": 10,
            "ay": 11,
            "au": 12,
            # ă
            "ă": 13,
            "ăc": 14,
            "ăm": 15,
            "ăn": 16,
            "ăng": 17,
            "ăp": 18,
            "ăt": 19,
            # â
            "â": 20,
            "âc": 21,
            "âm": 22,
            "ân": 23,
            "âng": 24,
            "âp": 25,
            "ât": 26,
            "âu": 27,
            "ây": 28,
            # e
            "e": 29,
            "ec": 30,
            "em": 31,
            "en": 32,
            "eng": 33,
            "eo": 34,
            "ep": 35,
            "et": 36,
            # ê
            "ê": 37,
            "êch": 38,
            "êm": 39,
            "ên": 40,
            "ênh": 41,
            "êp": 42,
            "êt": 43,
            "êu": 44,
            # i            
            "i": 45,
            "ia": 46,
            "ich": 47,
            "iêc": 48,
            "iêm": 49,
            "iên": 50,
            "iêng": 51,
            "iêp": 52,
            "iêt": 53,
            "iêu": 54,
            "im": 55,
            "in": 56,
            "inh": 57,
            "ip": 58,
            "it": 59,
            "iu": 60,
            # o
            "o": 61,
            "oa": 62,
            "oac": 63,
            "oach": 64,
            "oai": 65,
            "oam": 66,
            "oan": 67,
            "oang": 68,
            "oanh": 69,
            "oao": 70,
            "oap": 71,
            "oat": 72,
            "oay": 73,
            "oăc": 74,
            "oăm": 75,
            "oăn": 76,
            "oăng": 77,
            "oăt": 78,
            "oc": 79,
            "oe": 80,
            "oen": 81,
            "oeo": 82,
            "oet": 83,
            "oi": 84,
            "om": 85,
            "on": 86,
            "ong": 87,
            "ooc": 88,
            "oong": 89,
            "op": 90,
            "ot": 91,
            # ô
            "ô": 92,
            "ôc": 93,
            "ôi": 94,
            "ôm": 95,
            "ôn": 96,
            "ông": 97,
            "ôp": 98,
            "ôt": 99,
            # ơ
            "ơ": 100,
            "ơi": 101,
            "ơm": 102,
            "ơn": 103,
            "ơp": 104,
            "ơt": 105,
            # u
            "u": 106,
            "ua": 107,
            "uân": 108,
            "uâng": 109,
            "uât": 110,
            "uây": 111,
            "uc": 112,
            "uê": 113,
            "uêch": 114,
            "uênh": 115,
            "ui": 116,
            "um": 117,
            "un": 118,
            "ung": 119,
            "uơ": 120,
            "uôc": 121,
            "uôi": 122,
            "uôm": 123,
            "uôn": 124,
            "uông": 125,
            "uôt": 126,
            "up": 127,
            "ut": 128,
            "uy": 129,
            "uya": 130,
            "uych": 131,
            "uyên": 132,
            "uyêt": 133,
            "uyn": 134,
            "uynh": 135,
            "uyp": 136,
            "uyt": 137,
            "uyu": 138,
            # ư
            "ư": 139,
            "ưa": 140,
            "ưc": 141,
            "ưi": 142,
            "ưng": 143,
            "ươc": 144,
            "ươi": 145,
            "ươm": 146,
            "ươn": 147,
            "ương": 148,
            "ươp": 149,
            "ươt": 150,
            "ươu": 151,
            "ưt": 152,
            "ưu": 153,
            # y
            "y": 154,
            "yêm": 155,
            "yên": 156,
            "yêng": 157,
            "yêt": 158,
            "yêu": 159,
        }

        self.tone_2_index = {
            "ngang": 0,
            "sắc": 1,
            "huyền": 2,
            "hỏi": 3,
            "ngã": 4,
            "nặng": 5
        }

        self.index_2_consonant = {i: c for c, i in self.consonant_2_index.items()}
        self.index_2_vowel = {i: v for v, i in self.vowel_2_index.items()}
        self.index_2_tone = {i: t for t, i in self.tone_2_index.items()}

    def __len__(self):
        return (len(self.consonant_2_index) * len(self.vowel_2_index) * len(self.tone_2_index))
    
    def __getindex__(self, word):
        tokenized_word = self._tokenized_word(word)
        encoded_word = self._encode_word(tokenized_word)
        return encoded_word

    def __getword__(self, index):
        item_of_word = self._encode_index(index)
        word = self._merge_item_of_word(item_of_word)
        return word

    def _tokenized_word(self, word):
        tokenized_word = []
        # Lấy phụ âm
        num_consonant = 0
        if word[:1] in ["b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "x"]:
            if word[:2] in ["ch", "gh", "ng", "tr", "qu", "ph", "th", "nh", "kh", "gi"]:
                if word[:3] == "ngh":
                    tokenized_word.append(word[:3])
                    num_consonant = 3
                else:
                    tokenized_word.append(word[:2])
                    num_consonant = 2
            else:
                tokenized_word.append(word[:1])
                num_consonant = 1
        else:  # không có phụ âm
            tokenized_word.append("")
            
        # Lấy vần
        map_tone = {
            "á": "a", "à": "a", "ả": "a", "ã": "a", "ạ": "a",
            "ắ": "ă", "ằ": "ă", "ẳ": "ă", "ẵ": "ă", "ặ": "ă",
            "ấ": "â", "ầ": "â", "ẩ": "â", "ẫ": "â", "ậ": "â",
            "í": "i", "ì": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
            "ý": "y", "ỳ": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
            "ó": "o", "ò": "o", "ỏ": "o", "õ": "o", "ọ": "o",
            "ố": "ô", "ồ": "ô", "ổ": "ô", "ỗ": "ô", "ộ": "ô",
            "ớ": "ơ", "ờ": "ơ", "ở": "ơ", "ỡ": "ơ", "ợ": "ơ",
            "é": "e", "è": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
            "ế": "ê", "ề": "ê", "ể": "ê", "ễ": "ê", "ệ": "ê",
            "ú": "u", "ù": "u", "ủ": "u", "ũ": "u", "ụ": "u",
            "ứ": "ư", "ừ": "ư", "ử": "ư", "ữ": "ư", "ự": "ư"
        }
        vowels = []
        for character in word[num_consonant:]:
            transformed_character = map_tone.get(character, character)
            vowels.append(transformed_character)
        tokenized_word.append("".join(vowels))

        # Lấy thanh điệu
        for index, character in enumerate(word[num_consonant:]):
            if re.match(r"[áắấíýóốớéếúứ]", character):
                tokenized_word.append("sắc")
                break
            elif re.match(r"[àằầìỳòồờèềùừ]", character):
                tokenized_word.append("huyền")
                break
            elif re.match(r"[ảẳẩỉỷỏổởẻểủử]", character):
                tokenized_word.append("hỏi")
                break
            elif re.match(r"[ãẵẫĩỹõỗỡẽễũữ]", character):
                tokenized_word.append("ngã")
                break
            elif re.match(r"[ạặậịỵọộợẹệụự]", character):
                tokenized_word.append("nặng")
                break
            if index == len(word[num_consonant:]) - 1:
                tokenized_word.append("ngang")
            
    def _encode_word(self, tokenized_word):
        return [self.consonant_2_index.get(tokenized_word[0]), self.vowel_2_index.get(tokenized_word[1]), self.tone_2_index.get(tokenized_word[2])]
        
    def _encode_index(self, index):
        return [self.index_2_consonant.get(index[0]), self.index_2_vowel.get(index[1]), self.index_2_tone.get(index[2])]
    
    def _merge_item_of_word(self, item_of_word):
        if item_of_word[-1] == "ngang":
            return "".join((item_of_word[0], item_of_word[1]))
        elif item_of_word[-1] == "sắc":
            pass
        elif item_of_word[-1] == "huyền":
            pass
        elif item_of_word[-1] == "hỏi":
            pass
        elif item_of_word[-1] == "ngã":
            pass
        elif item_of_word[-1] == "nặng":
            pass
