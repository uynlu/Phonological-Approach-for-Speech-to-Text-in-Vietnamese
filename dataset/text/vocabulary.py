import re


class Vocabulary:
    def __init__(self):
        self.consonant_2_index = {
            "": 0,
            # Phụ âm
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
            "ngh": 26,
            # Ngoại lệ
            "1": 27,
            "2": 28,
            "3": 29,
            "4": 30,
            "5": 31,
            "6": 32,
            "7": 33,
            "8": 34,
            "9": 35,
            "0": 36,
            "<pad>": 37,
            "<sos>": 38,
            "<eos>": 39,
            "Đắk Lắk": 40,
            "Đắk Nông": 41
        }

        self.vowel_2_index = {
            "": 0,
            # a
            "a": 1,
            "ac": 2,
            "ach": 3,
            "ai": 4,
            "am": 5,
            "an": 6,
            "ang": 7,
            "anh": 8,
            "ao": 9,
            "ap": 10,
            "at": 11,
            "ay": 12,
            "au": 13,
            # ă
            "ă": 14,
            "ăc": 15,
            "ăm": 16,
            "ăn": 17,
            "ăng": 18,
            "ăp": 19,
            "ăt": 20,
            # â
            "â": 21,
            "âc": 22,
            "âm": 23,
            "ân": 24,
            "âng": 25,
            "âp": 26,
            "ât": 27,
            "âu": 28,
            "ây": 29,
            # e
            "e": 30,
            "ec": 31,
            "em": 32,
            "en": 33,
            "eng": 34,
            "eo": 35,
            "ep": 36,
            "et": 37,
            # ê
            "ê": 38,
            "êch": 39,
            "êm": 40,
            "ên": 41,
            "ênh": 42,
            "êp": 43,
            "êt": 44,
            "êu": 45,
            # i            
            "i": 46,
            "ia": 47,
            "ich": 48,
            "iêc": 49,
            "iêm": 50,
            "iên": 51,
            "iêng": 52,
            "iêp": 53,
            "iêt": 54,
            "iêu": 55,
            "im": 56,
            "in": 57,
            "inh": 58,
            "ip": 59,
            "it": 60,
            "iu": 61,
            # o
            "o": 62,
            "oa": 63,
            "oac": 64,
            "oach": 65,
            "oai": 66,
            "oam": 67,
            "oan": 68,
            "oang": 69,
            "oanh": 70,
            "oao": 71,
            "oap": 72,
            "oat": 73,
            "oay": 74,
            "oăc": 75,
            "oăm": 76,
            "oăn": 77,
            "oăng": 78,
            "oăt": 79,
            "oc": 80,
            "oe": 81,
            "oen": 82,
            "oeo": 83,
            "oet": 84,
            "oi": 85,
            "om": 86,
            "on": 87,
            "ong": 88,
            "ooc": 89,
            "oong": 90,
            "op": 91,
            "ot": 92,
            # ô
            "ô": 93,
            "ôc": 94,
            "ôi": 95,
            "ôm": 96,
            "ôn": 97,
            "ông": 98,
            "ôp": 99,
            "ôt": 100,
            # ơ
            "ơ": 101,
            "ơi": 102,
            "ơm": 103,
            "ơn": 104,
            "ơp": 105,
            "ơt": 106,
            # u
            "u": 107,
            "ua": 108,
            "uân": 109,
            "uâng": 110,
            "uât": 111,
            "uây": 112,
            "uc": 113,
            "uê": 114,
            "uêch": 115,
            "uênh": 116,
            "ui": 117,
            "um": 118,
            "un": 119,
            "ung": 120,
            "uơ": 121,
            "uôc": 122,
            "uôi": 123,
            "uôm": 124,
            "uôn": 125,
            "uông": 126,
            "uôt": 127,
            "up": 128,
            "ut": 129,
            "uy": 130,
            "uya": 131,
            "uych": 132,
            "uyên": 133,
            "uyêt": 134,
            "uyn": 135,
            "uynh": 136,
            "uyp": 137,
            "uyt": 138,
            "uyu": 139,
            # ư
            "ư": 140,
            "ưa": 141,
            "ưc": 142,
            "ưi": 143,
            "ưng": 144,
            "ươc": 145,
            "ươi": 146,
            "ươm": 147,
            "ươn": 148,
            "ương": 149,
            "ươp": 150,
            "ươt": 151,
            "ươu": 152,
            "ưt": 153,
            "ưu": 154,
            # y
            "y": 155,
            "yêm": 156,
            "yên": 157,
            "yêng": 158,
            "yêt": 159,
            "yêu": 160
        }

        self.tone_2_index = {
            "": 0,
            "ngang": 1,
            "sắc": 2,
            "huyền": 3,
            "hỏi": 4,
            "ngã": 5,
            "nặng": 6
        }

        self.index_2_consonant = {i: c for c, i in self.consonant_2_index.items()}
        self.index_2_vowel = {i: v for v, i in self.vowel_2_index.items()}
        self.index_2_tone = {i: t for t, i in self.tone_2_index.items()}

    def __len__(self):
        # return (len(self.consonant_2_index) + len(self.vowel_2_index) + len(self.tone_2_index))
        pass  # xem lại
    
    def __getindex__(self, word):
        tokenized_word = self._tokenize_word(word)
        encoded_word = self._encode_word(tokenized_word)
        return encoded_word

    def __getword__(self, index):
        item_of_word = self._encode_index(index)
        word = self._merge_item_of_word(item_of_word)
        return word

    def _tokenize_word(self, word):
        tokenized_word = []
        # Lấy phụ âm
        num_consonant = 0
        if word in list(self.consonant_2_index.keys())[27:]:
            tokenized_word = [word, "", ""]
        else:
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

        return tokenized_word
            
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
