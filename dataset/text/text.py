from torch.utils.data import Dataset
import json
import re
from dataset.text.vocabulary import Vocabulary


class CustomText(Dataset):
    def __init__(self, data_directory, vocab=Vocabulary()):
        super(CustomText, self).__init__()

        with open(data_directory, mode="r", encoding="utf-8") as file:
            data = json.load(file)
        self.data = list(data.items())

        self.vocab = vocab

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        script = self._get_script(index)
        lowered_script = self._lower(script)
        tokenized_script = self._tokenize(lowered_script)
        encoded_script = self._encode(tokenized_script)
        return encoded_script
    
    def _get_script(self, index):
        return self.data[index][1]["script"]
    
    def _lower(self, script):
        return script.lower()

    def _tokenize(self, lowered_script):
        tokenized_script = []
        for word in lowered_script.split():
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

            tokenized_script.append(tokenized_word)
        return tokenized_script
    
    def _encode(self, tokenized_script):
        encoded_script = []
        for tokenized_word in tokenized_script:
            encoded_word = self.vocab._encode_word(tokenized_word)
            encoded_script.append(encoded_word)
        return encoded_script
 