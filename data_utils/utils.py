import torch
from torch import nn
from torchaudio import transforms
import unicodedata
import re

def normalize_script(script: str) -> str:
    script = script.replace("0", " không ")
    script = script.replace("1", " một ")
    script = script.replace("2", " hai ")
    script = script.replace("3", " ba ")
    script = script.replace("4", " bốn ")
    script = script.replace("5", " năm ")
    script = script.replace("6", " sáu ")
    script = script.replace("7", " bảy ")
    script = script.replace("8", " tám ")
    script = script.replace("9", " chín ")

    script = re.sub(r"[,\.!?@\'\"-_+=#$%^&*()~\\{}\[\];:<>/|]", " ", script)
    
    script = " ".join(script.split())

    return script

def get_tone(word: str):
    tone_map = {
        "\u0300": "<huyền>",
        "\u0301": "<sắc>",
        "\u0303": "<ngã>",
        "\u0309": "<hỏi>",
        "\u0323": "<nặng>",
    }
    decomposed_word = unicodedata.normalize("NFD", word)
    tone = None
    remaining_word = ""
    for char in decomposed_word:
        if char in tone_map:
            tone = tone_map[char]
        else:
            remaining_word += char
    remaining_word = unicodedata.normalize("NFC", remaining_word)
    
    return tone, remaining_word

def get_onset(word: str) -> tuple[str, str]:
    onsets = ["ngh", "tr", "th", "ph", "nh", "ng", "kh", 
              "gi", "gh", "ch", "q", "đ", "x", "v", "t", 
              "s", "r", "n", "m", "l", "k", "h", "g", "d", 
              "c", "b"]
    
    # get the onset
    for onset in onsets:
        if word.startswith(onset):
            if onset != "q":
                word = word.removeprefix(onset)
            return onset, word

    return None, word

def get_medial(word: str) -> tuple[str, str]:
    O_MEDIAL = "o"
    U_MEDIAL = "u"

    if word.startswith("q"):
        # in Vietnamese, words starting with "q" always has "u" as the medial
        word = word.removeprefix("qu")
        return U_MEDIAL, word
    
    o_medial_cases = ["oa", "oă", "oe"]
    for o_medial_case in o_medial_cases:
        if word.startswith(o_medial_case):
            word = word.removeprefix("o")
            return O_MEDIAL, word
        
    if (word.startswith("ua") and word != "ua") or word.startswith("uô"):
        return None, word
    
    nucleuses = ["ê", "y", "ơ", "a", "â", "ya"]
    for nucleus in nucleuses:
        component = U_MEDIAL + nucleus
        if word.startswith(component):
            word = word.removeprefix("u")
            return U_MEDIAL, word
        
    return None, word

def get_nucleus(word: str) -> tuple[str, str]:
    nucleuses = ["oo", "ươ", "ưa", "uô", "ua", "iê", "yê", 
                 "ia", "ya", "e", "ê", "u", "ư", "ô", "i", 
                 "y", "o", "ơ", "â", "a", "o", "ă"]
    
    for nucleus in nucleuses:
        if word.startswith(nucleus):
            word = word.removeprefix(nucleus)
            return nucleus, word
        
    return None, word
    
def get_coda(word: str) -> str:
    codas = ['ng', 'nh', 'ch', 'u', 'n', 'o', 'p', 'c', 'm', 'y', 'i', 't']
    
    if word in codas:
        return word
    
    return None

def split_phoneme(word: str) -> list[str, str, str]:
    onset, word = get_onset(word)
    
    medial, word = get_medial(word)

    nucleus, word = get_nucleus(word)

    coda = get_coda(word)
    
    return onset, medial, nucleus, coda

def is_Vietnamese(word: str) -> tuple[bool, tuple]:
    tone, word = get_tone(word)
    if not re.match(r"[a-zA-Zăâđưôơê]", word):
        return False, None

    # handling for special cases
    special_words_to_words = {
        "gin": "giin",     # gìn after being removed the tone 
        "giêng": "giiêng", # giếng after being removed the tone
        "giêt": "giiêt",   # giết after being removed the tone
        "giêc": "giiêc",   # giếc (diếc) after being removed the tone
        "gi": "gii"        # gì after removing the tone 
    }

    if word in special_words_to_words:
        word = special_words_to_words[word]

    # check the total number of nucleus in word
    vowels = ["oo", "ươ", "ưa", "uô", "ua", "iê", "yê", 
              "ia", "ya", "e", "ê", "u", "ư", "ô", "i", 
              "y", "o", "ơ", "â", "a", "o", "ă"]
    currentCharacterIsVowels = False
    previousCharacterIsVowels = word[0] in vowels
    foundVowels = 0
    
    for character in word[1:]:
        if character in vowels:
            currentCharacterIsVowels = True
        else:
            currentCharacterIsVowels = False
        
        if currentCharacterIsVowels and not previousCharacterIsVowels:
            foundVowels += 1

        # in Vietnamese, each word has only one syllable    
        if foundVowels > 2:
            return False, None
            
        previousCharacterIsVowels = currentCharacterIsVowels
    
    # in case the word has the structure of a Vietnamese word, we check whether it satisfies the rule of phoneme combination
    onset, medial, nucleus, coda = split_phoneme(word)

    if nucleus is None:
        return False, None
    
    former_word = ""
    for component in [onset, medial, nucleus, coda]:
        if component is not None:
            former_word += component
    if former_word != word:
        return False, None
    
    if onset == "k" and medial is None and nucleus not in ["i", "y", "e", "ê", "iê", "yê", "ia", "ya"]:
        return False, None
    
    if onset == "c" and medial is None and nucleus in ["i", "y", "e", "ê", "iê", "yê", "ia", "ya"]:
        return False, None
    
    if onset == "q" and not medial == "u":
        return False, None
    
    if onset == "gh" and medial is None and nucleus not in ["i", "e", "ê", "iê"]:
        return False, None
    
    if onset == "g" and medial is None and nucleus in ["i", "e", "ê", "iê"]:
        return False, None
    
    if onset == "ngh" and medial is None and nucleus not in ["i", "e", "ê", "iê", "yê", "ia", "ya"]:
        return False, None
    
    if onset == "ng" and medial is None and nucleus in ["i", "e", "ê", "iê", "yê", "ia", "ya"]:
        return False, None
    
    if onset in ["r", "gi"] and medial is not None:
        return False, None
    
    if medial == "o" and nucleus not in ["a", "ă", "e"]:
        return False, None
    
    if medial == "u" and nucleus not in ["yê", "ya", "e", "ê", "y", "ơ", "ô", "a", "â", "ă", "i"]:  # i => quí
        return False, None
    
    if nucleus == "oo" and coda not in ["ng", "c"]:
        return False, None
    
    if nucleus == "ua" and coda is not None:
        return False, None
    
    if nucleus == "ia" and coda is not None:
        return False, None
    
    if nucleus == "ya" and coda is not None:
        return False, None
    
    if nucleus in ["ua", "uô"] and coda == "ph":
        return False, None
    
    if nucleus in ["yê", "iê"] and coda is None:
        return False, None
    
    if nucleus in ["ă", "â"] and coda is None:
        return False, None
    
    if medial == "o" and nucleus in ["iê", "yê", "ia", "ya"]:
        return False, None
    
    if medial is not None:
        if nucleus in ["u", "oo", "o", "ua", "uô", "ươ", "ưa", "ư"]:
            return False, None
        
        if nucleus in ["i", "e", "ê", "ia", "ya", "iê", "yê"] and coda in ["m", "ph"]:
            return False, None
        
    if coda == "o" and nucleus not in ["a", "e"]:
        return False, None
    
    if coda == "y" and nucleus not in ["a", "â"]:
        return False, None
    
    if coda == "i" and nucleus in ["ă", "â", "i", "e", "iê", "yê", "ia", "ya"]:
        return False, None
    
    if coda == "nh" and nucleus not in ["a", "i", "y", "ê"]:
        return False, None
    
    if coda == "ng" and nucleus not in ["a", "o", "ô", "u", "ư", "e", "iê", "ươ", "â", "ă", "uô", "oo"]:
        return False, None

    if coda == "ch" and nucleus not in ["i", "a", "ê", "y"]:
        return False, None

    if coda == "c" and nucleus in ["i", "ê", "e", "ơ"]:
        return False, None

    if nucleus == coda:
        return False, None

    return True, (onset, medial, nucleus, coda, tone)

def decompose_non_vietnamese_word(word: str):
    vowels = ["a", "ă", "â",
              "e", "ê" "i", 
              "o", "ô", "ơ",
              "u", "ư"]
    chars = []
    for char in word:
        tone, char = get_tone(char)
        if char in vowels:
            chars.append([
                None,
                None,
                char,
                None,
                tone
            ])
        else:
            chars.append([
                char,
                None,
                None,
                None,
                tone
            ])

    return chars
    
def compose_word(onset: str, medial: str, nucleus: str, coda: str, tone: str) -> str:
    tone_map = {
            "<huyền>": "\u0300",
            "<sắc>": "\u0301",
            "<ngã>": "\u0303",
            "<hỏi>": "\u0309",
            "<nặng>": "\u0323"
        }
    if tone is not None:
        tone = tone_map[tone]

        # process for the special case of medial + coda (hỏa, thủy, thuở, thỏa, ...)
        # in this case, only "thuở" follows the general rule of tone marking, the others are the case that tones are marked on the medial.
        if onset != "q" and medial is not None and nucleus is not None and coda is None and nucleus != "ơ":
            medial += tone
        else:
            if coda is None:
                nucleus = nucleus[0] + tone + nucleus[1:]
            else:
                nucleus = nucleus + tone

    word = ""
    if onset:
        word += onset
    if medial:
        word += medial
    if nucleus:
        word += nucleus
    if coda:
        word += coda

    if "gii" in word:
        word = re.sub("gii", "gi", word)

    word = unicodedata.normalize("NFC", word)

    return word

def collate_fn(items: list[dict]) -> torch.Tensor:
    ids = [item["id"] for item in items]
    voices = [item["voice"] for item in items]
    scripts = [item["script"] for item in items]
    labels = [item["labels"] for item in items]

    # adding pad value for voice
    max_voice_len = max([voice.shape[-1] for voice in voices])
    for ith, voice in enumerate(voices):
        delta_len = max_voice_len - voice.shape[-1]
        pad_tensor = torch.zeros((1, delta_len)).float()
        voices[ith] = torch.cat([voice, pad_tensor], dim=-1)
    
    # adding pad value for encoded script
    max_label_len = max([label.shape[-1] for label in labels])
    for ith, label in enumerate(labels):
        delta_len = max_label_len - label.shape[-1]
        pad_tensor = torch.zeros((delta_len, )).float()
        labels[ith] = torch.cat([label, pad_tensor], dim=-1).unsqueeze(0)

    return {
        "id": ids,
        "voice": torch.tensor(voices),
        "script": scripts,
        "labels": torch.tensor(labels)
    }

class MelSpectrogram(nn.Module):
    def __init__(self, config):
        # sample_rate: tần suất lấy mẫu (16000 điểm/s)
        # n_mels: chiều cao của Mel Spectrogram?
        # win_length: (nếu chiều dài của data là 800 => 2 window)
        # hop_length: bước nhảy (stride)
        # n_ffts: chiều dài mỗi Time-Section

        super().__init__()

        self.transform = transforms.MelSpectrogram(
            sample_rate=config.sampling_rate,
            n_mels=config.n_mels, 
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_ffts
        )

    def forward(self, input):
        return self.transform(input)

class MFCC(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.transform = transforms.MFCC(
            sample_rate=config.sample_rate,
            n_mfcc=config.n_mfcc,
            melkwargs={
                "n_mels": config.n_mels,
                "n_fft": config.n_ffts,
                "hop_length": config.hop_length
            }
        )

    def forward(self, input):
        return self.transform(input)
