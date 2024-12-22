import torch
import itertools
import re
import string

from dataset.text.word_decomposition import is_Vietnamese, decompose_non_vietnamese_word, compose_word


class PhonemeVocabv2:
    def __init__(self):
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.blank_idx = 3

        onsets = [
            'ngh', 'tr', 'th', 'ph', 'nh', 'ng', 'kh', 
            'gi', 'gh', 'ch', 'q', 'đ', 'x', 'v', 't', 
            's', 'r', 'n', 'm', 'l', 'k', 'h', 'g', 'd', 
            'c', 'b'
        ]
        rhymes = [
            # a
            "a", "ac", "ach", "ai", 
            "am", "an", "ang", "anh", 
            "ao", "ap", "at", "ay", "au",
            # ă
            "ă", "ăc", "ăm", "ăn", "ăng", "ăp", "ăt",
            # â
            "â", "âc", "âm", "ân", "âng",
            "âp", "ât", "âu", "ây",
            # e
            "e", "ec", "em", "en",
            "eng", "eo", "ep", "et",
            # ê
            "ê", "êch", "êm", "ên", 
            "ênh", "êp", "êt", "êu",
            # i
            "i", "ia", "ich", "iêc", "iêm", "iên",
            "iêng", "iêp", "iêt", "iêu", "im", "in",
            "inh", "ip", "it", "iu",
            # o
            "o", "oa", "oac", "oach", "oai",
            "oam", "oan", "oang", "oanh",
            "oao", "oap", "oat", "oay",
            "oăc", "oăm", "oăn", "oăng",
            "oăt", "oc", "oe", "oen","oeo",
            "oet", "oi", "om", "on", "ong",
            "ooc", "oong", "op", "ot",
            # ô
            "ô", "ôc", "ôi",
            "ôm", "ôn", "ông",
            "ôp", "ôt",
            # ơ
            "ơ", "ơi", "ơm",
            "ơn", "ơp", "ơt",
            # u
            "u", "ua", "uân", "uâng", "uât",
            "uây", "uc", "uê", "uêch", "uênh",
            "ui", "um", "un", "ung", "uơ", "uôc",
            "uôi", "uôm", "uôn", "uông", "uôt",
            "up", "ut", "uy", "uya", "uych",
            "uyên", "uyêt", "uyn", "uynh",
            "uyp", "uyt", "uyu",

            "uach", "uai", "uan", "uang", "uanh", "uao", "uat", "uau", "uay",
            "uăc", "uăm", "uăn", "uăng", "uăp", "uăt", "uâc", "uât",
            "ue", "uen", "ueo", "uet", "uên", "uêt", "uêu",

            # ư
            "ư", "ưa", "ưc", "ưi",
            "ưng", "ươc", "ươi",
            "ươm", "ươn", "ương",
            "ươp", "ươt", "ươu",
            "ưt", "ưu",
            # y
            "y", "yêm", "yên", 
            "yêng", "yêt", "yêu"
        ]
        codas = ['ng', 'nh', 'ch', 'u', 'n', 'o', 'p', 'c', 'k', 'm', 'y', 'i', 't']
        tones = ['<huyền>', '<sắc>', '<ngã>', '<hỏi>', '<nặng>']
        phonemes = onsets + rhymes + codas + tones
        self.phoneme2idx = {
            phoneme: idx for idx, phoneme in enumerate(phonemes, start=4)
        }
        self.idx2phoneme = {idx: phoneme for phoneme, idx in self.phoneme2idx.items()}

    def __len__(self):
        return len(self.phoneme2idx) + 4
    
    def encode_script(self, script: str):
        script = script.lower()
        pattern = f"[{re.escape(string.punctuation)}]"
        script = re.sub(pattern, "", script) 
        words = script.split()
        word_components = []
        is_Vietnamese_words = []

        for word in words:
            is_Vietnamese_word, components = is_Vietnamese(word)
            is_Vietnamese_words.append(is_Vietnamese_word)
            if is_Vietnamese_word:
                word_components.append(components)
            else:
                word_components.append(decompose_non_vietnamese_word(word))

        phoneme_script = []
        for ith in range(len(words)):
            word_component = word_components[ith]
            if is_Vietnamese_words[ith]:
                onset, medial, nucleus, coda, tone = word_component
                vowel = compose_word(None, medial, nucleus, coda, None)
                phoneme_script.extend([
                    self.phoneme2idx[onset] if onset else self.blank_idx, 
                    self.phoneme2idx[vowel] if vowel else self.blank_idx, 
                    self.phoneme2idx[tone] if tone else self.blank_idx, 
                    self.blank_idx])
            else:
                for char in word_component:
                    onset, medial, nucleus, coda, tone = char
                    vowel = compose_word(None, medial, nucleus, coda, None)
                    phoneme_script.extend([
                        self.phoneme2idx[onset] if onset else self.blank_idx, 
                        self.phoneme2idx[vowel] if vowel else self.blank_idx, 
                        self.phoneme2idx[tone] if tone else self.blank_idx, 
                        self.blank_idx])

        vec = torch.tensor(phoneme_script[:-1]).long()  # remove the last blank token
        return vec

    def decode_script(self, tensor_script: torch.Tensor):
        '''
            tensorscript: (1, seq_len)
        '''
        # remove duplicated token
        tensor_script = tensor_script.squeeze(0).long().tolist()
        script = [self.idx2phoneme[idx] for idx in tensor_script]
        script = ' '.join([k for k, _ in itertools.groupby(script)])
