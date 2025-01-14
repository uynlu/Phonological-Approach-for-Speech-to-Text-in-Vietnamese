import torch

from data_utils.utils import is_Vietnamese, decompose_non_vietnamese_word, compose_word, split_phoneme


class PhonemeVocabv1:
    def __init__(self):
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.blank_token = "<blank>"

        onsets = [
            'ngh', 'tr', 'th', 'ph', 'nh', 'ng', 'kh', 
            'gi', 'gh', 'ch', 'q', 'đ', 'x', 'v', 't', 
            's', 'r', 'n', 'm', 'l', 'k', 'h', 'g', 'd', 
            'c', 'b', 'f', 'j', 'w', 'z'
        ]
        medials = ["u", "o"]
        nucleuses = [
            'oo', 'ươ', 'ưa', 'uô', 'ua', 'iê', 'yê', 
            'ia', 'ya', 'e', 'ê', 'u', 'ư', 'ô', 'i', 
            'y', 'o', 'ơ', 'â', 'a', 'o', 'ă'
        ]
        codas = ['ng', 'nh', 'ch', 'u', 'n', 'o', 'p', 'c', 'k', 'm', 'y', 'i', 't']
        tones = ['<huyền>', '<sắc>', '<ngã>', '<hỏi>', '<nặng>']
        phonemes = [self.pad_token, self.bos_token, self.eos_token, self.blank_token] +  onsets + medials + nucleuses + codas + tones
        self.phoneme2idx = {
            phoneme: idx for idx, phoneme in enumerate(phonemes)
        }
        self.idx2phoneme = {idx: phoneme for phoneme, idx in self.phoneme2idx.items()}
        
        self.pad_idx = self.phoneme2idx[self.pad_token]
        self.bos_idx = self.phoneme2idx[self.bos_token]
        self.eos_idx = self.phoneme2idx[self.eos_token]
        self.blank_idx = self.phoneme2idx[self.blank_token]

        self.special_ids = [self.pad_idx, self.bos_idx, self.eos_idx, self.blank_idx]
    
    def encode_script(self, script: str):
        words = script.split()
        word_components = []
        word_indices = [] # mark the token belonging to a word
        word_index = 1
        for word in words:
            is_Vietnamese_word, components = is_Vietnamese(word)
            if is_Vietnamese_word:
                word_components.append(components)
                word_indices.append(word_index)
            else:
                characters = decompose_non_vietnamese_word(word)
                word_components.extend(characters)
                word_indices.extend([word_index]*len(characters))

            word_index += 1

        phoneme_script = []
        for ith in range(len(word_components)):
            word_component = word_components[ith]
            onset, medial, nucleus, coda, tone = word_component
            phoneme_script.extend([
                self.phoneme2idx[onset] if onset else self.blank_idx, 
                self.phoneme2idx[medial] if medial else self.blank_idx, 
                self.phoneme2idx[nucleus] if nucleus else self.blank_idx,
                self.phoneme2idx[coda] if coda else self.blank_idx,
                self.phoneme2idx[tone] if tone else self.blank_idx,
                self.blank_idx])
        
        phoneme_script = phoneme_script[:-1] # skip the last blank token
        bos_token = [self.bos_idx, self.blank_idx, self.blank_idx, self.blank_idx, self.blank_idx, self.blank_idx]
        eos_token = [self.eos_idx, self.blank_idx, self.blank_idx, self.blank_idx, self.blank_idx, self.blank_idx]
        phoneme_script = bos_token + phoneme_script + eos_token
        # index for bos token and eos token
        word_indices = [0] + word_indices + [len(word_indices)+1]
        
        vec = torch.tensor(phoneme_script).long()
        word_indices = torch.tensor(word_indices).long()

        return vec, word_indices

    def decode_script(self, tensor_script: torch.Tensor, word_indices: torch.Tensor):
        '''
            tensorscript: (1, seq_len)
        '''
        # remove duplicated token
        ids = tensor_script.long().tolist()
        script = []
        word = []
        ith = 0
        while ith < len(ids):
            idx = ids[ith]
            if idx not in self.special_ids:
                word.append(self.idx2phoneme[idx])
            else:
                word.append(None)
            if len(word) == 5:
                onset, medial, nucleus, coda, tone = word
                word = compose_word(onset, medial, nucleus, coda, tone)
                script.append(word)
                word = []
                ith += 1 # skip the blank token
            ith += 1

        refined_script = [] # script that is preprocessed for non Vietnamese words
        prev_id = [word_indices[0]]
        word = [script[0]]
        for ith, current_id in enumerate(word_indices[1:], start=1):
            current_token = script[ith]
            if current_id == prev_id:
                word.append(current_token)
            else:
                refined_script.append("".join(word))
                prev_id = current_id
                word = [current_token]

        refined_script = ' '.join(refined_script[1:]) # skip the bos token

        return refined_script


class PhonemeVocabv2:
    '''
        Turn words into continuous senquences of phoneme
    '''
    
    def __init__(self):
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.blank_token = "<blank>"

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
            "uăc", "uăm", "uăn", "uăng", "uăp", "uăt", "uâc", "uât", "uoang",
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
        tones = ["<huyền>", "<sắc>", "<ngã>", "<hỏi>", "<nặng>"]
        phonemes = [self.pad_token, self.bos_token, self.eos_token, self.blank_token] +  onsets + rhymes + tones
        self.phoneme2idx = {
            phoneme: idx for idx, phoneme in enumerate(phonemes)
        }
        self.idx2phoneme = {idx: phoneme for phoneme, idx in self.phoneme2idx.items()}
        
        self.pad_idx = self.phoneme2idx[self.pad_token]
        self.bos_idx = self.phoneme2idx[self.bos_token]
        self.eos_idx = self.phoneme2idx[self.eos_token]
        self.blank_idx = self.phoneme2idx[self.blank_token]

        self.special_ids = [self.pad_idx, self.bos_idx, self.eos_idx, self.blank_idx]

    @property
    def size(self) -> int:
        return len(self.phoneme2idx)

    def encode_script(self, script: str):
        words = script.split()
        word_components = []
        word_indices = [] # mark the token belonging to a word
        word_index = 1
        for word in words:
            is_Vietnamese_word, components = is_Vietnamese(word)
            if is_Vietnamese_word:
                word_components.append(components)
                word_indices.append(word_index)
            else:
                characters = decompose_non_vietnamese_word(word)
                word_components.extend(characters)
                word_indices.extend([word_index]*len(characters))

            word_index += 1

        phoneme_script = []
        for ith in range(len(words)):
            word_component = word_components[ith]
            onset, medial, nucleus, coda, tone = word_component
            rhyme = compose_word(None, medial, nucleus, coda, None)
            phoneme_script.extend([
                self.phoneme2idx[onset] if onset else self.blank_idx, 
                self.phoneme2idx[rhyme] if rhyme else self.blank_idx, 
                self.phoneme2idx[tone] if tone else self.blank_idx, 
                self.blank_idx])

        phoneme_script = phoneme_script[:-1] # skip the last blank token
        phoneme_script = self.bos_idx + phoneme_script + self.eos_idx
        # index for bos token and eos token
        word_indices = [0] + word_indices + [len(word_indices)+1]
        
        vec = torch.tensor(phoneme_script).long()
        word_indices = torch.tensor(word_indices).long()
        
        return vec, word_indices

    def decode_script(self, tensor_script: torch.Tensor, word_indices: torch.Tensor):
        '''
            tensorscript: (1, seq_len)
        '''
        # remove duplicated token
        ids = tensor_script.long().tolist()
        script = []
        word = []
        ith = 0
        while ith < len(ids):
            idx = ids[ith]
            if idx not in self.special_ids:
                word.append(self.idx2phoneme[idx])
            else:
                word.append(None)
            if len(word) == 3:
                onset, rhyme, tone = word
                if rhyme:
                    _, medial, nucleus, coda = split_phoneme(rhyme)
                    word = compose_word(onset, medial, nucleus, coda, tone)
                else:
                    word = compose_word(onset, None, None, None, None)
                script.append(word)
                word = []
                ith += 1 # skip the blank token
            ith += 1

        refined_script = [] # script that is preprocessed for non Vietnamese words
        prev_id = [word_indices[0]]
        word = [script[0]]
        for ith, current_id in enumerate(word_indices[1:], start=1):
            current_token = script[ith]
            if current_id == prev_id:
                word.append(current_token)
            else:
                refined_script.append("".join(word))
                prev_id = current_id
                word = [current_token]

        refined_script = ' '.join(refined_script[1:]) # skip the bos token

        return refined_script
