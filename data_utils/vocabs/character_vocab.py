import torch

from builders.vocab_builder import META_VOCAB

@META_VOCAB.register()
class CharacterVocab:
    def __init__(self):
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        characters = [
            "a", "á", "à", "ả", "ã", "ạ",
            "ă", "ắ", "ằ", "ẳ", "ẵ", "ặ",
            "â", "ấ", "ầ", "ẩ", "ẫ", "ậ",
            "b", "c", "d", "đ", "g", "h",
            "e", "é", "è", "ẻ", "ẽ", "ẹ",
            "ê", "ế", "ề", "ể", "ễ", "ệ",
            "i", "í", "ì", "ỉ", "ĩ", "ị",
            "k", "l", "m", "n", "p", "q",
            "o", "ó", "ò", "ỏ", "õ", "ọ",
            "ô", "ố", "ồ", "ổ", "ỗ", "ộ",
            "ơ", "ớ", "ờ", "ở", "ỡ", "ợ",
            "y", "ý", "ỳ", "ỷ", "ỹ", "ỵ",
            "u", "ú", "ù", "ủ", "ũ", "ụ",
            "ư", "ứ", "ừ", "ử", "ữ", "ự",
            "r", "s", "t", "v", "x", 
            "f", "z", "w", "j"
        ]
        characters = [self.pad_token, self.bos_token, self.eos_token] +  characters
        self.character2idx = {
            phoneme: idx for idx, phoneme in enumerate(characters)
        }
        self.idx2character = {idx: phoneme for phoneme, idx in self.character2idx.items()}
        
        self.pad_idx = self.character2idx[self.pad_token]
        self.bos_idx = self.character2idx[self.bos_token]
        self.eos_idx = self.character2idx[self.eos_token]

        self.special_ids = [self.pad_idx, self.bos_idx, self.eos_idx]
    
    @property
    def size(self) -> int:
        return len(self.character2idx)

    def encode_script(self, script: str):
        words = script.split()
        tokens = [] # token in this vocab is character
        for word in words:
            tokens.extend(list(word))
            tokens.append(self.pad_token)

        token_ids = [self.character2idx[token] for token in tokens[:-1]] # skip the last blank token

        vec = torch.tensor(token_ids)

        return vec

    def decode_script(self, tensor_script: torch.Tensor):
        '''
            tensorscript: (1, seq_len)
        '''
        # remove duplicated token
        ids = tensor_script.squeeze(0).long().tolist()
        script = []
        for idx in ids:
            script.append(self.idx2character[idx])

        # script = [k for k, _ in itertools.groupby(script)]
        script = "".join(script)
        words = script.split(self.pad_token)
        words = " ".join(words)

        return words
