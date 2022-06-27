from .instance import NER_DependencyInstance
from .instance import Sentence
from .prepare_data import ROOT, END, MAX_CHAR_LENGTH

class Reader(object):
    def __init__(self, file_path, alphabets):
        self.__source_file = open(file_path, 'r')
        self.alphabets = alphabets

    def close(self):
        self.__source_file.close()

    def getNext(self, lower_case=False, symbolic_root=False, symbolic_end=False,use_aug=False):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None
        if not use_aug and "AUG" in line:
            return "SKIP"
        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            if not "#" in line:
                lines.append(line.split('\t'))                    
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        heads = []
        tokens_dict = {}
        ids_dict = {}
        word_ids = []
        sent_ids = []
        for alphabet_name in self.alphabets.keys():
            tokens_dict[alphabet_name] = []
            ids_dict[alphabet_name] = []
        if symbolic_root:
            for alphabet_name, alphabet in self.alphabets.items():
                if alphabet_name.startswith('char'):
                    tokens_dict[alphabet_name].append([ROOT, ])
                    ids_dict[alphabet_name].append([alphabet.get_index(ROOT), ])
                else:
                    tokens_dict[alphabet_name].append(ROOT)
                    ids_dict[alphabet_name].append(alphabet.get_index(ROOT))
            heads.append(0)
            word_ids.append(0)
            sent_ids.append("_")
        for tokens in lines:
            chars = []
            char_ids = []
            if lower_case:
                tokens[1] = tokens[1].lower()
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.alphabets['char_alphabet'].get_index(char))
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            tokens_dict['char_alphabet'].append(chars)
            ids_dict['char_alphabet'].append(char_ids)
            word = tokens[1]
            word_id = 0 
            try:                
                word_id = int(tokens[2])
            except ValueError:
                word_id = 0
            sent_id = tokens[3]
            word_pos_id = sent_id + ":" + str(int(tokens[0]) -1)
            pos_type = tokens[4]
            case = tokens[5]
            number = tokens[6]
            gender = tokens[7]
            verb_form = tokens[8]
            full_pos = tokens[9] 
            head = int(tokens[10])
            arc_tag = tokens[11]
            class_tag = tokens[12]
            punc_tag = tokens[13]            
            if len(tokens) > 14:
                auto_label = tokens[14]
                tokens_dict['auto_label_alphabet'].append(auto_label)
                ids_dict['auto_label_alphabet'].append(self.alphabets['auto_label_alphabet'].get_index(auto_label))
            tokens_dict['word_alphabet'].append(word)
            ids_dict['word_alphabet'].append(self.alphabets['word_alphabet'].get_index(word))
            tokens_dict['pos_type_alphabet'].append(pos_type)
            ids_dict['pos_type_alphabet'].append(self.alphabets['pos_type_alphabet'].get_index(pos_type))
            tokens_dict['case_alphabet'].append(case)
            ids_dict['case_alphabet'].append(self.alphabets['case_alphabet'].get_index(case))
            tokens_dict['sent_id_alphabet'].append(sent_id)
            ids_dict['sent_id_alphabet'].append(self.alphabets['sent_id_alphabet'].get_index(sent_id))
            tokens_dict['wordpos_id_alphabet'].append(sent_id)
            ids_dict['wordpos_id_alphabet'].append(self.alphabets['wordpos_id_alphabet'].get_index(word_pos_id))
            tokens_dict['number_alphabet'].append(number)
            ids_dict['number_alphabet'].append(self.alphabets['number_alphabet'].get_index(number))
            tokens_dict['gender_alphabet'].append(gender)
            ids_dict['gender_alphabet'].append(self.alphabets['gender_alphabet'].get_index(gender))
            tokens_dict['verb_form_alphabet'].append(verb_form)
            ids_dict['verb_form_alphabet'].append(self.alphabets['verb_form_alphabet'].get_index(verb_form))
            tokens_dict['full_pos_alphabet'].append(full_pos)
            ids_dict['full_pos_alphabet'].append(self.alphabets['full_pos_alphabet'].get_index(full_pos))            
            tokens_dict['arc_alphabet'].append(arc_tag)
            ids_dict['arc_alphabet'].append(self.alphabets['arc_alphabet'].get_index(arc_tag))
            tokens_dict['class_alphabet'].append(class_tag)
            ids_dict['class_alphabet'].append(self.alphabets['class_alphabet'].get_index(class_tag))
            tokens_dict['punc_alphabet'].append(punc_tag)
            ids_dict['punc_alphabet'].append(self.alphabets['punc_alphabet'].get_index(punc_tag))

            heads.append(head)
            word_ids.append(word_id)
        if symbolic_end:
            for alphabet_name, alphabet in self.alphabets.items():
                if alphabet_name.startswith('char'):
                    tokens_dict[alphabet_name].append([END, ])
                    ids_dict[alphabet_name].append([alphabet.get_index(END), ])
                else:
                    tokens_dict[alphabet_name] = [END]
                    ids_dict[alphabet_name] = [alphabet.get_index(END)]
            heads.append(0)

        return NER_DependencyInstance(Sentence(tokens_dict['word_alphabet'], ids_dict['word_alphabet'],
                                               tokens_dict['char_alphabet'], ids_dict['char_alphabet']),
                                      tokens_dict, ids_dict, heads, word_ids)
