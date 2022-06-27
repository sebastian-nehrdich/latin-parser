
class Writer(object):
    def __init__(self, alphabets):
        self.__source_file = None
        self.alphabets = alphabets

    def start(self, file_path):
        self.__source_file = open(file_path, 'w')

    def close(self):
        self.__source_file.close()

    def write(self, word, word_id, word_pos_id, sen_id, pos_type, case, number, gender, verb_form, full_pos, head, arc, class_tag, punc_tag, lengths, auto_label=None, symbolic_root=False, symbolic_end=False):
        batch_size, _ = word.shape
        start = 1 if symbolic_root else 0
        end = 1 if symbolic_end else 0
        for i in range(batch_size):
            for j in range(start, lengths[i] - end):
                w = self.alphabets['word_alphabet'].get_instance(word[i, j])
                pt = self.alphabets['pos_type_alphabet'].get_instance(pos_type[i, j])
                cs = self.alphabets['case_alphabet'].get_instance(case[i, j])
                nmb = self.alphabets['number_alphabet'].get_instance(number[i, j])
                gn = self.alphabets['gender_alphabet'].get_instance(gender[i, j])
                vf = self.alphabets['verb_form_alphabet'].get_instance(verb_form[i, j])
                fp = self.alphabets['full_pos_alphabet'].get_instance(full_pos[i, j])
                ct = self.alphabets['class_alphabet'].get_instance(class_tag[i, j])
                punc = self.alphabets['punc_alphabet'].get_instance(punc_tag[i, j])                                
                t = self.alphabets['arc_alphabet'].get_instance(arc[i, j])
                h = head[i, j]
                wi = word_id[i, j].item()
                si = self.alphabets['sent_id_alphabet'].get_instance(sen_id[i, j])
                wpi = self.alphabets['wordpos_id_alphabet'].get_instance(word_pos_id[i, j])
                if auto_label is not None:
                    m = self.alphabets['auto_label_alphabet'].get_instance(auto_label[i, j])
                    # gewollte Reihenfolge: j, w 
                    self.__source_file.write('%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\n' %
                                             (j,   w,  wi, si, pt, cs, nmb,gn, vf, fp, h,  t,  m,  ct, punc))
                else:
                    self.__source_file.write('%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n' %
                                             (j,   w,  wi, si, pt, cs, nmb,gn, vf, fp, h,  t,  ct, punc))

            self.__source_file.write('\n')

class Index2Instance(object):
    def __init__(self, alphabet):
        self.__alphabet = alphabet

    def index2instance(self, indices, lengths, symbolic_root=False, symbolic_end=False):
        batch_size, _ = indices.shape
        start = 1 if symbolic_root else 0
        end = 1 if symbolic_end else 0
        instnaces = []
        for i in range(batch_size):
            tmp_instances = []
            for j in range(start, lengths[i] - end):
                instamce = self.__alphabet.get_instance(indices[i, j])
                tmp_instances.append(instamce)
            instnaces.append(tmp_instances)
        return instnaces
